import sys

import torch_geometric
from torch_geometric.utils import segment, scatter
sys.path.append('../..')
sys.path.append('..') # Assuming test_utils is in the parent directory

import torch
from torch.autograd import Function

from equitorch.nn.functional.sparse_scale import (
    sparse_scale
)
from equitorch.structs import (
    SparseScaleInfo
)

from equitorch.utils._structs import sparse_scale_infos, prepare_so3
from equitorch.irreps import check_irreps, Irreps
from test_utils import FunctionTester, max_abs_diff_list, max_abs_diff # Import testing utilities

torch.random.manual_seed(0)
# torch.set_default_dtype(torch.float64)
torch.set_default_dtype(torch.float32)
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'
def ref_scale(input, info_fwd: SparseScaleInfo, *args, **kwargs):
    # Ensure input is batched for consistency if needed, though ref logic might handle it
    input_b = input if input.ndim == 3 else input.unsqueeze(0) # Assume (N, S, C) or (S, C)

    inter = input_b
    if info_fwd.index is not None:
        # Select along the sparse dimension (-2)
        inter = inter.index_select(-2, info_fwd.index) # (N, T, C) or (T, C)

    # Apply scaling
    inter = inter * info_fwd.scale.unsqueeze(-1) # (N, T, C) or (T, C)

    # Apply segmentation (summing contributions for the same output segment)
    if info_fwd.seg_out is not None:
        # segment expects input, segment_ids, num_segments (optional)
        # Assuming seg corresponds to the output segments after scaling
        # The sparse dimension is -2
        inter = segment(inter, info_fwd.seg_out.unsqueeze(0)) # (N, O, C) or (O, C)

    # Note: The original ref_scale had index_out and scatter, which seems incorrect
    # for a scale operation followed by segment. Segment already aggregates.
    # If scatter is needed, the logic of info generation or ref_scale needs clarification.
    # Assuming segment is the correct aggregation here based on indexed_scale_segment op.

    # Remove batch dim if input was not batched
    if input.ndim == 2:
        inter = inter.squeeze(0)

    return inter

def prepare_infos(irreps1, irreps2, irreps_out, device='cuda'):
    # Note: irreps1 and irreps2 are used by prepare_so3 to determine the scale structure,
    # even though sparse_scale itself only operates on one input tensor derived from irreps1*irreps2.
    irreps1 = check_irreps(irreps1)
    irreps2 = check_irreps(irreps2)
    irreps_out = check_irreps(irreps_out)
    scale, M, M1, M2, _, _, _, _, _ = prepare_so3(irreps_out, irreps1, irreps2)
    # M_in represents the indices in the flattened irreps1 x irreps2 space
    M_in = [M1_ * irreps2.dim + M2_ for M1_, M2_ in zip(M1,M2)]

    # sparse_scale_infos expects input indices (M_in), output indices (M), scale, out_size, in_size
    info_f, info_b = sparse_scale_infos(M_in, M, scale, irreps_out.dim, irreps1.dim*irreps2.dim)
    return info_f.to(device), info_b.to(device)

class TestWrapperModule(torch.nn.Module):
    def __init__(self, irreps1, irreps2, irreps_out, 
                 device='cuda'):
        super().__init__() # Initialize the base class
        info_fwd, info_bwd = prepare_infos(irreps1, irreps2, irreps_out, device=device)
        self.info_fwd = info_fwd
        self.info_bwd = info_bwd
        # The input size for sparse_scale corresponds to the output size of the backward info
        self.in_size = info_bwd.out_size
        self.out_size = info_fwd.out_size

    def forward(self, input):
        # sparse_scale expects input, info_fwd, info_bwd
        return sparse_scale(input, self.info_fwd, self.info_bwd)

# --- Reference Implementation Wrapper ---
class RefWrapperModule(torch.nn.Module):
    def __init__(self, irreps1, irreps2, irreps_out,
                 device='cuda'):
        super().__init__()
        info_fwd, info_bwd = prepare_infos(irreps1, irreps2, irreps_out, device=device)
        # Reference function only needs forward info
        self.info_fwd = info_fwd
        self.in_size = info_bwd.out_size # Keep for consistency in shape generation
        self.out_size = info_fwd.out_size

    def forward(self, input):
        # Ensure info is on the same device as input for ref function
        info_fwd_dev = self.info_fwd.to(input.device)
        return ref_scale(input, info_fwd_dev)

# --- Helper function to get shapes ---
def get_shapes(N, in_size, C, shared=False):
    # input shape: (N, in_size, C) or (in_size, C) if shared
    # output shape base: (out_size, C) - out_size determined by info_fwd
    shape = (in_size, C) if shared else (N, in_size, C)
    # Output shape base doesn't include N or the sparse dim (O) which is module-dependent
    out_shape_base = (C,)
    return shape, out_shape_base

# --- Core Test Function ---
def test_sparse_scale(irreps1, irreps2, irreps_out,
                      N, C, # Batch and Channel dimensions
                      shared=False, # Batch handling flag
                      device=DEVICE):

    print(f"\n--- Testing sparse_scale ---")
    print(f"Config: N={N}, C={C}, shared={shared}")
    print(f"Irreps: In1={irreps1}, In2={irreps2}, Out={irreps_out}") # In1*In2 -> Out conceptually
    dtype = torch.get_default_dtype()

    # 1. Prepare Modules
    eqt_module = TestWrapperModule(irreps1, irreps2, irreps_out, device=device)
    ref_module = RefWrapperModule(irreps1, irreps2, irreps_out, device=device)

    # 2. Prepare Input (x)
    in_size = eqt_module.in_size # Get the required input sparse dimension size
    O = eqt_module.out_size # Output sparse dimension size (for reference)

    shape, out_shape_base = get_shapes(N, in_size, C, shared)
    x = torch.randn(*shape, device=device, dtype=dtype)

    # 3. Test Forward
    print("Forward Pass:")
    tester_fwd = FunctionTester({
        'eqt': (eqt_module, [x], {}),
        'ref': (ref_module, [x], {}),
    })
    comp_fwd = tester_fwd.compare()
    print(comp_fwd)
    # tester_fwd.profile(repeat=10) # Optional profiling

    # 4. Test Backward
    print("Backward Pass:")
    x_eqt = x.clone().detach().requires_grad_(True)
    x_ref = x.clone().detach().requires_grad_(True)

    out_ref = ref_module(x_ref)
    out_eqt = eqt_module(x_eqt) # Also run eqt forward for grad calculation

    grad_out = torch.randn_like(out_ref) # Use reference output shape

    def backward_eqt(x, grad):
        res = eqt_module(x)
        res.backward(grad)
        g = x.grad.clone() if x.grad is not None else torch.zeros_like(x)
        return [g] # Return as list for consistency with tester

    def backward_ref(x, grad):
        res = ref_module(x)
        res.backward(grad)
        g = x.grad.clone() if x.grad is not None else torch.zeros_like(x)
        return [g] # Return as list

    tester_bwd = FunctionTester({
        'eqt': (backward_eqt, [x_eqt, grad_out], {}),
        'ref': (backward_ref, [x_ref, grad_out], {}),
    })

    comp_bwd = tester_bwd.compare(compare_func=max_abs_diff_list)
    print(comp_bwd)

    # Do not test gradgrad for linear operation

    # # 5. Test Second-Order Backward (Grad-Grad)
    # print("Second-Order Backward Pass (Grad-Grad):")
    # x_eqt_gg = x.clone().detach().requires_grad_(True)
    # x_ref_gg = x.clone().detach().requires_grad_(True)

    # # Need vector for grad product: dL/d(out) * v_out
    # v_out = torch.randn_like(out_ref)
    # # Need vector for second grad: d(dL/dx)/d(input) * v_in_x
    # # Compute first grad shape
    # temp_x_ref_shape = x_ref.clone().detach().requires_grad_(True)
    # res_ref_shape = ref_module(temp_x_ref_shape)
    # ref_grads = torch.autograd.grad(
    #     outputs=res_ref_shape,
    #     inputs=temp_x_ref_shape,
    #     grad_outputs=torch.ones_like(res_ref_shape) * v_out,
    #     create_graph=False,
    #     allow_unused=True
    # )
    # ref_grad_x = ref_grads[0] if ref_grads[0] is not None else torch.zeros_like(temp_x_ref_shape)
    # del temp_x_ref_shape, res_ref_shape # Clean up

    # v_in_x = torch.randn_like(ref_grad_x)

    # def gradgrad_eqt(x, v_out, v_in_x):
    #     res = eqt_module(x)
    #     grads_x, = torch.autograd.grad( # Unpack tuple
    #         outputs=res,
    #         inputs=x,
    #         grad_outputs=v_out,
    #         create_graph=True,
    #         allow_unused=True
    #     )
    #     grads_x = grads_x if grads_x is not None else torch.zeros_like(x)

    #     gradgrad_x, = torch.autograd.grad( # Unpack tuple
    #         outputs=grads_x,
    #         inputs=x,
    #         grad_outputs=v_in_x,
    #         allow_unused=True
    #     )
    #     gg_x = gradgrad_x if gradgrad_x is not None else torch.zeros_like(x)
    #     return [gg_x] # Return as list

    # def gradgrad_ref(x, v_out, v_in_x):
    #     res = ref_module(x)
    #     grads_x, = torch.autograd.grad(
    #         outputs=res,
    #         inputs=x,
    #         grad_outputs=v_out,
    #         create_graph=True,
    #         allow_unused=True
    #     )
    #     grads_x = grads_x if grads_x is not None else torch.zeros_like(x)

    #     gradgrad_x, = torch.autograd.grad(
    #         outputs=grads_x,
    #         inputs=x,
    #         grad_outputs=v_in_x,
    #         allow_unused=True
    #     )
    #     gg_x = gradgrad_x if gradgrad_x is not None else torch.zeros_like(x)
    #     return [gg_x] # Return as list

    # tester_gradgrad = FunctionTester({
    #     'eqt': (gradgrad_eqt, [x_eqt_gg, v_out, v_in_x], {}),
    #     'ref': (gradgrad_ref, [x_ref_gg, v_out, v_in_x], {}),
    # })

    # comp_gradgrad = tester_gradgrad.compare(compare_func=max_abs_diff_list)
    # print(comp_gradgrad)

# --- Define Test Cases ---
if __name__ == "__main__":
    N_test = 256 # Small batch size
    C_test = 128 # Channel dim

    # Define some irreps. The specific combination determines the scale structure.
    irreps1_test = Irreps('1e+2e') # Example input 1 irreps
    irreps2_test = Irreps('1e+2e') # Example input 2 irreps
    irreps_out_test = Irreps((0,4)) # Example output irreps
    # irreps1_test = Irreps('5') # Example input 1 irreps
    # irreps2_test = Irreps('0') # Example input 2 irreps
    # irreps_out_test = Irreps('5') # Example output irreps

    # Define a single configuration to test
    config = {
        'irreps1': irreps1_test,
        'irreps2': irreps2_test,
        'irreps_out': irreps_out_test,
        'N': N_test,
        'C': C_test,
        'shared': False, # Test non-shared batch dimension first
        'device': DEVICE
    }

    print("Running test with config:")
    print(config)
    test_sparse_scale(**config)

    # Example for shared batch dimension (optional)
    # config_shared = config.copy()
    # config_shared['shared'] = True
    # print("\nRunning test with shared config:")
    # print(config_shared)
    # test_sparse_scale(**config_shared)
