import torch
from torch import Tensor
import math

from equitorch.irreps import Irrep, Irreps, check_irreps
from equitorch.structs import IrrepsInfo
# Import the target function
from equitorch.nn.functional.norm import batch_mean_squared_norm
# Import Separable for reference
from equitorch.nn.others import Separable
from equitorch.utils._structs import irreps_info

# Import testing utilities
from test_utils import FunctionTester, max_abs_diff_list, max_abs_diff

# torch.set_default_dtype(torch.float64)
torch.set_default_dtype(torch.float32)
# DEVICE = 'cpu'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Reference Implementation ---

class BatchMeanSqNormPerIrrep(torch.nn.Module):
    r"""Computes mean(dim=0) of sum_sq(dim=-2) for a single irrep block."""
    def __init__(self, rsqrt_dim_val_sq: float, scaled: bool):
        super().__init__()
        self.rsqrt_dim_val_sq = rsqrt_dim_val_sq # This is 1/D_k
        self.scaled = scaled

    def forward(self, x_k: Tensor) -> Tensor:
        # x_k shape: (N, ..., D_k, C)
        N = x_k.shape[0]
        sum_sq_comp = torch.sum(x_k.pow(2), dim=-2, keepdim=True)
        sum_sq_batch = sum_sq_comp.sum(dim=0) # Keep the irrep dim (size 1)
        mean_sq_batch = sum_sq_batch / N
        if self.scaled:
            mean_sq_batch = mean_sq_batch * self.rsqrt_dim_val_sq
        return mean_sq_batch # Shape (..., 1, C)

class RefBatchMeanSquaredNorm(torch.nn.Module):
    r"""Reference implementation using Separable"""
    def __init__(self, irreps: Irreps, scaled: bool):
        super().__init__()
        self.irreps_obj = check_irreps(irreps)
        self.scaled = scaled

        rsqrt_dims_sq_vals = [(1.0 / ir.dim) if ir.dim > 0 else 0.0 for ir in self.irreps_obj]

        sub_modules = []
        for i in range(len(self.irreps_obj)):
            sub_modules.append(
                BatchMeanSqNormPerIrrep(rsqrt_dims_sq_vals[i], self.scaled)
            )
        
        self.separable_op = Separable(
            irreps=self.irreps_obj,
            split_num_irreps=[1] * len(self.irreps_obj), # one module per Irrep instance
            sub_modules=sub_modules, 
            cat_after=True, # Concatenate results
            dim=-2 # Specify irreps dim for split/cat
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        # Input: (N, ..., M, C)
        # Output of separable_op: (..., len(irreps), C)
        return self.separable_op(input_tensor)


# --- Equitorch Implementation ---
class EqtBatchMeanSquaredNormModule(torch.nn.Module):
    def __init__(self, irreps: Irreps, scaled: bool):
        super().__init__()
        self.irreps_obj = check_irreps(irreps)
        self.scaled = scaled
        self.irreps_info = irreps_info(self.irreps_obj) # Contains rsqrt_dims

    def forward(self, input_tensor: Tensor) -> Tensor:
        # Call the functional implementation for batch mean squared norm
        return batch_mean_squared_norm(input_tensor, self.irreps_info, self.scaled)

    def _apply(self, *args, **kwargs):
        # Handles .to(device), .cuda(), .float(), etc.
        n = super()._apply(*args, **kwargs)
        if hasattr(n, 'irreps_info') and n.irreps_info is not None:
            n.irreps_info = n.irreps_info._apply(*args, **kwargs)
        return n

# --- Test Suite ---
def run_tests(scaled_test: bool):
    print(f"\n--- Testing BatchMeanSquaredNorm with scaled={scaled_test} ---")
    
    irreps_str = "1x0e + 2x1o + 1x2e" # Dim: 1*1 + 2*3 + 1*5 = 12
    irreps = Irreps(irreps_str)

    print(f"Irreps: {irreps} (dim={irreps.dim}, num_irreps={len(irreps)})")
    print(f"Device: {DEVICE}")
    print(f"Dtype: {torch.get_default_dtype()}")

    N = 256 # Batch size
    C = 64  # Channels

    print(f"\n--- Testing with N={N}, C={C} ---")

    ref_module = RefBatchMeanSquaredNorm(irreps, scaled=scaled_test).to(device=DEVICE)
    eqt_module = EqtBatchMeanSquaredNormModule(irreps, scaled=scaled_test).to(device=DEVICE)

    # Test with standard shape
    x = torch.randn(N, irreps.dim, C, device=DEVICE)

    # Add some zero vectors
    if irreps.dim > 0 and C > 0: 
        x[0, :irreps[0].dim, :] = 0.0 
        if len(irreps) > 1:
                x[1, irreps[0].dim : irreps[0].dim + irreps[1].dim, 0 % C if C > 0 else 0] = 0.0
    
    x.requires_grad_(True)

    # --- Test Standard Shape ---
    print("\n--- Testing Standard Shape (N, D, C) ---")
    out_ref = ref_module(x) # Output shape (num_irreps, C) - assuming no intermediate dims
    grad_out = torch.randn_like(out_ref)
    
    tester_fwd = FunctionTester({'eqt': (eqt_module, [x], {}), 'ref': (ref_module, [x], {})})
    comp_fwd = tester_fwd.compare(compare_func=max_abs_diff); print("Forward:", comp_fwd)
    assert comp_fwd[('eqt', 'ref')] < 1e-6, "Forward failed"

    def backward_call(module, inp, grad):
        inp_clone = inp.clone().detach().requires_grad_(True)
        res = module(inp_clone); res.backward(grad, retain_graph=True)
        g_x = inp_clone.grad.clone() if inp_clone.grad is not None else torch.zeros_like(inp_clone)
        return [g_x]
        
    tester_bwd = FunctionTester({'eqt': (backward_call, [eqt_module, x, grad_out], {}), 'ref': (backward_call, [ref_module, x, grad_out], {})})
    comp_bwd = tester_bwd.compare(compare_func=max_abs_diff_list); print("Backward:", comp_bwd)
    assert all(c < 1e-6 for c in comp_bwd[('eqt', 'ref')]), "Backward failed"

    v_out = torch.randn_like(out_ref)
    grad_x_shape = backward_call(ref_module, x, torch.ones_like(out_ref))[0]
    v_in_x = torch.randn_like(grad_x_shape)

    def gradgrad_call(module, inp, v_out, v_in):
        inp_clone = inp.clone().detach().requires_grad_(True)
        res = module(inp_clone)
        grads, = torch.autograd.grad(res, inp_clone, v_out, create_graph=True, allow_unused=True)
        grads = grads if grads is not None else torch.zeros_like(inp_clone)
        gradgrads, = torch.autograd.grad(grads, inp_clone, v_in, allow_unused=True)
        return [gradgrads if gradgrads is not None else torch.zeros_like(inp_clone)]

    tester_gg = FunctionTester({'eqt': (gradgrad_call, [eqt_module, x, v_out, v_in_x], {}), 'ref': (gradgrad_call, [ref_module, x, v_out, v_in_x], {})})
    comp_gg = tester_gg.compare(compare_func=max_abs_diff_list); print("GradGrad:", comp_gg)
    assert all(c < 1e-6 for c in comp_gg[('eqt', 'ref')]), "GradGrad failed"


    print(f"\nAll tests for BatchMeanSquaredNorm with scaled={scaled_test} passed.")

if __name__ == "__main__":
    run_tests(scaled_test=True)
    run_tests(scaled_test=False)
    print("\nAll BatchMeanSquaredNorm tests finished.")
