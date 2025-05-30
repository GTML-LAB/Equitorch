import sys

import torch_geometric
from torch_geometric.utils import segment, scatter
sys.path.append('../..')
sys.path.append('..') # Assuming test_utils is in the parent directory

import torch
from torch.autograd import Function

from equitorch.nn.functional.sparse_product import (
    sparse_mul,
    sparse_outer,
    sparse_inner,
    sparse_vecmat,
    sparse_vecsca,
    sparse_scavec,
    sparse_mat_t_vec,
)

from equitorch.structs import SparseProductInfo
# Use sparse_product_infos for generating test data structures
from equitorch.utils._structs import sparse_product_info, sparse_product_infos, prepare_so3

from equitorch.irreps import check_irreps, Irreps
from test_utils import FunctionTester, max_abs_diff_list, max_abs_diff # Import testing utilities

torch.random.manual_seed(0)
torch.set_default_dtype(torch.float64) 
# torch.set_default_dtype(torch.float32) 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'
# --- Reference Dense Implementations (using einsum) ---
# Note: These assume 'scale' is the dense reference tensor [size_out, size1, size2]
#       and x, y have shapes like (N, size1, C1), (N, size2, C2) or shared versions.
#       The einsum strings need to match the actual operation.

def ref_product_mul(x, y, out_accumulated, info_fwd, *args, **kwargs):
    x_b = x if x.ndim == 3 else x.unsqueeze(0)
    y_b = y if y.ndim == 3 else y.unsqueeze(0)

    if info_fwd.index1 is not None:
        x_b = x_b[:, info_fwd.index1] # (N, T, C1)
    if info_fwd.index2 is not None:
        y_b = y_b[:, info_fwd.index2] # (N, T, C2)

    inter = info_fwd.scale.unsqueeze(-1) * x_b * y_b
    if info_fwd.seg_out is not None:
        inter = segment(inter, info_fwd.seg_out.unsqueeze(0))
    if info_fwd.index_out is not None:
        inter = scatter(inter, info_fwd.index_out, dim=-2,
                        dim_size=info_fwd.out_size)
    if out_accumulated:
        inter = inter.sum(dim=0)
    return inter

def ref_product_outer(x, y, out_accumulated, info_fwd, *args, **kwargs):
    # x: (N, S1, C1) or (S1, C1) -> vec
    # y: (N, S2, C2) or (S2, C2) -> vec
    # out: (N, O, C1, C2) or (O, C1, C2) -> mat
    x_b = x if x.ndim == 3 else x.unsqueeze(0)
    y_b = y if y.ndim == 3 else y.unsqueeze(0)
    N = max(x_b.shape[0], y_b.shape[0])
    if x_b.shape[0] == 1: x_b = x_b.expand(N, -1, -1)
    if y_b.shape[0] == 1: y_b = y_b.expand(N, -1, -1)

    # Input indexing
    if info_fwd.index1 is not None:
        x_b = x_b[:, info_fwd.index1] # (N, T, C1)
    if info_fwd.index2 is not None:
        y_b = y_b[:, info_fwd.index2] # (N, T, C2)

    # Core operation: Outer product
    inter = x_b.unsqueeze(-1) * y_b.unsqueeze(-2) # (N, T, C1, C2)

    # Apply scaling
    if info_fwd.scale is not None:
        inter = info_fwd.scale.view(1, -1, 1, 1) * inter # (N, T, C1, C2)

    # Apply segmentation and output indexing
    if info_fwd.seg_out is not None:
        # Segment sum along the sparse dimension T -> O
        inter = segment(inter, info_fwd.seg_out.unsqueeze(0)) # (N, O, C1, C2)
    if info_fwd.index_out is not None:
        # Scatter along the sparse dimension T -> O
        # Use index_add_ for scatter sum behavior if needed, or simple scatter
        # Assuming scatter should sum contributions for the same output index
        inter = scatter(inter, info_fwd.index_out, dim=1, dim_size=info_fwd.out_size) # (N, O, C1, C2)

    # Apply output accumulation
    if out_accumulated:
        inter = inter.sum(dim=0) # (O, C1, C2)

    return inter

def ref_product_inner(x, y, out_accumulated, info_fwd, *args, **kwargs):
    # x: (N, S1, C) or (S1, C) -> vec
    # y: (N, S2, C) or (S2, C) -> vec
    # out: (N, O) or (O,) -> scalar
    x_b = x if x.ndim == 3 else x.unsqueeze(0)
    y_b = y if y.ndim == 3 else y.unsqueeze(0)
    N = max(x_b.shape[0], y_b.shape[0])
    if x_b.shape[0] == 1: x_b = x_b.expand(N, -1, -1)
    if y_b.shape[0] == 1: y_b = y_b.expand(N, -1, -1)

    # Input indexing
    if info_fwd.index1 is not None:
        x_b = x_b[:, info_fwd.index1] # (N, T, C)
    if info_fwd.index2 is not None:
        y_b = y_b[:, info_fwd.index2] # (N, T, C)

    # Core operation: Inner product (element-wise mul + sum over C)
    inter = (x_b * y_b).sum(dim=-1) # (N, T)

    # Apply scaling
    if info_fwd.scale is not None:
        inter = info_fwd.scale.view(1, -1) * inter # (N, T)

    # Apply segmentation and output indexing
    if info_fwd.seg_out is not None:
        inter = segment(inter, info_fwd.seg_out.unsqueeze(0)) # (N, O)
    if info_fwd.index_out is not None:
        inter = scatter(inter, info_fwd.index_out, dim=1, dim_size=info_fwd.out_size) # (N, O)

    # Apply output accumulation
    if out_accumulated:
        inter = inter.sum(dim=0) # (O,)

    return inter

def ref_product_vecmat(x, y, out_accumulated, info_fwd, *args, **kwargs):
    # x: (N, S1, Cin) or (S1, Cin) -> vec
    # y: (N, S2, Cin, Cout) or (S2, Cin, Cout) -> mat
    # out: (N, O, Cout) or (O, Cout) -> vec
    x_b = x if x.ndim == 3 else x.unsqueeze(0)
    y_b = y if y.ndim == 4 else y.unsqueeze(0)
    N = max(x_b.shape[0], y_b.shape[0])
    if x_b.shape[0] == 1: x_b = x_b.expand(N, -1, -1)
    if y_b.shape[0] == 1: y_b = y_b.expand(N, -1, -1, -1)

    # Input indexing
    if info_fwd.index1 is not None:
        x_b = x_b[:, info_fwd.index1] # (N, T, Cin)
    if info_fwd.index2 is not None:
        y_b = y_b[:, info_fwd.index2] # (N, T, Cin, Cout)

    # Core operation: Vec-Mat product
    inter = torch.einsum('...i,...ij->...j', x_b, y_b) # (N, T, Cout)

    # Apply scaling
    if info_fwd.scale is not None:
        inter = info_fwd.scale.view(1, -1, 1) * inter # (N, T, Cout)

    # Apply segmentation and output indexing
    if info_fwd.seg_out is not None:
        inter = segment(inter, info_fwd.seg_out.unsqueeze(0)) # (N, O, Cout)
    if info_fwd.index_out is not None:
        inter = scatter(inter, info_fwd.index_out, dim=1, dim_size=info_fwd.out_size) # (N, O, Cout)

    # Apply output accumulation
    if out_accumulated:
        inter = inter.sum(dim=0) # (O, Cout)

    return inter

def ref_product_vecsca(x, y, out_accumulated, info_fwd, *args, **kwargs):
    # x: (N, S1, C) or (S1, C) -> vec
    # y: (N, S2) or (S2,) -> scalar
    # out: (N, O, C) or (O, C) -> vec
    x_b = x if x.ndim == 3 else x.unsqueeze(0)
    y_b = y if y.ndim == 2 else y.unsqueeze(0)
    N = max(x_b.shape[0], y_b.shape[0])
    if x_b.shape[0] == 1: x_b = x_b.expand(N, -1, -1)
    if y_b.shape[0] == 1: y_b = y_b.expand(N, -1)

    # Input indexing
    if info_fwd.index1 is not None:
        x_b = x_b[:, info_fwd.index1] # (N, T, C)
    if info_fwd.index2 is not None:
        y_b = y_b[:, info_fwd.index2] # (N, T)

    # Core operation: Vec-Scale product
    inter = x_b * y_b.unsqueeze(-1) # (N, T, C)

    # Apply scaling (from info_fwd.scale)
    if info_fwd.scale is not None:
        inter = info_fwd.scale.view(1, -1, 1) * inter # (N, T, C)

    # Apply segmentation and output indexing
    if info_fwd.seg_out is not None:
        inter = segment(inter, info_fwd.seg_out.unsqueeze(0)) # (N, O, C)
    if info_fwd.index_out is not None:
        inter = scatter(inter, info_fwd.index_out, dim=1, dim_size=info_fwd.out_size) # (N, O, C)

    # Apply output accumulation
    if out_accumulated:
        inter = inter.sum(dim=0) # (O, C)

    return inter

def ref_product_scavec(x, y, out_accumulated, info_fwd, *args, **kwargs):
    # x: (N, S1) or (S1,) -> scalar
    # y: (N, S2, C) or (S2, C) -> vec
    # out: (N, O, C) or (O, C) -> vec
    x_b = x if x.ndim == 2 else x.unsqueeze(0)
    y_b = y if y.ndim == 3 else y.unsqueeze(0)
    N = max(x_b.shape[0], y_b.shape[0])
    if x_b.shape[0] == 1: x_b = x_b.expand(N, -1)
    if y_b.shape[0] == 1: y_b = y_b.expand(N, -1, -1)

    # Input indexing
    if info_fwd.index1 is not None:
        x_b = x_b[:, info_fwd.index1] # (N, T)
    if info_fwd.index2 is not None:
        y_b = y_b[:, info_fwd.index2] # (N, T, C)

    # Core operation: Scale-Vec product
    inter = x_b.unsqueeze(-1) * y_b # (N, T, C)

    # Apply scaling
    if info_fwd.scale is not None:
        inter = info_fwd.scale.view(1, -1, 1) * inter # (N, T, C)

    # Apply segmentation and output indexing
    if info_fwd.seg_out is not None:
        inter = segment(inter, info_fwd.seg_out.unsqueeze(0)) # (N, O, C)
    if info_fwd.index_out is not None:
        inter = scatter(inter, info_fwd.index_out, dim=1, dim_size=info_fwd.out_size) # (N, O, C)

    # Apply output accumulation
    if out_accumulated:
        inter = inter.sum(dim=0) # (O, C)

    return inter

def ref_product_mat_t_vec(x, y, out_accumulated, info_fwd, *args, **kwargs):
    # x: (N, S1, Cin, Cout) or (S1, Cin, Cout) -> mat
    # y: (N, S2, Cin) or (S2, Cin) -> vec
    # out: (N, O, Cout) or (O, Cout) -> vec
    x_b = x if x.ndim == 4 else x.unsqueeze(0)
    y_b = y if y.ndim == 3 else y.unsqueeze(0)
    N = max(x_b.shape[0], y_b.shape[0])
    if x_b.shape[0] == 1: x_b = x_b.expand(N, -1, -1, -1)
    if y_b.shape[0] == 1: y_b = y_b.expand(N, -1, -1)

    # Input indexing
    if info_fwd.index1 is not None:
        x_b = x_b[:, info_fwd.index1] # (N, T, Cin, Cout)
    if info_fwd.index2 is not None:
        y_b = y_b[:, info_fwd.index2] # (N, T, Cin)

    # Core operation: Mat^T-Vec product
    inter = torch.einsum('...ij,...i->...j', x_b, y_b) # (N, T, Cout)

    # Apply scaling
    if info_fwd.scale is not None:
        inter = info_fwd.scale.view(1, -1, 1) * inter # (N, T, Cout)

    # Apply segmentation and output indexing
    if info_fwd.seg_out is not None:
        inter = segment(inter, info_fwd.seg_out.unsqueeze(0)) # (N, O, Cout)
    if info_fwd.index_out is not None:
        inter = scatter(inter, info_fwd.index_out, dim=1, dim_size=info_fwd.out_size) # (N, O, Cout)

    # Apply output accumulation
    if out_accumulated:
        inter = inter.sum(dim=0) # (O, Cout)

    return inter

sparse_func_dict = {
    'mul': sparse_mul,
    'outer': sparse_outer,
    'inner': sparse_inner,
    'vecmat': sparse_vecmat,
    'vecsca': sparse_vecsca,
    'scavec': sparse_scavec,
    'mat_t_vec': sparse_mat_t_vec,
}

ref_func_dict = {
    'mul': ref_product_mul,
    'outer': ref_product_outer,
    'inner': ref_product_inner,
    'vecmat': ref_product_vecmat,
    'vecsca': ref_product_vecsca,
    'scavec': ref_product_scavec,
    'mat_t_vec': ref_product_mat_t_vec,
}

def prepare_infos(irreps1, irreps2, irreps_out,
                  indexed1, indexed2, indexed_out, device='cuda'):

    irreps1 = check_irreps(irreps1)
    irreps2 = check_irreps(irreps2)
    irreps_out = check_irreps(irreps_out)
    scale, M, M1, M2, _, _, _, _, _ = prepare_so3(irreps_out, irreps1, irreps2)
    # scale = [1 for _ in scale]
    
    size_out = len(scale) if not indexed_out else irreps_out.dim
    size1 = len(scale) if not indexed1 else irreps1.dim
    size2 = len(scale) if not indexed2 else irreps2.dim

    M1 = M1 if indexed1 else None
    M2 = M2 if indexed2 else None
    M = M if indexed_out else None
    info_f, info_b1, info_b2 = sparse_product_infos(M1, M2, M, scale, size_out, size1, size2)

    return info_f.to(device), info_b1.to(device), info_b2.to(device)

class TestWrapperModule(torch.nn.Module):
    def __init__(self, irreps1, irreps2, irreps_out, prod_type,
                 indexed1, indexed2, indexed_out,
                 out_accumulated,
                 device='cuda'):
        super().__init__() # Initialize the base class
        self.f = sparse_func_dict[prod_type]
        info_fwd, info_bwd1, info_bwd2 = prepare_infos(irreps1, irreps2, irreps_out,
                            indexed1, indexed2, indexed_out, device=device)
        self.info_fwd = info_fwd
        self.info_bwd1 = info_bwd1
        self.info_bwd2 = info_bwd2
        self.out_accumulated = out_accumulated
        # Store sizes needed for input generation
        self.size1 = info_bwd1.out_size
        self.size2 = info_bwd2.out_size
        self.out_size = info_fwd.out_size

    def forward(self, x, y):
        return self.f(x, y, self.info_fwd, self.info_bwd1, self.info_bwd2, self.out_accumulated)

class RefWrapperModule(torch.nn.Module):
    def __init__(self, irreps1, irreps2, irreps_out, prod_type,
                 indexed1, indexed2, indexed_out,
                 out_accumulated,
                 device='cuda'):
        super().__init__() # Initialize the base class
        self.f = ref_func_dict[prod_type]
        info_fwd, info_bwd1, info_bwd2 = prepare_infos(irreps1, irreps2, irreps_out,
                            indexed1, indexed2, indexed_out, device=device)
        # Reference functions expect indices on CPU potentially, keep them there?
        # Let's try keeping them on the specified device for consistency
        self.info_fwd = info_fwd
        self.info_bwd1 = info_bwd1
        self.info_bwd2 = info_bwd2
        self.out_accumulated = out_accumulated
        # Store sizes needed for input generation
        self.size1 = info_bwd1.out_size
        self.size2 = info_bwd2.out_size
        self.out_size = info_fwd.out_size


    def forward(self, x, y):
        # Ensure infos are on the same device as inputs for ref functions
        info_fwd_dev = self.info_fwd.to(x.device)
        info_bwd1_dev = self.info_bwd1.to(x.device)
        info_bwd2_dev = self.info_bwd2.to(x.device)
        return self.f(x, y, self.out_accumulated, info_fwd_dev, info_bwd1_dev, info_bwd2_dev)


# Helper function to get shapes based on product type
def get_shapes(prod_type, N,
               indexed1, indexed2, irreps1_dim, irreps2_dim, num_scales, # New args
               C_common=None, Cin=None, Cout=None, shared1=False, shared2=False):
    # Determine sparse dimension size based on indexing
    sparse_dim1 = num_scales if not indexed1 else irreps1_dim
    sparse_dim2 = num_scales if not indexed2 else irreps2_dim

    # Returns shape1, shape2, out_shape_base (without N/O)
    if prod_type == 'mul': # (C), (C) -> (C)
        shape1 = (sparse_dim1, C_common) if shared1 else (N, sparse_dim1, C_common)
        shape2 = (sparse_dim2, C_common) if shared2 else (N, sparse_dim2, C_common)
        out_shape_base = (C_common,)
    elif prod_type == 'outer': # (C1), (C2) -> (C1, C2)
        shape1 = (sparse_dim1, Cin) if shared1 else (N, sparse_dim1, Cin)       # C1 = Cin
        shape2 = (sparse_dim2, Cout) if shared2 else (N, sparse_dim2, Cout)     # C2 = Cout
        out_shape_base = (Cin, Cout)
    elif prod_type == 'inner': # (C), (C) -> ()
        shape1 = (sparse_dim1, C_common) if shared1 else (N, sparse_dim1, C_common)
        shape2 = (sparse_dim2, C_common) if shared2 else (N, sparse_dim2, C_common)
        out_shape_base = ()
    elif prod_type == 'vecmat': # (Cin), (Cin, Cout) -> (Cout)
        shape1 = (sparse_dim1, Cin) if shared1 else (N, sparse_dim1, Cin)
        shape2 = (sparse_dim2, Cin, Cout) if shared2 else (N, sparse_dim2, Cin, Cout)
        out_shape_base = (Cout,)
    elif prod_type == 'vecsca': # (C), () -> (C)
        shape1 = (sparse_dim1, C_common) if shared1 else (N, sparse_dim1, C_common)
        shape2 = (sparse_dim2,) if shared2 else (N, sparse_dim2) # Sparse dim for scalar input
        out_shape_base = (C_common,)
    elif prod_type == 'scavec': # (), (C) -> (C)
        shape1 = (sparse_dim1,) if shared1 else (N, sparse_dim1) # Sparse dim for scalar input
        shape2 = (sparse_dim2, C_common) if shared2 else (N, sparse_dim2, C_common)
        out_shape_base = (C_common,)
    elif prod_type == 'mat_t_vec': # (Cin, Cout), (Cin) -> (Cout)
        shape1 = (sparse_dim1, Cin, Cout) if shared1 else (N, sparse_dim1, Cin, Cout)
        shape2 = (sparse_dim2, Cin) if shared2 else (N, sparse_dim2, Cin)
        out_shape_base = (Cout,)
    else:
        raise ValueError(f"Unknown prod_type: {prod_type}")
    return shape1, shape2, out_shape_base

# Core Test Function
def test_sparse_op(prod_type, irreps1, irreps2, irreps_out,
                   N, C_common=None, Cin=None, Cout=None, # Channel dimensions
                   indexed1=False, indexed2=False, indexed_out=False, # Indexing flags
                   shared1=False, shared2=False, out_accumulated=False, # Batch handling flags
                   device=DEVICE):

    print(f"\n--- Testing {prod_type} ---")
    print(f"Config: N={N}, C={C_common}, Cin={Cin}, Cout={Cout}, idx=({indexed1},{indexed2},{indexed_out}), shared=({shared1},{shared2}), accum={out_accumulated}")
    dtype = torch.get_default_dtype()

    # 1. Prepare Modules and Infos
    # Need to handle potential errors during info preparation if irreps are incompatible
    # try:
    eqt_module = TestWrapperModule(irreps1, irreps2, irreps_out, prod_type,
                                    indexed1, indexed2, indexed_out,
                                    out_accumulated, device=device)
    ref_module = RefWrapperModule(irreps1, irreps2, irreps_out, prod_type,
                                    indexed1, indexed2, indexed_out,
                                    out_accumulated, device=device)
    # except ValueError as e:
    #     print(f"Skipping test due to incompatible irreps for {prod_type}: {e}")
    #     return # Skip test if irreps combination is invalid for the product

    # 2. Prepare Inputs (x, y)
    # Get irreps dimensions and number of scales needed by get_shapes
    irreps1_dim = irreps1.dim
    irreps2_dim = irreps2.dim
    num_scales = len(eqt_module.info_fwd.scale) # Number of scales/paths
    O = eqt_module.out_size # Output sparse dimension size (for reference)

    # Get shapes using the new get_shapes logic
    shape1, shape2, out_shape_base = get_shapes(
        prod_type, N,
        indexed1, indexed2, irreps1_dim, irreps2_dim, num_scales, # Pass indexing info
        C_common, Cin, Cout, shared1, shared2
    )

    x = torch.randn(*shape1, device=device, dtype=dtype)
    y = torch.randn(*shape2, device=device, dtype=dtype)

    # 3. Test Forward
    print("Forward Pass:")
    tester_fwd = FunctionTester({
        'eqt': (eqt_module, [x, y], {}),
        'ref': (ref_module, [x, y], {}),
    })
    # try:
    comp_fwd = tester_fwd.compare()
    print(comp_fwd)
        # tester_fwd.profile(repeat=10)
    # except Exception as e:
    #     print(f"Forward pass failed: {e}")
    #     # Optionally print shapes for debugging
    #     print(f"  x shape: {x.shape}, y shape: {y.shape}")
    #     try:
    #         out_eqt = eqt_module(x,y)
    #         print(f"  eqt out shape: {out_eqt.shape}")
    #     except Exception as e_eqt:
    #         print(f"  eqt forward failed: {e_eqt}")
    #     try:
    #         out_ref = ref_module(x,y)
    #         print(f"  ref out shape: {out_ref.shape}")
    #     except Exception as e_ref:
    #         print(f"  ref forward failed: {e_ref}")
    #     return # Stop if forward fails

    # 4. Test Backward
    print("Backward Pass:")
    x_eqt = x.clone().detach().requires_grad_(True)
    y_eqt = y.clone().detach().requires_grad_(True)
    x_ref = x.clone().detach().requires_grad_(True)
    y_ref = y.clone().detach().requires_grad_(True)

    # Run forward to get output shape for grad_out
    # try:
    out_ref = ref_module(x_ref, y_ref)
    out_eqt = eqt_module(x_eqt, y_eqt) # Also run eqt forward for grad calculation
    # except Exception as e:
        # print(f"Forward pass failed during backward setup: {e}")
        # return

    grad_out = torch.randn_like(out_ref) # Use output shape

    def backward_eqt(x, y, grad):
        # Need to re-run forward pass inside for autograd
        res = eqt_module(x, y)
        res.backward(grad)
        # Handle cases where input doesn't require grad
        g1 = x.grad.clone() if x.grad is not None else torch.zeros_like(x)
        g2 = y.grad.clone() if y.grad is not None else torch.zeros_like(y)
        return [g1, g2]

    def backward_ref(x, y, grad):
        # Need to re-run forward pass inside for autograd
        res = ref_module(x, y)
        res.backward(grad)
        g1 = x.grad.clone() if x.grad is not None else torch.zeros_like(x)
        g2 = y.grad.clone() if y.grad is not None else torch.zeros_like(y)
        return [g1, g2]

    tester_bwd = FunctionTester({
        'eqt': (backward_eqt, [x_eqt, y_eqt, grad_out], {}),
        'ref': (backward_ref, [x_ref, y_ref, grad_out], {}),
    })

    comp_bwd = tester_bwd.compare(compare_func=max_abs_diff_list)
    print(comp_bwd)

    # 5. Test Second-Order Backward (Grad-Grad) - Full Hessian-Vector Product
    print("Second-Order Backward Pass (Grad-Grad - Full):")
    # Need fresh tensors for gradgrad
    x_eqt_gg = x.clone().detach().requires_grad_(True)
    y_eqt_gg = y.clone().detach().requires_grad_(True)
    x_ref_gg = x.clone().detach().requires_grad_(True)
    y_ref_gg = y.clone().detach().requires_grad_(True)

    # Need vectors for grad products
    v_out = torch.randn_like(out_ref) # Vector for first grad: dL/d(out) * v_out
    # Vectors for second grad: d(dL/dx)/d(inputs) * v_in_x, etc.
    # Need to compute first grads to get shapes for v_in
    # This needs to be done *outside* no_grad to build the graph for backward_ref
    temp_x_ref_shape = x_ref.clone().detach().requires_grad_(True)
    temp_y_ref_shape = y_ref.clone().detach().requires_grad_(True)
    # Calculate actual gradients using torch.autograd.grad to get their shapes reliably
    res_ref_shape = ref_module(temp_x_ref_shape, temp_y_ref_shape)
    ref_grads = torch.autograd.grad(
        outputs=res_ref_shape,
        inputs=(temp_x_ref_shape, temp_y_ref_shape),
        grad_outputs=torch.ones_like(res_ref_shape) * v_out, # Match grad_outputs pattern
        create_graph=False, # No need for graph here, just shapes
        allow_unused=True
    )
    # Handle None grads if inputs didn't require grad or weren't used
    ref_grad_x = ref_grads[0] if ref_grads[0] is not None else torch.zeros_like(temp_x_ref_shape)
    ref_grad_y = ref_grads[1] if ref_grads[1] is not None else torch.zeros_like(temp_y_ref_shape)
    ref_grads = [ref_grad_x, ref_grad_y] # Store shapes

    del temp_x_ref_shape, temp_y_ref_shape, res_ref_shape # Clean up temporary tensors

    # Now create random vectors with the correct shapes
    v_in_x = torch.randn_like(ref_grads[0])
    v_in_y = torch.randn_like(ref_grads[1])

    def gradgrad_eqt(x, y, v_out, v_in_x, v_in_y):
        # Re-run forward pass inside for autograd
        res = eqt_module(x, y)

        # First gradient computation (dL/dx, dL/dy)
        # grad_outputs=v_out corresponds to v^T J
        grads_x, grads_y = torch.autograd.grad(
            outputs=res,
            inputs=(x, y),
            grad_outputs=v_out,
            create_graph=True, # Need graph for second derivative
            allow_unused=True # Handle cases where input doesn't affect output
        )
        # Handle None grads explicitly before summing
        grads_x = grads_x if grads_x is not None else torch.zeros_like(x)
        grads_y = grads_y if grads_y is not None else torch.zeros_like(y)

        # Second gradient computation d/d(inputs) of (dL/dx * v_in_x + dL/dy * v_in_y)
        # This computes (v_in^T H v), where H is the Hessian of the output w.r.t inputs
        # and v_in = [v_in_x, v_in_y], v = [dx, dy] (implicitly)
        # We want the gradient w.r.t. the original inputs x, y
        gradgrad_x, gradgrad_y = torch.autograd.grad(
            outputs=(grads_x, grads_y),
            inputs=(x, y),
            grad_outputs=(v_in_x, v_in_y),
            allow_unused=True
        )
        # Handle None grads
        gg_x = gradgrad_x if gradgrad_x is not None else torch.zeros_like(x)
        gg_y = gradgrad_y if gradgrad_y is not None else torch.zeros_like(y)
        return [gg_x, gg_y]

    def gradgrad_ref(x, y, v_out, v_in_x, v_in_y):
        # Re-run forward pass inside for autograd
        res = ref_module(x, y)

        # First gradient computation
        grads_x, grads_y = torch.autograd.grad(
            outputs=res,
            inputs=(x, y),
            grad_outputs=v_out,
            create_graph=True,
            allow_unused=True
        )
        grads_x = grads_x if grads_x is not None else torch.zeros_like(x)
        grads_y = grads_y if grads_y is not None else torch.zeros_like(y)

        # Second gradient computation
        gradgrad_x, gradgrad_y = torch.autograd.grad(
            outputs=(grads_x, grads_y),
            inputs=(x, y),
            grad_outputs=(v_in_x, v_in_y),
            allow_unused=True
        )
        gg_x = gradgrad_x if gradgrad_x is not None else torch.zeros_like(x)
        gg_y = gradgrad_y if gradgrad_y is not None else torch.zeros_like(y)
        return [gg_x, gg_y]

    # try:
    tester_gradgrad = FunctionTester({
        'eqt': (gradgrad_eqt, [x_eqt_gg, y_eqt_gg, v_out, v_in_x, v_in_y], {}),
        'ref': (gradgrad_ref, [x_ref_gg, y_ref_gg, v_out, v_in_x, v_in_y], {}),
    })
    comp_gradgrad = tester_gradgrad.compare(compare_func=max_abs_diff_list)
    print(comp_gradgrad)

    # 6. Test Second-Order Cross Gradient Check (d(dL/dx)/dy)
    print("Second-Order Cross Gradient Check (d(dL/dx)/dy):")
    # We want to compute d/dy of (dL/dx * v_in_x)
    # This is achieved by setting v_in_y = 0 in the gradgrad computation
    # and looking at the second output (gradient w.r.t. y)
    v_in_y_zero = torch.zeros_like(v_in_y)
    # Need fresh tensors as grad accumulates
    x_eqt_gg_dxdy = x.clone().detach().requires_grad_(True)
    y_eqt_gg_dxdy = y.clone().detach().requires_grad_(True)
    x_ref_gg_dxdy = x.clone().detach().requires_grad_(True)
    y_ref_gg_dxdy = y.clone().detach().requires_grad_(True)

    tester_gradgrad_dxdy = FunctionTester({
        'eqt': (lambda: gradgrad_eqt(x_eqt_gg_dxdy, y_eqt_gg_dxdy, v_out, v_in_x, v_in_y_zero)[1], [], {}), # Get only grad w.r.t y
        'ref': (lambda: gradgrad_ref(x_ref_gg_dxdy, y_ref_gg_dxdy, v_out, v_in_x, v_in_y_zero)[1], [], {}), # Get only grad w.r.t y
    })
    comp_gradgrad_dxdy = tester_gradgrad_dxdy.compare(compare_func=max_abs_diff) # Compare single tensors
    print(comp_gradgrad_dxdy)

    # 7. Test Second-Order Cross Gradient Check (d(dL/dy)/dx)
    print("Second-Order Cross Gradient Check (d(dL/dy)/dx):")
    # We want to compute d/dx of (dL/dy * v_in_y)
    # This is achieved by setting v_in_x = 0 in the gradgrad computation
    # and looking at the first output (gradient w.r.t. x)
    v_in_x_zero = torch.zeros_like(v_in_x)
    # Need fresh tensors as grad accumulates
    x_eqt_gg_dydx = x.clone().detach().requires_grad_(True)
    y_eqt_gg_dydx = y.clone().detach().requires_grad_(True)
    x_ref_gg_dydx = x.clone().detach().requires_grad_(True)
    y_ref_gg_dydx = y.clone().detach().requires_grad_(True)

    tester_gradgrad_dydx = FunctionTester({
        'eqt': (lambda: gradgrad_eqt(x_eqt_gg_dydx, y_eqt_gg_dydx, v_out, v_in_x_zero, v_in_y)[0], [], {}), # Get only grad w.r.t x
        'ref': (lambda: gradgrad_ref(x_ref_gg_dydx, y_ref_gg_dydx, v_out, v_in_x_zero, v_in_y)[0], [], {}), # Get only grad w.r.t x
    })
    comp_gradgrad_dydx = tester_gradgrad_dydx.compare(compare_func=max_abs_diff) # Compare single tensors
    print(comp_gradgrad_dydx)

    # except Exception as e:
    #     print(f"Second-order backward pass failed: {e}")
    #     # Optionally print shapes for debugging
    #     print(f"  x shape: {x.shape}, y shape: {y.shape}")
    #     print(f"  grad_out shape: {grad_out.shape}")
    #     try:
    #         grads_eqt = backward_eqt(x_eqt, y_eqt, grad_out)
    #         print(f"  eqt grads shapes: {[g.shape for g in grads_eqt]}")
    #     except Exception as e_eqt:
    #         print(f"  eqt backward failed: {e_eqt}")
    #     try:
    #         grads_ref = backward_ref(x_ref, y_ref, grad_out)
    #         print(f"  ref grads shapes: {[g.shape for g in grads_ref]}")
    #     except Exception as e_ref:
    #         print(f"  ref backward failed: {e_ref}")


# --- Define Test Cases ---
if __name__ == "__main__":
    # N_test = 9 # Small batch size for testing
    C_test = 97 # Common channel dim
    N_test = 135
    C1 = 35
    C2 = 21

    irreps1 = irreps2 = irreps = Irreps((1,3))

    # irreps1 = Irreps('7+1')
    # irreps2 = Irreps('1+12')
    # irreps = Irreps('5+1+3')

    # Test configurations
    configs = []
    prod_types = ['mul', 'outer', 'inner', 'vecmat', 'vecsca', 'scavec', 'mat_t_vec']
    indexing_flags = [False, True] # For indexed1, indexed2, indexed_out
    batch_flags = [False, True] # For shared1, shared2, out_accumulated

    # prod_types = ['mat_t_vec']

    for prod in prod_types:
        for idx1 in indexing_flags:
            for idx2 in indexing_flags:
                for idx_out in indexing_flags:
                    for sh1 in batch_flags:
                        for sh2 in batch_flags:
                            # print(prod, idx1, idx2, idx_out, sh1, sh2)
                            for acc in batch_flags:
                                if not idx1 and not idx2 and not idx_out:
                                    continue
                                if (sh1 + sh2 + acc) >= 2:
                                    continue
                                # Determine appropriate irreps based on product type
                                # Use consistent channel dimensions for simplicity where possible
                                if prod == 'mul': ir1, ir2, ir_out = irreps1, irreps2, irreps; c_args={'C_common': C_test}
                                elif prod == 'outer': ir1, ir2, ir_out = irreps1, irreps2, irreps; c_args={'Cin': C1, 'Cout': C2} # C1=3, C2=9 -> C1,C2 = 3,9
                                elif prod == 'inner': ir1, ir2, ir_out = irreps1, irreps2, irreps; c_args={'C_common': C_test} # Use C_test for channel dim
                                elif prod == 'vecmat': ir1, ir2, ir_out = irreps1, irreps2, irreps; c_args={'Cin': C1, 'Cout': C2} # Cin=3, Cout=4
                                elif prod == 'vecsca': ir1, ir2, ir_out = irreps1, irreps2, irreps; c_args={'C_common': C_test} # Use C_test for channel dim
                                elif prod == 'scavec': ir1, ir2, ir_out = irreps1, irreps2, irreps; c_args={'C_common': C_test} # Use C_test for channel dim
                                elif prod == 'mat_t_vec': ir1, ir2, ir_out = irreps1, irreps2, irreps; c_args={'Cin': C1, 'Cout': C2} # Cin=3, Cout=4

                                # Skip invalid combinations (e.g., cannot accumulate if both inputs shared)
                                if acc and sh1 and sh2: continue
                                # Skip if indexed requires specific irreps dims (less critical here as prepare_so3 handles it)

                                configs.append({
                                    'prod_type': prod,
                                    'irreps1': ir1, 'irreps2': ir2, 'irreps_out': ir_out,
                                    'N': N_test, **c_args,
                                    'indexed1': idx1, 'indexed2': idx2, 'indexed_out': idx_out,
                                    'shared1': sh1, 'shared2': sh2, 'out_accumulated': acc,
                                    'device': DEVICE
                                })

    # Run tests
    # test_sparse_op(**configs[3])
    for config in configs:
        print(config)
    #     # try:
        test_sparse_op(**config)
        # except Exception as e:
        #     print('=============================================')
        #     print(e)
        #     print('=============================================')
