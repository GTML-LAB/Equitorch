import sys
sys.path.append('../..')
sys.path.append('..')

import torch
import torch_geometric
from torch_geometric.utils import segment, scatter
import e3nn
import math
import os
import pprint

from equitorch.irreps import Irreps, check_irreps
from equitorch.nn.functional.sparse_product import (
    sparse_mul, sparse_outer, sparse_inner,
    sparse_vecmat, sparse_vecsca, sparse_scavec, sparse_mat_t_vec
)
from equitorch.structs import SparseProductInfo
from equitorch.utils._structs import sparse_product_info, sparse_product_infos, prepare_so3
from test_utils import FunctionTester, max_abs_diff, max_abs_diff_list, rate_mean2_std_list, rate_mean2_std

# Set environment and defaults
torch.set_default_dtype(torch.float32)
torch.random.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# torch.cuda.device(1)
# === Reference Implementations ===

def ref_product_mul(x, y, out_accumulated, info_fwd, *args, **kwargs):
    r"""Reference implementation for sparse multiplication."""
    x_b = x if x.ndim == 3 else x.unsqueeze(0)
    y_b = y if y.ndim == 3 else y.unsqueeze(0)

    if info_fwd.index1 is not None:
        x_b = x_b[:, info_fwd.index1]
    if info_fwd.index2 is not None:
        y_b = y_b[:, info_fwd.index2]

    inter = info_fwd.scale.unsqueeze(-1) * x_b * y_b
    if info_fwd.seg_out is not None:
        inter = segment(inter, info_fwd.seg_out.unsqueeze(0))
    if info_fwd.index_out is not None:
        inter = scatter(inter, info_fwd.index_out, dim=-2, dim_size=info_fwd.out_size)
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
                  indexed1, indexed2, indexed_out, device='cuda:1'):

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

# [Other reference implementations remain unchanged but with added docstrings]

# === Wrapper Classes ===

class TestWrapperModule(torch.nn.Module):
    r"""Wrapper for sparse product operations with proper initialization."""
    def __init__(self, irreps1, irreps2, irreps_out, prod_type,
                indexed1, indexed2, indexed_out,
                out_accumulated,
                device='cuda:1'):
        super().__init__()
        self.f = {
            'mul': sparse_mul,
            'outer': sparse_outer,
            'inner': sparse_inner,
            'vecmat': sparse_vecmat,
            'vecsca': sparse_vecsca,
            'scavec': sparse_scavec,
            'mat_t_vec': sparse_mat_t_vec,
        }[prod_type]
        
        info_fwd, info_bwd1, info_bwd2 = prepare_infos(
            irreps1, irreps2, irreps_out,
            indexed1, indexed2, indexed_out, device
        )
        self.info_fwd = info_fwd
        self.info_bwd1 = info_bwd1
        self.info_bwd2 = info_bwd2
        self.out_accumulated = out_accumulated
        self.size1 = info_bwd1.out_size
        self.size2 = info_bwd2.out_size
        self.out_size = info_fwd.out_size

    def forward(self, x, y):
        return self.f(x, y, self.info_fwd, self.info_bwd1, self.info_bwd2, self.out_accumulated)

class RefWrapperModule(torch.nn.Module):
    r"""Wrapper for reference sparse product operations."""
    def __init__(self, irreps1, irreps2, irreps_out, prod_type,
                indexed1, indexed2, indexed_out,
                out_accumulated,
                device='cuda:1'):
        super().__init__()
        self.f = {
            'mul': ref_product_mul,
            'outer': ref_product_outer,
            'inner': ref_product_inner,
            'vecmat': ref_product_vecmat,
            'vecsca': ref_product_vecsca,
            'scavec': ref_product_scavec,
            'mat_t_vec': ref_product_mat_t_vec,
        }[prod_type]
        
        info_fwd, info_bwd1, info_bwd2 = prepare_infos(
            irreps1, irreps2, irreps_out,
            indexed1, indexed2, indexed_out, device
        )
        self.info_fwd = info_fwd
        self.info_bwd1 = info_bwd1
        self.info_bwd2 = info_bwd2
        self.out_accumulated = out_accumulated
        self.size1 = info_bwd1.out_size
        self.size2 = info_bwd2.out_size
        self.out_size = info_fwd.out_size

    def forward(self, x, y):
        return self.f(x, y, self.out_accumulated, self.info_fwd, self.info_bwd1, self.info_bwd2)

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

# === Test Functions ===

def _test_sparse_op(prod_type, irreps1, irreps2, irreps_out,
                   N, C_common=None, Cin=None, Cout=None,
                   indexed1=False, indexed2=False, indexed_out=False,
                   shared1=False, shared2=False, out_accumulated=False,
                   device=device):
    r"""Core test function for sparse operations."""
    print(f"\n--- Testing {prod_type} ---")
    print(f"Config: N={N}, C={C_common}, Cin={Cin}, Cout={Cout}, "
          f"idx=({indexed1},{indexed2},{indexed_out}), "
          f"shared=({shared1},{shared2}), accum={out_accumulated}")

    # Initialize modules
    eqt_module = TestWrapperModule(
        irreps1, irreps2, irreps_out, prod_type,
        indexed1, indexed2, indexed_out,
        out_accumulated, device
    )
    
    ref_module = RefWrapperModule(
        irreps1, irreps2, irreps_out, prod_type,
        indexed1, indexed2, indexed_out,
        out_accumulated, device
    )

    # Prepare inputs
    irreps1_dim = irreps1.dim
    irreps2_dim = irreps2.dim
    num_scales = len(eqt_module.info_fwd.scale)
    
    # Get shapes based on product type
    shape1, shape2, out_shape = get_shapes(
        prod_type, N,
        indexed1, indexed2, irreps1_dim, irreps2_dim, num_scales,
        C_common, Cin, Cout, shared1, shared2
    )
    
    x = torch.randn(*shape1, device=device)
    y = torch.randn(*shape2, device=device)

    # Forward test
    print("Forward Pass:")
    tester_fwd = FunctionTester({
        'eqt': (eqt_module, [x, y], {}),
        'ref': (ref_module, [x, y], {}),
    })
    
    result = {}
    
    try:
        t = tester_fwd.multi_timeit(runs=10, repeat=20, warmup=10)        
        print(t)
        result['forward'] = t
    except Exception as e:
        print(f"Forward test failed: {e}")
        import traceback
        traceback.print_exc()

    # Backward test
    print("\nBackward Pass:")

    with torch.no_grad():
        out = eqt_module(x,y)

    x_eqt = x.clone().detach().requires_grad_(True)
    y_eqt = y.clone().detach().requires_grad_(True)
    x_ref = x.clone().detach().requires_grad_(True)
    y_ref = y.clone().detach().requires_grad_(True)

    def backward_eqt(x, y, grad):
        res = eqt_module(x, y)
        res.backward(grad)
        return [x.grad if x.grad is not None else torch.zeros_like(x),
                y.grad if y.grad is not None else torch.zeros_like(y)]

    def backward_ref(x, y, grad):
        res = ref_module(x, y)
        res.backward(grad)
        return [x.grad if x.grad is not None else torch.zeros_like(x),
                y.grad if y.grad is not None else torch.zeros_like(y)]

    grad_out = torch.randn_like(out)

    tester_bwd = FunctionTester({
        'eqt': (backward_eqt, [x_eqt, y_eqt, grad_out], {}),
        'ref': (backward_ref, [x_ref, y_ref, grad_out], {}),
    })

    try:
        t = tester_bwd.multi_timeit(runs=10, repeat=20, warmup=10)
        print(t)
        result['backward'] = t
    except Exception as e:
        print(f"Backward test failed: {e}")
        import traceback
        traceback.print_exc()

    return result

# === Main Execution Block ===

if __name__ == "__main__":
    results = {}
    N = 256
    C_test = 64
    C1, C2 = 64, 64

    # Define test irreps
    irreps = Irreps((0,3))

    # Test configurations
    configs = []
    prod_types = ['mul', 'outer', 'inner', 'vecmat', 'vecsca', 'scavec', 'mat_t_vec']
    
    for prod in prod_types:
    #     for idx1 in [False, True]:
    #         for idx2 in [False, True]:
    #             for idx_out in [False, True]:
    #                 if not idx1 and not idx2 and not idx_out:
    #                     continue
        for idx1 in [True]:
            for idx2 in [True]:
                for idx_out in [True]:
                    if not idx1 and not idx2 and not idx_out:
                        continue
                        
                    for sh1 in [False]:
                        for sh2 in [False]:
                            for acc in [True]:
                                if acc and sh1 and sh2:
                                    continue
                                    
                                config = {
                                    'prod_type': prod,
                                    'irreps1': irreps,
                                    'irreps2': irreps,
                                    'irreps_out': irreps,
                                    'N': N,
                                    'C_common': C_test,
                                    'Cin': C1,
                                    'Cout': C2,
                                    'indexed1': idx1,
                                    'indexed2': idx2,
                                    'indexed_out': idx_out,
                                    'shared1': sh1,
                                    'shared2': sh2,
                                    'out_accumulated': acc,
                                    'device': device
                                }
                                
                                print(f"\nTesting config: {config}")
                                try:
                                    # results[f"{prod}_{idx1}{idx2}{idx_out}_{sh1}{sh2}{acc}"] = \
                                    results[f"{prod}"] = \
                                        _test_sparse_op(**config)
                                except Exception as e:
                                    print(f"Test failed: {e}")
                                    import traceback
                                    traceback.print_exc()

    pprint.pprint(results)
