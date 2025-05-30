import sys


sys.path.append('..')

import torch
from torch_geometric.utils import segment


from ops.accumulated_indexed_product_segment_op import (
# from ops.accumulated_indexed_product_segment_op import (
    # accumulated_indexed_inner_segment,
    accumulated_indexed_mul_segment,
    accumulated_indexed_outer_segment,
    # accumulated_indexed_vecmat_segment,
    accumulated_indexed_vecsca_segment
)
from utils._indices import expand_left
from utils._structs import tp_info
from test_utils import profile_funcs, compare_funcs, max_abs_diff
from irreps import check_irreps

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# torch.set_default_dtype(torch.float32)
torch.set_default_dtype(torch.float64)

torch.random.manual_seed(0)

def prepare_accumulated_indexed_segment(ir1, ir2, ir, device):
    print('tp info start')
    tp_info, num_paths = tp_info(ir, ir1, ir2)
    tp_info = tp_info.to(device)
    print('tp info calculated')
    M1 = tp_info.M1_MijM1M2
    M2 = tp_info.M2_MijM1M2
    seg = tp_info.Mij_seg_MijM1M2
    return M1, M2, seg

def accumulated_indexed_mul_segment_torch(input1, input2, index1, index2, seg):
    input1 = torch.index_select(input1, -2, index1, )
    input2 = torch.index_select(input2, -2, index2, )
    out = (input2 * input1)
    out=out.sum(dim=0)
    return segment(out, expand_left(out, seg, -2))

def init_mul(ir1, ir2, ir, C, N, ones=False, device='cuda'):
    M1, M2, seg = prepare_accumulated_indexed_segment(ir1, ir2, ir, device)

    input1 = torch.randn(N,ir1.dim,C).to(device)

    input2 = torch.randn(N, ir2.dim,C).to(device)

    print('shapes:')
    print(input1.shape, input2.shape)

    if ones:
        input1 = torch.ones_like(input1)
        input2 = torch.ones_like(input2)

    funcs = [
        accumulated_indexed_mul_segment, accumulated_indexed_mul_segment_torch
    ]

    return input1, input2, M1, M2, seg, funcs

def test_mul(ir1, ir2, ir, C, N, ones=False, ):

    input1, input2, M1, M2, seg, funcs = init_mul(ir1, ir2, ir, C, N, ones, )

    compare_funcs(funcs, max_abs_diff, 
                  input1, input2, M1, M2, seg)
    profile_funcs(funcs, ['triton', 'torch'], None, 5, 
                  input1, input2, M1, M2, seg)


def accumulated_indexed_outer_segment_torch(input1, input2, index1, index2, seg):
    input1 = torch.index_select(input1, -2, index1, ).unsqueeze(-1)
    input2 = torch.index_select(input2, -2, index2, ).unsqueeze(-2)
    out = (input2 * input1)
    out=out.sum(dim=0)
    return segment(out, expand_left(out, seg, -3))

def init_outer(ir1, ir2, ir, C1, C2, N, ones=False, device='cuda'):
    M1, M2, seg = prepare_accumulated_indexed_segment(ir1, ir2, ir, device)

    input1 = torch.randn(N,ir1.dim,C1).to(device)

    input2 = torch.randn(N, ir2.dim,C2).to(device)

    print('shapes:')
    print(input1.shape, input2.shape)

    if ones:
        input1 = torch.ones_like(input1)
        input2 = torch.ones_like(input2)

    funcs = [
        accumulated_indexed_outer_segment, accumulated_indexed_outer_segment_torch
    ]

    return input1, input2, M1, M2, seg, funcs

def test_outer(ir1, ir2, ir, C1, C2, N, ones=False, ):

    input1, input2, M1, M2, seg, funcs = init_outer(ir1, ir2, ir, C1, C2, N, ones, )

    compare_funcs(funcs, max_abs_diff, 
                  input1, input2, M1, M2, seg)
    profile_funcs(funcs, ['triton', 'torch'], None, 5, 
                  input1, input2, M1, M2, seg)
    

def accumulated_indexed_inner_segment_torch(input1, input2, index1, index2, seg):
    input1 = torch.index_select(input1, -2, index1, )
    input2 = torch.index_select(input2, -2, index2, )
    out = (input2 * input1).sum(dim=-1)
    out=out.sum(dim=0)
    return segment(out, expand_left(out, seg, -1))

def init_inner(ir1, ir2, ir, C, N, ones=False, device='cuda'):
    M1, M2, seg = prepare_accumulated_indexed_segment(ir1, ir2, ir, device)

    input1 = torch.randn(N,ir1.dim,C).to(device)

    input2 = torch.randn(N, ir2.dim,C).to(device)

    print('shapes:')
    print(input1.shape, input2.shape)

    if ones:
        input1 = torch.ones_like(input1)
        input2 = torch.ones_like(input2)

    funcs = [
        accumulated_indexed_inner_segment, accumulated_indexed_inner_segment_torch
    ]

    return input1, input2, M1, M2, seg, funcs

def test_inner(ir1, ir2, ir, C, N, ones=False, ):

    input1, input2, M1, M2, seg, funcs = init_inner(ir1, ir2, ir, C, N, ones, )

    compare_funcs(funcs, max_abs_diff, 
                  input1, input2, M1, M2, seg)
    profile_funcs(funcs, ['triton', 'torch'], None, 5, 
                  input1, input2, M1, M2, seg)

def accumulated_indexed_vecmat_segment_torch(input1, input2, idx1, idx2, seg):
    if input2.ndim == 3:
        input2 = input2.unsqueeze(0)
    out = (input1.index_select(1, idx1).unsqueeze(-2) @ input2.index_select(1, idx2)).squeeze(-2) 
    out=out.sum(dim=0)
    return segment(out, expand_left(out, seg, -2))

def init_vecmat(ir1, ir2, ir, C_in, C_out, N, ones=False, device='cuda'):
    
    M1, M2, seg = prepare_accumulated_indexed_segment(ir1, ir2, ir, device)

    input1 = torch.randn(N, ir1.dim, C_in).to(device)

    input2 = torch.randn(N, ir2.dim, C_in, C_out).to(device)

    print('shapes:')
    print(input1.shape, input2.shape)

    if ones:
        input1 = torch.ones_like(input1)
        input2 = torch.ones_like(input2)

    funcs = [
        accumulated_indexed_vecmat_segment, accumulated_indexed_vecmat_segment_torch
    ]

    return input1, input2, M1, M2, seg, funcs

def test_vecmat(ir1, ir2, ir, C_in, C_out, N, ones=False, ):

    input1, input2, M1, M2, seg, funcs = init_vecmat(ir1, ir2, ir, C_in, C_out, N, ones, )

    compare_funcs(funcs, max_abs_diff, 
                  input1, input2, M1, M2, seg)
    profile_funcs(funcs, ['triton', 'torch'], None, 5, 
                  input1, input2, M1, M2, seg)


def accumulated_indexed_vecsca_segment_torch(input1, input2, idx1, idx2, seg):
    input2 = input2.unsqueeze(-1)
    if input2.ndim == 2:
        input2 = input2.unsqueeze(0)
    out = input1.index_select(1, idx1) * input2.index_select(1, idx2)
    out=out.sum(dim=0)
    return segment(out, expand_left(out, seg, -2))

def init_vecsca(ir1, ir2, ir, C, N, ones=False, device='cuda'):
    M1, M2, seg = prepare_accumulated_indexed_segment(ir1, ir2, ir, device)
    
    input1 = torch.randn(N,ir1.dim,C).to(device)

    input2 = torch.randn(N, ir2.dim).to(device)

    print('shapes:')
    print(input1.shape, input2.shape)

    if ones:
        input1 = torch.ones_like(input1)
        input2 = torch.ones_like(input2)

    funcs = [
        accumulated_indexed_vecsca_segment, accumulated_indexed_vecsca_segment_torch
    ]

    return input1, input2, M1, M2, seg, funcs

def test_vecsca(ir1, ir2, ir, C, N, ones=False, ):

    input1, input2, M1, M2, seg, funcs = init_vecsca(ir1, ir2, ir, C, N, ones, )

    compare_funcs(funcs, max_abs_diff, 
                  input1, input2, M1, M2, seg)
    profile_funcs(funcs, ['triton', 'torch'], None, 5, 
                  input1, input2, M1, M2, seg)



if __name__ == '__main__':

    # irreps1 = irreps2 = irreps_out = check_irreps((0,0))
    # C = C1 = C2 = 64
    # C1 = 2
    # C2 = 1
    # C1 = 64
    # C2 = 1
    # N = 1

    # irreps1 = check_irreps((1,5))
    # irreps2 = check_irreps((2,7))
    # irreps_out = check_irreps((1,4))
    # C = 347
    # C1 = 17
    # C2 = 5
    # N = 51 

    irreps1 = irreps2 = irreps_out = check_irreps((0,4))
    C = 64
    C = 128
    C = 512
    C1 = C2 = 64
    # C1 = C2 = 32
    C1, C2 = 64, 32
    # # C1, C2 = 1, 64
    N = 256
    N = 256

    test_mul(irreps1, irreps2, irreps_out, C, N, )
    # test_outer(irreps1, irreps2, irreps_out, C1, C2, N, )
    # test_inner(irreps1, irreps2, irreps_out, C, N, )
    # test_vecmat(irreps1, irreps2, irreps_out, C1, C2, N, )
    # test_vecsca(irreps1, irreps2, irreps_out, C, N, )
