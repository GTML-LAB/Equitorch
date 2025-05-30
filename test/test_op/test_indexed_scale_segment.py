import sys


sys.path.append('..')

import torch
from torch_geometric.utils import segment

from test_utils import profile_funcs, compare_funcs, max_abs_diff
from irreps import check_irreps
# from so3_indicies import expand_left, tp_info
from ops.indexed_scale_segment_op import indexed_scale_segment
from utils._indices import expand_left
from utils._structs import tp_info

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

torch.set_default_dtype(torch.float32)
# torch.set_default_dtype(torch.float64)

torch.random.manual_seed(0)


def prepare_indexed_scale_segment(ir1, ir2, ir, device):
    print('tp info start')
    tp_info, num_paths = tp_info(ir, ir1, ir2)
    tp_info = tp_info.to(device)
    print('tp info calculated')
    kM1M2 = tp_info.kM1M2_MijM1M2
    # M = tp_info.M_MijM1M2
    seg = tp_info.M_seg_MijM1M2
    cg = tp_info.cg_vals
    return cg, kM1M2, None, seg

def indexed_scale_segment_torch(input, scale, index, seg):
    input = input.index_select(-2, index) * scale.unsqueeze(-1)
    return segment(input, expand_left(input, seg, -2))

def init_indexed_scale_segment(ir1, ir2, ir, C, N, ones=False, device='cuda'):
    cg, kM1M2, M, seg = prepare_indexed_scale_segment(ir1, ir2, ir, device)

    input = torch.randn(N,kM1M2.max()+1,C).to(device)

    print('shapes:')
    print(input.shape)

    if ones:
        input = torch.ones_like(input)

    funcs = [
        indexed_scale_segment, indexed_scale_segment_torch
    ]

    return input, cg, kM1M2, M, seg, funcs

def test_indexed_scale_segment(ir1, ir2, ir, C, N, ones=False):

    input, cg, kM1M2, M, seg, funcs = init_indexed_scale_segment(ir1, ir2, ir, C, N, ones)

    compare_funcs(funcs, max_abs_diff, 
                 input, cg, kM1M2, seg)
    profile_funcs(funcs, ['triton', 'torch'], None, 5, 
                  input, cg, kM1M2, seg)


if __name__ == '__main__':

    # irreps1 = irreps2 = irreps_out = check_irreps((0,0))
    # C = 1
    # N = 1

    # irreps1 = check_irreps((1,5))
    # irreps2 = check_irreps((2,7))
    # irreps_out = check_irreps((1,4))
    # C = 347
    # N = 51 

    irreps1 = irreps2 = irreps_out = check_irreps((0,4))
    # C = 64
    C = 128
    # C = 512
    N = 256

    test_indexed_scale_segment(irreps1, irreps2, irreps_out, C, N)
