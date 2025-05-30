import sys



sys.path.append('../..')
sys.path.append('..')

import torch
import e3nn

import math

from equitorch.irreps import Irreps, check_irreps
from equitorch.structs import IrrepsLinearInfo
# from nn.functional.linear import irreps_linear
from equitorch.nn.linears import IrrepsLinear, IrrepWiseLinear
# import torch.nn as nn
# from utils._structs import irreps_linear_info

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

torch.set_default_dtype(torch.float32)
# torch.set_default_dtype(torch.float64)

torch.random.manual_seed(0)


def e3nn_features_to_equitorch(input: torch.Tensor, irreps, channels=None):

    
    e3nn_irreps = e3nn.o3.Irreps(irreps)
    equitorch_irreps = Irreps(str(irreps))
    if channels is None:
        channels = equitorch_irreps.max_channels()
    equitorch_irreps_divided = equitorch_irreps.channels_divided(channels)

    split_size = [channels * irrep.dim for irrep in equitorch_irreps_divided]

    features = input.split(split_size, dim=-1)
    features = [t.unflatten(-1, (channels, -1)) for t in features]
    features = torch.cat(features, dim=-1)
    return  features.transpose(-1,-2)

def equitorch_features_to_e3nn(input: torch.Tensor, irreps):

    channels = input.shape[-1]
    equitorch_irreps = check_irreps(irreps)

    equitorch_irreps_multiplied = equitorch_irreps.channels_multiplied(channels)

    e3nn_irreps = e3nn.o3.Irreps(equitorch_irreps_multiplied.short_repr())

    features = input.transpose(-1, -2)
    features = features.split(list(equitorch_irreps.dims()), -1)
    features = [t.flatten(-2, -1) for t in features]
    features = torch.cat(features, dim=-1)

    return  features


def e3nn_irrep_idx_to_eqt(irreps_e3nn, channels_eqt):
    acc_irrep_idx_eqt = 0
    irrep_idx_eqt_list = []
    for i, (mul, irrep) in enumerate(irreps_e3nn):
        irrep_idx_eqt_list_in = []
        for mul_eqt in range(mul//channels_eqt):
            irrep_idx_eqt_list_in.append((mul_eqt, acc_irrep_idx_eqt))
            acc_irrep_idx_eqt += 1
            # irrep_idx_eqt_list_in.append(mul_eqt)
        irrep_idx_eqt_list.append(irrep_idx_eqt_list_in)
    return irrep_idx_eqt_list


def e3nn_linear_weights_to_eqt(lin_e3nn, lin_eqt, weight_e3nn):
    
    irrep_in, irrep_out = lin_e3nn.irreps_in, lin_e3nn.irreps_out
    channels_in, channels_out = lin_eqt.channels_in, lin_eqt.channels_out
    to_eqt_in = e3nn_irrep_idx_to_eqt(irrep_in, lin_eqt.channels_in)
    to_eqt_out = e3nn_irrep_idx_to_eqt(irrep_out, lin_eqt.channels_out)
    weights_info_eqt = {}

    for ins, weight in zip(lin_e3nn.instructions, lin_e3nn.weight_views(weight_e3nn)):
        scale = (
            channels_in**(0.5) * ins.path_weight if (getattr(lin_eqt, 'path_norm',None) is None)or(not lin_eqt.path_norm)
            else 1
        )
        for e3nn_i_out_idx, i_out_eqt in to_eqt_out[ins.i_out]:
            for e3nn_i_in_idx, i_in_eqt in to_eqt_in[ins.i_in]:
                weights_info_eqt[i_out_eqt, i_in_eqt] = (
                    scale * weight[
                        ...,e3nn_i_in_idx*channels_in:(e3nn_i_in_idx+1)*channels_in,
                            e3nn_i_out_idx*channels_out:(e3nn_i_out_idx+1)*channels_out])

    kij_sorted = sorted(weights_info_eqt.keys())
    return torch.stack([weights_info_eqt[key] for key in kij_sorted], dim=-3)

def e3nn_linear_weights_grad_to_eqt(lin_e3nn, lin_eqt, weight_e3nn):
    
    irrep_in, irrep_out = lin_e3nn.irreps_in, lin_e3nn.irreps_out
    channels_in, channels_out = lin_eqt.channels_in, lin_eqt.channels_out
    to_eqt_in = e3nn_irrep_idx_to_eqt(irrep_in, lin_eqt.channels_in)
    to_eqt_out = e3nn_irrep_idx_to_eqt(irrep_out, lin_eqt.channels_out)
    weights_info_eqt = {}

    for ins, weight in zip(lin_e3nn.instructions, lin_e3nn.weight_views(weight_e3nn)):
        scale = (
            channels_in**(0.5) * ins.path_weight if (getattr(lin_eqt, 'path_norm',None) is None)or(not lin_eqt.path_norm)
            else 1
        )
        for e3nn_i_out_idx, i_out_eqt in to_eqt_out[ins.i_out]:
            for e3nn_i_in_idx, i_in_eqt in to_eqt_in[ins.i_in]:
                weights_info_eqt[i_out_eqt, i_in_eqt] = (
                    weight[
                        ...,e3nn_i_in_idx*channels_in:(e3nn_i_in_idx+1)*channels_in,
                            e3nn_i_out_idx*channels_out:(e3nn_i_out_idx+1)*channels_out] / scale)

    kij_sorted = sorted(weights_info_eqt.keys())
    return torch.stack([weights_info_eqt[key] for key in kij_sorted], dim=-3)

def init_irreps(irreps_in_e3nn, irreps_out_e3nn,
                irreps_in_eqt, irreps_out_eqt,
                C_in, C_out, N,
                shared=False, device='cuda',
                need_grad=False):
    
    lin_e3nn = e3nn.o3.Linear(irreps_in_e3nn, irreps_out_e3nn,
                            internal_weights=False, shared_weights=shared).to(device)
    lin_eqt = IrrepsLinear(irreps_in_eqt, irreps_out_eqt, C_in, C_out, 
                           path_norm=True,
                        internal_weights=False, channel_norm=True).to(device)

    input_eqt = torch.randn(N, irreps_in_eqt.dim, C_in).to(device=device)
    input_e3nn = equitorch_features_to_e3nn(input_eqt, irreps_in_eqt)

    if shared:
        W_e3nn = torch.randn(lin_e3nn.weight_numel).to(device=device)
        W_eqt = e3nn_linear_weights_to_eqt(lin_e3nn, lin_eqt, W_e3nn)
    else:
        W_e3nn = torch.randn(N, lin_e3nn.weight_numel).to(device)
        W_eqt = e3nn_linear_weights_to_eqt(lin_e3nn, lin_eqt, W_e3nn)

    grad_eqt = torch.randn(N, irreps_out_eqt.dim, C_out).to(device=device)
    grad_e3nn = equitorch_features_to_e3nn(grad_eqt, irreps_out_eqt)

    if need_grad:
        input_e3nn.requires_grad_()
        input_eqt.requires_grad_()
        W_e3nn.requires_grad_()
        W_eqt.requires_grad_()

    # lin_eqt = torch.jit.script(lin_eqt)
    # lin_e3nn = torch.jit.script(lin_e3nn)
    return {
        'eqt': [lin_eqt, input_eqt, W_eqt, grad_eqt],
        'e3nn': [lin_e3nn, input_e3nn, W_e3nn, grad_e3nn]
    }

def test_irreps(irreps_in_e3nn, irreps_out_e3nn,
               irreps_in_eqt, irreps_out_eqt,
               C_in, C_out, N, shared=True):

    inputs = init_irreps(irreps_in_e3nn, irreps_out_e3nn,
                irreps_in_eqt, irreps_out_eqt,
                C_in, C_out, N, shared=shared, need_grad=False)

    tester = FunctionTester({
        'eqt': (inputs['eqt'][0], inputs['eqt'][1:-1], {}),
        'e3nn': (inputs['e3nn'][0], inputs['e3nn'][1:-1], {}),
    })
    comp = tester.compare(post_transforms=[
        lambda x: x,
        lambda x: e3nn_features_to_equitorch(x, irreps_out_e3nn, channels=C_out)
    ])
    print(comp)
    # tester.profile(repeat=10, trace_name_func=lambda name, r: f'irreps_linear_forward_{name}_{shared}_{C_in}_{C_out}.json')
    tester.profile(repeat=10)
    
    print('backward')

    inputs_backward = init_irreps(irreps_in_e3nn, irreps_out_e3nn,
                irreps_in_eqt, irreps_out_eqt,
                C_in, C_out, N, shared=shared, need_grad=True)
    
    def backward_eqt(x, W, grad):
        res = inputs_backward['eqt'][0](x, W)
        res.backward(grad)
        return [x.grad, W.grad]
    def backward_e3nn(x, W, grad):
        res = inputs_backward['e3nn'][0](x, W)
        res.backward(grad)
        return [x.grad, W.grad]

    tester = FunctionTester({
        'eqt': (backward_eqt, inputs_backward['eqt'][1:], {}),
        'e3nn': (backward_e3nn, inputs_backward['e3nn'][1:], {}),
    })
    
    comp = tester.compare(post_transforms=[
        lambda res: res,
        lambda res: (
            e3nn_features_to_equitorch(res[0], irreps_in_e3nn, channels=C_in),
            e3nn_linear_weights_grad_to_eqt(inputs_backward['e3nn'][0], inputs_backward['eqt'][0], res[1])
        )
    # ], compare_func=max_abs_diff_list)
    ], compare_func=max_abs_diff_list)
    print(comp)
    # tester.profile(repeat=10, trace_name_func=lambda name, r: f'irreps_linear_backward_{name}_{shared}_{C_in}_{C_out}.json')
    tester.profile(repeat=10)



def init_irrep_wise(irreps_in_e3nn, irreps_out_e3nn, 
                    irreps_eqt,
                    C_in, C_out, N,
                shared=False, device='cuda',
                need_grad=False):
    
    lin_e3nn = e3nn.o3.Linear(
        irreps_in_e3nn, irreps_out_e3nn,
        internal_weights=False, shared_weights=shared,
        instructions=[(i, i) for i, _ in enumerate(irreps_in_e3nn)]).to(device)
    lin_eqt = IrrepWiseLinear(irreps_eqt, C_in, C_out, 
                        internal_weights=False, channel_norm=True).to(device)

    input_eqt = torch.randn(N, irreps_eqt.dim, C_in).to(device=device)
    input_e3nn = equitorch_features_to_e3nn(input_eqt, irreps_eqt)

    if shared:
        W_e3nn = torch.randn(lin_e3nn.weight_numel).to(device=device)
        W_eqt = e3nn_linear_weights_to_eqt(lin_e3nn, lin_eqt, W_e3nn)
    else:
        W_e3nn = torch.randn(N, lin_e3nn.weight_numel).to(device)
        W_eqt = e3nn_linear_weights_to_eqt(lin_e3nn, lin_eqt, W_e3nn)

    grad_eqt = torch.randn(N, irreps_eqt.dim, C_out).to(device=device)
    grad_e3nn = equitorch_features_to_e3nn(grad_eqt, irreps_eqt)

    if need_grad:
        input_e3nn.requires_grad_()
        input_eqt.requires_grad_()
        W_e3nn.requires_grad_()
        W_eqt.requires_grad_()

    # lin_eqt = torch.jit.script(lin_eqt)
    # lin_e3nn = torch.jit.script(lin_e3nn)
    return {
        'eqt': [lin_eqt, input_eqt, W_eqt, grad_eqt],
        'e3nn': [lin_e3nn, input_e3nn, W_e3nn, grad_e3nn]
    }

def test_irrep_wise(
        irreps_in_e3nn, irreps_out_e3nn,
        irreps_eqt,
        C_in, C_out, N,
        shared=False):

    inputs = init_irrep_wise(
        irreps_in_e3nn, irreps_out_e3nn, 
        irreps_eqt,
        C_in, C_out, N,
        shared=shared,
        need_grad=False)

    tester = FunctionTester({
        'eqt': (inputs['eqt'][0], inputs['eqt'][1:-1], {}),
        'e3nn': (inputs['e3nn'][0], inputs['e3nn'][1:-1], {}),
    })
    comp = tester.compare(post_transforms=[
        lambda x: x,
        lambda x: e3nn_features_to_equitorch(x, irreps_out_e3nn, channels=C_out)
    ])
    print(comp)
    # tester.profile(repeat=10, trace_name_func=lambda name, r: f'irreps_linear_forward_{name}_{shared}_{C_in}_{C_out}.json')
    tester.profile(repeat=10)
    
    print('backward')

    inputs_backward = init_irrep_wise(
        irreps_in_e3nn, irreps_out_e3nn, 
        irreps_eqt,
        C_in, C_out, N,
        shared=shared,
        need_grad=True)
    
    def backward_eqt(x, W, grad):
        res = inputs_backward['eqt'][0](x, W)
        res.backward(grad)
        return [x.grad, W.grad]
    def backward_e3nn(x, W, grad):
        res = inputs_backward['e3nn'][0](x, W)
        res.backward(grad)
        return [x.grad, W.grad]

    tester = FunctionTester({
        'eqt': (backward_eqt, inputs_backward['eqt'][1:], {}),
        'e3nn': (backward_e3nn, inputs_backward['e3nn'][1:], {}),
    })
    
    comp = tester.compare(post_transforms=[
        lambda res: res,
        lambda res: (
            e3nn_features_to_equitorch(res[0], irreps_in_e3nn, channels=C_in),
            e3nn_linear_weights_grad_to_eqt(inputs_backward['e3nn'][0], inputs_backward['eqt'][0], res[1])
        )
    ], compare_func=max_abs_diff_list)
    # ], compare_func=rate_mean2_std_list)
    print(comp)
    # tester.profile(repeat=10, trace_name_func=lambda name, r: f'irreps_linear_backward_{name}_{shared}_{C_in}_{C_out}.json')
    tester.profile(repeat=10)



from test_utils import FunctionTester, max_abs_diff

def max_abs_diff_list(list1, list2):
    return [max_abs_diff(a,b) for a,b in zip(list1, list2)]
def rate_mean2_std(x, y):
    rate = (x/y).nan_to_num(nan=1)
    return (rate.mean().item()**2, rate.std().item())

def rate_mean2_std_list(list1, list2):
    return [rate_mean2_std(a,b) for a,b in zip(list1, list2)]


# irreps_in = [(2,'0e')]
# irreps_out = [(2,'0e')]

# irreps_in = [(2,'1e'), (1,'1e'), (4,'2o'), (2,'1o')]
# irreps_out = [(3,'1e'), (1,'1o'), (1,'2e'), (3,'2o')]
# irreps_in = [(1,'0e'), (1,'0o'), (1,'1e'), (1,'1o'), (1,'2e'), (1,'2o'), (1,'3e'), (1,'3o'), (1, '4e'), (1, '4o')]
# irreps_out = [(1,'0e'), (1,'0o'), (1,'1e'), (1,'1o'), (1,'2e'), (1,'2o'), (1,'3e'), (1,'3o'), (1, '4e'), (1, '4o')]
irreps_in = [(1,'0e'), (1,'1e'), (1,'2e'), (1,'3e'), (1, '4e')]
irreps_out = [(1,'0e'), (1,'1e'), (1,'2e'), (1,'3e'), (1, '4e')]

# C_in, C_out = 256, 256
# C_in, C_out = 1, 1
C_in = C_out = 128

irreps_in_e3nn = e3nn.o3.Irreps('+'.join((f'{C_in*mul}x{ir}' for mul, ir in irreps_in)))
irreps_out_e3nn = e3nn.o3.Irreps('+'.join((f'{C_out*mul}x{ir}' for mul, ir in irreps_out)))

irreps_in_eqt = Irreps('+'.join((f'{mul}x{ir}' for mul, ir in irreps_in)))
irreps_out_eqt = Irreps('+'.join((f'{mul}x{ir}' for mul, ir in irreps_out)))
# irreps_in_eqt=irreps_in_e3nn = '0e+0e'
# irreps_out_eqt=irreps_out_e3nn = '0e'
# irreps_in_eqt = Irreps(irreps_in_eqt)
# irreps_out_eqt = Irreps(irreps_out_eqt)
print(irreps_in_e3nn, irreps_out_e3nn)
print(irreps_in_eqt, irreps_out_eqt)

N =256
# N =1

print('irreps linear')
print('not shared')

test_irreps(irreps_in_e3nn, irreps_out_e3nn,
               irreps_in_eqt, irreps_out_eqt,
               C_in, C_out, N, shared=False)

print('irreps linear')
print('shared')

test_irreps(irreps_in_e3nn, irreps_out_e3nn,
               irreps_in_eqt, irreps_out_eqt,
               C_in, C_out, N, shared=True)



# # irreps = [(3,'1e'), (1,'1o'), (1,'2e'), (3,'2o')]
# irreps = [(1,'0e'), (1,'0o'), (1,'1e'), (1,'1o'), (1,'2e'), (1,'2o'), (1,'3e'), (1,'3o'), (1, '4e'), (1, '4o')]
# # irreps = [(1,'1e')]

# # C_in, C_out = 67, 31
# C_in = C_out = 128


# # irreps_in_e3nn = e3nn.o3.Irreps('+'.join((f'{C_in*mul}x{ir}' for mul, ir in irreps)))
# # irreps_out_e3nn = e3nn.o3.Irreps('+'.join((f'{C_out*mul}x{ir}' for mul, ir in irreps)))
# irreps_in_e3nn = e3nn.o3.Irreps('+'.join((f'{C_in}x{ir}'for mul, ir in irreps for _ in range(mul) )))
# irreps_out_e3nn = e3nn.o3.Irreps('+'.join((f'{C_out}x{ir}' for mul, ir in irreps for _ in range(mul) )))

# irreps_eqt = Irreps('+'.join((f'{mul}x{ir}' for mul, ir in irreps)))
# print(irreps_in_e3nn, irreps_out_e3nn)
# print(irreps_eqt)

# N =256
# # N =61

# print('irrep_wise linear')
# print('not shared')

# test_irrep_wise(irreps_in_e3nn, irreps_out_e3nn,
#                irreps_eqt,
#                C_in, C_out, N, shared=False)

# print('irrep_wise linear')
# print('shared')

# test_irrep_wise(irreps_in_e3nn, irreps_out_e3nn,
#                irreps_eqt,
#                C_in, C_out, N, shared=True)
