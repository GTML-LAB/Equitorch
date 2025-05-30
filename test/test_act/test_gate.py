# %%
import sys


sys.path.append('..')

import torch
import e3nn

import math

from equitorch.irreps import Irreps, check_irreps
from equitorch.structs import IrrepsLinearInfo
from equitorch.nn.functional.activations import gating
import torch.nn as nn
from torch import Tensor
from equitorch.utils._structs import irreps_info
from equitorch.nn.activations import Gate

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

torch.set_default_dtype(torch.float32)
# torch.set_default_dtype(torch.float64)
# 
torch.random.manual_seed(0)


# %%
from equitorch.structs import IrrepsInfo


# %%
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


# %%
from test_utils import FunctionTester


def init_gate(irreps_e3nn, irreps_eqt, C, N,
                device='cuda',
                need_grad=False):
    
    num_irreps = len(irreps_eqt)
    num_gates = num_irreps * C

    gate_irreps_e3nn = f'{num_gates}x0e'

    gate_e3nn = e3nn.nn.Gate("", [], gate_irreps_e3nn, [torch.nn.Tanh()], irreps_e3nn).to(device=device)
    gate_eqt = Gate(irreps_eqt, torch.nn.Tanh()).to(device=device)    

    gates = torch.randn(N, num_gates).to(device=device)
    input_eqt = torch.randn(N, irreps_eqt.dim, C).to(device=device)
    
    input_e3nn = equitorch_features_to_e3nn(input_eqt, irreps_eqt)
    input_e3nn = torch.concat([gates, input_e3nn], dim=-1)
    grad_eqt = torch.randn(N, irreps_eqt.dim, C).to(device=device)
    grad_e3nn = equitorch_features_to_e3nn(grad_eqt, irreps_eqt)

    if need_grad:
        input_e3nn.requires_grad_()
        input_eqt.requires_grad_()
        gates.requires_grad_()

    # lin_eqt = torch.jit.script(lin_eqt)
    # lin_e3nn = torch.jit.script(lin_e3nn)
    return {
        'eqt': [gate_eqt, input_eqt, gates, grad_eqt],
        'e3nn': [gate_e3nn, input_e3nn, grad_e3nn]
    }

def test_gate(irreps_e3nn, irreps_eqt, C, N):

    inputs = init_gate(irreps_e3nn, irreps_eqt, C, N, need_grad=False)

    tester = FunctionTester({
        'eqt': (inputs['eqt'][0], inputs['eqt'][1:-1], {}),
        'e3nn': (inputs['e3nn'][0], inputs['e3nn'][1:-1], {}),
    })
    comp = tester.compare(post_transforms=[
        lambda x: x,
        lambda x: e3nn_features_to_equitorch(x, irreps_e3nn, channels=C)
    ], compare_func=rate_mean2_std)
    print(comp)
    # tester.profile(repeat=10, trace_name_func=lambda name, r: f'irreps_linear_forward_{name}_{shared}_{C_in}_{C_out}.json')
    tester.profile(repeat=10)
    
    print('backward')

    inputs_backward = init_gate(irreps_e3nn, irreps_eqt, C, N, need_grad=True)

    def backward_eqt(x, g, grad):
        res = inputs_backward['eqt'][0](x, g)
        res.backward(grad)
        return [x.grad, g.grad]
    def backward_e3nn(x, grad):
        res = inputs_backward['e3nn'][0](x)
        res.backward(grad)
        return x.grad

    tester = FunctionTester({
        'eqt': (backward_eqt, inputs_backward['eqt'][1:], {}),
        'e3nn': (backward_e3nn, inputs_backward['e3nn'][1:], {}),
    })
    # x.grad[:,len(irreps_eqt):], x.grad[:,:len(irreps_eqt)]
    comp = tester.compare(post_transforms=[
        lambda res: res,
        lambda res: (
            e3nn_features_to_equitorch(res[:,len(irreps_eqt)*C:], irreps_e3nn, channels=C),
            # e3nn_linear_weights_grad_to_eqt(inputs_backward['e3nn'][0], inputs_backward['eqt'][0], res[1])
            res[:,:len(irreps_eqt)*C]
        )
    # ], compare_func=max_abs_diff_list)
    ], compare_func=rate_mean2_std_list)
    print(comp)
    # tester.profile(repeat=10, trace_name_func=lambda name, r: f'irreps_linear_backward_{name}_{shared}_{C_in}_{C_out}.json')
    tester.profile(repeat=10)


# %%


from test_utils import FunctionTester, max_abs_diff

def max_abs_diff_list(list1, list2):
    return [max_abs_diff(a,b) for a,b in zip(list1, list2)]
def rate_mean2_std(x, y):
    rate = (x/y).nan_to_num(nan=1)
    return (rate.mean().item()**2, rate.std().item())

def rate_mean2_std_list(list1, list2):
    return [rate_mean2_std(a,b) for a,b in zip(list1, list2)]


# irreps_in = [(2,'1e'), (1,'1e'), (4,'2o'), (2,'1o')]
# irreps_out = [(3,'1e'), (1,'1o'), (1,'2e'), (3,'2o')]
# irreps = [(1,'0e'), (1,'0o'), (1,'1e'), (1,'1o'), (1,'2e'), (1,'2o'), (1,'3e'), (1,'3o'), (1, '4e'), (1, '4o')]
# irreps = [(1,'0o')]
# irreps = [(1,'0o')]
irreps = [(1,'0e'), (1,'1e'), (1,'2e'), (1,'3e'), (1, '4e')]
# irreps = [(4,'0e'), (2,'1e'), (1,'2e'), (3,'3e'), (2, '4e')]

C = 256
# C = 3
# C_in, C_out = 67, 31
# C_in = C_out = 128

irreps_e3nn = e3nn.o3.Irreps('+'.join((f'{C*mul}x{ir}' for mul, ir in irreps)))

irreps_eqt = Irreps('+'.join((f'{mul}x{ir}' for mul, ir in irreps)))
print(irreps_e3nn)
print(irreps_eqt)

N =256
# N =61


# %%
test_gate(irreps_e3nn, irreps_eqt, C, N)

# %%


# %%



