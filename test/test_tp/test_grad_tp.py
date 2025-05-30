import torch
import math
import os
from torch.autograd import gradcheck, gradgradcheck

from equitorch.irreps import Irreps, check_irreps
from equitorch.nn.tensor_products import TensorProduct
from equitorch.nn.linears import SO3Linear

# Set environment and defaults
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.set_default_dtype(torch.float64)  # Use float64 for gradcheck stability
torch.random.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

EPS = 1e-9


def _init_test_case(eqt_cls, feature_mode, irreps_in1, irreps_in2, irreps_out, 
                   channels_in1, channels_in2, channels_out, batch_size, device, shared_weight=False):
    r"""Initialize test case for gradcheck."""
    kwargs = {
        'irreps_in1': irreps_in1,
        'irreps_in2': irreps_in2,
        'irreps_out': irreps_out,
        'feature_mode': feature_mode,
        'path_norm': True,
        'internal_weights': False,
    }
    
    if eqt_cls == SO3Linear:
        kwargs.update({'channels_in': channels_in1, 'channels_out': channels_out})
    else:
        kwargs.update({'channels_in1': channels_in1, 'channels_in2': channels_in2, 'channels_out': channels_out})

    layer = eqt_cls(**kwargs).to(device)
    
    # Create inputs with requires_grad=True and smaller values for stability
    x = torch.randn(batch_size, Irreps(irreps_in1).dim, channels_in1, 
                   device=device, dtype=torch.float64, requires_grad=True) 
    
    if eqt_cls == SO3Linear:
        y = torch.randn(batch_size, Irreps(irreps_in2).dim, 
                       device=device, dtype=torch.float64, requires_grad=True) 
    else:
        y = torch.randn(batch_size, Irreps(irreps_in2).dim, channels_in2,
                       device=device, dtype=torch.float64, requires_grad=True) 
    
    # Create weights with requires_grad=True and smaller values
    weight_shape = layer.weight_shape
    if not shared_weight:
        weight_shape = (batch_size,)+weight_shape
    W = torch.randn(*weight_shape, device=device, dtype=torch.float64, requires_grad=True) 
    
    return layer, x, y, W

def _run_gradcheck(layer, x, y, W):
    r"""Run gradcheck and gradgradcheck for the given inputs."""
    def func(x, y, W):
        out = layer(x, y, W)
        # print(f"Forward output shape: {out.shape}")
        return out
    
    # Ensure inputs are leaf variables
    x = x.detach().requires_grad_(True)
    y = y.detach().requires_grad_(True)
    W = W.detach().requires_grad_(True)
    
    # Forward pass
    out = func(x, y, W)
    print(f"Forward output sum: {out.sum().item()}")
    
    # Backward pass with gradient checking
    out.sum().backward()
    
    # Print gradient norms for debugging
    print(f"x.grad norm: {x.grad.norm().item() if x.grad is not None else 'None'}")
    print(f"y.grad norm: {y.grad.norm().item() if y.grad is not None else 'None'}")
    print(f"W.grad norm: {W.grad.norm().item() if W.grad is not None else 'None'}")
    
    # Reset gradients
    x.grad = None
    y.grad = None
    W.grad = None
    
    # Run gradcheck with relaxed tolerances
    gradcheck_success = gradcheck(
        func, (x, y, W),
        eps=EPS, atol=1e-5, rtol=1e-5,  # More relaxed tolerances
        nondet_tol=1e-3,
        check_undefined_grad=False
    )
    print('grad_check_passed')
    # Run gradgradcheck with same settings
    gradgradcheck_success = gradgradcheck(
        func, (x, y, W),
        eps=EPS, atol=1e-5, rtol=1e-5,
        nondet_tol=1e-3,
        check_undefined_grad=False
    )
    print('gradgrad_check_passed')
    
    return gradcheck_success, gradgradcheck_success

# === Test Cases ===

def test_tp_uvw(shared_weight=False):
    r"""Test TensorProduct in UVW mode with more complex case."""
    # irreps_in1 = '1x0e'# + 1x1e + 1x2e'
    # irreps_in2 = '1x0e'# + 1x1e + 1x2e'
    # irreps_out = '1x0e'# + 1x1e + 1x2e + 1x3e'
    irreps_in1 = '1x0e + 1x1e + 1x2e'
    irreps_in2 = '1x0e + 1x1e + 1x2e'
    irreps_out = '1x0e + 1x1e + 1x2e + 1x3e'
    C1, C2, Cout = 3, 4, 5
    batch_size = 6
    
    layer, x, y, W = _init_test_case(
        TensorProduct, 'uvw', irreps_in1, irreps_in2, irreps_out,
        C1, C2, Cout, batch_size, device, shared_weight
    )
    
    gradcheck_success, gradgradcheck_success = _run_gradcheck(layer, x, y, W)
    print(f"TensorProduct UVW - gradcheck: {gradcheck_success}, gradgradcheck: {gradgradcheck_success}")
    assert gradcheck_success and gradgradcheck_success

def test_tp_uuu(shared_weight=False):
    r"""Test TensorProduct in UUU mode."""
    irreps_in1 = '1x0e'# /+ 1x1e'
    # irreps_in2 = '1x0e'# + 1x1e'
    # irreps_out = '1x0e'# + 1x1e'
    # C = 1
    # batch_size = 1
    irreps_in1 = '1x0e + 1x1e + 1x2e'
    irreps_in2 = '1x0e + 1x1e + 1x2e'
    irreps_out = '1x0e + 1x1e + 1x2e + 1x3e'
    C = 5
    batch_size = 6    
    layer, x, y, W = _init_test_case(
        TensorProduct, 'uuu', irreps_in1, irreps_in2, irreps_out,
        C, C, C, batch_size, device, shared_weight
    )
    
    gradcheck_success, gradgradcheck_success = _run_gradcheck(layer, x, y, W)
    print(f"TensorProduct UUU - gradcheck: {gradcheck_success}, gradgradcheck: {gradgradcheck_success}")
    assert gradcheck_success and gradgradcheck_success

def test_so3_uv(shared_weight=False):
    r"""Test SO3Linear in UV mode."""
    # irreps_in1 = '1x0e + 1x1e'
    # irreps_in2 = '1x0e'
    # irreps_out = '1x0e + 1x1e'
    # Cin, Cout = 2, 2
    # batch_size = 3
    irreps_in1 = '1x0e + 1x1e + 1x2e'
    irreps_in2 = '1x0e + 1x1e + 1x2e'
    irreps_out = '1x0e + 1x1e + 1x2e + 1x3e'
    Cin, Cout = 3, 4
    batch_size = 6
    layer, x, y, W = _init_test_case(
        SO3Linear, 'uv', irreps_in1, irreps_in2, irreps_out,
        Cin, 1, Cout, batch_size, device, shared_weight
    )
    
    gradcheck_success, gradgradcheck_success = _run_gradcheck(layer, x, y, W)
    print(f"SO3Linear UV - gradcheck: {gradcheck_success}, gradgradcheck: {gradgradcheck_success}")
    assert gradcheck_success and gradgradcheck_success

def test_so3_uu(shared_weight=False):
    r"""Test SO3Linear in UU mode."""
    # irreps_in1 = '1x0e + 1x1e'
    # irreps_in2 = '1x0e'
    # irreps_out = '1x0e + 1x1e'
    # C = 2
    # batch_size = 3
    irreps_in1 = '1x0e + 1x1e + 1x2e'
    irreps_in2 = '1x0e + 1x1e + 1x2e'
    irreps_out = '1x0e + 1x1e + 1x2e + 1x3e'
    C = 4
    batch_size = 6
    layer, x, y, W = _init_test_case(
        SO3Linear, 'uu', irreps_in1, irreps_in2, irreps_out,
        C, 1, C, batch_size, device, shared_weight
    )
    
    gradcheck_success, gradgradcheck_success = _run_gradcheck(layer, x, y, W)
    print(f"SO3Linear UU - gradcheck: {gradcheck_success}, gradgradcheck: {gradgradcheck_success}")
    assert gradcheck_success and gradgradcheck_success

# === Main Execution ===

if __name__ == '__main__':
    print("Running gradient checks...")
    
    print("\nTesting TensorProduct UVW (shared_weight=False):")
    test_tp_uvw(False)
    
    print("\nTesting TensorProduct UVW (shared_weight=True):")
    test_tp_uvw(True)
    
    print("\nTesting TensorProduct UUU (shared_weight=False):")
    test_tp_uuu(False)
    
    print("\nTesting TensorProduct UUU (shared_weight=True):")
    test_tp_uuu(True)
    
    print("\nTesting SO3Linear UV (shared_weight=False):")
    test_so3_uv(False)
    
    print("\nTesting SO3Linear UV (shared_weight=True):")
    test_so3_uv(True)
    
    print("\nTesting SO3Linear UU (shared_weight=False):")
    test_so3_uu(False)
    
    print("\nTesting SO3Linear UU (shared_weight=True):")
    test_so3_uu(True)
    
    print("\nAll gradient checks completed.")
