import torch
import math
import os
from torch.autograd import gradcheck, gradgradcheck

from equitorch.irreps import Irreps, check_irreps
from equitorch.nn.linears import IrrepsLinear

# Set environment and defaults
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.set_default_dtype(torch.float64)  # Use float64 for gradcheck stability
torch.random.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

EPS = 1e-9

def _init_test_case(irreps_in, irreps_out, channels_in, channels_out, batch_size, device, shared_weight):
    r"""Initialize test case for gradcheck."""
    layer = IrrepsLinear(
        irreps_in=irreps_in,
        irreps_out=irreps_out,
        channels_in=channels_in,
        channels_out=channels_out,
        path_norm=True,
        internal_weights=False
    ).to(device)
    
    # Create inputs with requires_grad=True
    x = torch.randn(batch_size, Irreps(irreps_in).dim, channels_in,
                   device=device, dtype=torch.float64, requires_grad=True)
    
    # Create weights with requires_grad=True
    weight_shape = layer.weight_shape
    if not shared_weight:
        weight_shape = (batch_size,) + weight_shape
    W = torch.randn(*weight_shape, device=device, dtype=torch.float64, requires_grad=True)
    
    return layer, x, W

def _run_gradcheck(layer, x, W):
    r"""Run gradcheck and gradgradcheck for the given inputs."""
    def func(x, W):
        return layer(x, W)
    
    # Ensure inputs are leaf variables
    x = x.detach().requires_grad_(True)
    W = W.detach().requires_grad_(True)
    
    # Forward pass
    out = func(x, W)
    print(f"Forward output sum: {out.sum().item()}")
    
    # Backward pass with gradient checking
    out.sum().backward()
    
    # Print gradient norms for debugging
    print(f"x.grad norm: {x.grad.norm().item() if x.grad is not None else 'None'}")
    print(f"W.grad norm: {W.grad.norm().item() if W.grad is not None else 'None'}")
    
    # Reset gradients
    x.grad = None
    W.grad = None
    
    # Run gradcheck with relaxed tolerances
    gradcheck_success = gradcheck(
        func, (x, W),
        eps=EPS, atol=1e-5, rtol=1e-5,
        nondet_tol=1e-3,
        check_undefined_grad=False
    )
    print('grad_check_passed')
    
    # Run gradgradcheck with same settings
    gradgradcheck_success = gradgradcheck(
        func, (x, W),
        eps=EPS, atol=1e-5, rtol=1e-5,
        nondet_tol=1e-3,
        check_undefined_grad=False
    )
    print('gradgrad_check_passed')
    
    return gradcheck_success, gradgradcheck_success

# Test cases with different irreps and parameter combinations
def test_linear_case(irreps_in, irreps_out, channels_in, channels_out, batch_size, shared_weight):
    r"""Test linear layer with given parameters."""
    print(f"\nTesting IrrepsLinear with irreps_in={irreps_in}, irreps_out={irreps_out}, "
          f"channels_in={channels_in}, channels_out={channels_out}, "
          f"batch_size={batch_size}, shared_weight={shared_weight}")
    
    layer, x, W = _init_test_case(
        irreps_in, irreps_out, channels_in, channels_out, 
        batch_size, device, shared_weight
    )
    
    gradcheck_success, gradgradcheck_success = _run_gradcheck(layer, x, W)
    print(f"IrrepsLinear - gradcheck: {gradcheck_success}, gradgradcheck: {gradgradcheck_success}")
    assert gradcheck_success and gradgradcheck_success

# Main execution
if __name__ == '__main__':
    print("Running gradient checks for IrrepsLinear...")
    
    # Test different combinations of parameters
    test_configs = [
        ("1x0e", "1x0e", 3, 3, 5),  # scalar only
        ("1x0e + 1x1e", "1x0e + 1x1e", 4, 4, 6),  # scalar + vector
        ("1x0e + 1x1e + 1x2e", "1x0e + 1x1e + 1x2e", 5, 5, 7)  # scalar + vector + tensor
    ]
    
    for irreps_in, irreps_out, channels_in, channels_out, batch_size in test_configs:
        for shared_weight in [True, False]:
            test_linear_case(
                irreps_in, irreps_out, 
                channels_in, channels_out,
                batch_size, shared_weight
            )
    
    print("\nAll gradient checks completed.")
