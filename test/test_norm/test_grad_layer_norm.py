import torch
import math
import os
from torch.autograd import gradcheck, gradgradcheck

from equitorch.irreps import Irreps, check_irreps
from equitorch.nn.normalization import LayerRMSNorm

# Set environment and defaults
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.set_default_dtype(torch.float64)  # Use float64 for gradcheck stability
torch.random.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

EPS = 1e-9

def _init_test_case(irreps, channels, batch_size, device, affine, scaled):
    r"""Initialize test case for gradcheck."""
    layer = LayerRMSNorm(
        irreps=irreps,
        channels=channels,
        eps=EPS,
        affine=affine,
        scaled=scaled
    ).to(device)
    
    # Create input with requires_grad=True
    x = torch.randn(batch_size, Irreps(irreps).dim, channels,
                   device=device, dtype=torch.float64, requires_grad=True)
    
    return layer, x

def _run_gradcheck(layer, x):
    r"""Run gradcheck and gradgradcheck for the given inputs."""
    def func(x):
        return layer(x)
    
    # Ensure input is leaf variable
    x = x.detach().requires_grad_(True)
    
    # Forward pass
    out = func(x)
    print(f"Forward output sum: {out.sum().item()}")
    
    # Backward pass with gradient checking
    out.sum().backward()
    
    # Print gradient norms for debugging
    print(f"x.grad norm: {x.grad.norm().item() if x.grad is not None else 'None'}")
    
    # Reset gradients
    x.grad = None
    
    # Run gradcheck with relaxed tolerances
    gradcheck_success = gradcheck(
        func, (x,),
        eps=EPS, atol=1e-5, rtol=1e-5,
        nondet_tol=1e-3,
        check_undefined_grad=False
    )
    print('grad_check_passed')
    
    # Run gradgradcheck with same settings
    gradgradcheck_success = gradgradcheck(
        func, (x,),
        eps=EPS, atol=1e-5, rtol=1e-5,
        nondet_tol=1e-3,
        check_undefined_grad=False
    )
    print('gradgrad_check_passed')
    
    return gradcheck_success, gradgradcheck_success

# Test cases with different irreps and parameter combinations
def test_norm_case(irreps, channels, batch_size, affine, scaled):
    r"""Test normalization with given parameters."""
    print(f"\nTesting LayerRMSNorm with irreps={irreps}, channels={channels}, "
          f"batch_size={batch_size}, affine={affine}, scaled={scaled}")
    
    layer, x = _init_test_case(irreps, channels, batch_size, device, affine, scaled)
    gradcheck_success, gradgradcheck_success = _run_gradcheck(layer, x)
    
    print(f"LayerRMSNorm - gradcheck: {gradcheck_success}, gradgradcheck: {gradgradcheck_success}")
    assert gradcheck_success and gradgradcheck_success

# Main execution
if __name__ == '__main__':
    print("Running gradient checks for LayerRMSNorm...")
    
    # Test different combinations of affine and scaled
    test_configs = [
        ("1x0e", 3, 5),  # scalar only
        ("1x0e + 1x1e", 4, 6),  # scalar + vector
        ("1x0e + 1x1e + 1x2e", 5, 7)  # scalar + vector + tensor
    ]
    
    for irreps, channels, batch_size in test_configs:
        for affine in [True, False]:
            for scaled in [True, False]:
                test_norm_case(irreps, channels, batch_size, affine, scaled)
    
    print("\nAll gradient checks completed.")
