import torch
from torch import Tensor
import math
from typing import Optional

from equitorch.nn.sphericals import XYZToSpherical, SphericalToXYZ
from equitorch.nn.functional.sphericals import xyz_to_spherical as functional_xyz_to_spherical
from equitorch.nn.functional.sphericals import spherical_to_xyz as functional_spherical_to_xyz

from test_utils import FunctionTester, max_abs_diff_list, max_abs_diff # Assuming test_utils is accessible

torch.set_default_dtype(torch.float64)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TEST_EPS = 1e-7
FUNC_EPS = 1e-14

print(f"Device: {DEVICE}")

# --- Test XYZToSpherical Module ---
def test_module_xyz_to_spherical():
    print("\n--- Test Module: XYZToSpherical ---")
    batch_size = 10
    xyz_input = torch.randn(batch_size, 3, device=DEVICE, dtype=torch.get_default_dtype(), requires_grad=True)
    xyz_input_ref = xyz_input.clone().detach().requires_grad_(True)

    # Test with_r = True
    print("Case: with_r=True, normalize_input=True")
    module_xyz_to_sph_r = XYZToSpherical(with_r=True, normalize_input=True, eps=FUNC_EPS).to(DEVICE)
    
    # Forward
    theta_m, phi_m, r_m = module_xyz_to_sph_r(xyz_input)
    theta_f, phi_f, r_f = functional_xyz_to_spherical(xyz_input_ref, with_r=True, normalize_input=True, eps=FUNC_EPS)
    
    assert max_abs_diff(r_m, r_f) < TEST_EPS
    assert max_abs_diff(theta_m, theta_f) < TEST_EPS
    assert max_abs_diff(phi_m, phi_f) < TEST_EPS
    print("Forward pass (with_r=True) matches functional.")

    # Backward
    grad_r = torch.randn_like(r_m)
    grad_theta = torch.randn_like(theta_m)
    grad_phi = torch.randn_like(phi_m)

    r_m.backward(grad_r, retain_graph=True)
    theta_m.backward(grad_theta, retain_graph=True)
    phi_m.backward(grad_phi, retain_graph=False) # Last one
    grad_xyz_m = xyz_input.grad.clone()
    xyz_input.grad.zero_()

    r_f.backward(grad_r, retain_graph=True)
    theta_f.backward(grad_theta, retain_graph=True)
    phi_f.backward(grad_phi, retain_graph=False) # Last one
    grad_xyz_f = xyz_input_ref.grad.clone()
    xyz_input_ref.grad.zero_()
    
    assert max_abs_diff(grad_xyz_m, grad_xyz_f) < TEST_EPS
    print("Backward pass (with_r=True) matches functional.")

    # Test with_r = False
    print("Case: with_r=False, normalize_input=False")
    xyz_input.requires_grad_(True) # Re-enable grad
    xyz_input_ref.requires_grad_(True)
    module_xyz_to_sph_no_r = XYZToSpherical(with_r=False, normalize_input=False, eps=FUNC_EPS).to(DEVICE)

    # Forward
    theta_m_no_r, phi_m_no_r = module_xyz_to_sph_no_r(xyz_input)
    theta_f_no_r, phi_f_no_r = functional_xyz_to_spherical(xyz_input_ref, with_r=False, normalize_input=False, eps=FUNC_EPS)

    assert max_abs_diff(theta_m_no_r, theta_f_no_r) < TEST_EPS
    assert max_abs_diff(phi_m_no_r, phi_f_no_r) < TEST_EPS
    print("Forward pass (with_r=False) matches functional.")

    # Backward
    grad_theta_no_r = torch.randn_like(theta_m_no_r)
    grad_phi_no_r = torch.randn_like(phi_m_no_r)

    theta_m_no_r.backward(grad_theta_no_r, retain_graph=True)
    phi_m_no_r.backward(grad_phi_no_r, retain_graph=False)
    grad_xyz_m_no_r = xyz_input.grad.clone()
    xyz_input.grad.zero_()

    theta_f_no_r.backward(grad_theta_no_r, retain_graph=True)
    phi_f_no_r.backward(grad_phi_no_r, retain_graph=False)
    grad_xyz_f_no_r = xyz_input_ref.grad.clone()
    xyz_input_ref.grad.zero_()

    assert max_abs_diff(grad_xyz_m_no_r, grad_xyz_f_no_r) < TEST_EPS
    print("Backward pass (with_r=False) matches functional.")
    print("XYZToSpherical module tests passed.")


# --- Test SphericalToXYZ Module ---
def test_module_spherical_to_xyz():
    print("\n--- Test Module: SphericalToXYZ ---")
    batch_size = 10
    theta_input = torch.rand(batch_size, 1, device=DEVICE) * math.pi
    phi_input = (torch.rand(batch_size, 1, device=DEVICE) * 2 - 1) * math.pi
    r_input = torch.rand(batch_size, 1, device=DEVICE) * 5 + 0.1 # Ensure r > 0
    
    theta_input.requires_grad_(True)
    phi_input.requires_grad_(True)
    r_input.requires_grad_(True)

    theta_input_ref = theta_input.clone().detach().requires_grad_(True)
    phi_input_ref = phi_input.clone().detach().requires_grad_(True)
    r_input_ref = r_input.clone().detach().requires_grad_(True)

    # Test with r provided
    print("Case: r provided")
    module_sph_to_xyz_r = SphericalToXYZ().to(DEVICE)

    # Forward
    xyz_m = module_sph_to_xyz_r(theta_input, phi_input, r_input)
    xyz_f = functional_spherical_to_xyz(theta_input_ref, phi_input_ref, r_input_ref)
    
    assert max_abs_diff(xyz_m, xyz_f) < TEST_EPS
    print("Forward pass (r provided) matches functional.")

    # Backward
    grad_xyz = torch.randn_like(xyz_m)
    xyz_m.backward(grad_xyz)
    grad_theta_m = theta_input.grad.clone()
    grad_phi_m = phi_input.grad.clone()
    grad_r_m = r_input.grad.clone()
    theta_input.grad.zero_(); phi_input.grad.zero_(); r_input.grad.zero_()

    xyz_f.backward(grad_xyz)
    grad_theta_f = theta_input_ref.grad.clone()
    grad_phi_f = phi_input_ref.grad.clone()
    grad_r_f = r_input_ref.grad.clone()
    theta_input_ref.grad.zero_(); phi_input_ref.grad.zero_(); r_input_ref.grad.zero_()

    assert max_abs_diff(grad_theta_m, grad_theta_f) < TEST_EPS
    assert max_abs_diff(grad_phi_m, grad_phi_f) < TEST_EPS
    assert max_abs_diff(grad_r_m, grad_r_f) < TEST_EPS
    print("Backward pass (r provided) matches functional.")

    # Test with r = None (unit sphere)
    print("Case: r = None")
    theta_input.requires_grad_(True) # Re-enable grad
    phi_input.requires_grad_(True)
    # r_input is not used here, so no grad needed for it in this path
    
    theta_input_ref.requires_grad_(True)
    phi_input_ref.requires_grad_(True)

    module_sph_to_xyz_no_r = SphericalToXYZ().to(DEVICE)

    # Forward
    xyz_m_no_r = module_sph_to_xyz_no_r(theta_input, phi_input, r=None)
    xyz_f_no_r = functional_spherical_to_xyz(theta_input_ref, phi_input_ref, r=None)

    assert max_abs_diff(xyz_m_no_r, xyz_f_no_r) < TEST_EPS
    print("Forward pass (r=None) matches functional.")

    # Backward
    grad_xyz_no_r = torch.randn_like(xyz_m_no_r)
    xyz_m_no_r.backward(grad_xyz_no_r)
    grad_theta_m_no_r = theta_input.grad.clone()
    grad_phi_m_no_r = phi_input.grad.clone()
    theta_input.grad.zero_(); phi_input.grad.zero_()

    xyz_f_no_r.backward(grad_xyz_no_r)
    grad_theta_f_no_r = theta_input_ref.grad.clone()
    grad_phi_f_no_r = phi_input_ref.grad.clone()
    theta_input_ref.grad.zero_(); phi_input_ref.grad.zero_()

    assert max_abs_diff(grad_theta_m_no_r, grad_theta_f_no_r) < TEST_EPS
    assert max_abs_diff(grad_phi_m_no_r, grad_phi_f_no_r) < TEST_EPS
    print("Backward pass (r=None) matches functional.")
    print("SphericalToXYZ module tests passed.")

# --- Test Round Trip with Modules ---
def test_module_round_trip():
    print("\n--- Test Module: Round Trip ---")
    batch_size = 10
    xyz_orig = torch.randn(batch_size, 3, device=DEVICE, dtype=torch.get_default_dtype())
    # Ensure not all points are at origin for normalize_input tests
    xyz_orig[0] = torch.tensor([1.0, 2.0, 3.0], device=DEVICE) 

    # XYZ -> Sph (r) -> XYZ
    mod_xyz_to_sph_r = XYZToSpherical(with_r=True, normalize_input=True).to(DEVICE)
    mod_sph_to_xyz = SphericalToXYZ().to(DEVICE)
    
    theta, phi, r = mod_xyz_to_sph_r(xyz_orig)
    xyz_re = mod_sph_to_xyz(theta, phi, r)
    assert max_abs_diff(xyz_orig, xyz_re) < TEST_EPS
    print("Round trip XYZ -> Sph(r) -> XYZ passed.")

    # XYZ -> Sph (no r, normalize_input) -> XYZ (normalize_input)
    mod_xyz_to_sph_norm = XYZToSpherical(with_r=False, normalize_input=True).to(DEVICE)
    
    r_orig_norm = torch.norm(xyz_orig, dim=-1, keepdim=True)
    xyz_orig_normalized = xyz_orig / (r_orig_norm + FUNC_EPS)

    theta_n, phi_n = mod_xyz_to_sph_norm(xyz_orig) # Module handles normalization internally
    xyz_re_norm = mod_sph_to_xyz(theta_n, phi_n, r=None) # r=None for unit sphere
    assert max_abs_diff(xyz_orig_normalized, xyz_re_norm) < TEST_EPS
    print("Round trip XYZ -> Sph(norm) -> XYZ_normalized passed.")


if __name__ == "__main__":
    test_module_xyz_to_spherical()
    test_module_spherical_to_xyz()
    test_module_round_trip()
    print("\nAll spherical conversion module tests finished.")
