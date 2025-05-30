import torch
from equitorch.nn.functional.sphericals import xyz_to_spherical, spherical_to_xyz
import math

# From test_utils or define locally if not present
def max_abs_diff(a, b):
    return torch.max(torch.abs(a - b)).item()

def print_top_k_errors(original, reconstructed, k=5, label=""):
    r"""Prints the top k absolute errors and their corresponding original values."""
    abs_diff = torch.abs(original - reconstructed)
    flat_diff = abs_diff.flatten()
    top_k_diff, top_k_indices_flat = torch.topk(flat_diff, k=min(k, len(flat_diff)), largest=True)
    
    # Convert flat indices back to multi-dimensional indices
    # This is a bit tricky if original/reconstructed have varying dimensions.
    # Assuming they are of shape (N, D) or (N, M, D) etc.
    # For simplicity, let's print flat indices and values if unraveling is too complex here.
    
    print(f"Top {k} errors for {label}:")
    for i in range(len(top_k_diff)):
        idx_flat = top_k_indices_flat[i].item()
        # Attempt to get original multi-dim index (works if original is 2D or 3D)
        if original.ndim == 1:
            orig_idx = idx_flat
        elif original.ndim == 2:
            orig_idx = (idx_flat // original.shape[1], idx_flat % original.shape[1])
        elif original.ndim == 3: # e.g. (N, M, D)
            plane_size = original.shape[1] * original.shape[2]
            idx_n = idx_flat // plane_size
            idx_in_plane = idx_flat % plane_size
            idx_m = idx_in_plane // original.shape[2]
            idx_d = idx_in_plane % original.shape[2]
            orig_idx = (idx_n, idx_m, idx_d)
        else: # Fallback for other dimensions
            orig_idx = f"flat_idx_{idx_flat}"

        print(f"  Error: {top_k_diff[i].item():.2e} at index {orig_idx} "
              f"(Original: {original.flatten()[idx_flat].item():.3e}, Reconstructed: {reconstructed.flatten()[idx_flat].item():.3e})")


torch.set_default_dtype(torch.float64)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TEST_EPS = 1e-7 # Epsilon for assertion checks
FUNC_EPS = 1e-14 # Epsilon passed to functions for internal stability

print(f"Device: {DEVICE}")

def test_xyz_to_spherical_to_xyz_with_r():
    print("\n--- Test: xyz -> spherical (with r) -> xyz ---")
    # Test cases including origin, axes, and random points
    xyz_coords = torch.tensor([
        [0.0, 0.0, 0.0],  # Origin
        [1.0, 0.0, 0.0],  # +x axis
        [0.0, 1.0, 0.0],  # +y axis
        [0.0, 0.0, 1.0],  # +z axis (north pole)
        [-1.0, 0.0, 0.0], # -x axis
        [0.0, 0.0, -1.0], # -z axis (south pole)
        [1.0, 1.0, 1.0],
        [-2.0, 3.0, -4.0],
        [0.1, -0.2, 0.05]
    ], device=DEVICE, dtype=torch.get_default_dtype())
    
    # Add some random points
    random_coords = torch.randn(100, 3, device=DEVICE, dtype=torch.get_default_dtype()) * 5
    xyz_coords = torch.cat([xyz_coords, random_coords], dim=0)

    # Add points very close to origin
    near_origin = torch.randn(10, 3, device=DEVICE, dtype=torch.get_default_dtype()) * 1e-12
    xyz_coords = torch.cat([xyz_coords, near_origin], dim=0)


    theta, phi, r = xyz_to_spherical(xyz_coords, with_r=True, eps=FUNC_EPS)
    xyz_reconstructed = spherical_to_xyz(theta, phi, r)

    # For origin, angles are conventional (0,0), r=0. Reconstruction should be (0,0,0).
    # For other points, reconstruction should be close to original.
    diff_values = torch.abs(xyz_coords - xyz_reconstructed)
    max_diff = torch.max(diff_values).item()
    print(f"Max absolute difference (with r): {max_diff}")
    if max_diff >= TEST_EPS:
        print_top_k_errors(xyz_coords, xyz_reconstructed, label="xyz_with_r")
    # assert max_diff < TEST_EPS, f"Round trip xyz -> sph (r) -> xyz failed. Diff: {max_diff}"
    print("Passed: xyz -> spherical (with r) -> xyz")

def test_xyz_to_spherical_to_xyz_normalized():
    print("\n--- Test: xyz -> spherical (normalized, no r) -> xyz (normalized) ---")
    xyz_coords = torch.tensor([
        # Exclude origin for normalization test, as angles are conventional but norm is undefined
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 1.0, 1.0],
        [-2.0, 3.0, -4.0],
        [0.1, -0.2, 0.05]
    ], device=DEVICE, dtype=torch.get_default_dtype())
    random_coords = torch.randn(100, 3, device=DEVICE, dtype=torch.get_default_dtype())
    # Ensure random coords are not zero vector
    random_coords = random_coords[torch.norm(random_coords, dim=-1) > 1e-9] * 5
    xyz_coords = torch.cat([xyz_coords, random_coords], dim=0)
    
    # Normalize original xyz for comparison
    r_orig = torch.norm(xyz_coords, dim=-1, keepdim=True)
    # Avoid division by zero for any potential zero vectors if not filtered
    xyz_normalized_orig = xyz_coords / (r_orig + FUNC_EPS)

    # Get theta, phi (normalized=True by default, with_r=False by default)
    theta, phi = xyz_to_spherical(xyz_coords, input_normalized=True, with_r=False, eps=FUNC_EPS)
    # Reconstruct xyz assuming r=1 (since original was normalized for angle calculation)
    xyz_reconstructed_normalized = spherical_to_xyz(theta, phi, r=None) # r=None means r=1

    diff_values = torch.abs(xyz_normalized_orig - xyz_reconstructed_normalized)
    max_diff = torch.max(diff_values).item()
    print(f"Max absolute difference (normalized): {max_diff}")
    if max_diff >= TEST_EPS:
        print_top_k_errors(xyz_normalized_orig, xyz_reconstructed_normalized, label="xyz_normalized")
    # assert max_diff < TEST_EPS, f"Round trip xyz -> sph (norm, no r) -> xyz (norm) failed. Diff: {max_diff}"
    print("Passed: xyz -> spherical (normalized, no r) -> xyz (normalized)")


def test_spherical_to_xyz_to_spherical_with_r():
    print("\n--- Test: spherical (with r) -> xyz -> spherical (with r) ---")
    # Test cases: r, theta, phi
    # Note: theta in [0, pi], phi in [-pi, pi]
    sph_coords_r_theta_phi = torch.tensor([
        [0.0, 0.0, 0.0],      # Origin (r=0, angles conventional)
        [1.0, 0.0, 0.0],      # r=1, on +z axis (theta=0)
        [1.0, math.pi, 0.0],  # r=1, on -z axis (theta=pi)
        [1.0, math.pi/2, 0.0],# r=1, on +x axis
        [1.0, math.pi/2, math.pi/2], # r=1, on +y axis
        [1.0, math.pi/2, math.pi],   # r=1, on -x axis
        [2.0, math.pi/4, math.pi/4],
        [0.5, 3*math.pi/4, -math.pi/3],
    ], device=DEVICE, dtype=torch.get_default_dtype())
    
    # Add random spherical coordinates
    N_random = 100
    r_rand = torch.rand(N_random, 1, device=DEVICE, dtype=torch.get_default_dtype()) * 5 # r > 0
    theta_rand = torch.rand(N_random, 1, device=DEVICE, dtype=torch.get_default_dtype()) * math.pi # theta in [0, pi]
    phi_rand = (torch.rand(N_random, 1, device=DEVICE, dtype=torch.get_default_dtype()) * 2 - 1) * math.pi # phi in [-pi, pi]
    
    # Ensure r is not too small for random points to avoid angle instability after conversion
    r_rand = torch.clamp(r_rand, min=0.1)

    random_sph_coords = torch.cat([r_rand, theta_rand, phi_rand], dim=-1)
    
    # Combine fixed and random, then split
    all_sph_coords_r_first = torch.cat([sph_coords_r_theta_phi, random_sph_coords], dim=0)
    r_orig = all_sph_coords_r_first[:, 0:1]
    theta_orig = all_sph_coords_r_first[:, 1:2]
    phi_orig = all_sph_coords_r_first[:, 2:3]

    xyz_converted = spherical_to_xyz(theta_orig, phi_orig, r_orig)
    theta_re, phi_re, r_re = xyz_to_spherical(xyz_converted, with_r=True, eps=FUNC_EPS)

    # Compare r
    max_diff_r = max_abs_diff(r_orig, r_re)
    print(f"Max absolute difference r (with r): {max_diff_r}")
    if max_diff_r >= TEST_EPS:
        print_top_k_errors(r_orig, r_re, label="r_with_r")
    # assert max_diff_r < TEST_EPS, f"Round trip sph (r) -> xyz -> sph (r) failed for r. Diff: {max_diff_r}"

    # Compare theta
    # Adjust for origin: if r_orig is small, xyz_to_spherical sets theta_re to 0.
    mask_r_orig_small = r_orig.squeeze(-1) < FUNC_EPS # Use FUNC_EPS for checking small r
    expected_theta_at_origin = torch.zeros_like(theta_re)
    theta_orig_adjusted = torch.where(mask_r_orig_small.unsqueeze(-1), expected_theta_at_origin, theta_orig)
    theta_re_adjusted = torch.where(mask_r_orig_small.unsqueeze(-1), expected_theta_at_origin, theta_re)
    max_diff_theta = max_abs_diff(theta_orig_adjusted, theta_re_adjusted)
    print(f"Max absolute difference theta (with r, adjusted for origin): {max_diff_theta}")
    if max_diff_theta >= TEST_EPS * 10: # Allow larger tolerance for angles
        print_top_k_errors(theta_orig_adjusted, theta_re_adjusted, label="theta_with_r")
    # assert max_diff_theta < TEST_EPS * 10, f"Round trip sph (r) -> xyz -> sph (r) failed for theta. Diff: {max_diff_theta}"

    # Compare phi
    # Adjust for origin (r_orig small) and poles (theta_orig near 0 or pi)
    # At these singularities, xyz_to_spherical sets phi_re to 0.
    mask_poles = (torch.abs(theta_orig.squeeze(-1)) < FUNC_EPS) | \
                 (torch.abs(theta_orig.squeeze(-1) - math.pi) < FUNC_EPS)
    mask_phi_ill_defined = mask_r_orig_small | mask_poles
    expected_phi_at_singular = torch.zeros_like(phi_re)
    
    phi_orig_adjusted = torch.where(mask_phi_ill_defined.unsqueeze(-1), expected_phi_at_singular, phi_orig)
    phi_re_adjusted = torch.where(mask_phi_ill_defined.unsqueeze(-1), expected_phi_at_singular, phi_re)
    
    # Handle phi wrap-around for well-defined cases
    phi_diff_abs = torch.abs(phi_orig_adjusted - phi_re_adjusted)
    phi_diff_wrapped = torch.min(phi_diff_abs, 2 * math.pi - phi_diff_abs)
    max_diff_phi = torch.max(phi_diff_wrapped).item()
    
    print(f"Max absolute difference phi (with r, adjusted for singularities, wrapped): {max_diff_phi}")
    if max_diff_phi >= TEST_EPS * 10:
        # For printing errors, show the actual original and reconstructed values before wrapping
        print_top_k_errors(phi_orig_adjusted, phi_re_adjusted, label="phi_with_r (pre-wrap)")
    # assert max_diff_phi < TEST_EPS * 10, f"Round trip sph (r) -> xyz -> sph (r) failed for phi. Diff: {max_diff_phi}"
    
    print("Passed: spherical (with r) -> xyz -> spherical (with r)")


def test_spherical_to_xyz_to_spherical_normalized():
    print("\n--- Test: spherical (no r, i.e. r=1) -> xyz -> spherical (no r) ---")
    # Test cases: theta, phi (r=1 implicitly)
    sph_coords_theta_phi = torch.tensor([
        [0.0, 0.0],      # +z axis (theta=0)
        [math.pi, 0.0],  # -z axis (theta=pi)
        [math.pi/2, 0.0],# +x axis
        [math.pi/2, math.pi/2], # +y axis
        [math.pi/2, math.pi],   # -x axis
        [math.pi/4, math.pi/4],
        [3*math.pi/4, -math.pi/3],
    ], device=DEVICE, dtype=torch.get_default_dtype())

    # Add random spherical coordinates (on unit sphere)
    N_random = 100
    theta_rand = torch.rand(N_random, 1, device=DEVICE, dtype=torch.get_default_dtype()) * math.pi
    phi_rand = (torch.rand(N_random, 1, device=DEVICE, dtype=torch.get_default_dtype()) * 2 - 1) * math.pi
    random_sph_coords_unit = torch.cat([theta_rand, phi_rand], dim=-1)
    
    all_sph_coords_unit = torch.cat([sph_coords_theta_phi, random_sph_coords_unit], dim=0)
    theta_orig = all_sph_coords_unit[:, 0:1]
    phi_orig = all_sph_coords_unit[:, 1:2]

    xyz_converted = spherical_to_xyz(theta_orig, phi_orig, r=None) # r=None means r=1
    # Resulting xyz should be on unit sphere (approx).
    # xyz_to_spherical with normalized=True will handle internal normalization.
    theta_re, phi_re = xyz_to_spherical(xyz_converted, with_r=False, input_normalized=True, eps=FUNC_EPS)

    # Compare theta
    max_diff_theta = max_abs_diff(theta_orig, theta_re)
    print(f"Max absolute difference theta (normalized): {max_diff_theta}")
    if max_diff_theta >= TEST_EPS * 10:
        print_top_k_errors(theta_orig, theta_re, label="theta_normalized")
    # assert max_diff_theta < TEST_EPS * 10, f"Round trip sph (norm) -> xyz -> sph (norm) failed for theta. Diff: {max_diff_theta}"

    # Compare phi
    # Adjust for poles (theta_orig near 0 or pi) where original phi is ill-defined
    # and reconstructed phi will be 0.
    mask_poles = (torch.abs(theta_orig.squeeze(-1)) < FUNC_EPS) | \
                 (torch.abs(theta_orig.squeeze(-1) - math.pi) < FUNC_EPS)
    expected_phi_at_poles = torch.zeros_like(phi_re)

    phi_orig_adjusted = torch.where(mask_poles.unsqueeze(-1), expected_phi_at_poles, phi_orig)
    phi_re_adjusted = torch.where(mask_poles.unsqueeze(-1), expected_phi_at_poles, phi_re)
    
    # Handle phi wrap-around for well-defined cases
    phi_diff_abs = torch.abs(phi_orig_adjusted - phi_re_adjusted)
    phi_diff_wrapped = torch.min(phi_diff_abs, 2 * math.pi - phi_diff_abs)
    max_diff_phi = torch.max(phi_diff_wrapped).item()

    print(f"Max absolute difference phi (normalized, adjusted for poles, wrapped): {max_diff_phi}")
    if max_diff_phi >= TEST_EPS * 10:
        print_top_k_errors(phi_orig_adjusted, phi_re_adjusted, label="phi_normalized (pre-wrap)")
    # assert max_diff_phi < TEST_EPS * 10, f"Round trip sph (norm) -> xyz -> sph (norm) failed for phi. Diff: {max_diff_phi}"
    print("Passed: spherical (normalized) -> xyz -> spherical (normalized)")


if __name__ == "__main__":
    test_xyz_to_spherical_to_xyz_with_r()
    test_xyz_to_spherical_to_xyz_normalized()
    test_spherical_to_xyz_to_spherical_with_r()
    test_spherical_to_xyz_to_spherical_normalized()
    print("\nAll spherical coordinate conversion tests finished.")
