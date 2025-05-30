import torch
from torch import Tensor
import math

from equitorch.irreps import Irreps
from equitorch.nn.wigner_d import AlignToZWignerD
from equitorch.nn.sphericals import SphericalHarmonics

# Assuming test_utils is in the parent directory relative to test_modules
# Adjust the import path if necessary
from test_utils import max_abs_diff

torch.set_default_dtype(torch.float64)
# torch.set_default_dtype(torch.float64)
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'
TEST_EPS = 1e-7
FUNC_EPS = 0

print(f"Device: {DEVICE}")

def test_align_to_z_rotation():
    print("\n--- Test Module: AlignToZWignerD ---")
    batch_size = 10
    l_max = 4
    irreps = Irreps.spherical_harmonics(l_max) # SH irreps: 0e+1o+2e+3o+4e
    print(f"Testing Irreps: {irreps} (l_max={l_max}, dim={irreps.dim})")

    # --- Instantiate Modules ---
    # Use input_normalize=True for SH as it operates on direction vectors
    # Use integral_normalize=False as it's the standard SH definition often used
    sh_module = SphericalHarmonics(l_max=l_max, normalize_input=True, integral_normalize=False).to(DEVICE)
    # Use normalized=True for alignment D matrix as rotation depends only on direction
    align_module = AlignToZWignerD(irreps=irreps, normalized=True, eps=FUNC_EPS).to(DEVICE)

    # --- Prepare Inputs ---
    # Random xyz vectors
    xyz_input = torch.randn(batch_size, 3, device=DEVICE, dtype=torch.get_default_dtype())
    # Ensure vectors are not zero length for SH normalization
    xyz_input = xyz_input[torch.norm(xyz_input, dim=-1) > FUNC_EPS]
    # Add specific cases: axes
    axes = torch.tensor([
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0],
        [1.0, 1.0, 1.0], [-1.0, 2.0, -0.5]
    ], device=DEVICE, dtype=torch.get_default_dtype())
    xyz_input = torch.cat([xyz_input, axes], dim=0)
    
    # Target vector: positive z-axis
    z_axis = torch.tensor([[0.0, 0.0, 1.0]], device=DEVICE, dtype=torch.get_default_dtype()).expand(xyz_input.shape[0], -1)

    # --- Calculations ---
    # 1. Alignment D-matrix for each xyz vector
    # D should rotate xyz to align with +z
    D_align = align_module(xyz_input) # Shape (N, dim, dim)

    # 2. Spherical harmonics of the original vectors
    # sh_module normalizes input internally
    sh_orig = sh_module(xyz_input) # Shape (N, dim)

    # 3. Spherical harmonics of the target z-axis vector
    # Need to compute SH for z-axis for comparison.
    # Since sh_module expects batch, we expand z_axis
    sh_z = sh_module(z_axis) # Shape (N, dim) - should be the same for all rows

    # 4. Rotate the original spherical harmonics using the alignment matrix
    # We expect D_align @ sh_orig ~= sh_z
    # Need to unsqueeze sh_orig for matrix multiplication: (N, dim, dim) @ (N, dim, 1)
    sh_rotated = torch.matmul(D_align, sh_orig.unsqueeze(-1)).squeeze(-1) # Shape (N, dim)

    # --- Comparison ---
    # Compare the rotated SH with the SH of the z-axis vector
    diff = max_abs_diff(sh_rotated, sh_z)
    print(f"Max absolute difference between rotated SH and z-axis SH: {diff}")

    # Check specific components (optional, for debugging)
    # For l=0 (scalar), component should be invariant and equal sh_z[0,0]
    l0_dim = 1
    diff_l0 = max_abs_diff(sh_rotated[:, :l0_dim], sh_z[:, :l0_dim])
    print(f"Max absolute difference for l=0 component: {diff_l0}")
    
    # For m!=0 components of sh_z, they should be zero.
    # Check if rotated m!=0 components are close to zero.
    non_m0_mask = torch.ones(irreps.dim, dtype=torch.bool)
    offset = 0
    for l in range(l_max + 1):
        dim_l = 2 * l + 1
        m0_index_in_block = l # index of m=0 component within the block
        global_m0_index = offset + m0_index_in_block
        non_m0_mask[global_m0_index] = False
        offset += dim_l
        
    max_diff_non_m0_rotated = torch.max(torch.abs(sh_rotated[:, non_m0_mask])).item()
    max_diff_non_m0_z = torch.max(torch.abs(sh_z[:, non_m0_mask])).item()
    print(f"Max absolute value of non-m=0 components in rotated SH: {max_diff_non_m0_rotated}")
    print(f"Max absolute value of non-m=0 components in z-axis SH (should be ~0): {max_diff_non_m0_z}")
    assert max_diff_non_m0_z < TEST_EPS * 10 # Check reference calculation
    assert max_diff_non_m0_rotated < TEST_EPS * 10 # Check rotated result

    # Overall check
    assert diff < TEST_EPS * 10, f"Alignment rotation failed. Max diff: {diff}" # Allow slightly larger tolerance
    print("AlignToZWignerD module test passed.")


if __name__ == "__main__":
    test_align_to_z_rotation()
    print("\nAll AlignToZWignerD module tests finished.")
