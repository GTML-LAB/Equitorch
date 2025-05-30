import torch
import math
from equitorch.irreps import Irreps, check_irreps, element_degrees
# from equitorch.ops import rotate_z # Import the Triton kernel wrapper
from test_utils import FunctionTester, rate_mean2_std, rate_mean2_std_list, rand_irreps_feature # Import necessary test utils

torch.random.manual_seed(0)

def _z_rot_mat(angle, l):
    r"""
    Create the matrix representation of a z-axis rotation by the given angle,
    in the irrep l of dimension 2 * l + 1, in the basis of real centered
    spherical harmonics (RC basis in rep_bases.py).

    Note: this function is easy to use, but inefficient: only the entries
    on the diagonal and anti-diagonal are non-zero, so explicitly constructing
    this matrix is unnecessary.
    """
    shape, device, dtype = angle.shape, angle.device, angle.dtype
    M = angle.new_zeros((*shape, 2 * l + 1, 2 * l + 1))
    inds = torch.arange(0, 2 * l + 1, 1, device=device)
    reversed_inds = torch.arange(2 * l, -1, -1, device=device)
    frequencies = torch.arange(l, -l - 1, -1, dtype=dtype, device=device)
    M[..., inds, reversed_inds] = torch.sin(frequencies * angle[..., None])
    M[..., inds, inds] = torch.cos(frequencies * angle[..., None])
    return M


def prepare_z_rotation(irreps: Irreps, device=None, dtype=torch.float32):
    r"""
    Prepares tensors required for the rotate_z kernel based on input Irreps.

    Args:
        irreps (Irreps): The irreducible representations defining the tensor structure.
        device: The torch device for the output tensors.
        dtype: The torch dtype for the 'sign' tensor (should match input tensor dtype).

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - abs_m (torch.Tensor): Absolute value of m for each M index (dtype=torch.int32).
            - sign (torch.Tensor): Sign of m (+1.0, -1.0, or 0.0) for each M index (dtype=dtype).
            - idx_neg (torch.Tensor): Index mapping M to ~M (index for -m) (dtype=torch.int64).
    """
    irreps = check_irreps(irreps) # Ensure it's an Irreps object

    abs_m_list = []
    sign_list = []
    idx_to_irrep_m = [] # Stores (irrep_instance_idx, m) for each global index M
    m_to_idx_map = {}   # Stores global index M for each (irrep_instance_idx, m)
    current_idx = 0

    # First pass: Calculate abs_m, sign, and build mappings
    for irrep_instance_idx, irrep in enumerate(irreps): # Iterates through expanded irreps
        l = irrep.l
        start_idx = current_idx
        for m in range(-l, l + 1):
            abs_m_val = abs(m)
            sign_val = 1.0 if m > 0 else -1.0 if m < 0 else 0.0

            abs_m_list.append(abs_m_val)
            sign_list.append(sign_val)

            idx_to_irrep_m.append((irrep_instance_idx, m))
            m_to_idx_map[(irrep_instance_idx, m)] = current_idx
            current_idx += 1

    # Second pass: Calculate idx_neg using the map
    idx_neg_list = []
    M_total = irreps.dim
    for M in range(M_total):
        irrep_instance_idx, m = idx_to_irrep_m[M]
        # Find the global index corresponding to -m within the same irrep instance
        neg_m_global_idx = m_to_idx_map[(irrep_instance_idx, -m)]
        idx_neg_list.append(neg_m_global_idx)

    # Convert lists to tensors
    abs_m = torch.tensor(abs_m_list, dtype=torch.int32, device=device)
    sign = torch.tensor(sign_list, dtype=dtype, device=device)
    idx_neg = torch.tensor(idx_neg_list, dtype=torch.int64, device=device)

    return abs_m, sign, idx_neg


def rotate_z_pytorch(input: torch.Tensor,
                     cos: torch.Tensor,
                     sin: torch.Tensor,
                     abs_m: torch.Tensor,
                     sign: torch.Tensor,
                     idx_neg: torch.Tensor,
                     output: torch.Tensor = None):
    r"""
    PyTorch implementation of Z-axis rotation using index_select.

    Implements the transformation:
    x'_nMc = cos_nm * x_nMc + sign_M * sin_nm * x_n~Mc
    where cos_nm = cos[n, abs_m[M]] and sin_nm = sin[n, abs_m[M]].

    Args:
        input (torch.Tensor): Input tensor of shape (N, M_total, C).
        cos (torch.Tensor): Cosine values table of shape (N, max_abs_m + 1).
        sin (torch.Tensor): Sine values table of shape (N, max_abs_m + 1).
        abs_m (torch.Tensor): Absolute value of m for each M index, shape (M_total,).
        sign (torch.Tensor): Sign of m (+1.0, -1.0, or 0.0) for each M index, shape (M_total,).
        idx_neg (torch.Tensor): Index mapping M to ~M (index for -m), shape (M_total,).
        output (torch.Tensor, optional): Output tensor. If None, created internally. Defaults to None.

    Returns:
        torch.Tensor: Output tensor of shape (N, M_total, C) with rotations applied.
    """
    N, M_total, C = input.shape
    device = input.device

    # Gather cos and sin values corresponding to abs_m for each M using index_select
    # Shape: (N, M_total)
    cos_nm = cos.index_select(dim=-1, index=abs_m)
    sin_nm = sin.index_select(dim=-1, index=abs_m)

    # Gather input values corresponding to the negative m index (~M)
    # x_n~Mc has shape (N, M_total, C)
    # This is equivalent to torch.index_select(input, 1, idx_neg)
    x_neg_m = input.index_select(dim=-2, index=idx_neg)

    # Apply the rotation formula
    # cos_nm and sin_nm need broadcasting to (N, M_total, 1)
    # sign needs broadcasting to (1, M_total, 1)
    rotated_input = cos_nm[..., None] * input - sign[None, :, None] * sin_nm[..., None] * x_neg_m

    if output is None:
        output = rotated_input
    else:
        # Ensure output has correct shape, dtype, device if provided
        assert output.shape == input.shape
        assert output.dtype == input.dtype
        assert output.device == input.device
        output.copy_(rotated_input)

    return output


def rotate_z_reference(input: torch.Tensor,
                       angle: torch.Tensor,
                       irreps: Irreps,
                       output: torch.Tensor = None):
    r"""
    Reference implementation of Z-axis rotation using explicit matrix multiplication
    on each irrep slice.

    Args:
        input (torch.Tensor): Input tensor of shape (N, M_total, C).
        angle (torch.Tensor): Rotation angles for each batch element, shape (N,).
        irreps (Irreps): The irreducible representations defining the tensor structure.
        output (torch.Tensor, optional): Output tensor. If None, created internally. Defaults to None.

    Returns:
        torch.Tensor: Output tensor of shape (N, M_total, C) with rotations applied.
    """
    irreps = check_irreps(irreps)
    N, M_total, C = input.shape
    assert M_total == irreps.dim, "Input M dimension does not match irreps dimension"
    assert angle.shape == (N,), f"Angle shape mismatch: {angle.shape} vs {(N,)}"
    device = input.device
    dtype = input.dtype

    output_slices = []
    current_dim = 0
    for irrep_instance_idx, irrep in enumerate(irreps): # Iterate through expanded irreps
        l = irrep.l
        irrep_dim = irrep.dim
        input_slice = input[:, current_dim : current_dim + irrep_dim, :] # Shape (N, irrep_dim, C)

        # Get rotation matrix for this irrep and angles
        # Shape (N, irrep_dim, irrep_dim)
        rot_mat = _z_rot_mat(angle, l).to(dtype=dtype, device=device)

        # Apply rotation: einsum('nij,njk->nik', rot_mat, input_slice)
        # rot_mat: (N, irrep_dim, irrep_dim)
        # input_slice: (N, irrep_dim, C)
        # rotated_slice: (N, irrep_dim, C)
        rotated_slice = torch.einsum('nij,njk->nik', rot_mat, input_slice)
        output_slices.append(rotated_slice)

        current_dim += irrep_dim

    # Concatenate slices along the M dimension
    output = torch.cat(output_slices, dim=1) # Write directly to output tensor

    return output


def init_z_rotation(irreps: Irreps, N: int, C: int = None, device=None, dtype=torch.float32, need_grad=False):
    r"""Initializes random tensors for testing Z-rotation."""
    irreps = check_irreps(irreps)
    input_tensor = rand_irreps_feature(irreps, N, C, dtype=dtype, device=device)
    angle = torch.rand(N, device=device, dtype=dtype) * 2 * math.pi # Random angles from 0 to 2*pi

    if need_grad:
        input_tensor.requires_grad_(True)
        angle.requires_grad_(True) # For reference implementation gradient check

    # Prepare inputs for rotate_z_pytorch and rotate_z (Triton)
    abs_m, sign, idx_neg = prepare_z_rotation(irreps, device=device, dtype=dtype)

    # Calculate cos and sin tables
    max_abs_m = abs_m.max().item()
    m_range = torch.arange(max_abs_m + 1, device=device, dtype=dtype)
    # angle shape (N,), m_range shape (max_abs_m+1,) -> angle[:, None] * m_range[None, :] shape (N, max_abs_m+1)
    angle_m = angle[:, None] * m_range[None, :]
    cos_table = torch.cos(angle_m)
    sin_table = torch.sin(angle_m)

    # Prepare grad_output
    grad_output = torch.randn_like(input_tensor)

    return {
        "irreps": irreps,
        "input": input_tensor,
        "angle": angle,
        "abs_m": abs_m,
        "sign": sign,
        "idx_neg": idx_neg,
        "cos_table": cos_table,
        "sin_table": sin_table,
        "grad_output": grad_output
    }


# --- Wrapper Functions for Testing ---

def forward_ref(data):
    r"""Forward pass using the reference implementation."""
    return rotate_z_reference(data["input"], data["angle"], data["irreps"])

def forward_torch(data):
    r"""Forward pass using the optimized PyTorch implementation."""
    return rotate_z_pytorch(data["input"], data["cos_table"], data["sin_table"],
                            data["abs_m"], data["sign"], data["idx_neg"])

# Note: Triton version (rotate_z) would need a similar wrapper if tested here.

def backward_ref(data):
    r"""Backward pass for the reference implementation."""
    # Ensure grads are zero before backward
    if data["input"].grad is not None:
        data["input"].grad.zero_()
    if data["angle"].grad is not None:
        data["angle"].grad.zero_()

    res = rotate_z_reference(data["input"], data["angle"], data["irreps"])
    res.backward(data["grad_output"])
    # Return grads relevant for comparison (input and angle)
    return data["input"].grad, data["angle"].grad

def backward_torch(data):
    r"""Backward pass for the optimized PyTorch implementation."""
     # Ensure grads are zero before backward
    if data["input"].grad is not None:
        data["input"].grad.zero_()
    if data["angle"].grad is not None: # angle grad is implicitly computed via cos/sin tables
        data["angle"].grad.zero_()

    res = rotate_z_pytorch(data["input"], data["cos_table"], data["sin_table"],
                           data["abs_m"], data["sign"], data["idx_neg"])
    res.backward(data["grad_output"])
    # Return input and angle grads
    return data["input"].grad, data["angle"].grad


# --- Test Function ---

def test_z_rotation(irreps_str, N, C=None, device=None, dtype=torch.float32):
    r"""Tests consistency between reference and PyTorch implementations."""
    print(f"\n--- Testing Z Rotation: irreps='{irreps_str}', N={N}, C={C}, dtype={dtype} ---")
    irreps = Irreps(irreps_str)

    # -- Forward Test --
    print("Forward Pass Comparison:")
    inputs_forward = init_z_rotation(irreps, N, C, device=device, dtype=dtype, need_grad=False)
    tester_fwd = FunctionTester({
        'ref': (forward_ref, [inputs_forward], {}),
        'torch': (forward_torch, [inputs_forward], {}),
        # Add 'triton': (forward_triton, [inputs_forward], {}) here if testing Triton
    })
    comp_fwd = tester_fwd.compare(compare_func=rate_mean2_std)
    print(comp_fwd)

    # -- Backward Test --
    print("\nBackward Pass Comparison (Input Grad, Angle Grad):")
    inputs_backward = init_z_rotation(irreps, N, C, device=device, dtype=dtype, need_grad=True)
    tester_bwd = FunctionTester({
        'ref': (backward_ref, [inputs_backward], {}),
        'torch': (backward_torch, [inputs_backward], {}),
         # Add 'triton': (backward_triton, [inputs_backward], {}) here if testing Triton
    })
    # Compare both input and angle gradients
    comp_bwd = tester_bwd.compare(
        # No post_transforms needed as both return (input_grad, angle_grad)
        compare_func=rate_mean2_std_list # Use list comparison
    )
    print(comp_bwd)


# Example Usage (can be removed or kept for testing)
if __name__ == '__main__':
    # irreps_str = "1x0e + 2x1o + 1x2e"
    # irreps = Irreps(irreps_str)
    # print(f"Irreps: {irreps}")
    # print(f"Total dimension: {irreps.dim}")
    # abs_m, sign, idx_neg = prepare_z_rotation(irreps)


    N = 256
    C = 64
    # irreps = "1"
    irreps = (0,4)
    # --- Run Tests ---
    # test_z_rotation("1x0e+2x1o+1x2e", N=N, C=C, dtype=torch.float64)
    test_z_rotation(irreps, N=N, C=C, dtype=torch.float32)
    if torch.cuda.is_available():
        test_z_rotation(irreps, N=N, C=C, device='cuda', dtype=torch.float32)
