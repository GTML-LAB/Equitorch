import torch
from equitorch.irreps import Irrep, Irreps, check_irreps
from equitorch.utils._structs import wigner_d_info
from equitorch.nn.functional.sparse_product import sparse_vecsca
from equitorch.nn.functional.sparse_scale import sparse_scale # Import sparse_scale

import torch
from torch import Tensor

import e3nn
from sympy import Matrix
import numpy as np

torch.set_default_dtype(torch.float64)
# torch.set_default_dtype(torch.float32)


from functools import lru_cache


def prepare_sincos(angle, max_m):
    r"""Prepares the sin/cos tensor for z-rotation."""
    if max_m == 0:
        # Only scalar irreps, rotation is identity
        return torch.ones_like(angle).unsqueeze(-1)
    m = torch.arange(1, max_m+1, dtype=angle.dtype, device=angle.device)
    m_angle = angle.unsqueeze(-1) * m
    sin_abs_m = torch.sin(m_angle) # sin(|m|*angle)
    cos_abs_m = torch.cos(m_angle) # cos(|m|*angle)
    ones = torch.ones_like(angle).unsqueeze(-1)
    # [1.0, sin(1a), cos(1a), sin(2a), cos(2a), ...]
    return torch.cat([ones, torch.stack([sin_abs_m, cos_abs_m], dim=-1).flatten(-2, -1)], dim=-1)

# --- End Copied ---


# --- Reference Implementation ---
class RefWignerRotation(torch.nn.Module):
    def __init__(self, irreps):
        super().__init__()
        self.irreps = check_irreps(irreps)
        # Convert to e3nn Irreps string format for D_from_angles
        self.irreps_e3nn = e3nn.o3.Irreps("+".join(f"{mul}x{irrep.l}{'o' if irrep.p==-1 else 'e'}" for irrep, mul in irreps.irrep_groups))

    def forward(self, input: Tensor, alpha: Tensor, beta: Tensor, gamma: Tensor) -> Tensor:
        """
        Args:
            input (Tensor): Shape (N, irreps.dim, C) or (irreps.dim, C)
            alpha (Tensor): Shape (N,) or scalar
            beta (Tensor): Shape (N,) or scalar
            gamma (Tensor): Shape (N,) or scalar
        Returns:
            Tensor: Rotated tensor, shape matching input.
        """
        # suppose input is correctly shaped in testing


        D = self.irreps_e3nn.D_from_angles(alpha, beta, gamma) # Shape (N, dim, dim) or (dim, dim)

        return D @ input


# --- Equitorch Implementation ---
class WignerDRotation(torch.nn.Module):
    def __init__(self, irreps):
        super().__init__()
        self.irreps = check_irreps(irreps)
        self.max_m = max(ir.l for ir in self.irreps) if self.irreps else 0
        self.info = wigner_d_info(irreps)

    def forward(self, input: Tensor, alpha: Tensor, beta: Tensor, gamma: Tensor) -> Tensor:
        """Applies D(alpha, beta, gamma) = Dz(alpha) J Dz(beta) J Dz(gamma)"""
        # Prepare sin/cos tensors for each angle
        sincos_a = prepare_sincos(alpha, self.max_m)
        sincos_b = prepare_sincos(beta, self.max_m)
        sincos_g = prepare_sincos(gamma, self.max_m)

        # Unpack infos
        # Note: The backward infos stored are primarily for the autograd wrapper,
        # but we need the forward infos here.
        dz_info_fwd = self.info.rotate_z_info_fwd
        dz_info_bwd_x = self.info.rotate_z_info_bwd_input
        dz_info_bwd_cs = self.info.rotate_z_info_bwd_cs
        j_info = self.info.j_matrix_info # Use the stored forward info (since J is symmetric, forward and backward are the same)



        # Apply sequence: Dz(gamma) -> J -> Dz(beta) -> J -> Dz(alpha)
        # Note: sparse_scale expects (input, info_fwd, info_bwd)
        # Note: sparse_vecsca expects (input1, input2, info_fwd, info_bwd1, info_bwd2)

        x0 = input
        # 1. Dz(gamma)
        x1 = sparse_vecsca(x0, sincos_g, dz_info_fwd, dz_info_bwd_x, dz_info_bwd_cs)
        # 2. J
        x2 = sparse_scale(x1, j_info, j_info)
        # 3. Dz(beta)
        x3 = sparse_vecsca(x2, sincos_b, dz_info_fwd, dz_info_bwd_x, dz_info_bwd_cs)
        # 4. J^T (which is J, so use same infos)
        x4 = sparse_scale(x3, j_info, j_info)
        # 5. Dz(alpha)
        out = sparse_vecsca(x4, sincos_a, dz_info_fwd, dz_info_bwd_x, dz_info_bwd_cs)

        return out

    def _apply(self, *args, **kwargs):
        # Ensure info objects are moved to the correct device/dtype
        wig = super()._apply(*args, **kwargs)
        # Apply to the WignerDRotationInfo NamedTuple fields
        wig.info = wig.info._apply(*args, **kwargs)
        return wig


# --- Test Suite ---
from test_utils import FunctionTester, max_abs_diff_list, max_abs_diff # Adjusted import path

torch.random.manual_seed(0)
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'
if __name__ == "__main__":

    irreps = Irreps("1x0e + 2x1o + 1x2e") # Use a slightly more complex irrep
    # Expected dim: 1*1 + 2*3 + 1*5 = 1 + 6 + 5 = 12
    print(f"Testing Irreps: {irreps} (dim={irreps.dim})")
    print(f"Device: {DEVICE}")

    N = 256 # Batch size
    C = 128 # Channels
    dtype = torch.get_default_dtype()

    # --- Instantiate Modules ---
    ref_module = RefWignerRotation(irreps).to(device=DEVICE, dtype=dtype)
    eqt_module = WignerDRotation(irreps).to(device=DEVICE, dtype=dtype)

    # --- Prepare Inputs ---
    x = torch.randn(N, irreps.dim, C, device=DEVICE, dtype=dtype, requires_grad=True)
    alpha = torch.rand(N, device=DEVICE, dtype=dtype, requires_grad=True) * 2 * torch.pi
    beta = torch.rand(N, device=DEVICE, dtype=dtype, requires_grad=True) * torch.pi # Beta usually in [0, pi]
    gamma = torch.rand(N, device=DEVICE, dtype=dtype, requires_grad=True) * 2 * torch.pi

    # --- Forward Test ---
    print("\n--- Forward Test ---")
    tester_fwd = FunctionTester({
        'eqt': (eqt_module, [x, alpha, beta, gamma], {}),
        'ref': (ref_module, [x, alpha, beta, gamma], {}),
    })
    comp_fwd = tester_fwd.compare(compare_func=max_abs_diff)
    print(comp_fwd)
    # assert comp_fwd[('eqt', 'ref')] < 1e-7 # Check if results are close

    # --- Backward Test ---
    print("\n--- Backward Test ---")
    x_eqt = x.clone().detach().requires_grad_(True)
    a_eqt = alpha.clone().detach().requires_grad_(True)
    b_eqt = beta.clone().detach().requires_grad_(True)
    g_eqt = gamma.clone().detach().requires_grad_(True)

    x_ref = x.clone().detach().requires_grad_(True)
    a_ref = alpha.clone().detach().requires_grad_(True)
    b_ref = beta.clone().detach().requires_grad_(True)
    g_ref = gamma.clone().detach().requires_grad_(True)

    # Run forward to get output shape for grad_out
    out_ref = ref_module(x_ref, a_ref, b_ref, g_ref)
    grad_out = torch.randn_like(out_ref)

    def backward_eqt(x, a, b, g, grad):
        res = eqt_module(x, a, b, g)
        res.backward(grad, retain_graph=True)
        gx = x.grad.clone() if x.grad is not None else torch.zeros_like(x)
        ga = a.grad.clone() if a.grad is not None else torch.zeros_like(a)
        gb = b.grad.clone() if b.grad is not None else torch.zeros_like(b)
        gg = g.grad.clone() if g.grad is not None else torch.zeros_like(g)
        x.grad = None; a.grad = None; b.grad = None; g.grad = None # Clear grads
        return [gx, ga, gb, gg]

    def backward_ref(x, a, b, g, grad):
        res = ref_module(x, a, b, g)
        res.backward(grad, retain_graph=True)
        gx = x.grad.clone() if x.grad is not None else torch.zeros_like(x)
        ga = a.grad.clone() if a.grad is not None else torch.zeros_like(a)
        gb = b.grad.clone() if b.grad is not None else torch.zeros_like(b)
        gg = g.grad.clone() if g.grad is not None else torch.zeros_like(g)
        x.grad = None; a.grad = None; b.grad = None; g.grad = None # Clear grads
        return [gx, ga, gb, gg]

    tester_bwd = FunctionTester({
        'eqt': (backward_eqt, [x_eqt, a_eqt, b_eqt, g_eqt, grad_out], {}),
        'ref': (backward_ref, [x_ref, a_ref, b_ref, g_ref, grad_out], {}),
    })
    comp_bwd = tester_bwd.compare(compare_func=max_abs_diff_list)
    print(comp_bwd)
    # assert all(c < 1e-7 for c in comp_bwd[('eqt', 'ref')]) # Check if grads are close

    # --- Second-Order Backward Test (Grad-Grad) ---
    # Test all second-order derivatives implicitly by computing vjp of vjp
    print("\n--- Second-Order Backward Test (Grad-Grad) ---")
    x_eqt_gg = x.clone().detach().requires_grad_(True)
    a_eqt_gg = alpha.clone().detach().requires_grad_(True)
    b_eqt_gg = beta.clone().detach().requires_grad_(True)
    g_eqt_gg = gamma.clone().detach().requires_grad_(True)

    x_ref_gg = x.clone().detach().requires_grad_(True)
    a_ref_gg = alpha.clone().detach().requires_grad_(True)
    b_ref_gg = beta.clone().detach().requires_grad_(True)
    g_ref_gg = gamma.clone().detach().requires_grad_(True)

    # Need vectors for grad products
    v_out = torch.randn_like(out_ref)
    # Get shapes of first gradients
    grads_ref_shape = backward_ref(x_ref.clone().detach().requires_grad_(True),
                                   a_ref.clone().detach().requires_grad_(True),
                                   b_ref.clone().detach().requires_grad_(True),
                                   g_ref.clone().detach().requires_grad_(True),
                                   torch.ones_like(out_ref))
    v_in_x = torch.randn_like(grads_ref_shape[0])
    v_in_a = torch.randn_like(grads_ref_shape[1])
    v_in_b = torch.randn_like(grads_ref_shape[2])
    v_in_g = torch.randn_like(grads_ref_shape[3])

    def gradgrad_eqt(x, a, b, g, v_out, v_in_x, v_in_a, v_in_b, v_in_g):
        res = eqt_module(x, a, b, g)
        # Compute first gradients w.r.t all inputs
        grads = torch.autograd.grad(
            outputs=res, inputs=(x, a, b, g), grad_outputs=v_out, create_graph=True, allow_unused=True
        )
        # Replace None grads with zeros
        grads_x = grads[0] if grads[0] is not None else torch.zeros_like(x)
        grads_a = grads[1] if grads[1] is not None else torch.zeros_like(a)
        grads_b = grads[2] if grads[2] is not None else torch.zeros_like(b)
        grads_g = grads[3] if grads[3] is not None else torch.zeros_like(g)

        # Compute second gradients (vjp of vjp)
        gradgrads = torch.autograd.grad(
            outputs=(grads_x, grads_a, grads_b, grads_g),
            inputs=(x, a, b, g),
            grad_outputs=(v_in_x, v_in_a, v_in_b, v_in_g),
            allow_unused=True
        )
        # Replace None grad-grads with zeros
        gg_x = gradgrads[0] if gradgrads[0] is not None else torch.zeros_like(x)
        gg_a = gradgrads[1] if gradgrads[1] is not None else torch.zeros_like(a)
        gg_b = gradgrads[2] if gradgrads[2] is not None else torch.zeros_like(b)
        gg_g = gradgrads[3] if gradgrads[3] is not None else torch.zeros_like(g)
        return [gg_x, gg_a, gg_b, gg_g]

    def gradgrad_ref(x, a, b, g, v_out, v_in_x, v_in_a, v_in_b, v_in_g):
        res = ref_module(x, a, b, g)
        # Compute first gradients w.r.t all inputs
        grads = torch.autograd.grad(
            outputs=res, inputs=(x, a, b, g), grad_outputs=v_out, create_graph=True, allow_unused=True
        )
        # Replace None grads with zeros
        grads_x = grads[0] if grads[0] is not None else torch.zeros_like(x)
        grads_a = grads[1] if grads[1] is not None else torch.zeros_like(a)
        grads_b = grads[2] if grads[2] is not None else torch.zeros_like(b)
        grads_g = grads[3] if grads[3] is not None else torch.zeros_like(g)

        # Compute second gradients (vjp of vjp)
        gradgrads = torch.autograd.grad(
            outputs=(grads_x, grads_a, grads_b, grads_g),
            inputs=(x, a, b, g),
            grad_outputs=(v_in_x, v_in_a, v_in_b, v_in_g),
            allow_unused=True
        )
        # Replace None grad-grads with zeros
        gg_x = gradgrads[0] if gradgrads[0] is not None else torch.zeros_like(x)
        gg_a = gradgrads[1] if gradgrads[1] is not None else torch.zeros_like(a)
        gg_b = gradgrads[2] if gradgrads[2] is not None else torch.zeros_like(b)
        gg_g = gradgrads[3] if gradgrads[3] is not None else torch.zeros_like(g)
        return [gg_x, gg_a, gg_b, gg_g]

    tester_gradgrad = FunctionTester({
        'eqt': (gradgrad_eqt, [x_eqt_gg, a_eqt_gg, b_eqt_gg, g_eqt_gg, v_out, v_in_x, v_in_a, v_in_b, v_in_g], {}),
        'ref': (gradgrad_ref, [x_ref_gg, a_ref_gg, b_ref_gg, g_ref_gg, v_out, v_in_x, v_in_a, v_in_b, v_in_g], {}),
    })
    comp_gradgrad = tester_gradgrad.compare(compare_func=max_abs_diff_list)
    print(comp_gradgrad)
    # Compare all 4 resulting grad-grad tensors
    # assert all(c < 1e-6 for c in comp_gradgrad[('eqt', 'ref')]) # Check if grad-grads are close

    print("\nAll tests finished.")
