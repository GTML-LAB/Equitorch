# test/test_op/test_rotate_z.py
import torch
from typing import List, NamedTuple, Tuple, Dict, Optional

# Assume these imports are available based on the project structure
# Add try-except for robustness if run standalone
import math
from typing import Tuple # Add Tuple import
from equitorch.irreps import Irrep, Irreps, check_irreps
from equitorch.structs import SparseProductInfo, add_operation_methods
# Use the helper that generates all three infos needed for autograd
from equitorch.utils._structs import sparse_product_infos
from equitorch.nn.functional.sparse_product import sparse_vecsca # Add sparse_scavec import

import e3nn

def z_rotation_infos(irreps: Irreps) -> Tuple[SparseProductInfo, SparseProductInfo, SparseProductInfo]:
    """
    Generates SparseProductInfo for performing z-axis rotation on a tensor
    with the given Irreps structure using the sparse_scavec operation.

    Rotation formula:
        m=0: x'_{nimc} = x_{ni0c}
        m!=0: x'_{nimc} = cos(m*phi) * x_{nimc} + sin(m*phi) * x_{ni(-m)c}

    Assumes sparse_scavec computes:
        output[n, M] = sum_t scale[t] * input[n, index2[t]] * cs[n, index1[t]]
    where the sum is segmented by the output index M (implicit via index_out).

    Assumed `cs` Tensor Structure (Input 2): shape (N, cs_dim)
        cs_dim = 1 + 2 * max_l
        cs[:, 0] = 1.0
        cs[:, 2*m - 1] = sin(m*phi) for m = 1..max_l
        cs[:, 2*m]     = cos(m*phi) for m = 1..max_l

    Args:
        irreps: The Irreps object describing the geometric structure of the
                input tensor (Input 1, shape (N, irreps.dim, C)) and
                output tensor (shape (N, irreps.dim, C)).

    Returns:
        A tuple (info_fwd, info_bwd1, info_bwd2) containing SparseProductInfo
        objects for the forward pass and gradients w.r.t. x (input1) and
        cs (input2).
    """
    indices1: List[int] = [] # Index into x (input1)
    indices2: List[int] = [] # Index into cs (input2)
    indices_out: List[int] = [] # Output index M
    scales: List[float] = [] # Scale factor sigma_t

    # 1. Create mapping from M -> (irrep_idx, l, m) and (irrep_idx, l, m) -> M
    m_to_info: Dict[int, Tuple[int, int, int]] = {}
    lm_to_m_indices: Dict[Tuple[int, int, int], int] = {}
    current_m_index = 0
    max_l = 0
    for irrep_idx, irrep in enumerate(irreps):
        l = irrep.l
        max_l = max(max_l, l)
        for m in range(-l, l + 1):
            m_to_info[current_m_index] = (irrep_idx, l, m)
            # Use irrep_idx from the original list as part of the key
            lm_to_m_indices[(irrep_idx, l, m)] = current_m_index
            current_m_index += 1

    if irreps.dim != current_m_index:
        # This indicates an issue with Irreps definition or parsing
        raise RuntimeError(f"Calculated dimension ({current_m_index}) does not match irreps.dim ({irreps.dim})")

    cs_dim = 1 + 2 * max_l # Size of the dimension in cs tensor

    # 2. Iterate through output indices M and generate sparse interactions
    for M in range(irreps.dim):
        irrep_idx, l, m = m_to_info[M]

        if m == 0:
            # x'_{M} = 1.0 * x_{M}
            # Interaction t: M_out=M, scale=1.0, cs_idx=0 (for 1.0), x_idx=M
            indices_out.append(M)
            scales.append(1.0)
            indices1.append(M)
            indices2.append(0) # Index 0 in cs holds 1.0
        else:
            # x'_{M} = cos(m*phi) * x_{M} + sin(m*phi) * x_{M(-m)}
            abs_m = abs(m)
            sign_m = float(m > 0) - float(m < 0)

            # Find M_neg_m index for the same irrep_idx and l, but opposite m
            M_neg_m = lm_to_m_indices.get((irrep_idx, l, -m))
            if M_neg_m is None:
                 # This should not happen for valid irreps
                 raise ValueError(f"Could not find index for irrep_idx={irrep_idx}, l={l}, m={-m} corresponding to M={M}")

            # Interaction t1 (cos term): M_out=M, scale=1.0, cs_idx=2*|m|, x_idx=M
            indices_out.append(M)
            scales.append(1.0)
            indices1.append(M)
            indices2.append(2 * abs_m) # Index for cos(m*phi)

            # Interaction t2 (sin term): M_out=M, scale=-sign(m), cs_idx=2*|m|-1, x_idx=M_neg_m
            indices_out.append(M)
            scales.append(-sign_m)
            indices1.append(M_neg_m) # Index into x for the sin term
            indices2.append(2 * abs_m - 1) # Index for sin(|m|*phi)

    # 3. Create SparseProductInfo using the helper that returns all three infos
    # sparse_product_infos handles sorting by index_out ('index' arg) and creates segments.
    # It requires sizes for input1 (x), input2 (cs), and output (x').
    # Input1 (x) size = irreps.dim
    # Input2 (cs) size = cs_dim
    # Output (x') size = irreps.dim
    infos = sparse_product_infos(
        index1=indices1,    # Indices into x tensor (Input 1)
        index2=indices2,    # Indices into cs tensor (Input 2)
        index=indices_out,   # Defines output index M for segmentation
        scale=scales,        # Scaling factor sigma_t
        out_size=irreps.dim, # Size of the output sparse dimension (x')
        in1_size=irreps.dim,  # Size of the input1 sparse dimension (x)
        in2_size=cs_dim,     # Size of the input2 sparse dimension (cs)
    )
    # info_fwd, info_bwd1 (grad_cs), info_bwd2 (grad_x)
    return infos

def prepare_sincos(angle, max_m):
    m = torch.arange(1, max_m+1, dtype=angle.dtype, device=angle.device)
    m_angle = angle.unsqueeze(-1) * m
    sin = torch.sin(m_angle)
    cos = torch.cos(m_angle)
    ones = torch.ones_like(angle).unsqueeze(-1)
    return torch.cat([ones, torch.stack([sin, cos], dim=-1).flatten(-2, -1)], dim=-1)

def prepare_wigner(angle, irreps: Irreps):
    irreps_e3nn = e3nn.o3.Irreps("+".join(f"{mul}x{irrep.l}{'o' if irrep.p==-1 else 'e'}" for irrep, mul in irreps.irrep_groups))
    return irreps_e3nn.D_from_angles(torch.zeros_like(angle), torch.zeros_like(angle), angle)

class RefWignerRotationZ(torch.nn.Module):

    def __init__(self, irreps):
        super().__init__()
        self.irreps = check_irreps(irreps)

    def forward(self, input, angle):
        D = prepare_wigner(angle, self.irreps)
        return D @ input

class WignerRotationZ(torch.nn.Module):

    def __init__(self, irreps):
        super().__init__()
        self.irreps = check_irreps(irreps)
        self.max_m = max(ir.l for ir in self.irreps)
        self.info_fwd, self.info_bwd_input, self.info_bwd_sincos = z_rotation_infos(irreps)

    def forward(self, input, angle):
        sincos = prepare_sincos(angle, self.max_m)
        return sparse_vecsca(
            input, sincos, self.info_fwd, 
            self.info_bwd_input, self.info_bwd_sincos)

    def _apply(self, *args, **kwargs):
        wig = super()._apply(*args, **kwargs)
        wig.info_fwd = self.info_fwd._apply(*args, **kwargs)
        wig.info_bwd_input = self.info_bwd_input._apply(*args, **kwargs)
        wig.info_bwd_sincos = self.info_bwd_sincos._apply(*args, **kwargs)
        return wig


# --- Test Suite ---
from test_utils import FunctionTester, max_abs_diff_list, max_abs_diff

torch.random.manual_seed(0)
torch.set_default_dtype(torch.float64)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":

    irreps = Irreps("1x0e + 2x1o + 1x2e") # Use a slightly more complex irrep
    # Expected dim: 1*1 + 2*3 + 1*5 = 1 + 6 + 5 = 12
    print(f"Testing Irreps: {irreps} (dim={irreps.dim})")
    print(f"Device: {DEVICE}")

    N = 5 # Batch size
    C = 7 # Channels
    dtype = torch.get_default_dtype()

    # --- Instantiate Modules ---
    ref_module = RefWignerRotationZ(irreps).to(device=DEVICE, dtype=dtype)
    eqt_module = WignerRotationZ(irreps).to(device=DEVICE, dtype=dtype)

    # --- Prepare Inputs ---
    # Note: RefWignerRotationZ expects input (x) first, then angle (sincos)
    # WignerRotationZ expects input (x) first, then angle
    x = torch.randn(N, irreps.dim, C, device=DEVICE, dtype=dtype, requires_grad=True)
    angle = torch.rand(N, device=DEVICE, dtype=dtype, requires_grad=True) * 2 * torch.pi # Angle per batch item

    # --- Forward Test ---
    print("\n--- Forward Test ---")
    tester_fwd = FunctionTester({
        'eqt': (eqt_module, [x, angle], {}),
        'ref': (ref_module, [x, angle], {}),
    })
    comp_fwd = tester_fwd.compare(compare_func=max_abs_diff)
    print(comp_fwd)
    # assert comp_fwd[('eqt', 'ref')] < 1e-7 # Check if results are close

    # --- Backward Test ---
    print("\n--- Backward Test ---")
    x_eqt = x.clone().detach().requires_grad_(True)
    angle_eqt = angle.clone().detach().requires_grad_(True)
    x_ref = x.clone().detach().requires_grad_(True)
    angle_ref = angle.clone().detach().requires_grad_(True)

    # Run forward to get output shape for grad_out
    out_ref = ref_module(x_ref, angle_ref)
    grad_out = torch.randn_like(out_ref)

    def backward_eqt(x, ang, grad):
        res = eqt_module(x, ang)
        res.backward(grad, retain_graph=True) # Retain graph for potential second order
        g_x = x.grad.clone() if x.grad is not None else torch.zeros_like(x)
        g_ang = ang.grad.clone() if ang.grad is not None else torch.zeros_like(ang)
        x.grad = None # Clear grads for next test
        ang.grad = None
        return [g_x, g_ang]

    def backward_ref(x, ang, grad):
        res = ref_module(x, ang)
        res.backward(grad, retain_graph=True)
        g_x = x.grad.clone() if x.grad is not None else torch.zeros_like(x)
        g_ang = ang.grad.clone() if ang.grad is not None else torch.zeros_like(ang)
        x.grad = None
        ang.grad = None
        return [g_x, g_ang]

    tester_bwd = FunctionTester({
        'eqt': (backward_eqt, [x_eqt, angle_eqt, grad_out], {}),
        'ref': (backward_ref, [x_ref, angle_ref, grad_out], {}),
    })
    comp_bwd = tester_bwd.compare(compare_func=max_abs_diff_list)
    print(comp_bwd)
    # assert all(c < 1e-7 for c in comp_bwd[('eqt', 'ref')]) # Check if grads are close

    # --- Second-Order Backward Test (Grad-Grad) ---
    print("\n--- Second-Order Backward Test (Grad-Grad) ---")
    x_eqt_gg = x.clone().detach().requires_grad_(True)
    angle_eqt_gg = angle.clone().detach().requires_grad_(True)
    x_ref_gg = x.clone().detach().requires_grad_(True)
    angle_ref_gg = angle.clone().detach().requires_grad_(True)

    # Need vectors for grad products
    v_out = torch.randn_like(out_ref)
    # Get shapes of first gradients
    grads_ref_shape = backward_ref(x_ref.clone().detach().requires_grad_(True),
                                   angle_ref.clone().detach().requires_grad_(True),
                                   torch.ones_like(out_ref))
    v_in_x = torch.randn_like(grads_ref_shape[0])
    v_in_angle = torch.randn_like(grads_ref_shape[1])

    def gradgrad_eqt(x, ang, v_out, v_in_x, v_in_angle):
        res = eqt_module(x, ang)
        grads_x, grads_angle = torch.autograd.grad(
            outputs=res, inputs=(x, ang), grad_outputs=v_out, create_graph=True, allow_unused=True
        )
        grads_x = grads_x if grads_x is not None else torch.zeros_like(x)
        grads_angle = grads_angle if grads_angle is not None else torch.zeros_like(ang)

        gradgrad_x, gradgrad_angle = torch.autograd.grad(
            outputs=(grads_x, grads_angle), inputs=(x, ang), grad_outputs=(v_in_x, v_in_angle), allow_unused=True
        )
        gg_x = gradgrad_x if gradgrad_x is not None else torch.zeros_like(x)
        gg_angle = gradgrad_angle if gradgrad_angle is not None else torch.zeros_like(ang)
        return [gg_x, gg_angle]

    def gradgrad_ref(x, ang, v_out, v_in_x, v_in_angle):
        res = ref_module(x, ang)
        grads_x, grads_angle = torch.autograd.grad(
            outputs=res, inputs=(x, ang), grad_outputs=v_out, create_graph=True, allow_unused=True
        )
        grads_x = grads_x if grads_x is not None else torch.zeros_like(x)
        grads_angle = grads_angle if grads_angle is not None else torch.zeros_like(ang)

        gradgrad_x, gradgrad_angle = torch.autograd.grad(
            outputs=(grads_x, grads_angle), inputs=(x, ang), grad_outputs=(v_in_x, v_in_angle), allow_unused=True
        )
        gg_x = gradgrad_x if gradgrad_x is not None else torch.zeros_like(x)
        gg_angle = gradgrad_angle if gradgrad_angle is not None else torch.zeros_like(ang)
        return [gg_x, gg_angle]

    tester_gradgrad = FunctionTester({
        'eqt': (gradgrad_eqt, [x_eqt_gg, angle_eqt_gg, v_out, v_in_x, v_in_angle], {}),
        'ref': (gradgrad_ref, [x_ref_gg, angle_ref_gg, v_out, v_in_x, v_in_angle], {}),
    })
    comp_gradgrad = tester_gradgrad.compare(compare_func=max_abs_diff_list)
    print(comp_gradgrad)
    # assert all(c < 1e-6 for c in comp_gradgrad[('eqt', 'ref')]) # Check if grad-grads are close

    # --- Second-Order Cross Gradient Check (d(dL/dx)/d(angle)) ---
    print("\n--- Second-Order Cross Gradient Check (d(dL/dx)/d(angle)) ---")
    v_in_angle_zero = torch.zeros_like(v_in_angle)
    x_eqt_gg_dxda = x.clone().detach().requires_grad_(True)
    angle_eqt_gg_dxda = angle.clone().detach().requires_grad_(True)
    x_ref_gg_dxda = x.clone().detach().requires_grad_(True)
    angle_ref_gg_dxda = angle.clone().detach().requires_grad_(True)

    tester_gradgrad_dxda = FunctionTester({
        'eqt': (lambda: gradgrad_eqt(x_eqt_gg_dxda, angle_eqt_gg_dxda, v_out, v_in_x, v_in_angle_zero)[1], [], {}), # Get grad w.r.t angle
        'ref': (lambda: gradgrad_ref(x_ref_gg_dxda, angle_ref_gg_dxda, v_out, v_in_x, v_in_angle_zero)[1], [], {}), # Get grad w.r.t angle
    })
    comp_gradgrad_dxda = tester_gradgrad_dxda.compare(compare_func=max_abs_diff)
    print(comp_gradgrad_dxda)
    # assert comp_gradgrad_dxda[('eqt', 'ref')] < 1e-6

    # --- Second-Order Cross Gradient Check (d(dL/d(angle))/dx) ---
    print("\n--- Second-Order Cross Gradient Check (d(dL/d(angle))/dx) ---")
    v_in_x_zero = torch.zeros_like(v_in_x)
    x_eqt_gg_dadx = x.clone().detach().requires_grad_(True)
    angle_eqt_gg_dadx = angle.clone().detach().requires_grad_(True)
    x_ref_gg_dadx = x.clone().detach().requires_grad_(True)
    angle_ref_gg_dadx = angle.clone().detach().requires_grad_(True)

    tester_gradgrad_dadx = FunctionTester({
        'eqt': (lambda: gradgrad_eqt(x_eqt_gg_dadx, angle_eqt_gg_dadx, v_out, v_in_x_zero, v_in_angle)[0], [], {}), # Get grad w.r.t x
        'ref': (lambda: gradgrad_ref(x_ref_gg_dadx, angle_ref_gg_dadx, v_out, v_in_x_zero, v_in_angle)[0], [], {}), # Get grad w.r.t x
    })
    comp_gradgrad_dadx = tester_gradgrad_dadx.compare(compare_func=max_abs_diff)
    print(comp_gradgrad_dadx)
    # assert comp_gradgrad_dadx[('eqt', 'ref')] < 1e-6

    print("\nAll tests finished.")
