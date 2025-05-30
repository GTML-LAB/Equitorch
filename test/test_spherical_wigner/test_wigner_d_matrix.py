import torch
from torch import Tensor
import e3nn
import os

from equitorch.irreps import Irreps, check_irreps
from equitorch.nn.wigner_d import WignerD
# Import helper from functional layer if needed, or redefine locally
from test_utils import FunctionTester, max_abs_diff_list, max_abs_diff

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# torch.set_default_dtype(torch.float64)
torch.set_default_dtype(torch.float32)

# --- Helper Functions ---

def prepare_wigner_ref(alpha, beta, gamma, irreps: Irreps):
    r"""Reference implementation using e3nn"""
    irreps_e3nn = e3nn.o3.Irreps(irreps.short_repr()) # Convert equitorch Irreps to e3nn string
    return irreps_e3nn.D_from_angles(alpha, beta, gamma)

# --- Test Suite ---

torch.random.manual_seed(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":

    irreps = Irreps("0e+1e+2e+3e+4e") # Example Irreps
    print(f"Testing Irreps: {irreps} (dim={irreps.dim})")
    print(f"Device: {DEVICE}")

    N = 128 # Batch size
    dtype = torch.get_default_dtype()

    # --- Instantiate Modules ---
    # Reference is just a function call
    eqt_module = WignerD(irreps).to(device=DEVICE, dtype=dtype)

    # --- Prepare Inputs ---
    # Angles are the inputs we differentiate w.r.t.
    alpha = torch.rand(N, device=DEVICE, dtype=dtype, requires_grad=True) * 2 * torch.pi
    beta = torch.rand(N, device=DEVICE, dtype=dtype, requires_grad=True) * torch.pi # Beta usually in [0, pi]
    gamma = torch.rand(N, device=DEVICE, dtype=dtype, requires_grad=True) * 2 * torch.pi

    # --- Forward Test ---
    print("\n--- Forward Test ---")
    # eqt_module = torch.jit.script(eqt_module) # Scripting might be tricky with optional inputs

    # Define forward functions for tester
    def forward_eqt(a, b, g):
        # Module expects angles directly
        return eqt_module(alpha=a, beta=b, gamma=g)

    def forward_ref(a, b, g):
        return prepare_wigner_ref(a, b, g, irreps)

    tester_fwd = FunctionTester({
        'eqt': (forward_eqt, [alpha, beta, gamma], {}),
        'ref': (forward_ref, [alpha, beta, gamma], {}),
    })
    comp_fwd = tester_fwd.compare(compare_func=max_abs_diff)
    print(comp_fwd)
    tester_fwd.profile(repeat=10) # Profiling might need adjustments
    # assert comp_fwd[('eqt', 'ref')] < 1e-7 # Check if results are close

    # --- Backward Test ---
    print("\n--- Backward Test ---")
    a_eqt = alpha.clone().detach().requires_grad_(True)
    b_eqt = beta.clone().detach().requires_grad_(True)
    g_eqt = gamma.clone().detach().requires_grad_(True)

    a_ref = alpha.clone().detach().requires_grad_(True)
    b_ref = beta.clone().detach().requires_grad_(True)
    g_ref = gamma.clone().detach().requires_grad_(True)

    # Run forward to get output shape for grad_out
    out_ref = prepare_wigner_ref(a_ref, b_ref, g_ref, irreps)
    # Gradient w.r.t. the D-matrix elements
    grad_out = torch.randn_like(out_ref)

    def backward_eqt(a, b, g, grad):
        res = eqt_module(alpha=a, beta=b, gamma=g)
        res.backward(grad, retain_graph=True)
        ga = a.grad.clone() if a.grad is not None else torch.zeros_like(a)
        gb = b.grad.clone() if b.grad is not None else torch.zeros_like(b)
        gg = g.grad.clone() if g.grad is not None else torch.zeros_like(g)
        a.grad = None; b.grad = None; g.grad = None # Clear grads
        return [ga, gb, gg] # Gradients w.r.t. angles

    def backward_ref(a, b, g, grad):
        res = prepare_wigner_ref(a, b, g, irreps)
        res.backward(grad, retain_graph=True)
        ga = a.grad.clone() if a.grad is not None else torch.zeros_like(a)
        gb = b.grad.clone() if b.grad is not None else torch.zeros_like(b)
        gg = g.grad.clone() if g.grad is not None else torch.zeros_like(g)
        a.grad = None; b.grad = None; g.grad = None # Clear grads
        return [ga, gb, gg] # Gradients w.r.t. angles

    tester_bwd = FunctionTester({
        'eqt': (backward_eqt, [a_eqt, b_eqt, g_eqt, grad_out], {}),
        'ref': (backward_ref, [a_ref, b_ref, g_ref, grad_out], {}),
    })
    comp_bwd = tester_bwd.compare(compare_func=max_abs_diff_list)
    print(comp_bwd)
    # assert all(c < 1e-7 for c in comp_bwd[('eqt', 'ref')]) # Check if grads are close
    tester_bwd.profile(repeat=10) # Profiling might need adjustments

    # --- Second-Order Backward Test (Grad-Grad) ---
    print("\n--- Second-Order Backward Test (Grad-Grad) ---")
    a_eqt_gg = alpha.clone().detach().requires_grad_(True)
    b_eqt_gg = beta.clone().detach().requires_grad_(True)
    g_eqt_gg = gamma.clone().detach().requires_grad_(True)

    a_ref_gg = alpha.clone().detach().requires_grad_(True)
    b_ref_gg = beta.clone().detach().requires_grad_(True)
    g_ref_gg = gamma.clone().detach().requires_grad_(True)

    # Need vectors for grad products
    v_out = torch.randn_like(out_ref) # Vector for first vjp
    # Get shapes of first gradients (grads w.r.t angles)
    grads_ref_shape = backward_ref(a_ref.clone().detach().requires_grad_(True),
                                   b_ref.clone().detach().requires_grad_(True),
                                   g_ref.clone().detach().requires_grad_(True),
                                   torch.ones_like(out_ref))
    v_in_a = torch.randn_like(grads_ref_shape[0])
    v_in_b = torch.randn_like(grads_ref_shape[1])
    v_in_g = torch.randn_like(grads_ref_shape[2])

    def gradgrad_eqt(a, b, g, v_out, v_in_a, v_in_b, v_in_g):
        res = eqt_module(alpha=a, beta=b, gamma=g)
        # Compute first gradients w.r.t angles
        grads = torch.autograd.grad(
            outputs=res, inputs=(a, b, g), grad_outputs=v_out, create_graph=True, allow_unused=True
        )
        # Replace None grads with zeros
        grads_a = grads[0] if grads[0] is not None else torch.zeros_like(a)
        grads_b = grads[1] if grads[1] is not None else torch.zeros_like(b)
        grads_g = grads[2] if grads[2] is not None else torch.zeros_like(g)

        # Compute second gradients (vjp of vjp)
        gradgrads = torch.autograd.grad(
            outputs=(grads_a, grads_b, grads_g),
            inputs=(a, b, g),
            grad_outputs=(v_in_a, v_in_b, v_in_g),
            allow_unused=True
        )
        # Replace None grad-grads with zeros
        gg_a = gradgrads[0] if gradgrads[0] is not None else torch.zeros_like(a)
        gg_b = gradgrads[1] if gradgrads[1] is not None else torch.zeros_like(b)
        gg_g = gradgrads[2] if gradgrads[2] is not None else torch.zeros_like(g)
        return [gg_a, gg_b, gg_g] # Grad-grads w.r.t angles

    def gradgrad_ref(a, b, g, v_out, v_in_a, v_in_b, v_in_g):
        res = prepare_wigner_ref(a, b, g, irreps)
        # Compute first gradients w.r.t angles
        grads = torch.autograd.grad(
            outputs=res, inputs=(a, b, g), grad_outputs=v_out, create_graph=True, allow_unused=True
        )
        # Replace None grads with zeros
        grads_a = grads[0] if grads[0] is not None else torch.zeros_like(a)
        grads_b = grads[1] if grads[1] is not None else torch.zeros_like(b)
        grads_g = grads[2] if grads[2] is not None else torch.zeros_like(g)

        # Compute second gradients (vjp of vjp)
        gradgrads = torch.autograd.grad(
            outputs=(grads_a, grads_b, grads_g),
            inputs=(a, b, g),
            grad_outputs=(v_in_a, v_in_b, v_in_g),
            allow_unused=True
        )
        # Replace None grad-grads with zeros
        gg_a = gradgrads[0] if gradgrads[0] is not None else torch.zeros_like(a)
        gg_b = gradgrads[1] if gradgrads[1] is not None else torch.zeros_like(b)
        gg_g = gradgrads[2] if gradgrads[2] is not None else torch.zeros_like(g)
        return [gg_a, gg_b, gg_g] # Grad-grads w.r.t angles

    tester_gradgrad = FunctionTester({
        'eqt': (gradgrad_eqt, [a_eqt_gg, b_eqt_gg, g_eqt_gg, v_out, v_in_a, v_in_b, v_in_g], {}),
        'ref': (gradgrad_ref, [a_ref_gg, b_ref_gg, g_ref_gg, v_out, v_in_a, v_in_b, v_in_g], {}),
    })
    comp_gradgrad = tester_gradgrad.compare(compare_func=max_abs_diff_list)
    print(comp_gradgrad)
    # Compare all 3 resulting grad-grad tensors (w.r.t angles)
    # assert all(c < 1e-6 for c in comp_gradgrad[('eqt', 'ref')]) # Check if grad-grads are close

    print("\nAll tests finished.")
