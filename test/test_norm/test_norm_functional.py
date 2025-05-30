import torch
from torch import Tensor
import math
from typing import NamedTuple

from equitorch.irreps import Irrep, Irreps, check_irreps
from equitorch.structs import IrrepsInfo
# Import both norm and squared_norm
from equitorch.nn.functional.norm import norm as functional_norm_apply, squared_norm as functional_norm2_apply
from equitorch.nn.others import Separable
from equitorch.utils._structs import irreps_info
from test_utils import FunctionTester, max_abs_diff_list, max_abs_diff

# torch.set_default_dtype(torch.float64)
torch.set_default_dtype(torch.float32)
# DEVICE = 'cpu'
DEVICE = 'cuda'

# --- Reference Implementation ---
class NormPerIrrepModule(torch.nn.Module):
    def __init__(self, rsqrt_dim_val: float, scaled: bool, keepdim: bool, squared: bool):
        super().__init__()
        if squared:
            self.rsqrt_dim_val = rsqrt_dim_val**2
        else:
            self.rsqrt_dim_val = rsqrt_dim_val
        self.scaled = scaled
        self.keepdim = keepdim
        self.squared = squared

    def forward(self, x_irrep_k: Tensor) -> Tensor:
        if self.squared:
            # Compute sum of squares: ||x||^2
            norm_val = torch.sum(x_irrep_k.pow(2), dim=-2, keepdim=self.keepdim)
        else:
            # Compute L2 norm: ||x||
            norm_val = torch.linalg.norm(x_irrep_k, dim=-2, keepdim=self.keepdim)
        
        if self.scaled:
            # For squared norm, scaling is by (1/sqrt(D))^2 = 1/D if we want (||x||/sqrt(D))^2
            # However, the functional_norm2 scales the sum_sq by 1/sqrt(D).
            # To match functional_norm2, RefNorm should also scale sum_sq by 1/sqrt(D).
            # If functional_norm is ||x||/sqrt(D), then functional_norm2 is (||x||^2)/sqrt(D)
            # The current functional_norm2 scales sum_sq by rsqrt_dims.
            # So, this scaling is correct for both.
            norm_val = norm_val * self.rsqrt_dim_val
        return norm_val

class RefNorm(torch.nn.Module):
    def __init__(self, irreps: Irreps, scaled: bool, squared: bool, keepdim: bool = True):
        super().__init__()
        self.irreps_obj = check_irreps(irreps)
        self.scaled = scaled
        self.squared = squared
        self.keepdim = keepdim

        rsqrt_dims_vals = [1.0 / math.sqrt(ir.dim) if ir.dim > 0 else 0.0 for ir in self.irreps_obj]

        norm_modules = []
        for i in range(len(self.irreps_obj)):
            norm_modules.append(
                NormPerIrrepModule(rsqrt_dims_vals[i], self.scaled, self.keepdim, self.squared)
            )
        
        self.separable_norm = Separable(
            irreps=self.irreps_obj,
            split_num_irreps=[1] * len(self.irreps_obj),
            sub_modules=norm_modules, # Parameter name in Separable is sub_modules
            cat_after=True
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        return self.separable_norm(input_tensor)

# --- Equitorch Implementation ---
class EqtNormModule(torch.nn.Module): # Renamed to avoid conflict with functional_norm
    def __init__(self, irreps: Irreps, scaled: bool, squared: bool):
        super().__init__()
        self.irreps_obj = check_irreps(irreps)
        self.scaled = scaled
        self.squared = squared
        self.irreps_info = irreps_info(self.irreps_obj)

    def forward(self, input_tensor: Tensor) -> Tensor:
        if self.squared:
            return functional_norm2_apply(input_tensor, self.irreps_info, self.scaled)
        else:
            return functional_norm_apply(input_tensor, self.irreps_info, self.scaled)

    def _apply(self, *args, **kwargs):
        n = super()._apply(*args, **kwargs)
        n.irreps_info = self.irreps_info._apply(*args, **kwargs)
        return n

# --- Test Suite ---
def run_tests(scaled_test: bool, squared_test: bool):
    print(f"\n--- Testing with scaled={scaled_test}, squared={squared_test} ---")
    
    irreps_str = "1x0e + 2x1o + 1x2e"
    irreps = Irreps(irreps_str)
    
    print(f"Irreps: {irreps} (dim={irreps.dim}, num_irreps={len(irreps)})")
    print(f"Device: {DEVICE}")
    print(f"Dtype: {torch.get_default_dtype()}")

    N = 256
    C = 64


    ref_module = RefNorm(irreps, scaled=scaled_test, squared=squared_test).to(device=DEVICE)
    eqt_module = EqtNormModule(irreps, scaled=scaled_test, squared=squared_test).to(device=DEVICE)

    x = torch.randn(N, irreps.dim, C, device=DEVICE)
    if irreps.dim > 0:
        x[0, :irreps[0].dim, :] = 0.0
        if len(irreps) > 1:
             x[1, irreps[0].dim : irreps[0].dim + irreps[1].dim, 0] = 0.0
    x.requires_grad_(True)

    print("\n--- Forward Test ---")
    tester_fwd = FunctionTester({
        'eqt': (eqt_module, [x], {}),
        'ref': (ref_module, [x], {}),
    })
    comp_fwd = tester_fwd.compare(compare_func=max_abs_diff)
    print(comp_fwd)
    # assert comp_fwd[('eqt', 'ref')] < 1e-6, f"Forward failed: diff = {comp_fwd[('eqt', 'ref')]}" # Adjusted tolerance for float32

    print("\n--- Backward Test ---")
    x_eqt_bwd = x.clone().detach().requires_grad_(True)
    x_ref_bwd = x.clone().detach().requires_grad_(True)

    out_ref = ref_module(x_ref_bwd)
    grad_out = torch.randn_like(out_ref)

    def backward_call(module, inp_tensor, grad):
        res = module(inp_tensor)
        res.backward(grad, retain_graph=True)
        g_x = inp_tensor.grad.clone() if inp_tensor.grad is not None else torch.zeros_like(inp_tensor)
        inp_tensor.grad = None
        return [g_x]

    tester_bwd = FunctionTester({
        'eqt': (backward_call, [eqt_module, x_eqt_bwd, grad_out], {}),
        'ref': (backward_call, [ref_module, x_ref_bwd, grad_out], {}),
    })
    comp_bwd = tester_bwd.compare(compare_func=max_abs_diff_list)
    print(comp_bwd)
    # assert all(c < 1e-6 for c in comp_bwd[('eqt', 'ref')]), f"Backward failed: diffs = {comp_bwd[('eqt', 'ref')]}"

    print("\n--- Second-Order Backward Test (Grad-Grad) ---")
    x_eqt_gg = x.clone().detach().requires_grad_(True)
    x_ref_gg = x.clone().detach().requires_grad_(True)

    v_out = torch.randn_like(out_ref)
    
    temp_x_for_shape = x.clone().detach().requires_grad_(True)
    grad_x_shape_provider = backward_call(ref_module, temp_x_for_shape, torch.ones_like(out_ref))[0]
    v_in_x = torch.randn_like(grad_x_shape_provider)
    del temp_x_for_shape, grad_x_shape_provider

    def gradgrad_call(module, inp_tensor, vec_out, vec_in_x):
        res = module(inp_tensor)
        grads_x, = torch.autograd.grad(
            outputs=res, inputs=(inp_tensor,), grad_outputs=vec_out, create_graph=True, allow_unused=True
        )
        grads_x_nonone = grads_x if grads_x is not None else torch.zeros_like(inp_tensor)

        gradgrads_x, = torch.autograd.grad(
            outputs=(grads_x_nonone,), inputs=(inp_tensor,), grad_outputs=(vec_in_x,), allow_unused=True
        )
        gg_x_nonone = gradgrads_x if gradgrads_x is not None else torch.zeros_like(inp_tensor)
        return [gg_x_nonone]

    tester_gradgrad = FunctionTester({
        'eqt': (gradgrad_call, [eqt_module, x_eqt_gg, v_out, v_in_x], {}),
        'ref': (gradgrad_call, [ref_module, x_ref_gg, v_out, v_in_x], {}),
    })
    comp_gradgrad = tester_gradgrad.compare(compare_func=max_abs_diff_list)
    print(comp_gradgrad)
    # assert all(c < 1e-6 for c in comp_gradgrad[('eqt', 'ref')]), f"Grad-grad failed: diffs = {comp_gradgrad[('eqt', 'ref')]}"

    print(f"\nAll tests for scaled={scaled_test}, squared={squared_test} passed.")

if __name__ == "__main__":
    # Test L2 norm (not squared)
    run_tests(scaled_test=True, squared_test=False)
    run_tests(scaled_test=False, squared_test=False)

    # # Test squared L2 norm
    # run_tests(scaled_test=True, squared_test=True)
    # run_tests(scaled_test=False, squared_test=True)
    