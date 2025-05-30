import torch
from torch import Tensor, nn
import math

from equitorch.irreps import Irrep, Irreps, check_irreps
from equitorch.utils._structs import irreps_info

# Import the target module
from equitorch.nn.normalization import BatchRMSNorm
# Import Separable for reference
from equitorch.nn.others import Separable

# Import testing utilities
# Assuming test_utils.py is in the same directory or accessible
from test_utils import FunctionTester, max_abs_diff_list, max_abs_diff # TODO: Ensure this path is correct

torch.set_default_dtype(torch.float64)
# torch.set_default_dtype(torch.float32)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# --- Reference Implementation ---

class PerIrrepRMSNorm(nn.Module):
    def __init__(self, 
                 irrep_dim: int, 
                 eps: float, 
                 scaled_norm_factor: float, # 1.0/irrep_dim if scaled, else 1.0
                 momentum: float,
                 affine: bool):
        super().__init__()
        self.irrep_dim = irrep_dim
        self.eps = eps
        self.scaled_norm_factor = scaled_norm_factor
        self.momentum = momentum
        self.affine = affine
        # These will be set by the parent RefBatchRMSNorm
        self.running_sq_norm_k: Tensor = None 
        self.weight_k: Tensor = None # Shape (C,)

    def forward(self, x_k: Tensor) -> Tensor:
        # x_k shape: (N, D_k, C)
        # self.running_sq_norm_k shape: (C,)
        # self.weight_k shape: (C,)
        
        N, Dk, C = x_k.shape
        assert Dk == self.irrep_dim

        # current_batch_sq_norm_per_element = x_k.pow(2) # (N, Dk, C)
        # sum_sq_over_dim = torch.sum(current_batch_sq_norm_per_element, dim=1) # (N, C)
        # scaled_sum_sq_over_dim = sum_sq_over_dim * self.scaled_norm_factor # (N, C)
        
        # More direct: SquaredNorm part for one irrep
        # Equivalent to functional.squared_norm(x_k_reshaped_for_single_irrep, single_irrep_info, scaled=self.scaled_norm_factor!=1.0)
        # For simplicity here, we implement it directly:
        if Dk == 0: # Handle 0-dim irreps if they can occur
             return x_k.clone()

        current_batch_sq_norm_val_k = torch.sum(x_k.pow(2), dim=1) # Sum over D_k -> (N, C)
        if self.scaled_norm_factor != 1.0 and Dk > 0 : # Dk > 0 check for safety
            current_batch_sq_norm_val_k = current_batch_sq_norm_val_k / Dk


        if self.training:
            batch_mean_sq_norm_k = current_batch_sq_norm_val_k.mean(dim=0) # (C,)
            if self.running_sq_norm_k is not None: # Should always be set by parent
                 self.running_sq_norm_k.data.lerp_(batch_mean_sq_norm_k, self.momentum)
            norm_to_use_k = batch_mean_sq_norm_k
        else:
            norm_to_use_k = self.running_sq_norm_k
        
        inv_sigma_k = torch.rsqrt(norm_to_use_k + self.eps) # (C,)
        
        # Unsqueeze for broadcasting: (1, 1, C)
        output_k = x_k * inv_sigma_k.unsqueeze(0).unsqueeze(0) 
        
        if self.affine and self.weight_k is not None:
            output_k = output_k * self.weight_k.unsqueeze(0).unsqueeze(0)
            
        return output_k

class RefBatchRMSNorm(nn.Module):
    def __init__(self, irreps: Irreps, channels: int, eps: float, momentum: float, affine: bool, scaled: bool):
        super().__init__()
        self.irreps_obj = check_irreps(irreps)
        self.channels = channels
        self.num_irreps = len(self.irreps_obj)
        self.momentum = momentum # Store for PerIrrepRMSNorm
        self.affine = affine     # Store for PerIrrepRMSNorm
        self.eps = eps           # Store for PerIrrepRMSNorm
        self.scaled = scaled # Store for PerIrrepRMSNorm

        sub_modules = []
        for i, ir in enumerate(self.irreps_obj):
            scaled_factor = (1.0 / ir.dim) if scaled and ir.dim > 0 else 1.0
            # scaled_norm_factor for PerIrrepRMSNorm is 1.0 because scaling is done before mean
            # Correction: scaled_norm_factor should be based on self.scaled for consistency
            # The PerIrrepRMSNorm will use scaled_norm_factor to scale sum(x_k^2)
            # No, the PerIrrepRMSNorm's scaled_norm_factor is effectively what functional.squared_norm's `scaled` param does.
            # So, if self.scaled is True, PerIrrepRMSNorm should use 1.0/ir.dim.
            
            # The PerIrrepRMSNorm will compute sum(x_k^2) and then scale it by scaled_norm_factor.
            # This is equivalent to `squared_norm(..., scaled=True)` if scaled_norm_factor = 1/D_k
            # Or `squared_norm(..., scaled=False)` if scaled_norm_factor = 1.0
            # The `scaled` argument in the main BatchRMSNorm refers to how squared_norm is computed.
            # So, PerIrrepRMSNorm's `scaled_norm_factor` should be 1.0, and it should internally divide by ir.dim if self.scaled is true.
            # Let's simplify: PerIrrepRMSNorm will take `scaled_norm_bool`
            
            # Re-evaluating PerIrrepRMSNorm: it should compute sum(x_k.pow(2)) / D_k if scaled is true.
            # The `scaled_norm_factor` in its init was a bit confusing. Let's pass `scaled` bool directly.

            per_irrep_module = PerIrrepRMSNorm(
                irrep_dim=ir.dim, 
                eps=self.eps, 
                # scaled_norm_factor is effectively handled by how current_batch_sq_norm_val_k is calculated
                # Let's rename scaled_norm_factor to apply_scaling_to_sq_norm_bool
                scaled_norm_factor = (1.0 / ir.dim) if self.scaled and ir.dim > 0 else 1.0, # This factor is applied to sum_sq
                momentum=self.momentum,
                affine=self.affine
            )
            sub_modules.append(per_irrep_module)
        
        self.separable_op = Separable(
            irreps=self.irreps_obj,
            split_num_irreps=[1] * self.num_irreps,
            sub_modules=sub_modules, 
            cat_after=True,
            dim=-2 
        )

        # Buffers and parameters for the reference module itself
        self.register_buffer(
            "running_squared_norm", torch.ones(self.num_irreps, self.channels)
        )
        if self.affine:
            self.weight = nn.Parameter(torch.ones(self.num_irreps, self.channels))
        else:
            self.register_parameter("weight", None)

    def train(self, mode: bool = True):
        super().train(mode)
        for m in self.separable_op.sub_modules:
            if isinstance(m, PerIrrepRMSNorm):
                m.train(mode) # Propagate training mode
        return self

    def eval(self):
        return self.train(False)

    def forward(self, input_tensor: Tensor) -> Tensor:
        # Pass the sliced running_norm and weight to each PerIrrepRMSNorm module
        for i, m_irrep in enumerate(self.separable_op.sub_modules):
            if isinstance(m_irrep, PerIrrepRMSNorm):
                m_irrep.running_sq_norm_k = self.running_squared_norm[i] # Shape (C,)
                if self.affine and self.weight is not None:
                    m_irrep.weight_k = self.weight[i] # Shape (C,)
                else:
                    m_irrep.weight_k = None
        
        return self.separable_op(input_tensor)


# --- Test Suite ---
def run_tests(irreps_str: str, N: int, C: int, affine_test: bool, scaled_norm_test: bool, eps_test: float = 1e-5, momentum_test: float = 0.1):
    print(f"\n--- Testing BatchRMSNorm with Irreps='{irreps_str}', N={N}, C={C}, affine={affine_test}, scaled={scaled_norm_test}, eps={eps_test} ---")
    
    irreps = Irreps(irreps_str)
    if irreps.dim == 0:
        print("Skipping test for 0-dim irreps.")
        return

    print(f"Device: {DEVICE}, Dtype: {torch.get_default_dtype()}")

    # Instantiate modules
    eqt_module = BatchRMSNorm(
        irreps, channels=C, eps=eps_test, momentum=momentum_test, affine=affine_test, scaled=scaled_norm_test
    ).to(device=DEVICE)
    
    ref_module = RefBatchRMSNorm(
        irreps, channels=C, eps=eps_test, momentum=momentum_test, affine=affine_test, scaled=scaled_norm_test
    ).to(device=DEVICE)

    # Ensure weights are the same if affine
    if affine_test:
        random_weights = torch.randn_like(eqt_module.weight.data)
        eqt_module.weight.data.copy_(random_weights)
        ref_module.weight.data.copy_(random_weights)
    
    # Ensure running_squared_norm starts the same (default is ones)
    # eqt_module.running_squared_norm.data.fill_(1.0) # Already done in init
    # ref_module.running_squared_norm.data.fill_(1.0)


    # Test in training mode
    print("\n-- Training Mode --")
    eqt_module.train()
    ref_module.train()

    x = torch.randn(N, irreps.dim, C, device=DEVICE) * 2 + 0.5 # Add some variance and offset
    x.requires_grad_(True)

    out_eqt = eqt_module(x)
    out_ref = ref_module(x)
    
    fwd_diff = max_abs_diff(out_eqt, out_ref)
    print(f"Forward Pass Max Abs Diff: {fwd_diff}")
    assert fwd_diff < 1e-5, "Forward pass mismatch in training mode"

    # Check running_squared_norm update
    running_norm_diff = max_abs_diff(eqt_module.running_squared_norm, ref_module.running_squared_norm)
    print(f"Running Squared Norm Max Abs Diff after 1st iter: {running_norm_diff}")
    assert running_norm_diff < 1e-5, "Running squared norm mismatch after 1st iter"

    # Multiple iterations to check running_squared_norm convergence
    for _ in range(5):
        x_i = torch.randn(N, irreps.dim, C, device=DEVICE)
        eqt_module(x_i)
        ref_module(x_i)
    
    running_norm_diff_multi = max_abs_diff(eqt_module.running_squared_norm, ref_module.running_squared_norm)
    print(f"Running Squared Norm Max Abs Diff after multiple iters: {running_norm_diff_multi}")
    assert running_norm_diff_multi < 1e-5, "Running squared norm mismatch after multiple iters"


    # Backward pass
    grad_out = torch.randn_like(out_eqt)
    out_eqt.backward(grad_out, retain_graph=True)
    grad_x_eqt = x.grad.clone()
    x.grad.zero_()
    
    out_ref.backward(grad_out, retain_graph=True)
    grad_x_ref = x.grad.clone()
    x.grad.zero_()

    bwd_diff_x = max_abs_diff(grad_x_eqt, grad_x_ref)
    print(f"Backward Pass (grad_x) Max Abs Diff: {bwd_diff_x}")
    assert bwd_diff_x < 1e-5, "Backward pass (grad_x) mismatch in training mode"

    if affine_test:
        grad_w_eqt = eqt_module.weight.grad.clone()
        grad_w_ref = ref_module.weight.grad.clone()
        bwd_diff_w = max_abs_diff(grad_w_eqt, grad_w_ref)
        print(f"Backward Pass (grad_w) Max Abs Diff: {bwd_diff_w}")
        assert bwd_diff_w < 2e-5, "Backward pass (grad_w) mismatch in training mode" # Increased tolerance
        eqt_module.weight.grad.zero_()
        ref_module.weight.grad.zero_()


    # Test in eval mode
    print("\n-- Eval Mode --")
    eqt_module.eval()
    ref_module.eval()
    
    # Store running stats from training
    final_running_norm_eqt = eqt_module.running_squared_norm.clone().detach()
    final_running_norm_ref = ref_module.running_squared_norm.clone().detach()

    x_eval = torch.randn(N, irreps.dim, C, device=DEVICE) * 3 - 0.2
    x_eval.requires_grad_(True)

    out_eqt_eval = eqt_module(x_eval)
    out_ref_eval = ref_module(x_eval)

    fwd_diff_eval = max_abs_diff(out_eqt_eval, out_ref_eval)
    print(f"Eval Forward Pass Max Abs Diff: {fwd_diff_eval}")
    assert fwd_diff_eval < 1e-5, "Forward pass mismatch in eval mode"
    
    # Check that running_squared_norm did not change
    assert torch.equal(eqt_module.running_squared_norm, final_running_norm_eqt), "EQT running_squared_norm changed in eval mode"
    assert torch.equal(ref_module.running_squared_norm, final_running_norm_ref), "REF running_squared_norm changed in eval mode"

    # Backward pass in eval mode
    grad_out_eval = torch.randn_like(out_eqt_eval)
    out_eqt_eval.backward(grad_out_eval)
    grad_x_eqt_eval = x_eval.grad.clone()
    x_eval.grad.zero_()

    out_ref_eval.backward(grad_out_eval)
    grad_x_ref_eval = x_eval.grad.clone()
    x_eval.grad.zero_()
    
    bwd_diff_x_eval = max_abs_diff(grad_x_eqt_eval, grad_x_ref_eval)
    print(f"Eval Backward Pass (grad_x) Max Abs Diff: {bwd_diff_x_eval}")
    assert bwd_diff_x_eval < 1e-5, "Backward pass (grad_x) mismatch in eval mode"

    if affine_test:
        # Gradients for weights should still be computed in eval mode if they exist
        grad_w_eqt_eval = eqt_module.weight.grad.clone() if eqt_module.weight.grad is not None else torch.zeros_like(eqt_module.weight)
        grad_w_ref_eval = ref_module.weight.grad.clone() if ref_module.weight.grad is not None else torch.zeros_like(ref_module.weight)
        bwd_diff_w_eval = max_abs_diff(grad_w_eqt_eval, grad_w_ref_eval)
        print(f"Eval Backward Pass (grad_w) Max Abs Diff: {bwd_diff_w_eval}")
        assert bwd_diff_w_eval < 2e-5, "Backward pass (grad_w) mismatch in eval mode" # Increased tolerance


    print(f"All tests for BatchRMSNorm with Irreps='{irreps_str}', affine={affine_test}, scaled={scaled_norm_test} passed.")

if __name__ == "__main__":
    test_configs = [
        {"irreps_str": "1x0e + 2x1o + 1x2e", "N": 256, "C": 64}, # Basic
        {"irreps_str": "3x0e", "N": 256, "C": 64},              # Scalars only
        {"irreps_str": "2x1e", "N": 256, "C": 64},              # Vectors only
        {"irreps_str": "1x0e+1x1o+1x2e+1x3o", "N": 256, "C": 64},# Mixed, more irreps
        {"irreps_str": "1x0e", "N":1, "C":1}                 # Minimal case
    ]

    for affine in [True, False]:
        for scaled in [True, False]:
            for config in test_configs:
                run_tests(
                    irreps_str=config["irreps_str"], 
                    N=config["N"], 
                    C=config["C"], 
                    affine_test=affine, 
                    scaled_norm_test=scaled
                )
    
    # Test with 0-dim irrep (should be handled gracefully, e.g. by skipping or identity)
    # run_tests(irreps_str="0x0e", N=2, C=2, affine_test=True, scaled_norm_test=True) # BatchRMSNorm init will fail if irreps.dim is 0
    
    print("\nAll BatchRMSNorm tests finished.")
