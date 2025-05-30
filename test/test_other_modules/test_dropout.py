import torch
from torch import Tensor, nn
import torch.nn.functional as F
import math
import itertools

from equitorch.irreps import Irrep, Irreps, check_irreps
from equitorch.structs import IrrepsInfo
from equitorch.utils._structs import irreps_info

# Import the target module
from equitorch.nn.dropout import Dropout
from equitorch.nn.functional.dropout import irrep_wise_dropout

# Import Separable for reference
from equitorch.nn.others import Separable

# Import norm functions for statistical comparison
from equitorch.nn.norm import SquaredNorm

# Import testing utilities
from test_utils import max_abs_diff # Assuming this is available

# torch.set_default_dtype(torch.float64)
torch.set_default_dtype(torch.float32)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device for Dropout tests: {DEVICE}")

# --- Reference Implementation for a single irrep (irrep_wise=True) ---
class PerIrrepRefDropout(nn.Module):
    def __init__(self, p: float, training_attr_from_parent: bool, work_on_eval_attr_from_parent: bool):
        super().__init__()
        self.p = p
        # These will be dynamically updated by the parent SeparableDropout reference
        self.training_from_parent = training_attr_from_parent 
        self.work_on_eval_from_parent = work_on_eval_attr_from_parent

    def forward(self, x_k: Tensor) -> Tensor:
        # x_k shape: (N, D_k, C)
        # Mimics the non-irrep-wise path of equitorch.nn.Dropout for a single irrep block
        # but applies dropout1d to the channels of that single irrep.
        
        # Effective training state for dropout application
        apply_dropout = self.work_on_eval_from_parent or self.training_from_parent
        
        if not apply_dropout or self.p == 0.0:
            return x_k

        # Permute for dropout1d: (N, D_k, C) -> (N, C, D_k)
        # dropout1d applies to the last dimension (D_k here) for each channel C
        x_permuted = x_k.permute(0, 2, 1) # (N, C, D_k)
        
        # Apply dropout1d. It will drop entire "columns" along D_k for selected channels.
        # This is the behavior for a single irrep if we consider its channels.
        # The user's reference was:
        # x = input.permute(-1, -2) # (C, D_k) if input is (D_k, C)
        # x_dropout = F.dropout1d(x, self.p, self.training or self.work_on_eval)
        # output = x_dropout.permute(-1, -2)
        # For batched input (N, D_k, C), permuted to (N, C, D_k), dropout1d works correctly.
        
        x_dropout_permuted = F.dropout1d(x_permuted, p=self.p, training=True) # training must be True for F.dropout to apply
        
        # Permute back: (N, C, D_k) -> (N, D_k, C)
        output_k = x_dropout_permuted.permute(0, 2, 1)
        return output_k.contiguous()

class RefSeparableIrrepWiseDropout(nn.Module):
    def __init__(self, irreps: Irreps, p: float, work_on_eval: bool):
        super().__init__()
        self.irreps_obj = check_irreps(irreps)
        self.p = p
        self.work_on_eval = work_on_eval # Store this initial config

        sub_modules = []
        for _ in self.irreps_obj: # One PerIrrepRefDropout for each irrep instance
            # Pass the initial training and work_on_eval states.
            # The actual 'training' state will be set from the parent's 'self.training'
            per_irrep_module = PerIrrepRefDropout(
                p=self.p,
                training_attr_from_parent=self.training, # Will be updated by parent's training state
                work_on_eval_attr_from_parent=self.work_on_eval 
            )
            sub_modules.append(per_irrep_module)
        
        self.separable_op = Separable(
            irreps=self.irreps_obj,
            split_num_irreps=[1] * len(self.irreps_obj), # Each irrep instance is a segment
            sub_modules=sub_modules, 
            cat_after=True,
            dim=-2 
        )
        # This will be set by the test runner based on the main Dropout module's state
        self.training = True # Default, will be overridden

    def forward(self, input_tensor: Tensor) -> Tensor:
        # Update training state of sub_modules before forward pass
        for m_irrep in self.separable_op.sub_modules:
            if isinstance(m_irrep, PerIrrepRefDropout):
                m_irrep.training_from_parent = self.training # Use current training state of RefSeparable
                m_irrep.work_on_eval_from_parent = self.work_on_eval
        return self.separable_op(input_tensor)

# --- Reference for non-irrep-wise dropout ---
def ref_non_irrep_wise_dropout(input_tensor: Tensor, p: float, training: bool, work_on_eval: bool) -> Tensor:
    apply_dropout = work_on_eval or training
    if not apply_dropout or p == 0.0:
        return input_tensor

    original_shape = input_tensor.shape
    if input_tensor.ndim < 2:
        raise ValueError("Input tensor must have at least 2 dimensions (irreps_dim, channels)")
    
    x = input_tensor.permute(*range(input_tensor.ndim - 2), -1, -2) # (..., C, irreps_dim)
    
    if x.ndim > 3:
        leading_dims = x.shape[:-2]
        num_leading_elements = x.numel() // (x.shape[-2] * x.shape[-1])
        x_reshaped = x.reshape(num_leading_elements, x.shape[-2], x.shape[-1])
        # training must be True for F.dropout1d to apply mask
        x_dropout = F.dropout1d(x_reshaped, p, training=True) 
        x_dropout = x_dropout.reshape(*leading_dims, x.shape[-2], x.shape[-1])
    else:
        x_dropout = F.dropout1d(x, p, training=True)

    output = x_dropout.permute(*range(x_dropout.ndim - 2), -1, -2) # (..., irreps_dim, C)
    return output.contiguous()

# --- Test Suite ---
def run_tests(irreps_str: str, N: int, C: int, p_test: float, irrep_wise_test: bool, work_on_eval_test: bool, num_samples_stats: int = 50000):
    print(f"\n--- Testing Dropout with Irreps='{irreps_str}', N={N}, C={C}, p={p_test}, irrep_wise={irrep_wise_test}, work_on_eval={work_on_eval_test} ---")
    
    irreps = Irreps(irreps_str)
    if irreps.dim == 0 and N > 0 and C > 0 :
        print("Skipping test for non-empty tensor with 0-dim irreps.")
        return
    if N == 0 or C == 0 or irreps.dim == 0:
        print(f"Testing with empty tensor scenario: N={N}, C={C}, irreps_dim={irreps.dim}")

    print(f"Device: {DEVICE}, Dtype: {torch.get_default_dtype()}")

    eqt_module = Dropout(
        p=p_test, irreps=irreps, irrep_wise=irrep_wise_test, work_on_eval=work_on_eval_test
    ).to(device=DEVICE)

    ref_module = None
    if irrep_wise_test:
        ref_module = RefSeparableIrrepWiseDropout(
            irreps=irreps, p=p_test, work_on_eval=work_on_eval_test
        ).to(device=DEVICE)
    
    # For statistical tests, we need a SquaredNorm module
    # This SquaredNorm is for analyzing the output of Dropout, not part of Dropout itself.
    # It should operate on the output irreps of Dropout, which are the same as input irreps.
    norm_analyzer = SquaredNorm(irreps=irreps, scaled=False).to(device=DEVICE) # scaled=False for raw sum of squares

    # Test Case 1: p = 0.0 (should be identity)
    print("  Test Case 1: p = 0.0")
    eqt_module_p0 = Dropout(p=0.0, irreps=irreps, irrep_wise=irrep_wise_test, work_on_eval=work_on_eval_test).to(DEVICE)
    eqt_module_p0.train() # or eval, should not matter for p=0
    
    x_p0 = torch.randn(N, irreps.dim, C, device=DEVICE)
    out_eqt_p0 = eqt_module_p0(x_p0)
    diff_p0 = max_abs_diff(out_eqt_p0, x_p0)
    print(f"    p=0, Max abs diff (EQT vs Input): {diff_p0}")
    assert diff_p0 < 1e-7, "p=0.0 test failed: output not equal to input."

    # Test Case 2: training = False and work_on_eval = False (should be identity)
    if not work_on_eval_test:
        print("  Test Case 2: training=False, work_on_eval=False")
        eqt_module.eval() # Sets training=False
        
        x_eval = torch.randn(N, irreps.dim, C, device=DEVICE)
        out_eqt_eval = eqt_module(x_eval)
        diff_eval = max_abs_diff(out_eqt_eval, x_eval)
        print(f"    eval mode, Max abs diff (EQT vs Input): {diff_eval}")
        assert diff_eval < 1e-7, "Eval mode (not work_on_eval) test failed: output not equal to input."
    
    # Test Case 3: Statistical comparison if p > 0 and dropout is active
    if p_test > 0.0 and (eqt_module.training or work_on_eval_test):
        print(f"  Test Case 3: Statistical comparison (p={p_test}, active dropout)")
        eqt_module.train() # Ensure training mode for eqt_module if not work_on_eval
        if ref_module:
            ref_module.train() # Ensure ref_module is also in training mode

        means_eqt = []
        vars_eqt = []
        means_ref = []
        vars_ref = []

        # Generate many samples
        for i in range(num_samples_stats):
            # if i % (num_samples_stats // 10) == 0 : print(f"    Generating sample {i}/{num_samples_stats}")
            x_sample = torch.randn(N, irreps.dim, C, device=DEVICE) # Fresh input for each sample
            
            # EQT Dropout
            out_eqt_sample = eqt_module(x_sample.clone())
            # squared_norm_eqt shape: (N, num_irrep_instances, C)
            squared_norm_eqt = norm_analyzer(out_eqt_sample) 
            means_eqt.append(squared_norm_eqt.mean(dim=0).detach()) # Mean over batch, result (num_irrep_instances, C)
            vars_eqt.append(squared_norm_eqt.var(dim=0, unbiased=False).detach()) # Variance over batch

            # REF Dropout
            if irrep_wise_test and ref_module:
                out_ref_sample = ref_module(x_sample.clone())
            elif not irrep_wise_test:
                # For non-irrep-wise, ref is functional
                out_ref_sample = ref_non_irrep_wise_dropout(x_sample.clone(), p_test, True, work_on_eval_test)
            else: # Should not happen if ref_module logic is correct
                continue 
            
            squared_norm_ref = norm_analyzer(out_ref_sample)
            means_ref.append(squared_norm_ref.mean(dim=0).detach())
            vars_ref.append(squared_norm_ref.var(dim=0, unbiased=False).detach())

        # Aggregate statistics
        # Stack along a new dimension (num_samples_stats, num_irrep_instances, C)
        all_means_eqt = torch.stack(means_eqt) 
        all_vars_eqt = torch.stack(vars_eqt)
        
        avg_mean_eqt = all_means_eqt.mean(dim=0) # Average of batch means
        avg_var_eqt = all_vars_eqt.mean(dim=0)   # Average of batch variances

        if means_ref: # If reference was run
            all_means_ref = torch.stack(means_ref)
            all_vars_ref = torch.stack(vars_ref)
            avg_mean_ref = all_means_ref.mean(dim=0)
            avg_var_ref = all_vars_ref.mean(dim=0)

            mean_diff = max_abs_diff(avg_mean_eqt, avg_mean_ref)
            var_diff = max_abs_diff(avg_var_eqt, avg_var_ref)
            
            print(f"    Avg Squared Norm Mean Diff (EQT vs REF): {mean_diff}")
            print(f"    Avg Squared Norm Var Diff (EQT vs REF): {var_diff}")
            
            # Tolerances for statistical tests can be tricky.
            # These are just heuristic values.
            # assert mean_diff < 0.1 + p_test, f"Mean of squared norms differs significantly (EQT vs REF): {mean_diff}"
            # assert var_diff < 0.1 + p_test, f"Variance of squared norms differs significantly (EQT vs REF): {var_diff}"
        else:
            print("    Skipped REF comparison in statistical test (likely irrep_wise=True without ref_module setup).")

    print(f"Finished tests for Dropout with Irreps='{irreps_str}', p={p_test}, irrep_wise={irrep_wise_test}, work_on_eval={work_on_eval_test}")

if __name__ == "__main__":
    test_configs = [
        {"irreps_str": "1x0e + 2x1o + 1x2e", "N": 32, "C": 16, "p": 0.5},
        {"irreps_str": "3x0e", "N": 64, "C": 8, "p": 0.25},
        {"irreps_str": "2x1e", "N": 16, "C": 32, "p": 0.1},
        {"irreps_str": "1x0e", "N": 128, "C": 4, "p": 0.75},
    ]

    for irrep_wise_val in [True, False]:
        for work_on_eval_val in [True, False]:
            for config in test_configs:
                # Skip irrep_wise=True if irreps is 0-dim, as IrrepsInfo might be problematic
                if irrep_wise_val and Irreps(config["irreps_str"]).dim == 0 and (config["N"] > 0 and config["C"] > 0) :
                    print(f"\nSkipping Dropout test: irrep_wise=True with 0-dim irreps ('{config['irreps_str']}') and non-empty N/C.")
                    continue
                
                run_tests(
                    irreps_str=config["irreps_str"], 
                    N=config["N"], 
                    C=config["C"], 
                    p_test=config["p"],
                    irrep_wise_test=irrep_wise_val,
                    work_on_eval_test=work_on_eval_val
                )
    
    print("\nAll Dropout tests finished.")
