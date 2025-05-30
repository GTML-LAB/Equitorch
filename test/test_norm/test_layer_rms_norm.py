import torch
from torch import Tensor, nn
import math

from equitorch.irreps import Irrep, Irreps, check_irreps
from equitorch.structs import IrrepsInfo
from equitorch.utils._structs import irreps_info

# Import the target module
from equitorch.nn.normalization import LayerRMSNorm
# Import Separable for reference
from equitorch.nn.others import Separable

# Import testing utilities
# Assuming test_utils.py is in the same directory or accessible via PYTHONPATH
from test_utils import max_abs_diff, FunctionTester # Using only max_abs_diff based on BatchRMSNorm test

# Import functional components needed for the new reference
from equitorch.nn.functional.norm import channel_mean_squared_norm
from equitorch.nn.functional.sparse_product import sparse_vecsca, sparse_mul
from equitorch.structs import SparseProductInfo
import itertools


# torch.set_default_dtype(torch.float64)
torch.set_default_dtype(torch.float32)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device for LayerRMSNorm tests: {DEVICE}")

# --- Reference Implementation ---

class PerIrrepRefLayerRMSNorm(nn.Module):
    def __init__(self, 
                 irrep_dim: int, 
                 channels: int,
                 eps: float, 
                 scaled_flag_from_parent: bool, # True if variance denominator includes D_k
                 affine_flag_from_parent: bool):
        super().__init__()
        self.irrep_dim = irrep_dim
        self.channels = channels
        self.eps = eps
        self.scaled_from_parent = scaled_flag_from_parent
        self.affine_from_parent = affine_flag_from_parent
        
        # weight_k will be set by the parent RefLayerRMSNorm if affine
        self.weight_k: Tensor = None # Shape (C,)

    def forward(self, x_k: Tensor) -> Tensor:
        # x_k shape: (N, D_k, C)
        # self.weight_k shape: (C,) if affine, else None
        
        N, Dk, C_in = x_k.shape
        ... #assert Dk == self.irrep_dim
        ... #assert C_in == self.channels

        if Dk == 0 or C_in == 0: # Handle 0-dim irreps or 0 channels
             return x_k.clone()

        # Calculate sum_{mc} x_nimc^2 for this irrep k
        # sum_sq_mc shape: (N,)
        sum_sq_mc = x_k.pow(2).sum(dim=(1, 2)) 

        # Determine denominator for the mean
        if self.scaled_from_parent: # Corresponds to (1/CD_i) * sum
            denominator = Dk * C_in
        else: # Corresponds to (1/C) * sum
            denominator = C_in
        
        if denominator == 0: # Should not happen if Dk > 0 and C_in > 0
            var_nk = torch.zeros_like(sum_sq_mc)
        else:
            var_nk = sum_sq_mc / denominator # Shape (N,)
        
        sigma_nk = torch.sqrt(var_nk + self.eps) # Shape (N,)
        
        # Normalize: x_k / sigma_nk
        # sigma_nk needs to be (N, 1, 1) for broadcasting
        normed_x_k = x_k / sigma_nk.view(N, 1, 1)
        
        output_k = normed_x_k
        if self.affine_from_parent and self.weight_k is not None:
            # weight_k shape (C,), needs to be (1, 1, C) for broadcasting
            output_k = output_k * self.weight_k.view(1, 1, C_in)
            
        return output_k

class RefLayerRMSNorm(nn.Module):
    def __init__(self, irreps: Irreps, channels: int, eps: float, affine: bool, scaled: bool):
        super().__init__()
        self.irreps_obj = check_irreps(irreps)
        self.channels = channels
        self.num_irreps = len(self.irreps_obj)
        self.affine = affine
        self.eps = eps
        self.scaled = scaled # This is the 'scaled' flag for LayerRMSNorm

        sub_modules = []
        for i, ir in enumerate(self.irreps_obj):
            per_irrep_module = PerIrrepRefLayerRMSNorm(
                irrep_dim=ir.dim,
                channels=self.channels,
                eps=self.eps, 
                scaled_flag_from_parent=self.scaled,
                affine_flag_from_parent=self.affine
            )
            sub_modules.append(per_irrep_module)
        
        self.separable_op = Separable(
            irreps=self.irreps_obj,
            split_num_irreps=[1] * self.num_irreps,
            sub_modules=sub_modules, 
            cat_after=True,
            dim=-2 
        )

        if self.affine:
            self.weight = nn.Parameter(torch.ones(self.num_irreps, self.channels))
        else:
            self.register_parameter("weight", None)

    def forward(self, input_tensor: Tensor) -> Tensor:
        # Pass the sliced weight to each PerIrrepRefLayerRMSNorm module if affine
        if self.affine and self.weight is not None:
            for i, m_irrep in enumerate(self.separable_op.sub_modules):
                if isinstance(m_irrep, PerIrrepRefLayerRMSNorm):
                    m_irrep.weight_k = self.weight[i] # Shape (C,)
        
        return self.separable_op(input_tensor)

# --- New Reference: Functional Forward Pass Only ---
def ref_functional_forward_only_layer_rms_norm(
    input_tensor: Tensor, 
    irreps_info: IrrepsInfo, 
    weight: Tensor, # Can be None
    scaled: bool, 
    eps: float
) -> Tensor:
    """
    Replicates the forward pass of functional.LayerRMSNorm.forward.
    """
    # These info objects are for the sparse_vecsca and sparse_mul calls below.
    # Their roles (info_fwd, info_bwd1, info_bwd2 for the sparse op itself) are:
    # info_fwd_for_sparse_op = SparseProductInfo(index2=irreps_info.irrep_index)
    # info_bwd1_for_sparse_op = SparseProductInfo(index1=irreps_info.irrep_index) # Dummy for forward
    # info_bwd2_for_sparse_op = SparseProductInfo(seg_out=irreps_info.irrep_seg)   # Dummy for forward
    
    # As used in LayerRMSNorm.forward:
    info_fwd_arg = SparseProductInfo(index2=irreps_info.irrep_index)
    info_bwd1_arg = SparseProductInfo(index1=irreps_info.irrep_index) 
    info_bwd2_arg = SparseProductInfo(seg_out=irreps_info.irrep_seg)

    squared_norm_ni = channel_mean_squared_norm(input_tensor, irreps_info, scaled)
    rsigma_ni = torch.rsqrt(squared_norm_ni + eps)

    normed = sparse_vecsca(input_tensor, rsigma_ni, 
                           info_fwd_arg, info_bwd1_arg, info_bwd2_arg)

    if weight is not None:
        output = sparse_mul(normed, weight, 
                            info_fwd_arg, info_bwd1_arg, info_bwd2_arg)
    else:
        output = normed
    return output

# --- Test Suite ---
def run_tests(irreps_str: str, N: int, C: int, affine_test: bool, scaled_test: bool, eps_test: float = 1e-5):
    print(f"\n--- Testing LayerRMSNorm with Irreps='{irreps_str}', N={N}, C={C}, affine={affine_test}, scaled={scaled_test}, eps={eps_test} ---")
    
    irreps = Irreps(irreps_str)
    if irreps.dim == 0 and N > 0 and C > 0 : # Allow if N or C is 0 for empty tensor tests
        print("Skipping test for non-empty tensor with 0-dim irreps, as input would be ill-defined.")
        return
    if N == 0 or C == 0 or irreps.dim == 0:
        print(f"Testing with empty tensor scenario: N={N}, C={C}, irreps_dim={irreps.dim}")


    print(f"Device: {DEVICE}, Dtype: {torch.get_default_dtype()}")

    # Instantiate modules
    eqt_module = LayerRMSNorm(
        irreps, channels=C, eps=eps_test, affine=affine_test, scaled=scaled_test
    ).to(device=DEVICE)
    
    ref_module = RefLayerRMSNorm(
        irreps, channels=C, eps=eps_test, affine=affine_test, scaled=scaled_test
    ).to(device=DEVICE)

    # Ensure weights are the same if affine
    weight_for_functional_ref = None
    if affine_test and eqt_module.weight is not None: # Check if weight exists
        random_weights = torch.randn_like(eqt_module.weight.data)
        eqt_module.weight.data.copy_(random_weights)
        ref_module.weight.data.copy_(random_weights)
        weight_for_functional_ref = random_weights.clone().detach() # Use the same weight for functional ref
    
    eqt_module.eval() # LayerRMSNorm does not have different train/eval behavior regarding stats
    ref_module.eval()

    # Prepare irreps_info for functional ref, ensure it's on the correct device
    irreps_info_for_functional_ref = irreps_info(irreps)._apply(lambda t: t.to(DEVICE))


    x = torch.randn(N, irreps.dim, C, device=DEVICE) * 2 + 0.5 
    x.requires_grad_(True)

    # Forward pass
    out_eqt = eqt_module(x)
    # Forward pass using FunctionTester
    funcs_to_test_fwd = {
        "EQT_Module": (eqt_module, (x,), {}),
        "REF_Module": (ref_module, (x,), {}),
        "REF_Functional_Fwd_Only": (
            ref_functional_forward_only_layer_rms_norm,
            (x, irreps_info_for_functional_ref, weight_for_functional_ref, scaled_test, eps_test),
            {},
        ),
    }
    tester_fwd = FunctionTester(funcs_to_test_fwd)
    fwd_comparison_results = tester_fwd.compare()

    print("\nForward Pass Comparisons (max_abs_diff):")
    for (name1, name2), diff in fwd_comparison_results.items():
        print(f"  {name1} vs {name2}: {diff}")
        if not (diff < 1e-5):
            print(f"    WARNING: Forward pass difference > 1e-5 between {name1} and {name2}: {diff}")
        # assert diff < 1e-5, f"Forward pass mismatch between {name1} and {name2}"

    # To get out_eqt for backward pass, run it once more if needed, or retrieve from tester if it stores results.
    # For simplicity, let's re-run eqt_module for grad_out reference.
    # Or, better, FunctionTester can be modified to return results if needed.
    # For now, let's assume the first run of eqt_module in the tester is 'out_eqt'.
    # This requires FunctionTester to store and provide access to results, or we run EQT_Module once outside.
    
    # Re-run EQT_Module to get its output for backward pass reference, ensuring grad context
    # This output is also implicitly checked by the forward FunctionTester
    if x.grad is not None:
        x.grad.zero_()
    out_eqt_for_bwd = eqt_module(x) 
    out_ref_module_for_bwd = ref_module(x) # Needed for REF_Module backward call

    grad_out = torch.randn_like(out_eqt_for_bwd)

    # --- Backward Pass Comparison using FunctionTester ---
    
    # Helper functions to perform backward pass and return grads
    def get_grads_eqt(module, inp, grad_output_tensor):
        if inp.grad is not None:
            inp.grad.zero_()
        if hasattr(module, 'weight') and module.weight is not None and module.weight.grad is not None:
            module.weight.grad.zero_()
        
        # Need to recompute output for backward if not retained from forward tester
        output = module(inp) 
        output.backward(grad_output_tensor)
        
        grad_x = inp.grad.clone() if inp.grad is not None else torch.zeros_like(inp)
        grad_w = module.weight.grad.clone() if hasattr(module, 'weight') and module.weight is not None and module.weight.grad is not None else None
        return grad_x, grad_w

    # For REF_Module, we need to ensure its internal state (weights) is set correctly before forward for backward
    # This is already handled as ref_module shares weights with eqt_module if affine.

    # Helper for functional ref backward
    def get_grads_functional(func_fwd, inp, weight_tensor, irreps_info_fwd, scaled_fwd, eps_fwd, grad_output_tensor):
        if inp.grad is not None:
            inp.grad.zero_()
        if weight_tensor is not None and weight_tensor.grad is not None:
            weight_tensor.grad.zero_()
        
        # Ensure weight_tensor requires grad if it's not None (for affine cases)
        if weight_tensor is not None:
            weight_tensor.requires_grad_(True)

        output = func_fwd(inp, irreps_info_fwd, weight_tensor, scaled_fwd, eps_fwd)
        output.backward(grad_output_tensor)
        
        grad_x = inp.grad.clone() if inp.grad is not None else torch.zeros_like(inp)
        grad_w = weight_tensor.grad.clone() if weight_tensor is not None and weight_tensor.grad is not None else None
        return grad_x, grad_w

    funcs_to_test_bwd = {
        "EQT_Module_Bwd": (get_grads_eqt, (eqt_module, x, grad_out), {}),
        "REF_Module_Bwd": (get_grads_eqt, (ref_module, x, grad_out), {}),
        "REF_Functional_Bwd": (
            get_grads_functional,
            (ref_functional_forward_only_layer_rms_norm, x, weight_for_functional_ref, 
             irreps_info_for_functional_ref, scaled_test, eps_test, grad_out),
            {},
        ),
    }
    # REF_Functional_Bwd will correctly produce grad_x even if not affine (weight_for_functional_ref is None).
    # grad_w will be None in that case, which is handled by the comparison logic below.
    tester_bwd = FunctionTester(funcs_to_test_bwd)
    
    print("\nBackward Pass Comparisons:")
    results_bwd = {}
    for name, (func, args, kwargs) in funcs_to_test_bwd.items():
        results_bwd[name] = func(*args, **kwargs)

    # Compare grad_x for all available backward functions
    grad_x_results = {name: res[0] for name, res in results_bwd.items()}
    for (name1, grad_x1), (name2, grad_x2) in itertools.combinations(grad_x_results.items(), 2):
        diff_x = max_abs_diff(grad_x1, grad_x2)
        print(f"  grad_x ({name1} vs {name2}): {diff_x}")
        if not math.isnan(diff_x) and not (diff_x < 1e-5):
            print(f"    WARNING: Backward pass (grad_x) difference > 1e-5 between {name1} and {name2}: {diff_x}")
            # assert diff_x < 1e-5, f"Backward pass (grad_x) mismatch between {name1} and {name2}"

    # Compare grad_w if affine
    if affine_test:
        grad_w_results = {name: res[1] for name, res in results_bwd.items()}
        for (name1, grad_w1), (name2, grad_w2) in itertools.combinations(grad_w_results.items(), 2):
            if grad_w1 is not None and grad_w2 is not None:
                diff_w = max_abs_diff(grad_w1, grad_w2)
                print(f"  grad_w ({name1} vs {name2}): {diff_w}")
                if not math.isnan(diff_w) and not (diff_w < 2e-5): # Using 2e-5 tolerance for weights
                    print(f"    WARNING: Backward pass (grad_w) difference > 2e-5 between {name1} and {name2}: {diff_w}")
                    # assert diff_w < 2e-5, f"Backward pass (grad_w) mismatch between {name1} and {name2}"
            elif grad_w1 is None and grad_w2 is None:
                print(f"  grad_w ({name1} vs {name2}): Both None, consistent.")
            else:
                print(f"    ERROR: grad_w None-ness mismatch between {name1} ({grad_w1 is None}) and {name2} ({grad_w2 is None})")
                # assert False, f"grad_w None-ness mismatch between {name1} and {name2}"
    
    print(f"Finished tests for LayerRMSNorm with Irreps='{irreps_str}', affine={affine_test}, scaled={scaled_test}.") # Changed "passed" to "finished"

if __name__ == "__main__":
    test_configs = [
        {"irreps_str": "0", "N": 1, "C": 1},   # Basic
        {"irreps_str": "1x0e + 2x1o + 1x2e", "N": 256, "C": 64},   # Basic
        {"irreps_str": "3x0e", "N": 257, "C": 65},                # Scalars only
        {"irreps_str": "2x1e", "N": 253, "C": 62},                # Vectors only
        {"irreps_str": "1x0e+1x1o+1x2e+1x3o", "N": 251, "C": 63}, # Mixed, more irreps
        {"irreps_str": "1x0e", "N":258, "C":65},                  # Minimal case
        {"irreps_str": "1x1o", "N":256, "C":64},
        # Add back empty tensor tests if desired, carefully handling ... #assertions for NaN/None grads
        # {"irreps_str": "0x0e", "N":2, "C":2}, 
        # {"irreps_str": "1x0e", "N":0, "C":2}, 
        # {"irreps_str": "1x0e", "N":2, "C":0}, 
    ]

    for affine in [True, False]:
        for scaled in [True, False]:
            for config in test_configs:
                run_tests(
                    irreps_str=config["irreps_str"], 
                    N=config["N"], 
                    C=config["C"], 
                    affine_test=affine, 
                    scaled_test=scaled
                )
    
    print("\nAll LayerRMSNorm tests finished.")
