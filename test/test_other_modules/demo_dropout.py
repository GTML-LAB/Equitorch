import torch
from equitorch.irreps import Irreps
from equitorch.nn.dropout import Dropout
from equitorch.nn.norm import Norm # Import Norm

# Setup
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# torch.manual_seed(0) # For reproducible dropout masks if needed for inspection
PRINT_TOLERANCE = 1e-6

def analyze_dropout_output(output_tensor: torch.Tensor, 
                           input_tensor: torch.Tensor, 
                           p: float, 
                           active_dropout: bool,
                           norm_analyzer: Norm): # Pass the Norm module
    """
    Analyzes the output of a dropout layer.
    Checks the element-wise condition and reports norm ratio per irrep/channel.
    """
    # Calculate norm per irrep instance and channel
    # norm_analyzer(tensor) will have shape (N, num_irrep_instances, C)
    norm_output_per_irrep = norm_analyzer(output_tensor)
    norm_input_per_irrep = norm_analyzer(input_tensor)
    
    norm_ratio_matrix = norm_output_per_irrep / (norm_input_per_irrep + 1e-16)
    print(f"    Norm_irrep(output) / Norm_irrep(input) matrix :\n{norm_ratio_matrix}")

    # Overall Frobenius norm ratio for a quick summary (optional, can be removed)
    # overall_norm_output = torch.norm(output_tensor)
    # overall_norm_input = torch.norm(input_tensor)
    # overall_norm_ratio = overall_norm_output / (overall_norm_input + 1e-16)
    print(f"    mean square norm ratio: {norm_ratio_matrix.mean()}")


    if active_dropout and p > 0.0:
        kept_mask = torch.abs(output_tensor) > PRINT_TOLERANCE
        
        input_elements_if_kept = input_tensor[kept_mask]
        output_elements_kept_scaled_back = output_tensor[kept_mask] * (1.0 - p)
        
        output_elements_dropped = output_tensor[~kept_mask]

        all_elements_match_condition = True
        if input_elements_if_kept.numel() > 0:
            if not torch.allclose(output_elements_kept_scaled_back, input_elements_if_kept, atol=PRINT_TOLERANCE):
                all_elements_match_condition = False
                max_diff_kept = torch.max(torch.abs(output_elements_kept_scaled_back - input_elements_if_kept)).item()
                print(f"    WARNING: Kept elements condition not met. Max diff: {max_diff_kept:.2e}")
        
        if output_elements_dropped.numel() > 0:
            if not torch.allclose(output_elements_dropped, torch.zeros_like(output_elements_dropped), atol=PRINT_TOLERANCE):
                all_elements_match_condition = False
                max_val_dropped = torch.max(torch.abs(output_elements_dropped)).item()
                print(f"    WARNING: Dropped elements condition not met. Max val of dropped: {max_val_dropped:.2e}")

        if all_elements_match_condition:
            print(f"    Element-wise check (output*(1-p) ~ input OR output ~ 0): Passed (tolerance {PRINT_TOLERANCE})")
        else:
            print(f"    Element-wise check (output*(1-p) ~ input OR output ~ 0): Failed (see warnings)")
    elif p == 0.0:
        assert torch.allclose(output_tensor, input_tensor, atol=PRINT_TOLERANCE), "Output should be identical to input when p=0"
        print("    Output is identical to input (as p=0).")
    elif not active_dropout:
        assert torch.allclose(output_tensor, input_tensor, atol=PRINT_TOLERANCE), "Output should be identical to input when dropout is not active"
        print("    Output is identical to input (dropout not active).")
    print()


# Define some sample Irreps and input tensor
irreps_str = "2x0e + 1x1o" # Example: 2 scalars (dim=1 each), 1 vector (dim=3)
# irreps_str = "100x0" # Example: 2 scalars (dim=1 each), 1 vector (dim=3)
irreps_global = Irreps(irreps_str) # num_irrep_instances = 3
# N, C = 100, 500 # Batch size, Channels
N, C = 2, 5 # Batch size, Channels
input_tensor_global = torch.randn(N, irreps_global.dim, C, device=DEVICE)

# Create Norm analyzer instance
# scaled=False means it computes sqrt(sum(x_i^2)) for each irrep block, not scaled by 1/sqrt(D_ir)
norm_analyzer_global = Norm(irreps=irreps_global, scaled=False).to(DEVICE)
# Ensure IrrepsInfo within norm_analyzer is on the correct device
norm_analyzer_global.irreps_info = norm_analyzer_global.irreps_info.to(DEVICE)


print(f"Input Irreps: {irreps_str} (dim={irreps_global.dim}, num_instances={len(irreps_global)})")
print(f"Input Tensor Shape: {input_tensor_global.shape}")
# print(f"Input Tensor:\n{input_tensor_global}\n") # Can be verbose

dropout_p = 0.333

# --- Demo Case 1: Irrep-wise Dropout ---
print(f"--- Case 1: Irrep-wise Dropout (p={dropout_p}) ---")
dropout_irrep_wise = Dropout(p=dropout_p, irreps=irreps_global, irrep_wise=True, work_on_eval=False).to(DEVICE)
input_clone_1 = input_tensor_global.clone()

# Training mode
dropout_irrep_wise.train()
output_train_iw = dropout_irrep_wise(input_clone_1)
print("Training mode (irrep_wise=True):")
# print(f"  Output:\n{output_train_iw}")
analyze_dropout_output(output_train_iw, input_clone_1, dropout_p, active_dropout=True, norm_analyzer=norm_analyzer_global)

# Evaluation mode (work_on_eval=False, so dropout is off)
dropout_irrep_wise.eval()
input_clone_2 = input_tensor_global.clone()
output_eval_iw_off = dropout_irrep_wise(input_clone_2)
print("Evaluation mode (irrep_wise=True, work_on_eval=False):")
# print(f"  Output:\n{output_eval_iw_off}")
analyze_dropout_output(output_eval_iw_off, input_clone_2, dropout_p, active_dropout=False, norm_analyzer=norm_analyzer_global)


# Evaluation mode (work_on_eval=True, so dropout is on)
dropout_irrep_wise_eval_on = Dropout(p=dropout_p, irreps=irreps_global, irrep_wise=True, work_on_eval=True).to(DEVICE)
dropout_irrep_wise_eval_on.eval() # Still in eval mode, but work_on_eval=True
input_clone_3 = input_tensor_global.clone()
output_eval_iw_on = dropout_irrep_wise_eval_on(input_clone_3)
print("Evaluation mode (irrep_wise=True, work_on_eval=True):")
# print(f"  Output:\n{output_eval_iw_on}")
analyze_dropout_output(output_eval_iw_on, input_clone_3, dropout_p, active_dropout=True, norm_analyzer=norm_analyzer_global)


# --- Demo Case 2: Non-Irrep-wise Dropout (Standard Dropout1D on features) ---
print(f"--- Case 2: Non-Irrep-wise Dropout (p={dropout_p}) ---")
dropout_non_irrep_wise = Dropout(p=dropout_p, irreps=irreps_global, irrep_wise=False, work_on_eval=False).to(DEVICE)
input_clone_4 = input_tensor_global.clone()

# Training mode
dropout_non_irrep_wise.train()
output_train_niw = dropout_non_irrep_wise(input_clone_4)
print("Training mode (irrep_wise=False):")
# print(f"  Output:\n{output_train_niw}")
analyze_dropout_output(output_train_niw, input_clone_4, dropout_p, active_dropout=True, norm_analyzer=norm_analyzer_global)

# Evaluation mode (work_on_eval=False, so dropout is off)
dropout_non_irrep_wise.eval()
input_clone_5 = input_tensor_global.clone()
output_eval_niw_off = dropout_non_irrep_wise(input_clone_5)
print("Evaluation mode (irrep_wise=False, work_on_eval=False):")
# print(f"  Output:\n{output_eval_niw_off}")
analyze_dropout_output(output_eval_niw_off, input_clone_5, dropout_p, active_dropout=False, norm_analyzer=norm_analyzer_global)

# --- Demo Case 3: Dropout with p=0 ---
print("--- Case 3: Dropout with p=0.0 (should be identity) ---")
dropout_p0 = Dropout(p=0.0, irreps=irreps_global, irrep_wise=True).to(DEVICE) # irrep_wise doesn't matter for p=0
dropout_p0.train() # Training or eval, should be identity
input_clone_6 = input_tensor_global.clone()
output_p0 = dropout_p0(input_clone_6)
print("Training mode (p=0.0):")
# print(f"  Output:\n{output_p0}")
analyze_dropout_output(output_p0, input_clone_6, 0.0, active_dropout=False, norm_analyzer=norm_analyzer_global)

print("Dropout demo finished.")
