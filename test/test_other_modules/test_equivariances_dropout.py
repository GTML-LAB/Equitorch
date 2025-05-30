import sys
sys.path.append('../..')

import torch
from equitorch.irreps import Irreps
from equitorch.nn.dropout import Dropout
from equitorch.nn.functional.wigner_d import dense_wigner_rotation
from test_utils import FunctionTester, rand_rotation_dict, max_abs_diff, max_abs_diff_list

# Set default dtype and seed for reproducibility of test data generation
torch.set_default_dtype(torch.float32)
# torch.set_default_dtype(torch.float64)
torch.random.manual_seed(0) # For test data generation

# Use the same diff functions as in other equivariance tests
diff_func = max_abs_diff
diff_func_list = max_abs_diff_list

def get_dropout_test_data(irreps_in_obj: Irreps, C: int, N: int, device: str = 'cuda', dtype: torch.dtype = torch.float64):
    r"""
    Generates input_data_val (raw data, requires_grad=False), 
    grad_output, Wigner D matrix, and VJPs for tests.
    """
    input_data_val = torch.randn(N, irreps_in_obj.dim, C, device=device, dtype=dtype)
    grad_output = torch.randn(N, irreps_in_obj.dim, C, device=device, dtype=dtype) # For backward dL/dy

    rotation_dict = rand_rotation_dict(irreps_in_obj, N, dtype=dtype, device=device)
    wigner_D = rotation_dict['wigner_D']

    # VJPs for second-order tests
    vjp1_dl_dy = torch.randn_like(input_data_val, device=device, dtype=dtype)
    vjp2_for_dx = torch.randn_like(input_data_val, device=device, dtype=dtype)
    
    return input_data_val, grad_output, wigner_D, vjp1_dl_dy, vjp2_for_dx

# --- Forward Pass Test Functions ---
def forward_dropout_original(DropoutClass, input_data_raw, irreps_str_for_layer, C, dropout_p, irrep_wise_flag, work_on_eval_flag, training_mode_bool, dropout_seed):
    input_for_op = input_data_raw.clone().requires_grad_(True)
    
    current_irreps_obj = Irreps(irreps_str_for_layer) if irrep_wise_flag else None

    torch.manual_seed(dropout_seed) # Ensure same mask
    layer = DropoutClass(p=dropout_p, irreps=current_irreps_obj, irrep_wise=irrep_wise_flag, work_on_eval=work_on_eval_flag).to(input_for_op.device)
    layer.train(training_mode_bool)
    return layer(input_for_op)

def forward_dropout_rotated(DropoutClass, input_data_raw, wigner_D, irreps_str_for_layer, C, dropout_p, irrep_wise_flag, work_on_eval_flag, training_mode_bool, dropout_seed):
    input_for_op = input_data_raw.clone().requires_grad_(True)
    current_irreps_obj = Irreps(irreps_str_for_layer) if irrep_wise_flag else None

    torch.manual_seed(dropout_seed) # Ensure same mask
    layer = DropoutClass(p=dropout_p, irreps=current_irreps_obj, irrep_wise=irrep_wise_flag, work_on_eval=work_on_eval_flag).to(input_for_op.device)
    layer.train(training_mode_bool)
    
    rotated_input = dense_wigner_rotation(input_for_op, wigner_D)
    rotated_output = layer(rotated_input)
    return dense_wigner_rotation(rotated_output, wigner_D.transpose(-1, -2))

# --- Backward Pass Test Functions (Input Gradients) ---
def backward_input_grad_original(DropoutClass, input_data_raw, grad_out, irreps_str_for_layer, C, dropout_p, irrep_wise_flag, work_on_eval_flag, training_mode_bool, dropout_seed):
    input_tensor_clone = input_data_raw.clone().requires_grad_(True)
    current_irreps_obj = Irreps(irreps_str_for_layer) if irrep_wise_flag else None

    torch.manual_seed(dropout_seed)
    layer = DropoutClass(p=dropout_p, irreps=current_irreps_obj, irrep_wise=irrep_wise_flag, work_on_eval=work_on_eval_flag).to(input_tensor_clone.device)
    layer.train(training_mode_bool)

    output = layer(input_tensor_clone)
    output.backward(grad_out)
    return [input_tensor_clone.grad]

def backward_input_grad_rotated(DropoutClass, input_data_raw, grad_out, wigner_D, irreps_str_for_layer, C, dropout_p, irrep_wise_flag, work_on_eval_flag, training_mode_bool, dropout_seed):
    input_tensor_clone = input_data_raw.clone().requires_grad_(True)
    current_irreps_obj = Irreps(irreps_str_for_layer) if irrep_wise_flag else None

    torch.manual_seed(dropout_seed)
    layer = DropoutClass(p=dropout_p, irreps=current_irreps_obj, irrep_wise=irrep_wise_flag, work_on_eval=work_on_eval_flag).to(input_tensor_clone.device)
    layer.train(training_mode_bool)

    rotated_input = dense_wigner_rotation(input_tensor_clone, wigner_D)
    rotated_output = layer(rotated_input)
    rotated_grad_out = dense_wigner_rotation(grad_out, wigner_D)
    rotated_output.backward(rotated_grad_out)
    return [input_tensor_clone.grad]

# --- Second Order Test Functions (Self-Input: d/dx (dL/dx)) ---
def so_self_x_original_dropout(DropoutClass, input_data_raw, irreps_str_for_layer, C, dropout_p, irrep_wise_flag, work_on_eval_flag, training_mode_bool, vjp1_dl_dy, vjp2_for_dx, dropout_seed):
    x_in = input_data_raw.clone().requires_grad_(True)
    current_irreps_obj = Irreps(irreps_str_for_layer) if irrep_wise_flag else None

    torch.manual_seed(dropout_seed)
    current_layer = DropoutClass(p=dropout_p, irreps=current_irreps_obj, irrep_wise=irrep_wise_flag, work_on_eval=work_on_eval_flag).to(x_in.device)
    current_layer.train(training_mode_bool)
    
    out = current_layer(x_in)
    grad_x_1st_tuple = torch.autograd.grad(out, x_in, grad_outputs=vjp1_dl_dy, create_graph=True, allow_unused=True)
    grad_x_1st = grad_x_1st_tuple[0]

    if grad_x_1st is None:
        grad_xx_2nd = torch.zeros_like(x_in)
    elif not grad_x_1st.requires_grad:
        grad_xx_2nd = torch.zeros_like(x_in)
    else:
        grad_xx_2nd_tuple = torch.autograd.grad(grad_x_1st, x_in, grad_outputs=vjp2_for_dx, retain_graph=False, allow_unused=True)
        grad_xx_2nd = grad_xx_2nd_tuple[0]
        if grad_xx_2nd is None: 
            grad_xx_2nd = torch.zeros_like(x_in)
    return [grad_xx_2nd]

def so_self_x_rotated_dropout(DropoutClass, input_data_raw, wigner_D, irreps_str_for_layer, C, dropout_p, irrep_wise_flag, work_on_eval_flag, training_mode_bool, vjp1_dl_dy, vjp2_for_dx, dropout_seed):
    x_in_orig = input_data_raw.clone().requires_grad_(True)
    current_irreps_obj = Irreps(irreps_str_for_layer) if irrep_wise_flag else None

    torch.manual_seed(dropout_seed)
    current_layer = DropoutClass(p=dropout_p, irreps=current_irreps_obj, irrep_wise=irrep_wise_flag, work_on_eval=work_on_eval_flag).to(x_in_orig.device)
    current_layer.train(training_mode_bool)
        
    x_transformed = dense_wigner_rotation(x_in_orig, wigner_D)
    out_intermediate = current_layer(x_transformed)
    vjp1_transformed = dense_wigner_rotation(vjp1_dl_dy, wigner_D)
    
    G_equiv_x_tuple = torch.autograd.grad(out_intermediate, x_in_orig, grad_outputs=vjp1_transformed, create_graph=True, allow_unused=True)
    G_equiv_x = G_equiv_x_tuple[0]
    
    if G_equiv_x is None:
        grad_xx_2nd_rot = torch.zeros_like(x_in_orig)
    elif not G_equiv_x.requires_grad:
        grad_xx_2nd_rot = torch.zeros_like(x_in_orig)
    else:
        grad_xx_2nd_rot_tuple = torch.autograd.grad(G_equiv_x, x_in_orig, grad_outputs=vjp2_for_dx, retain_graph=False, allow_unused=True)
        grad_xx_2nd_rot = grad_xx_2nd_rot_tuple[0]
        if grad_xx_2nd_rot is None:
            grad_xx_2nd_rot = torch.zeros_like(x_in_orig)
    return [grad_xx_2nd_rot]

# --- Main Test Runner ---
def run_equivariance_tests_for_dropout_layer(
        DropoutClass, 
        irreps_in_str: str, 
        C: int, N: int, 
        dropout_p_val: float, 
        dropout_seed_val: int, 
        device: str = 'cuda'):
    
    print(f"\n--- Testing {DropoutClass.__name__} with Irreps: {irreps_in_str}, C: {C}, N: {N}, p: {dropout_p_val}, Device: {device} ---")
    
    irreps_for_data_gen = Irreps(irreps_in_str) 
    input_data_raw, grad_out_data, wigner_D_data, vjp1_data, vjp2_x_data = \
        get_dropout_test_data(irreps_for_data_gen, C, N, device=device, dtype=torch.get_default_dtype())

    irrep_wise_options = [True, False]
    work_on_eval_options = [True, False]
    training_mode_options = [True, False] # Corresponds to layer.train(True/False)

    for iw_flag in irrep_wise_options:
        # Skip if irrep_wise=True but irreps_in_str is trivial or would lead to no irreps_info
        if iw_flag and not irreps_in_str: # Or more robust check if Irreps(irreps_in_str) is empty
             print(f"Skipping irrep_wise=True for empty/invalid irreps_str: {irreps_in_str}")
             continue

        for woe_flag in work_on_eval_options:
            for train_bool in training_mode_options:
                
                # If not training and not work_on_eval, dropout is identity. Equivariance is trivial.
                # Can skip for brevity or test for completeness. Let's test for completeness.
                # if not train_bool and not woe_flag and dropout_p_val > 0.0:
                #     print(f"Skipping config: irrep_wise={iw_flag}, work_on_eval={woe_flag}, training={train_bool} (dropout is identity)")
                #     continue
                
                print(f"\nConfig: irrep_wise={iw_flag}, work_on_eval={woe_flag}, training_mode={train_bool}")

                # Forward Test
                print("Forward test:")
                tester_fwd = FunctionTester({
                    'ori': (forward_dropout_original, [DropoutClass, input_data_raw, irreps_in_str, C, dropout_p_val, iw_flag, woe_flag, train_bool, dropout_seed_val], {}),
                    'rot': (forward_dropout_rotated, [DropoutClass, input_data_raw, wigner_D_data, irreps_in_str, C, dropout_p_val, iw_flag, woe_flag, train_bool, dropout_seed_val], {}),
                })
                comp_fwd = tester_fwd.compare(compare_func=diff_func)
                print(comp_fwd)

                # Backward Test (Input Gradient)
                print("Backward test (input grad):")
                tester_bwd_input = FunctionTester({
                    'ori': (backward_input_grad_original, [DropoutClass, input_data_raw, grad_out_data, irreps_in_str, C, dropout_p_val, iw_flag, woe_flag, train_bool, dropout_seed_val], {}),
                    'rot': (backward_input_grad_rotated, [DropoutClass, input_data_raw, grad_out_data, wigner_D_data, irreps_in_str, C, dropout_p_val, iw_flag, woe_flag, train_bool, dropout_seed_val], {}),
                })
                comp_bwd_input = tester_bwd_input.compare(compare_func=diff_func_list)
                print(comp_bwd_input)

                # Second Order Test (Self-Input)
                print("SO test (d/dx (dL/dx)):")
                tester_so_self_x = FunctionTester({
                    'ori': (so_self_x_original_dropout, [DropoutClass, input_data_raw, irreps_in_str, C, dropout_p_val, iw_flag, woe_flag, train_bool, vjp1_data, vjp2_x_data, dropout_seed_val], {}),
                    'rot': (so_self_x_rotated_dropout, [DropoutClass, input_data_raw, wigner_D_data, irreps_in_str, C, dropout_p_val, iw_flag, woe_flag, train_bool, vjp1_data, vjp2_x_data, dropout_seed_val], {}),
                })
                comp_so_self_x = tester_so_self_x.compare(compare_func=diff_func_list)
                print(comp_so_self_x)


if __name__ == '__main__':
    C_test = 4 
    N_test = 2   
    device_test = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device_test = 'cpu' # For easier debugging if needed
    
    # irreps_options_str_main = ['0e+1o', '2x0e+1x1o+1x2e', '1e'] # Test with various irreps
    irreps_options_str_main = ['2x0e+1x1+1x2e'] # Test with various irreps
    dropout_p_test = 0.5
    dropout_fixed_seed = 42 # Seed for deterministic dropout masks within comparisons

    for irreps_str_main_loop in irreps_options_str_main:
        run_equivariance_tests_for_dropout_layer(
            Dropout, 
            irreps_str_main_loop, 
            C_test, N_test, 
            dropout_p_val=dropout_p_test,
            dropout_seed_val=dropout_fixed_seed,
            device=device_test
        )

    # Test p=0 (should be identity)
    print("\nTesting with p=0.0 (should be identity operation)")
    run_equivariance_tests_for_dropout_layer(
        Dropout, '1x1o', C_test, N_test, dropout_p_val=0.0, dropout_seed_val=dropout_fixed_seed, device=device_test
    )
    
    # Test p=1 (should be all zeros)
    # Note: dL/dx will be zero if output is always zero. d/dx(dL/dx) will also be zero.
    # The comparison should still hold (zero vs zero).
    print("\nTesting with p=1.0 (should be zero output)")
    run_equivariance_tests_for_dropout_layer(
        Dropout, '1x1o', C_test, N_test, dropout_p_val=1.0, dropout_seed_val=dropout_fixed_seed, device=device_test
    )

    print("\nEquivariance tests for Dropout layer completed.")
