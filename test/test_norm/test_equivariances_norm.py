import sys
sys.path.append('../..')

import torch
from equitorch.irreps import Irreps
from equitorch.nn.normalization import BatchRMSNorm, LayerRMSNorm
from equitorch.nn.functional.wigner_d import dense_wigner_rotation
from test_utils import FunctionTester, rand_rotation_dict, max_abs_diff, max_abs_diff_list, rate_mean2_std, rate_mean2_std_list

# Set default dtype and seed for reproducibility
torch.set_default_dtype(torch.float64)
torch.random.manual_seed(0)

# Use the same diff functions as in test_equivariances_gate.py
diff_func = max_abs_diff
diff_func_list = max_abs_diff_list

def print_graph_debug_info(tensor, tensor_name: str, context_message: str = ""):
    if tensor is None:
        print(f"Debug info for {tensor_name} ({context_message}): Tensor is None.")
        return
    print(f"Debug info for {tensor_name} ({context_message}):")
    print(f"  Shape: {tensor.shape}")
    print(f"  dtype: {tensor.dtype}")
    print(f"  Device: {tensor.device}")
    print(f"  requires_grad: {tensor.requires_grad}")
    print(f"  is_leaf: {tensor.is_leaf}")
    grad_fn_obj = tensor.grad_fn
    print(f"  grad_fn: {type(grad_fn_obj).__name__ if grad_fn_obj else None}")
    if grad_fn_obj:
        print(f"  grad_fn.next_functions:")
        for i, (next_fn, grad_idx) in enumerate(grad_fn_obj.next_functions):
            if next_fn is not None:
                print(f"    [{i}] fn_name: {type(next_fn).__name__}, grad_idx: {grad_idx}, variable_requires_grad: {next_fn.variable.requires_grad if hasattr(next_fn, 'variable') else 'N/A'}")
            else:
                # This case typically means the input to the grad_fn was a leaf tensor that requires grad.
                print(f"    [{i}] fn_name: None (points to a leaf tensor or a tensor without grad_fn), grad_idx: {grad_idx}")
    else:
        print(f"  No grad_fn, so no next_functions.")
    print("-" * 40)

def get_norm_test_data(irreps_in: Irreps, C: int, N: int, device: str = 'cuda', dtype: torch.dtype = torch.float64):
    r"""
    Generates input_data_val (raw data, requires_grad=False), 
    grad_output, Wigner D matrix, and VJPs for tests.
    """
    input_data_val = torch.randn(N, irreps_in.dim, C, device=device, dtype=dtype) # requires_grad is False by default
    grad_output = torch.randn(N, irreps_in.dim, C, device=device, dtype=dtype) # For backward dL/dy

    # For Wigner D matrix, use the original Irreps object passed to the function
    rotation_dict = rand_rotation_dict(irreps_in, N, dtype=dtype, device=device)
    wigner_D = rotation_dict['wigner_D']

    # VJPs for second-order tests
    vjp1_dl_dy = torch.randn_like(input_data_val, device=device, dtype=dtype) # VJP for dL/dy (first autograd.grad)
    vjp2_for_dx = torch.randn_like(input_data_val, device=device, dtype=dtype) # VJP for dL/dx (when it's an input to a second grad)
    
    # vjp2_for_dw will be created inside test functions as its shape depends on the layer's weight

    return input_data_val, grad_output, wigner_D, vjp1_dl_dy, vjp2_for_dx

# --- Forward Pass Test Functions ---
def forward_norm_original(NormLayerClass, input_data_raw, irreps_str, C, affine, scaled, training_mode_for_batch_norm):
    input_for_op = input_data_raw.clone().requires_grad_(True) # Output of forward needs graph for backward
    layer = NormLayerClass(Irreps(irreps_str), C, affine=affine, scaled=scaled).to(input_for_op.device)
    if NormLayerClass == BatchRMSNorm:
        layer.train(training_mode_for_batch_norm)
    else:
        layer.eval() # LayerRMSNorm doesn't have a training mode that affects output differently for this test
    return layer(input_for_op)

def forward_norm_rotated(NormLayerClass, input_data_raw, wigner_D, irreps_str, C, affine, scaled, training_mode_for_batch_norm):
    input_for_op = input_data_raw.clone().requires_grad_(True) # Output of forward needs graph for backward
    layer = NormLayerClass(Irreps(irreps_str), C, affine=affine, scaled=scaled).to(input_for_op.device)
    if NormLayerClass == BatchRMSNorm:
        layer.train(training_mode_for_batch_norm)
    else:
        layer.eval()
    
    rotated_input = dense_wigner_rotation(input_for_op, wigner_D) # Use input_for_op
    rotated_output = layer(rotated_input)
    return dense_wigner_rotation(rotated_output, wigner_D.transpose(-1, -2))

# --- Backward Pass Test Functions (Input Gradients) ---
def backward_input_grad_original(NormLayerClass, input_data_raw, grad_out, irreps_str, C, affine, scaled, training_mode_for_batch_norm):
    input_tensor_clone = input_data_raw.clone().requires_grad_(True) # This is the tensor we differentiate w.r.t
    layer = NormLayerClass(Irreps(irreps_str), C, affine=affine, scaled=scaled).to(input_tensor_clone.device)
    if NormLayerClass == BatchRMSNorm:
        layer.train(training_mode_for_batch_norm)
    else:
        layer.eval()

    output = layer(input_tensor_clone)
    output.backward(grad_out)
    return [input_tensor_clone.grad]

def backward_input_grad_rotated(NormLayerClass, input_data_raw, grad_out, wigner_D, irreps_str, C, affine, scaled, training_mode_for_batch_norm):
    input_tensor_clone = input_data_raw.clone().requires_grad_(True) # This is the tensor we differentiate w.r.t
    layer = NormLayerClass(Irreps(irreps_str), C, affine=affine, scaled=scaled).to(input_tensor_clone.device)
    if NormLayerClass == BatchRMSNorm:
        layer.train(training_mode_for_batch_norm)
    else:
        layer.eval()

    rotated_input = dense_wigner_rotation(input_tensor_clone, wigner_D)
    rotated_output = layer(rotated_input)
    rotated_grad_out = dense_wigner_rotation(grad_out, wigner_D)
    rotated_output.backward(rotated_grad_out)
    return [input_tensor_clone.grad] # This is D^T @ (dL/dx_rot)

# --- Backward Pass Test Functions (Weight Gradients, if affine=True) ---
def backward_weight_grad_original(NormLayerClass, input_data_raw, grad_out, irreps_str, C, scaled, training_mode_for_batch_norm):
    input_for_op = input_data_raw.clone() # Does not need grad itself for dL/dw
    # Affine must be True for this test
    layer = NormLayerClass(Irreps(irreps_str), C, affine=True, scaled=scaled).to(input_for_op.device)
    if NormLayerClass == BatchRMSNorm:
        layer.train(training_mode_for_batch_norm)
    else:
        layer.eval()
    
    layer.weight.requires_grad_(True)
    
    output = layer(input_for_op)
    output.backward(grad_out)
    return [layer.weight.grad]

def backward_weight_grad_rotated(NormLayerClass, input_data_raw, grad_out, wigner_D, irreps_str, C, scaled, training_mode_for_batch_norm):
    input_for_op = input_data_raw.clone() # Does not need grad itself for dL/dw
    # Affine must be True for this test
    layer = NormLayerClass(Irreps(irreps_str), C, affine=True, scaled=scaled).to(input_for_op.device)
    if NormLayerClass == BatchRMSNorm:
        layer.train(training_mode_for_batch_norm)
    else:
        layer.eval()
        
    layer.weight.requires_grad_(True)

    rotated_input = dense_wigner_rotation(input_for_op, wigner_D) # Use input_for_op
    rotated_output = layer(rotated_input)
    rotated_grad_out = dense_wigner_rotation(grad_out, wigner_D)
    rotated_output.backward(rotated_grad_out)
    return [layer.weight.grad] # dL/dw should be invariant

# --- Second Order Test Functions (Self-Input: d/dx (dL/dx)) ---
def so_self_x_original(NormLayerClass, input_data_raw, irreps_str, C, affine, scaled, training_mode_for_batch_norm, vjp1_dl_dy, vjp2_for_dx):
    x_in = input_data_raw.clone().requires_grad_(True) # This is the x we differentiate w.r.t. twice
    current_layer = NormLayerClass(Irreps(irreps_str), C, affine=affine, scaled=scaled).to(x_in.device)
    if NormLayerClass == BatchRMSNorm:
        current_layer.train(training_mode_for_batch_norm)
    else:
        current_layer.eval()
    
    if affine and current_layer.weight is not None: # Ensure weight also requires grad if it exists
        current_layer.weight.requires_grad_(True)

    out = current_layer(x_in)
    grad_x_1st_tuple = torch.autograd.grad(out, x_in, grad_outputs=vjp1_dl_dy, create_graph=True, allow_unused=True)
    grad_x_1st = grad_x_1st_tuple[0]

    if grad_x_1st is None:
        # This implies 'out' does not depend on 'x_in', which is highly unlikely for norm layers.
        print(f"Warning: In so_self_x_original, grad_x_1st (dL/dx) is None. This is unexpected. Setting d/dx(dL/dx) to zero.")
        grad_xx_2nd = torch.zeros_like(x_in)
    elif not grad_x_1st.requires_grad:
        print(f"Info: In so_self_x_original, grad_x_1st (dL/dx) does not require grad w.r.t. x. "
              f"This can occur if dL/dx is constant w.r.t x (e.g., for {NormLayerClass.__name__} "
              f"with affine={affine}, training={training_mode_for_batch_norm}, scaled={scaled}). " # More context
              f"Analytically, d/dx(dL/dx) is zero. Skipping second autograd.grad.")
        print_graph_debug_info(grad_x_1st, "grad_x_1st", f"so_self_x_original ({NormLayerClass.__name__} aff={affine} sc={scaled} tr={training_mode_for_batch_norm})")
        grad_xx_2nd = torch.zeros_like(x_in)
    else:
        grad_xx_2nd_tuple = torch.autograd.grad(grad_x_1st, x_in, grad_outputs=vjp2_for_dx, retain_graph=False, allow_unused=True)
        grad_xx_2nd = grad_xx_2nd_tuple[0]
        if grad_xx_2nd is None: 
            grad_xx_2nd = torch.zeros_like(x_in)
    return [grad_xx_2nd]

def so_self_x_rotated(NormLayerClass, input_data_raw, wigner_D, irreps_str, C, affine, scaled, training_mode_for_batch_norm, vjp1_dl_dy, vjp2_for_dx):
    x_in_orig = input_data_raw.clone().requires_grad_(True) # This is the x_orig we differentiate w.r.t. twice
    current_layer = NormLayerClass(Irreps(irreps_str), C, affine=affine, scaled=scaled).to(x_in_orig.device)
    if NormLayerClass == BatchRMSNorm:
        current_layer.train(training_mode_for_batch_norm)
    else:
        current_layer.eval()

    if affine and current_layer.weight is not None:
        current_layer.weight.requires_grad_(True)
        
    x_transformed = dense_wigner_rotation(x_in_orig, wigner_D)
    # x_transformed.requires_grad_(True) # Not needed if x_in_orig.requires_grad is True and D is not learnable

    out_intermediate = current_layer(x_transformed)
    vjp1_transformed = dense_wigner_rotation(vjp1_dl_dy, wigner_D)
    
    # We need grad w.r.t. x_in_orig. If weight exists, it's a parameter of current_layer.
    # grad_inputs should be x_in_orig. If affine, also current_layer.weight.
    grad_inputs_tuple = (x_in_orig,)
    if affine and current_layer.weight is not None:
        # Ensure weight is part of the graph if it's to be differentiated through or w.r.t.
        # For dL/dx, weight is a parameter. For dL/dw, it's an input.
        # Here we only care about dL/dx.
        pass

    # G_equiv_x is dL_rot/dx_orig = D^T dL/dx_rot
    G_equiv_x_tuple = torch.autograd.grad(out_intermediate, grad_inputs_tuple, grad_outputs=vjp1_transformed, create_graph=True, allow_unused=True)
    G_equiv_x = G_equiv_x_tuple[0]
    
    if G_equiv_x is None:
        print(f"Warning: In so_self_x_rotated, G_equiv_x (dL_rot/dx_orig) is None. This is unexpected. Setting d/dx(dL_rot/dx_orig) to zero.")
        grad_xx_2nd_rot = torch.zeros_like(x_in_orig)
    elif not G_equiv_x.requires_grad:
        print(f"Info: In so_self_x_rotated, G_equiv_x (dL_rot/dx_orig) does not require grad w.r.t. x_orig. "
              f"This can occur if dL_rot/dx_orig is constant w.r.t x_orig (e.g., for {NormLayerClass.__name__} "
              f"with affine={affine}, training={training_mode_for_batch_norm}, scaled={scaled}). " # More context
              f"Analytically, d/dx_orig(dL_rot/dx_orig) is zero. Skipping second autograd.grad.")
        print_graph_debug_info(G_equiv_x, "G_equiv_x", f"so_self_x_rotated ({NormLayerClass.__name__} aff={affine} sc={scaled} tr={training_mode_for_batch_norm})")
        grad_xx_2nd_rot = torch.zeros_like(x_in_orig)
    else:
        grad_xx_2nd_rot_tuple = torch.autograd.grad(G_equiv_x, x_in_orig, grad_outputs=vjp2_for_dx, retain_graph=False, allow_unused=True)
        grad_xx_2nd_rot = grad_xx_2nd_rot_tuple[0]
        if grad_xx_2nd_rot is None:
            grad_xx_2nd_rot = torch.zeros_like(x_in_orig)
    return [grad_xx_2nd_rot]


# --- Second Order Test Functions (Self-Weight: d/dw (dL/dw)) ---
def so_self_w_original(NormLayerClass, input_data_raw, irreps_str, C, scaled, training_mode_for_batch_norm, vjp1_dl_dy):
    x_in = input_data_raw.clone() # This x is data, does not need grad for dL/dw
    current_layer = NormLayerClass(Irreps(irreps_str), C, affine=True, scaled=scaled).to(x_in.device)
    if NormLayerClass == BatchRMSNorm:
        current_layer.train(training_mode_for_batch_norm)
    else:
        current_layer.eval()
    
    current_layer.weight.requires_grad_(True)
    vjp2_for_dw = torch.randn_like(current_layer.weight)


    out = current_layer(x_in)
    # allow_unused=True for robustness, though not expected if affine=True
    grad_w_1st_tuple = torch.autograd.grad(out, current_layer.weight, grad_outputs=vjp1_dl_dy, create_graph=True, allow_unused=True)
    grad_w_1st = grad_w_1st_tuple[0]

    if grad_w_1st is None:
        # This case implies 'out' did not depend on 'current_layer.weight', which shouldn't happen if affine=True.
        print(f"Warning: In so_self_w_original, grad_w_1st (dL/dw) is None. This is unexpected for affine=True. Setting d/dw(dL/dw) to zero.")
        grad_ww_2nd = torch.zeros_like(current_layer.weight)
    elif not grad_w_1st.requires_grad:
        print(f"Info: In so_self_w_original, grad_w_1st (dL/dw) does not require grad w.r.t. weight (w). "
              f"This is expected as dL/dw = (x/sigma) * dL/dy_norm (approx), which is not a function of w. " # Clarified approx
              f"Analytically, d/dw(dL/dw) is zero. Skipping second autograd.grad.")
        print_graph_debug_info(grad_w_1st, "grad_w_1st", f"so_self_w_original ({NormLayerClass.__name__} aff=True sc={scaled} tr={training_mode_for_batch_norm})")
        grad_ww_2nd = torch.zeros_like(current_layer.weight)
    else:
        # This case is unlikely if dL/dw is truly constant w.r.t w for these layers.
        print(f"Warning: In so_self_w_original, grad_w_1st (dL/dw) unexpectedly requires_grad w.r.t w. Proceeding with autograd for d/dw(dL/dw).")
        grad_ww_2nd_tuple = torch.autograd.grad(grad_w_1st, current_layer.weight, grad_outputs=vjp2_for_dw, retain_graph=False, allow_unused=True)
        grad_ww_2nd = grad_ww_2nd_tuple[0]
        if grad_ww_2nd is None: # If derivative is None (e.g. zero and unused by PyTorch)
            grad_ww_2nd = torch.zeros_like(current_layer.weight)
    return [grad_ww_2nd]

def so_self_w_rotated(NormLayerClass, input_data_raw, wigner_D, irreps_str, C, scaled, training_mode_for_batch_norm, vjp1_dl_dy):
    x_in_orig = input_data_raw.clone() # This x is data
    current_layer = NormLayerClass(Irreps(irreps_str), C, affine=True, scaled=scaled).to(x_in_orig.device)
    if NormLayerClass == BatchRMSNorm:
        current_layer.train(training_mode_for_batch_norm)
    else:
        current_layer.eval()

    current_layer.weight.requires_grad_(True)
    vjp2_for_dw = torch.randn_like(current_layer.weight)
        
    x_transformed = dense_wigner_rotation(x_in_orig, wigner_D)
    out_intermediate = current_layer(x_transformed) 
    vjp1_transformed = dense_wigner_rotation(vjp1_dl_dy, wigner_D) 
    
    # G_equiv_w is dL_rot/dw_orig. Since w is invariant, dL_rot/dw_orig should be dL_orig/dw_orig.
    G_equiv_w_tuple = torch.autograd.grad(out_intermediate, current_layer.weight, grad_outputs=vjp1_transformed, create_graph=True, allow_unused=True)
    G_equiv_w = G_equiv_w_tuple[0]
    
    if G_equiv_w is None:
        print(f"Warning: In so_self_w_rotated, G_equiv_w (dL_rot/dw) is None. This is unexpected for affine=True. Setting d/dw(dL_rot/dw) to zero.")
        grad_ww_2nd_rot = torch.zeros_like(current_layer.weight)
    elif not G_equiv_w.requires_grad:
        # dL_rot/dw should be invariant and equal to dL_orig/dw.
        # As dL_orig/dw is not a function of w, dL_rot/dw is also not a function of w.
        print(f"Info: In so_self_w_rotated, G_equiv_w (dL_rot/dw) does not require grad w.r.t. weight (w). "
              f"This is expected as dL_rot/dw = dL_orig/dw, and dL_orig/dw is not a function of w (approx). " # Clarified approx
              f"Analytically, d/dw(dL_rot/dw) is zero. Skipping second autograd.grad.")
        print_graph_debug_info(G_equiv_w, "G_equiv_w", f"so_self_w_rotated ({NormLayerClass.__name__} aff=True sc={scaled} tr={training_mode_for_batch_norm})")
        grad_ww_2nd_rot = torch.zeros_like(current_layer.weight)
    else:
        print(f"Warning: In so_self_w_rotated, G_equiv_w (dL_rot/dw) unexpectedly requires_grad w.r.t w. Proceeding with autograd for d/dw(dL_rot/dw).")
        grad_ww_2nd_rot_tuple = torch.autograd.grad(G_equiv_w, current_layer.weight, grad_outputs=vjp2_for_dw, retain_graph=False, allow_unused=True)
        grad_ww_2nd_rot = grad_ww_2nd_rot_tuple[0]
        if grad_ww_2nd_rot is None: # If derivative is None
            grad_ww_2nd_rot = torch.zeros_like(current_layer.weight)
    return [grad_ww_2nd_rot]


# --- Second Order Test Functions (Cross XdW: d/dw (dL/dx)) ---
def so_cross_xdw_original(NormLayerClass, input_data_raw, irreps_str, C, scaled, training_mode_for_batch_norm, vjp1_dl_dy, vjp2_for_dx):
    x_in = input_data_raw.clone().requires_grad_(True) # x for dL/dx
    current_layer = NormLayerClass(Irreps(irreps_str), C, affine=True, scaled=scaled).to(x_in.device)
    if NormLayerClass == BatchRMSNorm:
        current_layer.train(training_mode_for_batch_norm)
    else:
        current_layer.eval()
    current_layer.weight.requires_grad_(True)

    out = current_layer(x_in)
    # grad_x_1st is dL/dx
    grad_x_1st, = torch.autograd.grad(out, x_in, grad_outputs=vjp1_dl_dy, create_graph=True, allow_unused=True)
    if grad_x_1st is None: grad_x_1st = torch.zeros_like(x_in) # Should not happen if vjp1 makes output depend on x

    # grad_xdw_2nd is d/dw (dL/dx)
    # vjp2_for_dx is the VJP for dL/dx when taking grad w.r.t weight.
    grad_xdw_2nd, = torch.autograd.grad(grad_x_1st, current_layer.weight, grad_outputs=vjp2_for_dx, retain_graph=False, allow_unused=True) # Differentiate dL/dx w.r.t weight
    if grad_xdw_2nd is None: grad_xdw_2nd = torch.zeros_like(current_layer.weight)
    return [grad_xdw_2nd]

def so_cross_xdw_rotated(NormLayerClass, input_data_raw, wigner_D, irreps_str, C, scaled, training_mode_for_batch_norm, vjp1_dl_dy, vjp2_for_dx):
    x_in_orig = input_data_raw.clone().requires_grad_(True) # x_orig for dL_rot/dx_orig
    current_layer = NormLayerClass(Irreps(irreps_str), C, affine=True, scaled=scaled).to(x_in_orig.device)
    if NormLayerClass == BatchRMSNorm:
        current_layer.train(training_mode_for_batch_norm)
    else:
        current_layer.eval()
    current_layer.weight.requires_grad_(True)
        
    x_transformed = dense_wigner_rotation(x_in_orig, wigner_D)
    out_intermediate = current_layer(x_transformed)
    vjp1_transformed = dense_wigner_rotation(vjp1_dl_dy, wigner_D)
    
    # G_equiv_x is dL_rot/dx_orig = D^T dL/dx_rot
    G_equiv_x, = torch.autograd.grad(out_intermediate, x_in_orig, grad_outputs=vjp1_transformed, create_graph=True, allow_unused=True)
    if G_equiv_x is None: G_equiv_x = torch.zeros_like(x_in_orig)

    # grad_xdw_2nd_rot is d/dw_orig (G_equiv_x) = d/dw_orig (D^T dL/dx_rot)
    # vjp2_for_dx is for G_equiv_x (which is dL/dx like). It should be transformed by D^T if it's a covector to dL/dx.
    # However, vjp2_for_dx is the vector for the VJP d(G_equiv_x)/dw . (vjp2_for_dx).
    # G_equiv_x is dL/dx_orig. So vjp2_for_dx is for dL/dx_orig.
    # The comparison is d/dw (dL/dx_orig) vs d/dw_orig (D^T dL/dx_rot).
    # Since D^T dL/dx_rot = dL/dx_orig, the VJPs should match.
    grad_xdw_2nd_rot, = torch.autograd.grad(G_equiv_x, current_layer.weight, grad_outputs=vjp2_for_dx, retain_graph=False, allow_unused=True)
    if grad_xdw_2nd_rot is None: grad_xdw_2nd_rot = torch.zeros_like(current_layer.weight)
    return [grad_xdw_2nd_rot]

# --- Second Order Test Functions (Cross WdX: d/dx (dL/dw)) ---
def so_cross_wdx_original(NormLayerClass, input_data_raw, irreps_str, C, scaled, training_mode_for_batch_norm, vjp1_dl_dy):
    x_in = input_data_raw.clone().requires_grad_(True) # x we differentiate dL/dw w.r.t
    current_layer = NormLayerClass(Irreps(irreps_str), C, affine=True, scaled=scaled).to(x_in.device)
    if NormLayerClass == BatchRMSNorm:
        current_layer.train(training_mode_for_batch_norm)
    else:
        current_layer.eval()
    current_layer.weight.requires_grad_(True)
    vjp2_for_dw = torch.randn_like(current_layer.weight)


    out = current_layer(x_in)
    # grad_w_1st is dL/dw
    grad_w_1st, = torch.autograd.grad(out, current_layer.weight, grad_outputs=vjp1_dl_dy, create_graph=True, allow_unused=True)
    if grad_w_1st is None: grad_w_1st = torch.zeros_like(current_layer.weight)

    # grad_wdx_2nd is d/dx (dL/dw)
    # vjp2_for_dw is the VJP for dL/dw when taking grad w.r.t x.
    if grad_w_1st is None: # Should have been caught by previous check if affine=True
        print(f"Warning: In so_cross_wdx_original, grad_w_1st (dL/dw) is None. Setting d/dx(dL/dw) to zero.")
        grad_wdx_2nd = torch.zeros_like(x_in)
    elif not grad_w_1st.requires_grad:
        # This implies dL/dw does not depend on x_in in a differentiable manner through the graph.
        # For LayerRMSNorm, this would be unexpected if x_in requires grad.
        print(f"Info: In so_cross_wdx_original, grad_w_1st (dL/dw) does not require grad w.r.t. x_in. "
              f"This implies dL/dw is constant w.r.t x_in for {NormLayerClass.__name__} with affine={True}, training={training_mode_for_batch_norm}, scaled={scaled}. " # More context
              f"Analytically, d/dx(dL/dw) would be zero. Skipping second autograd.grad.")
        print_graph_debug_info(grad_w_1st, "grad_w_1st", f"so_cross_wdx_original ({NormLayerClass.__name__} aff=True sc={scaled} tr={training_mode_for_batch_norm})")
        grad_wdx_2nd = torch.zeros_like(x_in)
    else:
        grad_wdx_2nd_tuple = torch.autograd.grad(grad_w_1st, x_in, grad_outputs=vjp2_for_dw, retain_graph=False, allow_unused=True)
        grad_wdx_2nd = grad_wdx_2nd_tuple[0]
        if grad_wdx_2nd is None:
            grad_wdx_2nd = torch.zeros_like(x_in)
    return [grad_wdx_2nd]

def so_cross_wdx_rotated(NormLayerClass, input_data_raw, wigner_D, irreps_str, C, scaled, training_mode_for_batch_norm, vjp1_dl_dy):
    x_in_orig = input_data_raw.clone().requires_grad_(True) # x_orig we differentiate dL_rot/dw w.r.t
    current_layer = NormLayerClass(Irreps(irreps_str), C, affine=True, scaled=scaled).to(x_in_orig.device)
    if NormLayerClass == BatchRMSNorm:
        current_layer.train(training_mode_for_batch_norm)
    else:
        current_layer.eval()
    current_layer.weight.requires_grad_(True)
    vjp2_for_dw = torch.randn_like(current_layer.weight)
        
    x_transformed = dense_wigner_rotation(x_in_orig, wigner_D)
    out_intermediate = current_layer(x_transformed)
    vjp1_transformed = dense_wigner_rotation(vjp1_dl_dy, wigner_D)
    
    # G_equiv_w is dL_rot/dw_orig = dL_orig/dw_orig
    G_equiv_w_tuple = torch.autograd.grad(out_intermediate, current_layer.weight, grad_outputs=vjp1_transformed, create_graph=True, allow_unused=True)
    G_equiv_w = G_equiv_w_tuple[0]
    
    if G_equiv_w is None:
        print(f"Warning: In so_cross_wdx_rotated, G_equiv_w (dL_rot/dw) is None. Setting d/dx(dL_rot/dw) to zero.")
        grad_wdx_2nd_rot = torch.zeros_like(x_in_orig)
    elif not G_equiv_w.requires_grad:
        # This implies dL_rot/dw does not depend on x_in_orig in a differentiable manner.
        print(f"Info: In so_cross_wdx_rotated, G_equiv_w (dL_rot/dw) does not require grad w.r.t. x_in_orig. "
              f"This implies dL_rot/dw is constant w.r.t x_in_orig for {NormLayerClass.__name__} with affine={True}, training={training_mode_for_batch_norm}, scaled={scaled}. " # More context
              f"Analytically, d/dx_orig(dL_rot/dw) would be zero. Skipping second autograd.grad.")
        print_graph_debug_info(G_equiv_w, "G_equiv_w", f"so_cross_wdx_rotated ({NormLayerClass.__name__} aff=True sc={scaled} tr={training_mode_for_batch_norm})")
        grad_wdx_2nd_rot = torch.zeros_like(x_in_orig)
    else:
        grad_wdx_2nd_rot_tuple = torch.autograd.grad(G_equiv_w, x_in_orig, grad_outputs=vjp2_for_dw, retain_graph=False, allow_unused=True)
        grad_wdx_2nd_rot = grad_wdx_2nd_rot_tuple[0]
        if grad_wdx_2nd_rot is None:
            grad_wdx_2nd_rot = torch.zeros_like(x_in_orig)
    return [grad_wdx_2nd_rot]


# --- Main Test Runner ---
def run_equivariance_tests_for_norm_layer(NormLayerClass, irreps_in_str: str, C: int, N: int, device: str = 'cuda'):
    print(f"\n--- Testing {NormLayerClass.__name__} with Irreps: {irreps_in_str}, C: {C}, N: {N}, Device: {device} ---")
    
    # Convert irreps_in_str to Irreps object for get_norm_test_data
    # The NormLayerClass constructor will handle its own Irreps parsing and simplification.
    irreps_for_data = Irreps(irreps_in_str) 

    # This input_data_raw will not require grad. Cloning and setting requires_grad will happen in test functions.
    input_data_raw, grad_out_data, wigner_D_data, vjp1_data, vjp2_x_data = \
        get_norm_test_data(irreps_for_data, C, N, device=device, dtype=torch.get_default_dtype())

    affine_options = [True, False]
    scaled_options = [True, False]
    
    # Determine training options based on NormLayerClass
    if NormLayerClass == BatchRMSNorm:
        training_options = [True, False]
        training_desc = {True: "Train Mode", False: "Eval Mode"}
    else: # LayerRMSNorm
        training_options = [False] # LayerRMSNorm behavior isn't training-dependent for this test
        training_desc = {False: "N/A (Eval Mode)"}

    for affine in affine_options:
        for scaled in scaled_options:
            for training_mode_bool in training_options:
                mode_name = training_desc[training_mode_bool]
                print(f"\nConfig: affine={affine}, scaled={scaled}, Training={mode_name}")

                # Forward Test
                print("Forward test:")
                tester_fwd = FunctionTester({
                    'ori': (forward_norm_original, [NormLayerClass, input_data_raw, irreps_in_str, C, affine, scaled, training_mode_bool], {}),
                    'rot': (forward_norm_rotated, [NormLayerClass, input_data_raw, wigner_D_data, irreps_in_str, C, affine, scaled, training_mode_bool], {}),
                })
                comp_fwd = tester_fwd.compare(compare_func=diff_func)
                print(comp_fwd)

                # Backward Test (Input Gradient)
                print("Backward test (input grad):")
                tester_bwd_input = FunctionTester({
                    'ori': (backward_input_grad_original, [NormLayerClass, input_data_raw, grad_out_data, irreps_in_str, C, affine, scaled, training_mode_bool], {}),
                    'rot': (backward_input_grad_rotated, [NormLayerClass, input_data_raw, grad_out_data, wigner_D_data, irreps_in_str, C, affine, scaled, training_mode_bool], {}),
                })
                comp_bwd_input = tester_bwd_input.compare(compare_func=diff_func_list)
                print(comp_bwd_input)

                # Second Order Test (Self-Input)
                print("SO test (d/dx (dL/dx)):")
                tester_so_self_x = FunctionTester({
                    'ori': (so_self_x_original, [NormLayerClass, input_data_raw, irreps_in_str, C, affine, scaled, training_mode_bool, vjp1_data, vjp2_x_data], {}),
                    'rot': (so_self_x_rotated, [NormLayerClass, input_data_raw, wigner_D_data, irreps_in_str, C, affine, scaled, training_mode_bool, vjp1_data, vjp2_x_data], {}),
                })
                comp_so_self_x = tester_so_self_x.compare(compare_func=diff_func_list)
                print(comp_so_self_x)

                if affine:
                    print("Backward test (weight grad):")
                    tester_bwd_weight = FunctionTester({
                        'ori': (backward_weight_grad_original, [NormLayerClass, input_data_raw, grad_out_data, irreps_in_str, C, scaled, training_mode_bool], {}),
                        'rot': (backward_weight_grad_rotated, [NormLayerClass, input_data_raw, grad_out_data, wigner_D_data, irreps_in_str, C, scaled, training_mode_bool], {}),
                    })
                    comp_bwd_weight = tester_bwd_weight.compare(compare_func=diff_func_list)
                    print(comp_bwd_weight)
                    
                    print("SO test (d/dw (dL/dw)):")
                    tester_so_self_w = FunctionTester({
                        'ori': (so_self_w_original, [NormLayerClass, input_data_raw, irreps_in_str, C, scaled, training_mode_bool, vjp1_data], {}),
                        'rot': (so_self_w_rotated, [NormLayerClass, input_data_raw, wigner_D_data, irreps_in_str, C, scaled, training_mode_bool, vjp1_data], {}),
                    })
                    comp_so_self_w = tester_so_self_w.compare(compare_func=diff_func_list)
                    print(comp_so_self_w)

                    print("SO test (d/dw (dL/dx)):") # xdw
                    tester_so_cross_xdw = FunctionTester({
                        'ori': (so_cross_xdw_original, [NormLayerClass, input_data_raw, irreps_in_str, C, scaled, training_mode_bool, vjp1_data, vjp2_x_data], {}),
                        'rot': (so_cross_xdw_rotated, [NormLayerClass, input_data_raw, wigner_D_data, irreps_in_str, C, scaled, training_mode_bool, vjp1_data, vjp2_x_data], {}),
                    })
                    comp_so_cross_xdw = tester_so_cross_xdw.compare(compare_func=diff_func_list)
                    print(comp_so_cross_xdw)
                    
                    print("SO test (d/dx (dL/dw)):") # wdx
                    tester_so_cross_wdx = FunctionTester({
                        'ori': (so_cross_wdx_original, [NormLayerClass, input_data_raw, irreps_in_str, C, scaled, training_mode_bool, vjp1_data], {}),
                        'rot': (so_cross_wdx_rotated, [NormLayerClass, input_data_raw, wigner_D_data, irreps_in_str, C, scaled, training_mode_bool, vjp1_data], {}),
                    })
                    comp_so_cross_wdx = tester_so_cross_wdx.compare(compare_func=diff_func_list)
                    print(comp_so_cross_wdx)


if __name__ == '__main__':
    C_test = 16 # Reduced for faster testing, gate test uses 256
    N_test = 8   # Reduced for faster testing, gate test uses 256
    device_test = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # irreps_options = ['0e+1o', '2x0e+1x1o+1x2e', Irreps((0,2))] # Test with string and Irreps object
    irreps_options_str = ['0e+0o+2+1e+1+2o']


    for irreps_str_main in irreps_options_str:
        run_equivariance_tests_for_norm_layer(BatchRMSNorm, irreps_str_main, C_test, N_test, device=device_test)
        run_equivariance_tests_for_norm_layer(LayerRMSNorm, irreps_str_main, C_test, N_test, device=device_test)

    print("\nEquivariance tests for normalization layers completed.")
