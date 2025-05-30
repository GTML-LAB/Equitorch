# %%
import sys


sys.path.append('../..')

import torch
import e3nn

import math

from equitorch.irreps import Irreps, check_irreps

from equitorch.nn.activations import Gate
from equitorch.nn.functional.wigner_d import dense_wigner_rotation
import os

# torch.set_default_dtype(torch.float32)
torch.set_default_dtype(torch.float64)
# 
torch.random.manual_seed(0)


# %%
from test_utils import FunctionTester, rand_rotation_dict, rate_mean2_std, rate_mean2_std_list, max_abs_diff, max_abs_diff_list

diff_func = max_abs_diff
diff_func_list = max_abs_diff_list


def init_gate(irreps_in, C, N, need_grad = False,
              device='cuda', ):
    
    gate = Gate(irreps_in, torch.nn.Tanh()).to(device=device)    

    gates = torch.randn(N, gate.num_gates, C).to(device=device)
    input = torch.randn(N, irreps_in.dim, C).to(device=device)
    
    grad = torch.randn(N, irreps_in.dim, C).to(device=device)

    if need_grad:
        gates.requires_grad_()
        input.requires_grad_()


    rotation_dict = rand_rotation_dict(irreps_in, N, dtype=input.dtype,device=device)

    return gate, input, gates, grad, rotation_dict['wigner_D']

def test_gate(irreps_in, C, N):

    gate, input, gates, grad, wigner_D = init_gate(irreps_in, C, N)

    def forward_original(x, g):
        return gate(x, g)

    def forward_rotated(x, g): 
        return dense_wigner_rotation(gate(dense_wigner_rotation(x, wigner_D), g), wigner_D.transpose(-1,-2))

    tester = FunctionTester({
        'ori': (forward_original, [input, gates], {}),
        'rot': (forward_rotated, [input, gates], {}),
    })
    comp = tester.compare(compare_func=diff_func)
    print(comp)
    
    print('backward')


    gate, input, gates, grad, wigner_D = init_gate(irreps_in, C, N)

    def backward_original(x, g, grad):
        x = x.clone().requires_grad_()
        g = g.clone().requires_grad_()
        res = gate(x, g)
        res.backward(grad)
        return [x.grad, g.grad]
    
    def backward_rotated(x, g, grad):
        x = x.clone().requires_grad_()
        g = g.clone().requires_grad_()
        x_rot = dense_wigner_rotation(x, wigner_D)
        res = gate(x_rot, g)
        res.backward(dense_wigner_rotation(grad, wigner_D))
        return [x.grad, g.grad]

    tester = FunctionTester({
        'ori': (backward_original, [input, gates, grad], {}),
        'rot': (backward_rotated, [input, gates, grad], {}),
    })
    comp = tester.compare(compare_func=diff_func_list)
    print(comp)

    print('second order backward tests')
    # Initialize fresh tensors for second order tests to avoid graph issues
    # gate_so, input_so, gates_so, grad_so_1st_vjp, wigner_D_so = init_gate(irreps_in, C, N, need_grad=True)
    # Using the same variable names as the first order test for clarity inside functions,
    # but they are new tensors from a fresh init_gate call.
    # The `gate` object can be reused, but inputs need to be fresh for requires_grad.
    
    # Common setup for second order VJPs
    # grad_so_1st_vjp is the VJP for the first autograd.grad call (d_out / d_var1)
    # grad_so_2nd_vjp_for_x is the VJP for the second autograd.grad call, when differentiating d_out/d_x again.
    # grad_so_2nd_vjp_for_g is the VJP for the second autograd.grad call, when differentiating d_out/d_g again.

    # Self-grad for x: d/dx (d_out/dx)
    def so_self_x_original(current_gate, x_in, g_in, vjp1, vjp2_x):
        x_in = x_in.clone().requires_grad_()
        g_in = g_in.clone().requires_grad_() 
        
        out = current_gate(x_in, g_in)
        grad_x_1st, = torch.autograd.grad(out, x_in, grad_outputs=vjp1, create_graph=True)
        grad_xx_2nd, = torch.autograd.grad(grad_x_1st, x_in, grad_outputs=vjp2_x, retain_graph=False, allow_unused=True)
        if grad_xx_2nd is None:
            grad_xx_2nd = torch.zeros_like(x_in)
        return [grad_xx_2nd]

    def so_self_x_rotated(current_gate, x_in, g_in, vjp1, vjp2_x, D): # D is wigner_D_so
        x_in_orig = x_in.clone().requires_grad_()
        g_in_orig = g_in.clone().requires_grad_() 
        
        x_transformed = dense_wigner_rotation(x_in_orig, D)
        x_transformed.requires_grad_(True) 
        
        out_intermediate = current_gate(x_transformed, g_in_orig) 
        vjp1_transformed = dense_wigner_rotation(vjp1, D) 
        
        G_equiv_x, _ = torch.autograd.grad(out_intermediate, (x_in_orig, g_in_orig), grad_outputs=vjp1_transformed, create_graph=True, allow_unused=True)
        
        if G_equiv_x is None: 
            # This case implies G_equiv_x (d(gate(Dx,g))/dx) is not dependent on x_in_orig after all VJPs.
            # This can happen if, for example, vjp1 makes the effective output scalar and invariant.
            # For safety, return zeros, though ideally the test setup avoids this for meaningful G_equiv_x.
            # However, G_equiv_x is (d_gate/d_Dx)D. If d_gate/d_Dx is zero, G_equiv_x is zero.
            # If G_equiv_x is truly zero or None, its derivative will also be zero.
             G_equiv_x = torch.zeros_like(x_in_orig) # Ensure it's a tensor for the next grad call

        grad_xx_2nd_rot, = torch.autograd.grad(G_equiv_x, x_in_orig, grad_outputs=vjp2_x, retain_graph=False, allow_unused=True)
        if grad_xx_2nd_rot is None:
            grad_xx_2nd_rot = torch.zeros_like(x_in_orig)
        return [grad_xx_2nd_rot]

    # Self-grad for g: d/dg (d_out/dg)
    def so_self_g_original(current_gate, x_in, g_in, vjp1, vjp2_g):
        x_in = x_in.clone().requires_grad_()
        g_in = g_in.clone().requires_grad_()
        
        out = current_gate(x_in, g_in)
        grad_g_1st, = torch.autograd.grad(out, g_in, grad_outputs=vjp1, create_graph=True)
        grad_gg_2nd, = torch.autograd.grad(grad_g_1st, g_in, grad_outputs=vjp2_g, retain_graph=False, allow_unused=True)
        if grad_gg_2nd is None:
            grad_gg_2nd = torch.zeros_like(g_in)
        return [grad_gg_2nd]

    def so_self_g_rotated(current_gate, x_in, g_in, vjp1, vjp2_g, D):
        x_in_orig = x_in.clone().requires_grad_()
        g_in_orig = g_in.clone().requires_grad_()

        x_transformed = dense_wigner_rotation(x_in_orig, D)
        
        out_intermediate = current_gate(x_transformed, g_in_orig) 
        vjp1_transformed = dense_wigner_rotation(vjp1, D) 
        
        _, G_equiv_g = torch.autograd.grad(out_intermediate, (x_in_orig, g_in_orig), grad_outputs=vjp1_transformed, create_graph=True, allow_unused=True)

        if G_equiv_g is None:
             G_equiv_g = torch.zeros_like(g_in_orig) # Match shape of g_in

        # vjp2_g is for G_equiv_g w.r.t. g_in_orig. g_in_orig is invariant, so vjp2_g is not transformed by D.
        vjp2_g_transformed = vjp2_g 
        grad_gg_2nd_rot, = torch.autograd.grad(G_equiv_g, g_in_orig, grad_outputs=vjp2_g_transformed, retain_graph=False, allow_unused=True)
        if grad_gg_2nd_rot is None:
            grad_gg_2nd_rot = torch.zeros_like(g_in_orig)
        return [grad_gg_2nd_rot]

    # Cross-grad: d/dg (d_out/dx)
    def so_cross_xdg_original(current_gate, x_in, g_in, vjp1, vjp2_for_dx): 
        x_in = x_in.clone().requires_grad_()
        g_in = g_in.clone().requires_grad_()
        
        out = current_gate(x_in, g_in)
        grad_x_1st, = torch.autograd.grad(out, x_in, grad_outputs=vjp1, create_graph=True) 
        grad_xdg_2nd, = torch.autograd.grad(grad_x_1st, g_in, grad_outputs=vjp2_for_dx, retain_graph=False, allow_unused=True) 
        if grad_xdg_2nd is None:
            grad_xdg_2nd = torch.zeros_like(g_in) # Gradient w.r.t g_in
        return [grad_xdg_2nd]

    def so_cross_xdg_rotated(current_gate, x_in, g_in, vjp1, vjp2_for_dx, D):
        x_in_orig = x_in.clone().requires_grad_()
        g_in_orig = g_in.clone().requires_grad_()

        x_transformed = dense_wigner_rotation(x_in_orig, D)
        x_transformed.requires_grad_(True)
        
        out_intermediate = current_gate(x_transformed, g_in_orig) 
        vjp1_transformed = dense_wigner_rotation(vjp1, D) 
        
        G_equiv_x, _ = torch.autograd.grad(out_intermediate, (x_in_orig, g_in_orig), grad_outputs=vjp1_transformed, create_graph=True, allow_unused=True)
        
        if G_equiv_x is None:
            G_equiv_x = torch.zeros_like(x_in_orig)

        grad_xdg_2nd_rot, = torch.autograd.grad(G_equiv_x, g_in_orig, grad_outputs=vjp2_for_dx, retain_graph=False, allow_unused=True)
        if grad_xdg_2nd_rot is None:
            grad_xdg_2nd_rot = torch.zeros_like(g_in_orig) # Gradient w.r.t g_in_orig
        return [grad_xdg_2nd_rot]

    # Cross-grad: d/dx (d_out/dg)
    def so_cross_gdx_original(current_gate, x_in, g_in, vjp1, vjp2_for_dg): 
        x_in = x_in.clone().requires_grad_()
        g_in = g_in.clone().requires_grad_()
        
        out = current_gate(x_in, g_in)
        grad_g_1st, = torch.autograd.grad(out, g_in, grad_outputs=vjp1, create_graph=True) 
        grad_gdx_2nd, = torch.autograd.grad(grad_g_1st, x_in, grad_outputs=vjp2_for_dg, retain_graph=False, allow_unused=True) 
        if grad_gdx_2nd is None:
            grad_gdx_2nd = torch.zeros_like(x_in) # Gradient w.r.t x_in
        return [grad_gdx_2nd]

    def so_cross_gdx_rotated(current_gate, x_in, g_in, vjp1, vjp2_for_dg, D):
        x_in_orig = x_in.clone().requires_grad_()
        g_in_orig = g_in.clone().requires_grad_()

        x_transformed = dense_wigner_rotation(x_in_orig, D)
        
        out_intermediate = current_gate(x_transformed, g_in_orig) 
        vjp1_transformed = dense_wigner_rotation(vjp1, D) 
        
        _, G_equiv_g = torch.autograd.grad(out_intermediate, (x_in_orig, g_in_orig), grad_outputs=vjp1_transformed, create_graph=True, allow_unused=True)

        if G_equiv_g is None:
            G_equiv_g = torch.zeros_like(g_in_orig) # This grad is d_out/dg like, so shape of g_in if g_in was output, or vjp1 if vjp1 is output-like
                                                 # More robust: shape of vjp1 if G_equiv_g is output-like
            G_equiv_g = torch.zeros_like(vjp1)


        # vjp2_for_dg is for G_equiv_g (d_out/dg like). G_equiv_g is g-like in its indices.
        # So vjp2_for_dg should not be transformed by D (which acts on x-like indices).
        vjp2_for_dg_transformed = vjp2_for_dg
        grad_gdx_2nd_rot, = torch.autograd.grad(G_equiv_g, x_in_orig, grad_outputs=vjp2_for_dg_transformed, retain_graph=False, allow_unused=True)
        if grad_gdx_2nd_rot is None:
            grad_gdx_2nd_rot = torch.zeros_like(x_in_orig) # Gradient w.r.t x_in_orig
        return [grad_gdx_2nd_rot]

    # Re-initialize tensors for second-order tests
    gate_so, input_so, gates_so, grad_so_1, wigner_D_so = init_gate(irreps_in, C, N, need_grad=True)
    
    # VJPs for the second differentiation step
    grad_so_2_x = torch.randn_like(input_so) 
    grad_so_2_g = torch.randn_like(gates_so)
    # For cross terms, d/dg(d_out/dx), the VJP for d_out/dx is grad_so_2_x.
    # For cross terms, d/dx(d_out/dg), the VJP for d_out/dg (which is grad_g_1st) should match grad_g_1st, i.e. be g-like.
    # We use grad_so_2_g as it's already randn_like(gates_so).

    print("Second order backward (self-x: d/dx (d_out/dx))")
    tester_so_self_x = FunctionTester({
        'ori': (so_self_x_original, [gate_so, input_so, gates_so, grad_so_1, grad_so_2_x], {}),
        'rot': (so_self_x_rotated, [gate_so, input_so, gates_so, grad_so_1, grad_so_2_x, wigner_D_so], {}),
    })
    comp_so_self_x = tester_so_self_x.compare(compare_func=diff_func_list)
    print(comp_so_self_x)

    print("Second order backward (self-g: d/dg (d_out/dg))")
    tester_so_self_g = FunctionTester({
        'ori': (so_self_g_original, [gate_so, input_so, gates_so, grad_so_1, grad_so_2_g], {}),
        'rot': (so_self_g_rotated, [gate_so, input_so, gates_so, grad_so_1, grad_so_2_g, wigner_D_so], {}),
    })
    comp_so_self_g = tester_so_self_g.compare(compare_func=diff_func_list)
    print(comp_so_self_g)
    
    print("Second order backward (cross xdg: d/dg (d_out/dx))")
    tester_so_cross_xdg = FunctionTester({
        'ori': (so_cross_xdg_original, [gate_so, input_so, gates_so, grad_so_1, grad_so_2_x], {}), 
        'rot': (so_cross_xdg_rotated, [gate_so, input_so, gates_so, grad_so_1, grad_so_2_x, wigner_D_so], {}),
    })
    comp_so_cross_xdg = tester_so_cross_xdg.compare(compare_func=diff_func_list)
    print(comp_so_cross_xdg)

    print("Second order backward (cross gdx: d/dx (d_out/dg))")
    # vjp2_for_dg (the VJP for grad_g_1st) should be g-like. grad_so_2_g is g-like.
    tester_so_cross_gdx = FunctionTester({
        'ori': (so_cross_gdx_original, [gate_so, input_so, gates_so, grad_so_1, grad_so_2_g], {}), 
        'rot': (so_cross_gdx_rotated, [gate_so, input_so, gates_so, grad_so_1, grad_so_2_g, wigner_D_so], {}),
    })
    comp_so_cross_gdx = tester_so_cross_gdx.compare(compare_func=diff_func_list)
    print(comp_so_cross_gdx)

    return gate, input, gates, grad, wigner_D # Return the original gate and tensors from the first init_gate for any downstream notebook cells



# %%


from test_utils import FunctionTester, max_abs_diff

C = 256

# irreps_in = Irreps('0e+0o+1e+1o+2e+2o+3e+3o+4e+4o')
irreps_in = Irreps((0,5))
# irreps_in = Irreps('4x0e+2x1e+2e+3x3e+2x4e')
# irreps_in = Irreps('0')
print(irreps_in)

N =256
# N =2


# %%
gate, input, gates, grad, wigner_D = test_gate(irreps_in, C, N)
