import sys
sys.path.append('../..')
sys.path.append('../../..')
sys.path.append('..')
from test_utils import FunctionTester, max_abs_diff, max_abs_diff_list, rate_mean2_std_list, rate_mean2_std

import torch
import e3nn
import math
import os

from equitorch.irreps import Irreps, check_irreps
from equitorch.nn.tensor_products import TensorProduct
from equitorch.nn.linears import  SO3Linear
# Import e3nn comparison classes from the existing test file

# Set environment and defaults
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.set_default_dtype(torch.float32)
# torch.set_default_dtype(torch.float64) # Use float32 for consistency
torch.random.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DepthWiseTensorProduct(e3nn.o3.TensorProduct):

    def __init__(
        self, irreps_in1, irreps_in2, irreps_out, irrep_normalization: str = None, path_normalization: str = None, **kwargs
    ) -> None:
        irreps_in1 = e3nn.o3.Irreps(irreps_in1)
        irreps_in2 = e3nn.o3.Irreps(irreps_in2)
        irreps_out = e3nn.o3.Irreps(irreps_out)

        instr = [
            (i_1, i_2, i_out, "uuu", True, 1.0)
            for i_1, (_, ir_1) in enumerate(irreps_in1)
            for i_2, (_, ir_2) in enumerate(irreps_in2)
            for i_out, (_, ir_out) in enumerate(irreps_out)
            if ir_out in ir_1 * ir_2
        ]
        super().__init__(
            irreps_in1,
            irreps_in2,
            irreps_out,
            instr,
            irrep_normalization=irrep_normalization,
            path_normalization=path_normalization,
            **kwargs,
        )



class DepthWiseSO3Linear(e3nn.o3.TensorProduct):

    def __init__(
        self, irreps_in1, irreps_in2, irreps_out, irrep_normalization: str = None, path_normalization: str = None, **kwargs
    ) -> None:
        irreps_in1 = e3nn.o3.Irreps(irreps_in1)
        irreps_in2 = e3nn.o3.Irreps(irreps_in2)
        irreps_out = e3nn.o3.Irreps(irreps_out)

        instr = [
            (i_1, i_2, i_out, "uvu", True, 1.0)
            for i_1, (_, ir_1) in enumerate(irreps_in1)
            for i_2, (_, ir_2) in enumerate(irreps_in2)
            for i_out, (_, ir_out) in enumerate(irreps_out)
            if ir_out in ir_1 * ir_2
        ]
        super().__init__(
            irreps_in1,
            irreps_in2,
            irreps_out,
            instr,
            irrep_normalization=irrep_normalization,
            path_normalization=path_normalization,
            **kwargs,
        )


# === Tensor Format Conversion Functions (Copied from test_linear_e3nn.py) ===

def e3nn_features_to_equitorch(input: torch.Tensor, irreps, channels=None):
    r"""Converts e3nn feature tensor format to equitorch format."""
    e3nn_irreps = e3nn.o3.Irreps(irreps)
    equitorch_irreps = Irreps(str(irreps))
    if channels is None:
        # Infer channels if not provided (assumes uniform multiplicity)
        if len(e3nn_irreps) > 0:
             channels = e3nn_irreps[0].mul
        else:
             channels = 1 # Default if irreps is empty
        # Fallback if inference fails or irreps are mixed
        if not all(ir.mul == channels for ir in e3nn_irreps):
             print("Warning: Could not reliably infer channels for e3nn_features_to_equitorch. Assuming max_channels.")
             channels = equitorch_irreps.max_channels()


    equitorch_irreps_divided = equitorch_irreps.channels_divided(channels)

    # Ensure split sizes match the input dimension
    expected_dim = sum(channels * irrep.dim for irrep in equitorch_irreps_divided)
    if input.shape[-1] != expected_dim:
        raise ValueError(f"Input tensor last dimension {input.shape[-1]} does not match expected dimension {expected_dim} for irreps {irreps} and channels {channels}")

    split_size = [channels * irrep.dim for irrep in equitorch_irreps_divided]

    features = input.split(split_size, dim=-1)
    # Add check for empty features list
    if not features or not all(t.numel() > 0 for t in features):
         # Handle cases where split results in empty tensors (e.g., zero dim input)
         # Return tensor with correct shape but potentially zero size in irreps dim
         return torch.zeros(*input.shape[:-1], equitorch_irreps.dim, channels, device=input.device, dtype=input.dtype)

    features = [t.unflatten(-1, (channels, -1)) for t in features]
    features = torch.cat(features, dim=-1)
    return features.transpose(-1, -2)


def equitorch_features_to_e3nn(input: torch.Tensor, irreps):
    r"""Converts equitorch feature tensor format to e3nn format."""
    if input.numel() == 0:
         # Handle empty input tensor
         equitorch_irreps = check_irreps(irreps)
         return torch.zeros(*input.shape[:-2], equitorch_irreps.dim * input.shape[-1], device=input.device, dtype=input.dtype)

    channels = input.shape[-1]
    equitorch_irreps = check_irreps(irreps)

    # Handle case where irreps might be empty
    if not equitorch_irreps:
        return torch.empty(*input.shape[:-2], 0, device=input.device, dtype=input.dtype)


    equitorch_irreps_multiplied = equitorch_irreps.channels_multiplied(channels)
    # e3nn_irreps = e3nn.o3.Irreps(equitorch_irreps_multiplied.short_repr()) # Not needed for conversion logic

    features = input.transpose(-1, -2)
    # Ensure split sizes are positive
    split_dims = [dim for dim in equitorch_irreps.dims() if dim > 0]
    if not split_dims or sum(split_dims) != features.shape[-1]:
         # This can happen if irreps contains only 0-dim irreps or doesn't match tensor
         if all(d == 0 for d in equitorch_irreps.dims()):
              return torch.empty(*features.shape[:-1], 0, device=features.device, dtype=features.dtype)
         else:
              raise ValueError(f"Tensor dimension {features.shape[-1]} doesn't match sum of irrep dimensions {sum(equitorch_irreps.dims())}")

    features_split = features.split(split_dims, -1)
    features_flattened = [t.flatten(-2, -1) for t in features_split]
    features_cat = torch.cat(features_flattened, dim=-1)

    return features_cat


# === Irrep Index Mapping (Copied from test_linear_e3nn.py) ===

def e3nn_irrep_idx_to_eqt(irreps_e3nn, channels_eqt):
    r"""Maps e3nn irrep index to corresponding equitorch irrep index."""
    acc_irrep_idx_eqt = 0
    irrep_idx_eqt_list = []
    for i, (mul, irrep) in enumerate(irreps_e3nn):
        irrep_idx_eqt_list_in = []
        # Avoid division by zero if channels_eqt is 0
        if channels_eqt > 0:
            for mul_eqt in range(mul // channels_eqt):
                irrep_idx_eqt_list_in.append((mul_eqt, acc_irrep_idx_eqt))
                acc_irrep_idx_eqt += 1
        irrep_idx_eqt_list.append(irrep_idx_eqt_list_in)
    return irrep_idx_eqt_list

# === Weight Conversion Functions (Adapted for Tensor Products) ===

def get_tp_fan_in(tp_eqt):
    r"""Helper to get fan_in based on equitorch layer type and mode."""
    if isinstance(tp_eqt, TensorProduct):
        if tp_eqt.feature_mode == 'uvw':
            return tp_eqt.channels_in1 * tp_eqt.channels_in2
        elif tp_eqt.feature_mode == 'uuu':
            # Fan in for uuu is just channels_in according to e3nn logic? Let's assume C_in * C_in
            # E3nn path_weight uses fan_in = mul_ir1 * mul_ir2 which translates to C*C here
            #  return tp_eqt.channels_in1 * tp_eqt.channels_in2 # C*C
             return 1
        else:
            raise ValueError(f"Unknown TensorProduct feature mode: {tp_eqt.feature_mode}")
    elif isinstance(tp_eqt, SO3Linear):
        if tp_eqt.feature_mode == 'uv':
            return tp_eqt.channels_in # Fan_in is C_in for Linear/SO3Linear
        elif tp_eqt.feature_mode == 'uu':
             # Fan in for uu is just channels_in? E3nn path_weight uses fan_in = mul_ir1 * mul_ir2 = C*1 = C
            #  return tp_eqt.channels_in # Fan_in is C_in
             return 1 # Fan_in is C_in
        else:
            raise ValueError(f"Unknown SO3Linear feature mode: {tp_eqt.feature_mode}")
    else:
        raise TypeError(f"Unsupported layer type for fan_in calculation: {type(tp_eqt)}")

def e3nn_tp_weights_to_eqt(tp_e3nn, tp_eqt, weight_e3nn):
    r"""Converts e3nn TensorProduct weights to equitorch format."""
    weights_info_eqt = {}
    is_so3linear = isinstance(tp_eqt, SO3Linear)
    is_uvw = isinstance(tp_eqt, TensorProduct) and tp_eqt.feature_mode == 'uvw'
    is_uuu = isinstance(tp_eqt, TensorProduct) and tp_eqt.feature_mode == 'uuu'
    is_uv = isinstance(tp_eqt, SO3Linear) and tp_eqt.feature_mode == 'uv'
    is_uu = isinstance(tp_eqt, SO3Linear) and tp_eqt.feature_mode == 'uu'

    # Get channel info
    C1 = tp_eqt.channels_in1 if hasattr(tp_eqt, 'channels_in1') else tp_eqt.channels_in
    C2 = tp_eqt.channels_in2 if hasattr(tp_eqt, 'channels_in2') else 1 # 1 for SO3Linear
    C_out = tp_eqt.channels_out

    # Get irrep index mappings
    to_eqt1 = e3nn_irrep_idx_to_eqt(tp_e3nn.irreps_in1, C1)
    # Handle C2=1 case for SO3Linear
    to_eqt2 = e3nn_irrep_idx_to_eqt(tp_e3nn.irreps_in2, C2 if not is_so3linear else 1)
    to_eqt_out = e3nn_irrep_idx_to_eqt(tp_e3nn.irreps_out, C_out)
    irrep_out = tp_e3nn.irreps_out
    fan_in = get_tp_fan_in(tp_eqt)

    weight_views = list(tp_e3nn.weight_views(weight_e3nn))

    for ins_idx, ins in enumerate(tp_e3nn.instructions):
        weight = weight_views[ins_idx]
        # path_count = 1#ins.path_shape[0]
        # path_weight_e3nn = ( fan_in) ** (-0.5) if fan_in > 0 else 1.0 # Default e3nn path weight

        if tp_eqt.path_norm:
            # eqt_weight = e3nn_weight * fan_in**(-0.5) / irrep_out[ins.i_out][1].dim
            # scale =(fan_in * irrep_out[ins.i_out][1].dim)**(-0.5) 
            scale =(fan_in)**(-0.5) 
        else:
            # eqt_weight = e3nn_weight * path_weight_e3nn / irrep_out[ins.i_out][1].dim
            # scale = ins.path_weight * irrep_out[ins.i_out][1].dim ** (-0.5) 
            scale = ins.path_weight  / irrep_out[ins.i_out][1].dim ** (0.5) 

        # Extract indices based on mode
        if is_uvw: # uvw: (out, in1, in2)
            for e3nn_i_out_idx, i_out_eqt in to_eqt_out[ins.i_out]:
                for e3nn_i_in1_idx, i_in1_eqt in to_eqt1[ins.i_in1]:
                    for e3nn_i_in2_idx, i_in2_eqt in to_eqt2[ins.i_in2]:
                        weights_info_eqt[i_out_eqt, i_in1_eqt, i_in2_eqt] = scale * weight[
                            ...,
                            e3nn_i_in1_idx*C1:(e3nn_i_in1_idx+1)*C1,
                            e3nn_i_in2_idx*C2:(e3nn_i_in2_idx+1)*C2,
                            e3nn_i_out_idx*C_out:(e3nn_i_out_idx+1)*C_out
                        ]
        elif is_uuu: # uuu: (out, ) - weight shape is (..., C_out)
             # Ensure indices align correctly for depthwise
             if len(to_eqt_out[ins.i_out]) == len(to_eqt1[ins.i_in1]) == len(to_eqt2[ins.i_in2]):
                 for (e3nn_i_out_idx, i_out_eqt), (_, i_in1_eqt), (_, i_in2_eqt) in zip(to_eqt_out[ins.i_out], to_eqt1[ins.i_in1], to_eqt2[ins.i_in2]):
                     weights_info_eqt[i_out_eqt, i_in1_eqt, i_in2_eqt] = scale * weight[
                         ..., e3nn_i_out_idx*C_out:(e3nn_i_out_idx+1)*C_out
                     ]
             else:
                  print(f"Warning: Index alignment mismatch for UUU mode instruction {ins}. Skipping weight conversion for this path.")

        elif is_uv: # uv: (out, in1) - weight shape is (..., C_in, 1, C_out)
            i_in2_eqt = to_eqt2[ins.i_in2][0][1] # Only one index for irrep2
            for e3nn_i_out_idx, i_out_eqt in to_eqt_out[ins.i_out]:
                for e3nn_i_in1_idx, i_in1_eqt in to_eqt1[ins.i_in1]:
                     weights_info_eqt[i_out_eqt, i_in1_eqt, i_in2_eqt] = scale * weight[
                         ...,
                         e3nn_i_in1_idx*C1:(e3nn_i_in1_idx+1)*C1,
                         0, # Index for irrep2 is always 0
                         e3nn_i_out_idx*C_out:(e3nn_i_out_idx+1)*C_out
                     ]
        elif is_uu: # uu: (out, ) - weight shape is (..., C_out, 1)
             # Ensure indices align correctly for depthwise
             if len(to_eqt_out[ins.i_out]) == len(to_eqt1[ins.i_in1]) == len(to_eqt2[ins.i_in2]):
                 i_in2_eqt = to_eqt2[ins.i_in2][0][1] # Only one index for irrep2
                 for (e3nn_i_out_idx, i_out_eqt), (_, i_in1_eqt) in zip(to_eqt_out[ins.i_out], to_eqt1[ins.i_in1]):
                     weights_info_eqt[i_out_eqt, i_in1_eqt, i_in2_eqt] = scale * weight[
                         ..., e3nn_i_out_idx*C_out:(e3nn_i_out_idx+1)*C_out, 0
                     ]
             else:
                  print(f"Warning: Index alignment mismatch for UU mode instruction {ins}. Skipping weight conversion for this path.")


    if not weights_info_eqt: # Handle case where no paths were generated
        # Determine expected shape based on tp_eqt.weight_shape
        return torch.zeros(tp_eqt.weight_shape, device=weight_e3nn.device, dtype=weight_e3nn.dtype)


    kij_sorted = sorted(weights_info_eqt.keys())
    # Stack along the path dimension (-4 for uvw, -3 for uv, -2 for uuu/uu)
    stack_dim = -4 if is_uvw else (-3 if is_uv else -2)
    return torch.stack([weights_info_eqt[key] for key in kij_sorted], dim=stack_dim)


def e3nn_tp_weights_grad_to_eqt(tp_e3nn, tp_eqt, weight_grad_e3nn):
    r"""Converts e3nn TensorProduct weight gradients to equitorch format."""
    # Gradient scaling is the inverse of the forward weight scaling
    weights_info_eqt = {}
    is_so3linear = isinstance(tp_eqt, SO3Linear)
    is_uvw = isinstance(tp_eqt, TensorProduct) and tp_eqt.feature_mode == 'uvw'
    is_uuu = isinstance(tp_eqt, TensorProduct) and tp_eqt.feature_mode == 'uuu'
    is_uv = isinstance(tp_eqt, SO3Linear) and tp_eqt.feature_mode == 'uv'
    is_uu = isinstance(tp_eqt, SO3Linear) and tp_eqt.feature_mode == 'uu'

    C1 = tp_eqt.channels_in1 if hasattr(tp_eqt, 'channels_in1') else tp_eqt.channels_in
    C2 = tp_eqt.channels_in2 if hasattr(tp_eqt, 'channels_in2') else 1
    C_out = tp_eqt.channels_out

    to_eqt1 = e3nn_irrep_idx_to_eqt(tp_e3nn.irreps_in1, C1)
    to_eqt2 = e3nn_irrep_idx_to_eqt(tp_e3nn.irreps_in2, C2 if not is_so3linear else 1)
    to_eqt_out = e3nn_irrep_idx_to_eqt(tp_e3nn.irreps_out, C_out)

    fan_in = get_tp_fan_in(tp_eqt)
    weight_grad_views = list(tp_e3nn.weight_views(weight_grad_e3nn))
    irrep_out = tp_e3nn.irreps_out
    for ins_idx, ins in enumerate(tp_e3nn.instructions):
        weight_grad = weight_grad_views[ins_idx]
        path_count = ins.path_shape[0]
        # path_weight_e3nn = (path_count * fan_in) ** (-0.5) 

        if tp_eqt.path_norm:
            # eqt_grad = e3nn_grad * irrep_out[ins.i_out][1].dim / fan_in**(-0.5)
            # inv_scale = irrep_out[ins.i_out][1].dim / (fan_in**(-0.5)) if fan_in > 0 else irrep_out[ins.i_out][1].dim
            inv_scale = fan_in**(-0.5)
        else:
            # eqt_grad = e3nn_grad * irrep_out[ins.i_out][1].dim / path_weight_e3nn
            # inv_scale = irrep_out[ins.i_out][1].dim / path_weight_e3nn if path_weight_e3nn != 0 else irrep_out[ins.i_out][1].dim
            inv_scale = ins.path_weight / irrep_out[ins.i_out][1].dim ** (0.5)
 
        # Extract indices based on mode (similar to forward)
        if is_uvw:
            for e3nn_i_out_idx, i_out_eqt in to_eqt_out[ins.i_out]:
                for e3nn_i_in1_idx, i_in1_eqt in to_eqt1[ins.i_in1]:
                    for e3nn_i_in2_idx, i_in2_eqt in to_eqt2[ins.i_in2]:
                        weights_info_eqt[i_out_eqt, i_in1_eqt, i_in2_eqt] = weight_grad[
                            ...,
                            e3nn_i_in1_idx*C1:(e3nn_i_in1_idx+1)*C1,
                            e3nn_i_in2_idx*C2:(e3nn_i_in2_idx+1)*C2,
                            e3nn_i_out_idx*C_out:(e3nn_i_out_idx+1)*C_out
                        ] / inv_scale
        elif is_uuu:
             if len(to_eqt_out[ins.i_out]) == len(to_eqt1[ins.i_in1]) == len(to_eqt2[ins.i_in2]):
                 for (e3nn_i_out_idx, i_out_eqt), (_, i_in1_eqt), (_, i_in2_eqt) in zip(to_eqt_out[ins.i_out], to_eqt1[ins.i_in1], to_eqt2[ins.i_in2]):
                     weights_info_eqt[i_out_eqt, i_in1_eqt, i_in2_eqt] = weight_grad[
                         ..., e3nn_i_out_idx*C_out:(e3nn_i_out_idx+1)*C_out
                     ] / inv_scale
             else:
                  print(f"Warning: Index alignment mismatch for UUU mode instruction {ins}. Skipping weight grad conversion for this path.")

        elif is_uv:
            i_in2_eqt = to_eqt2[ins.i_in2][0][1]
            for e3nn_i_out_idx, i_out_eqt in to_eqt_out[ins.i_out]:
                for e3nn_i_in1_idx, i_in1_eqt in to_eqt1[ins.i_in1]:
                     weights_info_eqt[i_out_eqt, i_in1_eqt, i_in2_eqt] = weight_grad[
                         ...,
                         e3nn_i_in1_idx*C1:(e3nn_i_in1_idx+1)*C1,
                         0,
                         e3nn_i_out_idx*C_out:(e3nn_i_out_idx+1)*C_out
                     ] / inv_scale
        elif is_uu:
             if len(to_eqt_out[ins.i_out]) == len(to_eqt1[ins.i_in1]) == len(to_eqt2[ins.i_in2]):
                 i_in2_eqt = to_eqt2[ins.i_in2][0][1]
                 for (e3nn_i_out_idx, i_out_eqt), (_, i_in1_eqt) in zip(to_eqt_out[ins.i_out], to_eqt1[ins.i_in1]):
                     weights_info_eqt[i_out_eqt, i_in1_eqt, i_in2_eqt] = weight_grad[
                         ..., e3nn_i_out_idx*C_out:(e3nn_i_out_idx+1)*C_out, 0
                     ] / inv_scale
             else:
                  print(f"Warning: Index alignment mismatch for UU mode instruction {ins}. Skipping weight grad conversion for this path.")


    if not weights_info_eqt:
         # Determine expected shape based on tp_eqt.weight_shape
         return torch.zeros(tp_eqt.weight_shape, device=weight_grad_e3nn.device, dtype=weight_grad_e3nn.dtype)


    kij_sorted = sorted(weights_info_eqt.keys())
    stack_dim = -4 if is_uvw else (-3 if is_uv else -2)
    return torch.stack([weights_info_eqt[key] for key in kij_sorted], dim=stack_dim)


# === Initialization Functions (Adapted for Tensor Products) ===

def _init_common(
    eqt_cls, e3nn_cls, feature_mode, path_norm,
    irreps1_eqt, irreps2_eqt, irreps_out_eqt,
    irreps1_e3nn, irreps2_e3nn, irreps_out_e3nn,
    C1, C2, C_out, N, shared, need_grad, device
):
    r"""Common initialization logic."""
    is_so3linear = eqt_cls == SO3Linear
    eqt_kwargs = {
        'irreps_in1': irreps1_eqt,
        'irreps_in2': irreps2_eqt,
        'irreps_out': irreps_out_eqt,
        'feature_mode': feature_mode,
        'path_norm': path_norm,
        'internal_weights': False,
    }
    if is_so3linear:
        eqt_kwargs.update({'channels_in': C1, 'channels_out': C_out})
    else:
        eqt_kwargs.update({'channels_in1': C1, 'channels_in2': C2, 'channels_out': C_out})

    tp_eqt = eqt_cls(**eqt_kwargs).to(device)
    tp_e3nn = e3nn_cls(irreps1_e3nn, irreps2_e3nn, irreps_out_e3nn, shared_weights=shared).to(device)

    # Create inputs
    x_eqt = torch.randn(N, irreps1_eqt.dim, C1, device=device)
    if is_so3linear: # y has no channel dim for SO3Linear
        y_eqt = torch.randn(N, irreps2_eqt.dim, device=device)
    else:
        y_eqt = torch.randn(N, irreps2_eqt.dim, C2, device=device)

    x_e3nn = equitorch_features_to_e3nn(x_eqt, irreps1_eqt)
    if is_so3linear:
        y_e3nn = y_eqt.clone() # No channel conversion needed
    else:
        y_e3nn = equitorch_features_to_e3nn(y_eqt, irreps2_eqt)

    # Create weights
    if shared:
        W_e3nn = torch.randn(tp_e3nn.weight_numel, device=device)
    else:
        W_e3nn = torch.randn(N, tp_e3nn.weight_numel, device=device)

    # Add check for zero weights before conversion
    if tp_e3nn.weight_numel == 0:
         W_eqt = torch.zeros(tp_eqt.weight_shape, device=device, dtype=W_e3nn.dtype)
    else:
         W_eqt = e3nn_tp_weights_to_eqt(tp_e3nn, tp_eqt, W_e3nn)


    # Create output gradients
    grad_eqt = torch.randn(N, irreps_out_eqt.dim, C_out, device=device)
    grad_e3nn = equitorch_features_to_e3nn(grad_eqt, irreps_out_eqt)

    if need_grad:
        x_eqt.requires_grad_(True)
        y_eqt.requires_grad_(True)
        W_eqt.requires_grad_(True)
        x_e3nn.requires_grad_(True)
        y_e3nn.requires_grad_(True)
        W_e3nn.requires_grad_(True)

    # JIT compilation can sometimes help, but remove if causing issues
    # try:
    #     tp_eqt = torch.jit.script(tp_eqt)
    #     tp_e3nn = torch.jit.script(tp_e3nn)
    # except Exception as e:
    #     print(f"JIT compilation failed: {e}")
    #     # Proceed without JIT

    return {
        'eqt': [tp_eqt, x_eqt, y_eqt, W_eqt, grad_eqt],
        'e3nn': [tp_e3nn, x_e3nn, y_e3nn, W_e3nn, grad_e3nn]
    }


def init_tp_uvw(irreps1_eqt, irreps2_eqt, irreps_out_eqt, irreps1_e3nn, irreps2_e3nn, irreps_out_e3nn,
                C1, C2, C_out, N, shared, path_norm, need_grad, device):
    return _init_common(
        TensorProduct, e3nn.o3.FullyConnectedTensorProduct, 'uvw', path_norm,
        irreps1_eqt, irreps2_eqt, irreps_out_eqt, irreps1_e3nn, irreps2_e3nn, irreps_out_e3nn,
        C1, C2, C_out, N, shared, need_grad, device
    )

def init_tp_uuu(irreps1_eqt, irreps2_eqt, irreps_out_eqt, irreps1_e3nn, irreps2_e3nn, irreps_out_e3nn,
                C, N, shared, path_norm, need_grad, device):
    return _init_common(
        TensorProduct, DepthWiseTensorProduct, 'uuu', path_norm,
        irreps1_eqt, irreps2_eqt, irreps_out_eqt, irreps1_e3nn, irreps2_e3nn, irreps_out_e3nn,
        C, C, C, N, shared, need_grad, device # C1=C, C2=C, C_out=C
    )

def init_so3_uv(irreps1_eqt, irreps2_eqt, irreps_out_eqt, irreps1_e3nn, irreps2_e3nn, irreps_out_e3nn,
                C_in, C_out, N, shared, path_norm, need_grad, device):
    return _init_common(
        SO3Linear, e3nn.o3.FullyConnectedTensorProduct, 'uv', path_norm,
        irreps1_eqt, irreps2_eqt, irreps_out_eqt, irreps1_e3nn, irreps2_e3nn, irreps_out_e3nn,
        C_in, 1, C_out, N, shared, need_grad, device # C2=1
    )

def init_so3_uu(irreps1_eqt, irreps2_eqt, irreps_out_eqt, irreps1_e3nn, irreps2_e3nn, irreps_out_e3nn,
                C, N, shared, path_norm, need_grad, device):
    return _init_common(
        SO3Linear, DepthWiseSO3Linear, 'uu', path_norm,
        irreps1_eqt, irreps2_eqt, irreps_out_eqt, irreps1_e3nn, irreps2_e3nn, irreps_out_e3nn,
        C, 1, C, N, shared, need_grad, device # C1=C, C2=1, C_out=C
    )


# === Test Functions (Adapted for Tensor Products) ===

def _test_common(
    init_func, test_name,
    irreps1_eqt, irreps2_eqt, irreps_out_eqt,
    irreps1_e3nn, irreps2_e3nn, irreps_out_e3nn,
    *args # Catches C1, C2, C_out or C_in, C_out or C
):
    N, shared, path_norm, device = args[-4:]
    channel_args = args[:-4] # Extract channel args

    print(f'\n--- Testing {test_name} (path_norm={path_norm}, shared={shared}) ---')

    # Forward Test
    print('Forward Pass:')
    inputs_fwd = init_func(
        irreps1_eqt, irreps2_eqt, irreps_out_eqt, irreps1_e3nn, irreps2_e3nn, irreps_out_e3nn,
        *channel_args, N, shared, path_norm, need_grad=False, device=device
    )

    tp_eqt_fwd, x_eqt_fwd, y_eqt_fwd, W_eqt_fwd, _ = inputs_fwd['eqt']
    tp_e3nn_fwd, x_e3nn_fwd, y_e3nn_fwd, W_e3nn_fwd, _ = inputs_fwd['e3nn']

    # Handle potential empty irreps_out
    C_out = channel_args[-1] if len(channel_args) > 0 else (channel_args[0] if len(channel_args) == 1 else 1) # Infer C_out


    tester_fwd = FunctionTester({
        'eqt': (tp_eqt_fwd, [x_eqt_fwd, y_eqt_fwd, W_eqt_fwd], {}),
        'e3nn': (tp_e3nn_fwd, [x_e3nn_fwd, y_e3nn_fwd, W_e3nn_fwd], {}),
    })

    try:
        comp_fwd = tester_fwd.compare(post_transforms=[
            lambda x: x,
            lambda x: e3nn_features_to_equitorch(x, irreps_out_e3nn, channels=C_out) if x.numel() > 0 else torch.zeros_like(inputs_fwd['eqt'][0](x_eqt_fwd, y_eqt_fwd, W_eqt_fwd)) # Handle empty output
        ], compare_func=max_abs_diff)
        print(f"Forward Max Abs Diff: {comp_fwd}")
        tester_fwd.profile(repeat=5)
    except Exception as e:
        print(f"Forward test failed: {e}")
        import traceback
        traceback.print_exc()


    # Backward Test
    print('\nBackward Pass:')
    inputs_bwd = init_func(
        irreps1_eqt, irreps2_eqt, irreps_out_eqt, irreps1_e3nn, irreps2_e3nn, irreps_out_e3nn,
        *channel_args, N, shared, path_norm, need_grad=True, device=device
    )

    tp_eqt_bwd, x_eqt_bwd, y_eqt_bwd, W_eqt_bwd, grad_eqt_bwd = inputs_bwd['eqt']
    tp_e3nn_bwd, x_e3nn_bwd, y_e3nn_bwd, W_e3nn_bwd, grad_e3nn_bwd = inputs_bwd['e3nn']

    # Check if weights require grad before defining backward functions
    weights_require_grad = W_eqt_bwd.requires_grad

    def backward_eqt(x, y, W, grad):
        # Detach inputs that don't require grad to avoid errors
        x_ = x if x.requires_grad else x.detach()
        y_ = y if y.requires_grad else y.detach()
        W_ = W if W.requires_grad else W.detach()

        # Ensure inputs requiring grad are used in computation
        res = tp_eqt_bwd(x_, y_, W_)
        # Handle cases where output might not have gradient path (e.g. empty irreps_out)
        if res.requires_grad:
             res.backward(grad, retain_graph=True) # Retain graph if needed for multiple backward calls
        grads = []
        if x.requires_grad: grads.append(x.grad)
        if y.requires_grad: grads.append(y.grad)
        if W.requires_grad: grads.append(W.grad)
        return grads


    def backward_e3nn(x, y, W, grad):
        x_ = x if x.requires_grad else x.detach()
        y_ = y if y.requires_grad else y.detach()
        W_ = W if W.requires_grad else W.detach()

        res = tp_e3nn_bwd(x_, y_, W_)
        if res.requires_grad:
             res.backward(grad, retain_graph=True)
        grads = []
        if x.requires_grad: grads.append(x.grad)
        if y.requires_grad: grads.append(y.grad)
        if W.requires_grad: grads.append(W.grad)
        return grads

    tester_bwd = FunctionTester({
        'eqt': (backward_eqt, [x_eqt_bwd, y_eqt_bwd, W_eqt_bwd, grad_eqt_bwd], {}),
        'e3nn': (backward_e3nn, [x_e3nn_bwd, y_e3nn_bwd, W_e3nn_bwd, grad_e3nn_bwd], {}),
    })

    # Determine channels for input gradient conversion
    C1 = channel_args[0]
    C2 = channel_args[1] if len(channel_args) > 2 else (1 if isinstance(tp_eqt_bwd, SO3Linear) else C1) # Infer C2

    try:
        comp_bwd = tester_bwd.compare(post_transforms=[
            lambda res: res, # eqt grads are already in correct format
            lambda res: [ # Convert e3nn grads
                e3nn_features_to_equitorch(g, irreps1_e3nn, channels=C1) if i == 0 and g is not None else \
                (e3nn_features_to_equitorch(g, irreps2_e3nn, channels=C2) if i == 1 and g is not None and not isinstance(tp_eqt_bwd, SO3Linear) else \
                 (g if i == 1 and g is not None and isinstance(tp_eqt_bwd, SO3Linear) else \
                  (e3nn_tp_weights_grad_to_eqt(tp_e3nn_bwd, tp_eqt_bwd, g) if i == 2 and g is not None and weights_require_grad else None)))
                for i, g in enumerate(res)
            ]
        ], compare_func=max_abs_diff_list) # Compare list of gradients
        print(f"Backward Max Abs Diff (Input1, Input2, Weight): {comp_bwd}")
        tester_bwd.profile(repeat=5)
    except Exception as e:
        print(f"Backward test failed: {e}")
        import traceback
        traceback.print_exc()


def test_tp_uvw(irreps1_eqt, irreps2_eqt, irreps_out_eqt, irreps1_e3nn, irreps2_e3nn, irreps_out_e3nn,
                C1, C2, C_out, N, shared, path_norm, device):
    _test_common(
        init_tp_uvw, 'TensorProduct UVW',
        irreps1_eqt, irreps2_eqt, irreps_out_eqt, irreps1_e3nn, irreps2_e3nn, irreps_out_e3nn,
        C1, C2, C_out, N, shared, path_norm, device
    )

def test_tp_uuu(irreps1_eqt, irreps2_eqt, irreps_out_eqt, irreps1_e3nn, irreps2_e3nn, irreps_out_e3nn,
                C, N, shared, path_norm, device):
    _test_common(
        init_tp_uuu, 'TensorProduct UUU',
        irreps1_eqt, irreps2_eqt, irreps_out_eqt, irreps1_e3nn, irreps2_e3nn, irreps_out_e3nn,
        C, N, shared, path_norm, device # Only one channel arg C
    )

def test_so3_uv(irreps1_eqt, irreps2_eqt, irreps_out_eqt, irreps1_e3nn, irreps2_e3nn, irreps_out_e3nn,
                C_in, C_out, N, shared, path_norm, device):
    _test_common(
        init_so3_uv, 'SO3Linear UV',
        irreps1_eqt, irreps2_eqt, irreps_out_eqt, irreps1_e3nn, irreps2_e3nn, irreps_out_e3nn,
        C_in, C_out, N, shared, path_norm, device # Channel args C_in, C_out
    )

def test_so3_uu(irreps1_eqt, irreps2_eqt, irreps_out_eqt, irreps1_e3nn, irreps2_e3nn, irreps_out_e3nn,
                C, N, shared, path_norm, device):
    _test_common(
        init_so3_uu, 'SO3Linear UU',
        irreps1_eqt, irreps2_eqt, irreps_out_eqt, irreps1_e3nn, irreps2_e3nn, irreps_out_e3nn,
        C, N, shared, path_norm, device # Only one channel arg C
    )


# === Main Execution Block ===

if __name__ == '__main__':
    # Define test parameters
    N = 256 # Smaller batch size for faster testing
    L_max = 4 # Smaller L_max
    L_min = 0
    # Define Irreps Strings
    irreps_str1 = '+'.join(f'1x{l}e' for l in range(L_min, L_max + 1))
    irreps_str2 = '+'.join(f'1x{l}e' for l in range(L_min, L_max + 1)) 
    irreps_str_out = '+'.join(f'1x{l}e' for l in range(L_min, L_max + 1))

    # irreps_str1 = '2e'
    # irreps_str2 = '1e'
    # irreps_str_out = '1e' # Example output with mixed parity allowed by TP
    # irreps_str_out = '1x0e + 1x1e + 1x2e' # Example output with mixed parity allowed by TP
    # irreps_str1 = irreps_str2 = irreps_str_out = '0e'
    # irreps_str2 = '0e+0e'
    # irreps_str2 = '0e'
    # Define Channels
    C1_uvw, C2_uvw, C_out_uvw = 16, 16, 16

    # Create Equitorch Irreps
    irreps1_eqt = Irreps(irreps_str1)
    irreps2_eqt = Irreps(irreps_str2)
    irreps_out_eqt = Irreps(irreps_str_out)

    # Create E3NN Irreps (multiplied by channels)
    irreps1_e3nn_uvw = e3nn.o3.Irreps(irreps1_eqt.channels_multiplied(C1_uvw).short_repr())
    irreps2_e3nn_uvw = e3nn.o3.Irreps(irreps2_eqt.channels_multiplied(C2_uvw).short_repr())
    irreps_out_e3nn_uvw = e3nn.o3.Irreps(irreps_out_eqt.channels_multiplied(C_out_uvw).short_repr())


    # Run tests for different configurations
    # Test UVW for all parameter combinations
    print(f"\n{'='*10} Testing UVW with all parameter combinations {'='*10}")

    # Case 1: path_norm=True, shared=True
    print(f"\n{'='*10} Testing path_norm=True, shared=True {'='*10}")
    test_tp_uvw(
        irreps1_eqt, irreps2_eqt, irreps_out_eqt,
        irreps1_e3nn_uvw, irreps2_e3nn_uvw, irreps_out_e3nn_uvw,
        C1_uvw, C2_uvw, C_out_uvw, N, True, True, device
    )

    # Case 2: path_norm=True, shared=False
    print(f"\n{'='*10} Testing path_norm=True, shared=False {'='*10}")
    test_tp_uvw(
        irreps1_eqt, irreps2_eqt, irreps_out_eqt,
        irreps1_e3nn_uvw, irreps2_e3nn_uvw, irreps_out_e3nn_uvw,
        C1_uvw, C2_uvw, C_out_uvw, N, False, True, device
    )

    # Case 3: path_norm=False, shared=True
    print(f"\n{'='*10} Testing path_norm=False, shared=True {'='*10}")
    test_tp_uvw(
        irreps1_eqt, irreps2_eqt, irreps_out_eqt,
        irreps1_e3nn_uvw, irreps2_e3nn_uvw, irreps_out_e3nn_uvw,
        C1_uvw, C2_uvw, C_out_uvw, N, True, False, device
    )

    # Case 4: path_norm=False, shared=False
    print(f"\n{'='*10} Testing path_norm=False, shared=False {'='*10}")
    test_tp_uvw(
        irreps1_eqt, irreps2_eqt, irreps_out_eqt,
        irreps1_e3nn_uvw, irreps2_e3nn_uvw, irreps_out_e3nn_uvw,
        C1_uvw, C2_uvw, C_out_uvw, N, False, False, device
    )



    # C_uuu = 256
    # irreps1_e3nn_uuu = e3nn.o3.Irreps(irreps1_eqt.channels_multiplied(C_uuu).short_repr())
    # irreps2_e3nn_uuu = e3nn.o3.Irreps(irreps2_eqt.channels_multiplied(C_uuu).short_repr())
    # irreps_out_e3nn_uuu = e3nn.o3.Irreps(irreps_out_eqt.channels_multiplied(C_uuu).short_repr())


    # # Test UUU for all parameter combinations
    # print(f"\n{'='*10} Testing UUU with all parameter combinations {'='*10}")

    # # Case 1: path_norm=True, shared=True
    # print(f"\n{'='*10} Testing path_norm=True, shared=True {'='*10}")
    # test_tp_uuu(
    #     irreps1_eqt, irreps2_eqt, irreps_out_eqt,
    #     irreps1_e3nn_uuu, irreps2_e3nn_uuu, irreps_out_e3nn_uuu,
    #     C_uuu, N, True, True, device
    # )

    # # Case 2: path_norm=True, shared=False
    # print(f"\n{'='*10} Testing path_norm=True, shared=False {'='*10}")
    # test_tp_uuu(
    #     irreps1_eqt, irreps2_eqt, irreps_out_eqt,
    #     irreps1_e3nn_uuu, irreps2_e3nn_uuu, irreps_out_e3nn_uuu,
    #     C_uuu, N, False, True, device
    # )

    # # Case 3: path_norm=False, shared=True
    # print(f"\n{'='*10} Testing path_norm=False, shared=True {'='*10}")
    # test_tp_uuu(
    #     irreps1_eqt, irreps2_eqt, irreps_out_eqt,
    #     irreps1_e3nn_uuu, irreps2_e3nn_uuu, irreps_out_e3nn_uuu,
    #     C_uuu, N, True, False, device
    # )

    # # Case 4: path_norm=False, shared=False
    # print(f"\n{'='*10} Testing path_norm=False, shared=False {'='*10}")
    # test_tp_uuu(
    #     irreps1_eqt, irreps2_eqt, irreps_out_eqt,
    #     irreps1_e3nn_uuu, irreps2_e3nn_uuu, irreps_out_e3nn_uuu,
    #     C_uuu, N, False, False, device
    # )

    # C_in_uv, C_out_uv = 64, 64

    # irreps1_e3nn_uv = e3nn.o3.Irreps(irreps1_eqt.channels_multiplied(C_in_uv).short_repr())
    # irreps2_e3nn_uv = e3nn.o3.Irreps(irreps2_eqt.short_repr()) # No channel multiplication for y in uv/uu
    # irreps_out_e3nn_uv = e3nn.o3.Irreps(irreps_out_eqt.channels_multiplied(C_out_uv).short_repr())

    # # Test UV for all parameter combinations
    # print(f"\n{'='*10} Testing UV with all parameter combinations {'='*10}")

    # # Case 1: path_norm=True, shared=True
    # print(f"\n{'='*10} Testing path_norm=True, shared=True {'='*10}")
    # test_so3_uv(
    #     irreps1_eqt, irreps2_eqt, irreps_out_eqt,
    #     irreps1_e3nn_uv, irreps2_e3nn_uv, irreps_out_e3nn_uv,
    #     C_in_uv, C_out_uv, N, True, True, device
    # )

    # # Case 2: path_norm=True, shared=False
    # print(f"\n{'='*10} Testing path_norm=True, shared=False {'='*10}")
    # test_so3_uv(
    #     irreps1_eqt, irreps2_eqt, irreps_out_eqt,
    #     irreps1_e3nn_uv, irreps2_e3nn_uv, irreps_out_e3nn_uv,
    #     C_in_uv, C_out_uv, N, False, True, device
    # )

    # # Case 3: path_norm=False, shared=True
    # print(f"\n{'='*10} Testing path_norm=False, shared=True {'='*10}")
    # test_so3_uv(
    #     irreps1_eqt, irreps2_eqt, irreps_out_eqt,
    #     irreps1_e3nn_uv, irreps2_e3nn_uv, irreps_out_e3nn_uv,
    #     C_in_uv, C_out_uv, N, True, False, device
    # )

    # # Case 4: path_norm=False, shared=False
    # print(f"\n{'='*10} Testing path_norm=False, shared=False {'='*10}")
    # test_so3_uv(
    #     irreps1_eqt, irreps2_eqt, irreps_out_eqt,
    #     irreps1_e3nn_uv, irreps2_e3nn_uv, irreps_out_e3nn_uv,
    #     C_in_uv, C_out_uv, N, False, False, device
    # )

    # C_uu = 256

    # irreps1_e3nn_uu = e3nn.o3.Irreps(irreps1_eqt.channels_multiplied(C_uu).short_repr())
    # irreps2_e3nn_uu = e3nn.o3.Irreps(irreps2_eqt.short_repr()) # No channel multiplication for y
    # irreps_out_e3nn_uu = e3nn.o3.Irreps(irreps_out_eqt.channels_multiplied(C_uu).short_repr())

    # # Test UU for all parameter combinations
    # print(f"\n{'='*10} Testing UU with all parameter combinations {'='*10}")

    # # Case 1: path_norm=True, shared=True
    # print(f"\n{'='*10} Testing path_norm=True, shared=True {'='*10}")
    # test_so3_uu(
    #     irreps1_eqt, irreps2_eqt, irreps_out_eqt,
    #     irreps1_e3nn_uu, irreps2_e3nn_uu, irreps_out_e3nn_uu,
    #     C_uu, N, True, True, device
    # )

    # # Case 2: path_norm=True, shared=False
    # print(f"\n{'='*10} Testing path_norm=True, shared=False {'='*10}")
    # test_so3_uu(
    #     irreps1_eqt, irreps2_eqt, irreps_out_eqt,
    #     irreps1_e3nn_uu, irreps2_e3nn_uu, irreps_out_e3nn_uu,
    #     C_uu, N, False, True, device
    # )

    # # Case 3: path_norm=False, shared=True
    # print(f"\n{'='*10} Testing path_norm=False, shared=True {'='*10}")
    # test_so3_uu(
    #     irreps1_eqt, irreps2_eqt, irreps_out_eqt,
    #     irreps1_e3nn_uu, irreps2_e3nn_uu, irreps_out_e3nn_uu,
    #     C_uu, N, True, False, device
    # )

    # # Case 4: path_norm=False, shared=False
    # print(f"\n{'='*10} Testing path_norm=False, shared=False {'='*10}")
    # test_so3_uu(
    #     irreps1_eqt, irreps2_eqt, irreps_out_eqt,
    #     irreps1_e3nn_uu, irreps2_e3nn_uu, irreps_out_e3nn_uu,
    #     C_uu, N, False, False, device
    # )
    # print("\nAll tests completed.")
