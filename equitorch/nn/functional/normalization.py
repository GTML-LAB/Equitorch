import torch
from torch import Tensor

from equitorch.structs import IrrepsInfo, SparseProductInfo
from equitorch.utils._structs import sparse_product_infos
from .sparse_product import sparse_mul, sparse_vecsca, sparse_inner
from .norm import squared_norm, channel_mean_squared_norm


def batch_rms_norm(
    input: Tensor,
    running_squared_norm: Tensor,  # Shape (num_irreps, C)
    irreps_info: IrrepsInfo,
    weight: Tensor = None,  # Optional, Shape (num_irreps, C)
    scaled: bool = True,  # Whether the squared_norm is scaled by 1/D_i
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-05,
) -> Tensor:
    """
    Applies Batch RMS Normalization for equivariant features.

    Args:
        input (Tensor): Input tensor of shape (N, irreps_dim, C).
        running_squared_norm (Tensor): Running average of squared norms,
            shape (num_irreps, C). Modified in-place if training is True.
        irreps_info (IrrepsInfo): Contains irreps structure information.
        weight (Tensor, optional): Learnable scaling factor gamma_ic,
            shape (num_irreps, C). Defaults to None (no scaling).
        scaled (bool, optional): If True, the internal squared_norm calculation
            divides by the dimension of the irrep (D_i). Defaults to True.
        training (bool, optional): If True, uses current batch statistics for
            normalization and updates running_squared_norm. Otherwise, uses
            running_squared_norm. Defaults to False.
        momentum (float, optional): Momentum for updating running_squared_norm.
            Defaults to 0.1.
        eps (float, optional): Epsilon for numerical stability. Defaults to 1e-05.

    Returns:
        Tensor: Normalized output tensor of the same shape as input.
    """
    # Skip shape checking, but keep these commented
    # if input.ndim != 3:
    #     raise ValueError(
    #         f"Input tensor must be 3D (N, irreps_dim, C), got {input.ndim}D"
    #     )
    
    # num_irreps = running_squared_norm.shape[0]
    # if irreps_info.irrep_index.max() >= num_irreps:
    #     raise ValueError(
    #         "irreps_info.irrep_index contains indices out of bounds "
    #         f"for running_squared_norm first dimension ({num_irreps})"
    #     )
    # if running_squared_norm.shape[1] != input.shape[2]:
    #     raise ValueError(
    #         f"Channel dimension of running_squared_norm ({running_squared_norm.shape[1]}) "
    #         f"must match input's channel dimension ({input.shape[2]})"
    #     )
    # if weight is not None:
    #     if weight.shape != running_squared_norm.shape:
    #         raise ValueError(
    #             f"Shape of weight {weight.shape} must match running_squared_norm {running_squared_norm.shape}"
    #         )

    # 1. Calculate current batch's squared norm
    # _squared_norm output shape: (N, num_irreps, C)
    current_batch_sq_norm_val = squared_norm(input, irreps_info, scaled)

    # 2. Determine norm to use and update running_squared_norm if training
    if training:
        # batch_mean_current_sq_norm shape: (num_irreps, C)
        batch_mean_current_sq_norm = current_batch_sq_norm_val.mean(dim=0)
        running_squared_norm.data.lerp_(batch_mean_current_sq_norm, momentum)
        norm_to_use = batch_mean_current_sq_norm
    else:
        norm_to_use = running_squared_norm

    # 3. Calculate sigma_ic
    # inv_sigma_ic shape: (num_irreps, C)
    inv_sigma_ic = torch.rsqrt(norm_to_use + eps)

    scale_values: Tensor  # Shape (num_irreps, C)
    if weight is not None:
        scale_values = weight * inv_sigma_ic
    else:
        scale_values = inv_sigma_ic

    # 5. Prepare for sparse_mul
    # input is (N, irreps_dim, C)
    # scale_values is (num_irreps, C)
    # We want to scale input[n, M, c] by scale_values[i[M], c]
        
    # sparse_product_infos will create tensors on the specified device.
    # irreps_info.irrep_index should ideally be on the same device.
    # We assume irreps_info has been moved to the correct device by the caller (e.g., nn.Module)
    info_fwd = SparseProductInfo(index2=irreps_info.irrep_index)
    info_bwd1 = SparseProductInfo(index1=irreps_info.irrep_index)
    info_bwd2 = SparseProductInfo(seg_out=irreps_info.irrep_seg)
    
    # 6. Perform scaling using sparse_mul
    # input1: input (N, irreps_dim, C)
    # input2: scale_values (num_irreps, C) -> this is broadcasted/indexed correctly by sparse_mul
    output = sparse_mul(input, scale_values, info_fwd, info_bwd1, info_bwd2)
    
    return output


def layer_rms_norm(input_tensor: Tensor, 
                   irreps_info: IrrepsInfo, 
                   weight: Tensor = None, # weight is gamma_ic, shape (num_irreps, C)
                   scaled: bool = True, # if True, sigma is scaled by 1/D_i
                   eps: float = 1e-05) -> Tensor:
    """
    Applies Layer Root Mean Square Normalization for equivariant features.

    Args:
        input_tensor (Tensor): Input tensor of shape (N, irreps_dim, C).
        irreps_info (IrrepsInfo): Contains irreps structure information.
        weight (Tensor, optional): Learnable scaling factor gamma_ic, shape (num_irreps, C).
                                   Defaults to None (no scaling).
        scaled (bool, optional): If True, the statistics calculation considers the norm
                                 to be scaled by 1/D_i. Defaults to True.
        eps (float, optional): Epsilon for numerical stability. Defaults to 1e-05.

    Returns:
        Tensor: Normalized output tensor of the same shape as input.
    """
    num_channels = input_tensor.shape[-1]
    
    # Calculate sigma_ni = sqrt( (1/(C*D_i)) * sum_mc(x_nimc^2) + eps )
    # channel_mean_squared_norm with scaled=True computes (1/(C*D_i)) * sum_mc(x_nimc^2)
    # Its output shape is (N, num_irreps)
    squared_norm_ni = channel_mean_squared_norm(input_tensor, irreps_info, scaled)
    
    rsigma_ni = torch.rsqrt(squared_norm_ni + eps) # Using rsqrt for 1/sigma_ni, shape (N, num_irreps)

    # Define SparseProductInfo objects as per user's convention for re-use
    # These will be used as the fwd_info, bwd_info1, bwd_info2 arguments in sparse_X.apply(...) calls
    info_fwd = SparseProductInfo(index2=irreps_info.irrep_index)
    info_bwd1 = SparseProductInfo(index1=irreps_info.irrep_index)
    info_bwd2 = SparseProductInfo(seg_out=irreps_info.irrep_seg)

    # Calculate x'_nimc = x_nimc / sigma_ni
    # input_tensor is x (N, irreps_dim, C)
    # rsigma_ni is 1/sigma_ni (N, num_irreps)
    # sparse_vecsca uses info_fwd_user.index2 to map irreps_dim to num_irreps for rsigma_ni
    normed = sparse_vecsca(input_tensor, rsigma_ni, 
                           info_fwd, info_bwd1, info_bwd2)

    # Calculate y'_nimc = gamma_ic * x'_nimc
    # normed is x' (N, irreps_dim, C)
    # weight is gamma (num_irreps, C)
    # sparse_mul uses info_fwd_user.index2 to map irreps_dim to num_irreps for weight
    if weight is not None:
        output_val = sparse_mul(normed, weight, 
                            info_fwd, info_bwd1, info_bwd2)
    else:
        output_val = normed
    
    return output_val
