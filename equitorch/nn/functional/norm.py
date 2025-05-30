import torch
from torch import Tensor
from torch.autograd import Function

from .sparse_product import sparse_mul, sparse_scavec

from ...structs import IrrepsInfo, SparseProductInfo 
from ...ops.product_segment_op import mul_segment 
from ...ops.indexed_product_op import indexed_mul
from ...utils._indices import expand_left
from torch_geometric.utils import segment # Retaining user's import, though mul_segment is used below



class SquaredNorm(Function):
    r"""
    Computes the squared L2 norm for each irrep block in an input tensor.
    Output_k = sum_{i in irrep_k} (input_i^2)
    Optionally scales the output by 1/sqrt(dim_irrep_k).
    """

    @staticmethod
    def forward(ctx, input_tensor: Tensor, 
                irreps_info: IrrepsInfo, scaled: bool = True):

        norm_sq = segment(input_tensor.pow(2), expand_left(input_tensor,irreps_info.irrep_seg, dim=-2))

        if scaled:
            scaling_factor = irreps_info.rdims.unsqueeze(-1) # (num_irreps, 1)
            norm_sq = norm_sq * scaling_factor

        ctx.save_for_backward(input_tensor if input_tensor.requires_grad else None)
        ctx.irreps_info = irreps_info
        ctx.scaled = scaled
        
        return norm_sq
    
    @staticmethod
    def backward(ctx, grad_output: Tensor):
        input_tensor, = ctx.saved_tensors # Comma to unpack single tensor
        scaled = ctx.scaled
        irreps_info = ctx.irreps_info
        info_fwd = SparseProductInfo(seg_out=irreps_info.irrep_seg)
        info_bwd1 = SparseProductInfo(index2=irreps_info.irrep_index)
        info_bwd2 = SparseProductInfo(index1=irreps_info.irrep_index)

        grad_output_effective = grad_output
        if scaled:
            scaling_factor = irreps_info.rdims.unsqueeze(-1) 
            grad_output_effective = grad_output_effective * scaling_factor # d(y*s)/dy = s

        grad_input = None
        if ctx.needs_input_grad[0] and input_tensor is not None:
            
            grad_input = sparse_mul(grad_output_effective, 2 * input_tensor, 
                                     info_bwd2, info_fwd, info_bwd1)
        
        return grad_input, None, None
def squared_norm(input_tensor: Tensor, irreps_info: IrrepsInfo, scaled: bool = True) -> Tensor:
    return SquaredNorm.apply(input_tensor, irreps_info, scaled)


class Norm(Function):
    r"""
    Computes the L2 norm for each irrep block in an input tensor.
    Output_k = sqrt(sum_{i in irrep_k} (input_i^2))
    Optionally scales the output by 1/sqrt(dim_irrep_k).
    Gradient at zero vector is zero.
    
    Hessian at zero vector is not supported (returns nan). 
    """
    @staticmethod
    def forward(ctx, input_tensor: Tensor, 
                irreps_info: IrrepsInfo, scaled: bool = True):

        norm_sq_val = segment(input_tensor.pow(2), expand_left(input_tensor,irreps_info.irrep_seg, dim=-2))

        # norm_val_before_scaling is y_k = sqrt(S_k)
        norm_val_before_scaling = torch.sqrt(norm_sq_val) 

        ctx.save_for_backward(input_tensor if input_tensor.requires_grad else None, 
                              norm_val_before_scaling) # Save y_k
        ctx.irreps_info = irreps_info
        ctx.scaled = scaled

        if scaled:
            scaling_factor_expanded = irreps_info.rsqrt_dims.unsqueeze(-1)
            return norm_val_before_scaling * scaling_factor_expanded
        else:
            return norm_val_before_scaling

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        input_tensor, norm_val_before_scaling = ctx.saved_tensors # norm_val_before_scaling is y_k
        irreps_info = ctx.irreps_info
        scaled = ctx.scaled
        info_fwd = SparseProductInfo(seg_out=irreps_info.irrep_seg)
        info_bwd1 = SparseProductInfo(index2=irreps_info.irrep_index)
        info_bwd2 = SparseProductInfo(index1=irreps_info.irrep_index)

        grad_wrt_final_output = grad_output

        # Step 1: Propagate gradient through the optional scaling
        # If y'_k = y_k * s_k, then dL/dy_k = (dL/dy'_k) * s_k
        if scaled:
            scaling_factor_expanded = irreps_info.rsqrt_dims.unsqueeze(-1)
            grad_wrt_norm_unscaled = grad_wrt_final_output * scaling_factor_expanded
        else:
            grad_wrt_norm_unscaled = grad_wrt_final_output
        
        # grad_wrt_norm_unscaled is now dL/dy_k, where y_k = sqrt(sum_i x_i^2)
        # We need dL/dx_j = (dL/dy_k) * (dy_k/dx_j)
        # And dy_k/dx_j = x_j / y_k

        grad_input = None
        if ctx.needs_input_grad[0] and input_tensor is not None:
            # Safely compute 1 / y_k
            # (norm_val_before_scaling is y_k)
            inv_norm_val_stable = torch.reciprocal(norm_val_before_scaling).nan_to_num(posinf=0, neginf=0)
            # Multiplier for input_tensor: (dL/dy_k) / y_k
            # This has shape (..., num_irreps, channels)
            multiplier_for_input = grad_wrt_norm_unscaled * inv_norm_val_stable
            
            # Expand multiplier to match input_tensor's irreps_dim using irrep_index
            # grad_input_j = multiplier_for_input_k * input_tensor_j
            # where k = irreps_info.irrep_index[j]
            grad_input = sparse_mul(multiplier_for_input, input_tensor, 
                                    info_bwd2, info_fwd, info_bwd1)
        
        return grad_input, None, None # for input_tensor, irreps_info, scaled

def norm(input_tensor: Tensor, irreps_info: IrrepsInfo, scaled: bool = True):
 return Norm.apply(input_tensor, irreps_info, scaled)


class ChannelMeanSquaredNorm(Function):
    r"""
    Computes the mean of squared L2 norms over channels for each irrep block.
    Output_k = (1/C) * sum_c ( sum_{i in irrep_k} (input_i_c^2) )
    Optionally scales the output by 1/dim_irrep_k.
    """
    @staticmethod
    def forward(ctx, input_tensor: Tensor, 
                irreps_info: IrrepsInfo, scaled: bool = True):
        
        num_channels = input_tensor.shape[-1]
        
        # norm_sq_segmented shape: (..., num_irreps, C)
        norm_sq_segmented = segment(input_tensor.pow(2), expand_left(input_tensor, irreps_info.irrep_seg, dim=-2))
        
        # Sum over channels: shape (..., num_irreps)
        norm_sq_summed_channels = norm_sq_segmented.sum(dim=-1)
        # Mandatory scaling by 1/num_channels
        mean_norm_sq = norm_sq_summed_channels / num_channels
        final_norm = mean_norm_sq

        if scaled:
            # Optional scaling by 1/dim_irrep_k (rsqrt_dims is 1/sqrt(D_k))
            # rsqrt_dims has shape (num_irreps)
            scaling_factor_irreps = irreps_info.rdims 
            final_norm = final_norm * scaling_factor_irreps
        
        ctx.save_for_backward(input_tensor if input_tensor.requires_grad else None)
        ctx.irreps_info = irreps_info
        ctx.scaled = scaled
        ctx.num_channels = num_channels
        
        return final_norm
    
    @staticmethod
    def backward(ctx, grad_output: Tensor): # grad_output has shape (..., num_irreps)
        input_tensor, = ctx.saved_tensors
        scaled = ctx.scaled
        irreps_info = ctx.irreps_info
        num_channels = ctx.num_channels

        grad_input = None
        if not ctx.needs_input_grad[0] or input_tensor is None:
            return grad_input, None, None

        # grad_output has shape (..., num_irreps)
        grad_wrt_final_norm = grad_output
        
        # Propagate gradient through optional irrep scaling
        grad_wrt_mean_norm_sq = grad_wrt_final_norm
        if scaled:
            scaling_factor_irreps = irreps_info.rdims
            grad_wrt_mean_norm_sq = grad_wrt_final_norm * scaling_factor_irreps
            
        # Propagate gradient through mandatory channel scaling (1/num_channels)
        grad_wrt_norm_sq_summed_channels = grad_wrt_mean_norm_sq / num_channels
                
        # Construct SparseProductInfo as in SquaredNorm
        info_fwd = SparseProductInfo(seg_out=irreps_info.irrep_seg)
        info_bwd1 = SparseProductInfo(index2=irreps_info.irrep_index) # for 2*input_tensor
        info_bwd2 = SparseProductInfo(index1=irreps_info.irrep_index) # for grad_wrt_norm_sq_segmented
            
        grad_input = sparse_scavec(grad_wrt_norm_sq_summed_channels, 2 * input_tensor, 
                                 info_bwd2, info_fwd, info_bwd1)
        
        return grad_input, None, None
def channel_mean_squared_norm(input_tensor: Tensor, irreps_info: IrrepsInfo, scaled: bool = True) -> Tensor:
    return ChannelMeanSquaredNorm.apply(input_tensor, irreps_info, scaled)


class BatchMeanSquaredNorm(Function):
    r"""
    Computes the mean of squared L2 norms over the batch dimension for each irrep block and channel.
    Output_ic = (1/N) * sum_n ( sum_{m in irrep_i} (input_n(im)c^2) )
    Optionally scales the output by 1/dim_irrep_i.
    """
    @staticmethod
    def forward(ctx, input_tensor: Tensor, 
                irreps_info: IrrepsInfo, scaled: bool = True):
        
        batch_size = input_tensor.shape[0]
        
        # norm_sq_segmented shape: (N, ..., num_irreps, C)
        norm_sq_segmented = segment(input_tensor.pow(2), expand_left(input_tensor, irreps_info.irrep_seg, dim=-2))
        
        # Sum over batch dim: shape (..., num_irreps, C)
        norm_sq_summed_batch = norm_sq_segmented.sum(dim=0)
        
        mean_norm_sq = norm_sq_summed_batch / batch_size

        final_norm = mean_norm_sq

        if scaled:
            scaling_factor_irreps = irreps_info.rdims 
            scaling_factor_expanded = scaling_factor_irreps.unsqueeze(-1)                  
            final_norm = final_norm * scaling_factor_expanded
        
        ctx.save_for_backward(input_tensor if input_tensor.requires_grad else None)
        ctx.irreps_info = irreps_info
        ctx.scaled = scaled
        ctx.batch_size = batch_size
        
        return final_norm # Shape (..., num_irreps, C)
    
    @staticmethod
    def backward(ctx, grad_output: Tensor): # grad_output has shape (..., num_irreps, C)
        input_tensor, = ctx.saved_tensors
        scaled = ctx.scaled
        irreps_info = ctx.irreps_info
        batch_size = ctx.batch_size

        grad_input = None
        if not ctx.needs_input_grad[0] or input_tensor is None:
            return grad_input, None, None
            
        # grad_output has shape (..., num_irreps, C)
        grad_wrt_final_norm = grad_output
        
        # Propagate gradient through optional irrep scaling
        grad_wrt_mean_norm_sq = grad_wrt_final_norm
        if scaled:
            scaling_factor_irreps = irreps_info.rdims
            scaling_factor_expanded = scaling_factor_irreps.unsqueeze(-1)
            grad_wrt_mean_norm_sq = grad_wrt_final_norm * scaling_factor_expanded
            
        # Propagate gradient through mandatory batch scaling (1/batch_size)
        grad_wrt_norm_sq_summed_batch = grad_wrt_mean_norm_sq / batch_size
                
        # Construct SparseProductInfo as in SquaredNorm
        info_fwd = SparseProductInfo(seg_out=irreps_info.irrep_seg)
        info_bwd1 = SparseProductInfo(index2=irreps_info.irrep_index) # for 2*input_tensor
        info_bwd2 = SparseProductInfo(index1=irreps_info.irrep_index) # for grad_wrt_norm_sq_segmented
            
        # The rest is like SquaredNorm backward
        grad_input = sparse_mul(grad_wrt_norm_sq_summed_batch, 2 * input_tensor, 
                                 info_bwd2, info_fwd, info_bwd1)
        
        return grad_input, None, None

def batch_mean_squared_norm(input_tensor: Tensor, irreps_info: IrrepsInfo, scaled: bool = True):
    return BatchMeanSquaredNorm.apply(input_tensor, irreps_info, scaled)
