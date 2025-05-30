import torch
from torch import Tensor
import torch.nn.functional as F

from .sparse_product import sparse_mul
from ...structs import IrrepsInfo, SparseProductInfo

def irrep_wise_dropout(input: Tensor, p: float, training: bool, irreps_info: IrrepsInfo) -> Tensor:
    r"""
    Apply dropout irrep-wise to an input tensor.

    This is the functional version of the irrep-wise mode of the
    :class:`~equitorch.nn.dropout.Dropout` module.
    See :class:`~equitorch.nn.dropout.Dropout` for more details on the dropout mechanism
    when ``irrep_wise=True``.

    Args:
        input (torch.Tensor): Input tensor of shape ``(..., irreps_dim, channels)``.
        p (float): Probability of an element to be zeroed.
        training (bool): Apply dropout if ``True``.
        irreps_info (IrrepsInfo): Contains irreps structure information, such as
            ``irreps_info.irrep_index`` to map components of ``input`` to their
            respective irrep instances, and ``irreps_info.num_irreps``.
            Must not be ``None``.

    Returns:
        torch.Tensor: Output tensor with dropout applied, of the same shape as ``input``.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
    if p == 0.0 or not training:
        return input
    
    if irreps_info is None:
        raise ValueError("irreps_info cannot be None for irrep_wise_dropout")

    num_irreps = irreps_info.num_irreps # num_irrep_instances
    num_channels = input.shape[-1]

    # Create a mask for each (irrep_instance, channel)
    # The mask should have shape (num_irreps, num_channels)
    # This mask will be applied to input elements input[n, M, c]
    # where M corresponds to an irrep instance i, and we use mask[i, c]
    
    # Bernoulli distribution for the mask
    # The mask tensor will have dimensions corresponding to (num_irreps, num_channels)
    # It needs to be broadcastable or correctly indexed by sparse_mul
    # sparse_mul with index2=irreps_info.irrep_index will map the irreps_dim of input
    # to the first dimension of the mask (num_irreps).
    # The channel dimension will align directly.
    mask_shape = (input.shape[0], num_irreps, num_channels)
    mask = torch.bernoulli(torch.full(mask_shape, 1.0 - p, dtype=input.dtype, device=input.device))
    
    # Scale the mask by 1 / (1 - p)
    mask = mask.div_(1.0 - p).nan_to_num_(nan=0)

    # Define SparseProductInfo similar to Gating or BatchRMSNorm
    # input1 is 'input', input2 is 'mask'
    # index2 maps input's irreps_dim to mask's first dim (num_irreps)
    info_fwd = SparseProductInfo(index2=irreps_info.irrep_index)
    # For backward pass of sparse_mul(A, B):
    # grad_A = sparse_mul(B, grad_output, info_bwd1_for_A, info_bwd2_for_A, info_fwd_for_A)
    # grad_B = sparse_mul(grad_output, A, info_bwd1_for_B, info_bwd2_for_B, info_fwd_for_B)
    # Here, A=input, B=mask.
    # info_bwd1 (for grad_input) needs to configure sparse_mul(mask, grad_output)
    #   - mask is input1, grad_output is input2
    #   - index1 for mask should be irreps_info.irrep_index
    info_bwd1 = SparseProductInfo(index1=irreps_info.irrep_index)
    # info_bwd2 (for grad_mask, though mask is not learnable) needs to configure sparse_mul(grad_output, input)
    #   - grad_output is input1, input is input2
    #   - seg_out for grad_output should be irreps_info.irrep_seg to sum contributions for each (irrep_instance, channel)
    info_bwd2 = SparseProductInfo(seg_out=irreps_info.irrep_seg)

    return sparse_mul(input, mask, info_fwd, info_bwd1, info_bwd2)
