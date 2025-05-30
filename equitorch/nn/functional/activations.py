import torch
from torch import Tensor

from .sparse_product import sparse_mul

from ...structs import IrrepsInfo, SparseProductInfo


def gating(input: Tensor, gates: Tensor, irreps_info: IrrepsInfo) -> Tensor:
    r"""
    Equivariant gating mechanism.

    Applies element-wise gates to features. This is the functional
    version of the :class:`~equitorch.nn.activations.Gate` module.

    See :class:`~equitorch.nn.activations.Gate` for more details on the gating mechanism,
    including how ``input`` and ``gates`` are structured and combined.

    Args:
        input (torch.Tensor): Tensor to be gated. Shape ``(..., irreps.dim, channels)``.
        gates (torch.Tensor): Gating values. Shape depends on how gates are applied
            (e.g., ``(..., num_gates, channels)`` for irrep-wise gating or
            ``(..., 1, channels)`` for global gating).
        irreps_info (IrrepsInfo): Contains ``irreps_info.irrep_index``, which maps
                                  each component of ``input``'s spherical dimension to an
                                  index in ``gates``' corresponding dimension (the gate dimension).

    Returns:
        torch.Tensor: The gated input tensor, shape ``(..., irreps.dim, channels)``.
    """
    info_fwd = SparseProductInfo(index2=irreps_info.irrep_index)
    info_bwd1 = SparseProductInfo(index1=irreps_info.irrep_index)
    info_bwd2 = SparseProductInfo(seg_out=irreps_info.irrep_seg)
    
    return sparse_mul(input, gates, 
                      info_fwd, info_bwd1, info_bwd2)
