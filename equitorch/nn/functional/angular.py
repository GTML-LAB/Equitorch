import torch
from torch import Tensor

def sincos(angle: Tensor, max_m: int, with_ones=True, component_normalize=False):
    r"""Prepares the sin/cos tensor for z-rotation.

    This is the functional version of the :class:`~equitorch.nn.angular.SinCos` module.
    See :class:`~equitorch.nn.angular.SinCos` for more details.

    Args:
        angle (torch.Tensor): Input angles.
        max_m (int): The maximum multiple of the angle to compute.
        with_ones (bool, optional): Whether to include the leading 1.0. Defaults to ``True``.
        component_normalize (bool, optional): If ``True``, normalizes sin/cos components by :math:`\sqrt{2}`.
                                         Defaults to ``False``.

    Returns:
        torch.Tensor: The computed sin/cos tensor.
    """
    if max_m == 0:
        # Only scalar irreps, rotation is identity
        return torch.ones_like(angle).unsqueeze(-1)
    m = torch.arange(1, max_m+1, dtype=angle.dtype, device=angle.device)
    m_angle = angle.unsqueeze(-1) * m
    sin_m = torch.sin(m_angle) # sin(|m|*angle)
    cos_m = torch.cos(m_angle) # cos(|m|*angle)
    if component_normalize:
        sin_m = sin_m * (2**0.5)
        cos_m = cos_m * (2**0.5)
    if with_ones:
        ones = torch.ones_like(angle).unsqueeze(-1)
        # [1.0, sin(1a), cos(1a), sin(2a), cos(2a), ...]
        return torch.cat([ones, torch.stack([sin_m, cos_m], dim=-1).flatten(-2, -1)], dim=-1)
    else:
        return torch.stack([sin_m, cos_m], dim=-1).flatten(-2, -1)

