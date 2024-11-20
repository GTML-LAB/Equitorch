import torch
import torch.nn as nn
from torch import Tensor

class SineBasisExpansion(nn.Module):
    r"""
    Sine Basis Expansion module.

    This module implements a sine basis expansion of the form:

    .. math::

        \sin(n x),\ n=1,2,\dots,\text{max_freq}.

    Parameters
    ----------
    max_freq : int
        The maximum frequency to use.
    """
    def __init__(self, 
                 max_freq: int):
        super().__init__()
        self.max_freq = max_freq
        freq = torch.arange(1, max_freq+1)
        self.register_buffer('freq', freq)

    def forward(self, x: Tensor):
        """
        """
        return (self.freq * x.unsqueeze(-1)).sin()
    
class CosineBasisExpansion(nn.Module):
    r"""
    Cosine Basis Expansion module.

    This module implements a cosine basis expansion of the form:

    .. math::

        \cos(n x), n=1,2,\dots,\text{max_freq}.

    Parameters
    ----------
    max_freq : int
        The maximum frequency to use.
    """
    def __init__(self, 
                 max_freq: int):
        super().__init__()
        self.max_freq = max_freq
        freq = torch.arange(1, max_freq+1)
        self.register_buffer('freq', freq)

    def forward(self, x: Tensor):
        """
        """
        return (self.freq * x.unsqueeze(-1)).cos()


class FourierBasisExpansion(nn.Module):
    r"""Fourier Basis Expansion module.

    This module implements a Fourier basis expansion of the form:

    .. math::

        [\sin(nx), \cos(nx)],\ n=1,2,\dots,\text{max_freq}.

    Parameters
    ----------
    max_freq : int
        The maximum frequency to use.
    include_freq_0 : bool, optional
        Whether to include a constant term (frequency 0) in the expansion. 
        Default is :obj:`False`.

    Note
    ----
    The output tensor is organized in the following frequency order:
    
    .. math::
        [\sin(\text{max_freq}\cdot x), ..., \sin(2x), \sin(x), \\
            1 \text{(if include_freq_0)}, \\
                \cos(x), \cos(2x), ..., \cos(\text{max_freq}\cdot x)]

    This arrangement places lower frequencies closer to the center of the output tensor,
    with sine terms in descending order followed by cosine terms in ascending order.

    It is recommended to not include frequency 0. It will be equivalent to add a bias in
    the following linear operation.
    """
    def __init__(self, 
                 max_freq: int,
                 include_freq_0: bool = False):
        super().__init__()
        self.max_freq = max_freq
        freq = torch.arange(1, max_freq+1)
        self.register_buffer('freq', freq)
        self.include_freq_0 = include_freq_0

    def forward(self, x: Tensor):
        """
        """
        x = x.unsqueeze(-1)
        if self.include_freq_0:
            return torch.cat([
                (self.freq.flip(0) * x).sin(),
                torch.ones_like(x),                                                                   
                (self.freq * x).cos()
            ], dim=-1)
        else:
            return torch.cat([
                (self.freq.flip(0) * x).sin(),
                (self.freq * x).cos()
            ], dim=-1)