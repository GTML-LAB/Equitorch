import torch
from torch import Tensor
import torch.nn as nn # Add nn import
from .functional.angular import sincos

class SinCos(nn.Module):
    r"""
    Module wrapper for the :func:`~equitorch.nn.functional.angular.sincos` function.

    Computes the sin/cos expansion of an angle \(a\):

    .. math::

        [1.0, \sin(a), \cos(a), \sin(2a), \cos(2a), \dots, \sin(\text{max_m} \cdot a), \cos(\text{max_m} \cdot a)]

    or

    .. math::
        [1.0, \sqrt{2}\sin(a), \sqrt{2}\cos(a), \sqrt{2}\sin(2a), \sqrt{2}\cos(2a), \dots, \sqrt{2}\sin(\text{max_m} \cdot a), \sqrt{2}\cos(\text{max_m} \cdot a)]
    
    The leading 1.0 is excluded if `with_ones` is ``False``.

    Args:
        max_m (int): The maximum multiple of the angle \(a\) to compute \(\sin\) and \(\cos\) for.
        with_ones (bool, optional): Whether to include the leading 1.0 in the expansion. Defaults to ``True``.
        component_normalize (bool, optional): If ``True``, multiplies the \(\sin\) and \(\cos\) values by \(\sqrt{2}\) 
                                         such that the expectation of the squared norm over \([0, 2\pi]\) is 1. 
                                         Defaults to ``False``.
    """
    def __init__(self, max_m: int, with_ones: bool = True, component_normalize: bool = False):
        super().__init__()
        if not isinstance(max_m, int) or max_m < 0:
            raise ValueError(f"max_m must be a non-negative integer, got {max_m}")
        self.max_m = max_m
        self.with_ones = with_ones
        self.component_normalize = component_normalize

    def forward(self, angle: Tensor) -> Tensor:
        r"""
        Args:
            angle (Tensor): Input angles.

        Returns:
            Tensor: The computed sin/cos tensor.
        """
        return sincos(angle, self.max_m, self.with_ones, component_normalize=self.component_normalize)

    def extra_repr(self) -> str:
        return f'max_m={self.max_m}, with_ones={self.with_ones}'
