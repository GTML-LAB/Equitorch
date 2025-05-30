import torch
from torch import nn, Tensor

from typing import Literal

from ..irreps import Irreps, check_irreps
from ..structs import IrrepsInfo
from ..utils._structs import irreps_info
# Import all functional norm versions
from .functional.norm import norm, squared_norm, channel_mean_squared_norm, batch_mean_squared_norm

class SquaredNorm(nn.Module):
    r"""Computes the squared L2 norm for each irrep block in an input tensor.

    .. math::

        \text{Output}_k = \sum_{m \in \text{irrep}_k} (\text{input}_{km}^2)

    Optionally scales the output by :math:`1/\text{irrep}_k\text{.dim}`.

    Args:
        irreps (Irreps): Irreducible representations of the input tensor.
        scaled (bool, optional): If ``True``, scales the output of each ``irrep_k``
                                 by :math:`1/\text{irrep}_k\text{.dim}`.
                                 Defaults to ``True``.
    """
    irreps_info: IrrepsInfo

    def __init__(self, irreps: Irreps, scaled: bool = True):
        super().__init__()
        self.irreps = check_irreps(irreps)
        self.scaled = scaled
        self.irreps_dim = self.irreps.dim
        # Store IrrepsInfo. It will be moved to device/dtype by _apply
        self.irreps_info = irreps_info(self.irreps)

    def forward(self, input_tensor: Tensor) -> Tensor:
        r"""
        Args:
            input_tensor (torch.Tensor): Input tensor of shape ``(..., irreps.dim, C)``.
        Returns:
            torch.Tensor: Output tensor of shape ``(..., len(irreps), C)``.
        """
        if input_tensor.shape[-2] != self.irreps_dim:
            raise ValueError(
                f"Input tensor's irreps dimension ({input_tensor.shape[-2]}) "
                f"does not match expected irreps dimension ({self.irreps.dim})"
            )
        return squared_norm(input_tensor, self.irreps_info, self.scaled)


    def _apply(self, *args, **kwargs):
        n = super()._apply(*args, **kwargs)
        n.irreps_info = self.irreps_info._apply(*args, **kwargs)
        return n

    def __extra_repr__(self):
        return f"irreps='{self.irreps}', scaled={self.scaled}"


class Norm(nn.Module):
    r"""Computes the L2 norm for each irrep block in an input tensor.

    .. math::

        \text{Output}_k = \sqrt{\sum_{m \in \text{irrep}_k} (\text{input}_{km}^2)}

    Optionally scales the output by :math:`\sqrt{1/\text{irrep}_k\text{.dim}}`.
    Gradient at zero vector is zero.

    Args:
        irreps (Irreps): Irreducible representations of the input tensor.
        scaled (bool, optional): If ``True``, scales the output of each ``irrep_k``
                                 by :math:`\sqrt{1/\text{irrep}_k\text{.dim}}`.
                                 Defaults to ``True``.
    """
    irreps_info: IrrepsInfo

    def __init__(self, irreps: Irreps, scaled: bool = True):
        super().__init__()
        self.irreps = check_irreps(irreps)
        self.scaled = scaled
        self.irreps_dim = self.irreps.dim
        self.irreps_info = irreps_info(self.irreps) # Use imported irreps_info

    def forward(self, input_tensor: Tensor) -> Tensor:
        r"""
        Args:
            input_tensor (torch.Tensor): Input tensor of shape ``(..., irreps.dim, C)``.
        Returns:
            torch.Tensor: Output tensor of shape ``(..., len(irreps), C)``.
        """
        if input_tensor.shape[-2] != self.irreps_dim:
            raise ValueError(
                f"Input tensor's irreps dimension ({input_tensor.shape[-2]}) "
                f"does not match expected irreps dimension ({self.irreps_dim})"
            )
        return norm(input_tensor, self.irreps_info, self.scaled) # Call imported norm

    def _apply(self, *args, **kwargs):
        # Apply the new _apply method style
        n = super()._apply(*args, **kwargs)
        if hasattr(n, 'irreps_info') and n.irreps_info is not None: # Check on 'n'
            n.irreps_info = n.irreps_info._apply(*args, **kwargs)
        return n

    def __extra_repr__(self): # Changed from __repr__
        return f"irreps='{self.irreps}', scaled={self.scaled}"


class MeanSquaredNorm(nn.Module):
    r"""Computes the mean of squared L2 norms over a specified dimension (batch or channel)
    for each irrep block.

    If ``dim=0`` (batch mean):

    .. math::

        \text{Output}_{ic} = \frac{1}{N} \sum_n \left( \sum_{m \in \text{irrep}_i} (\text{input}_n(im)c^2) \right)

    If ``dim=-1`` (channel mean):

    .. math::

        \text{Output}_{ni} = \frac{1}{C} \sum_c \left( \sum_{m \in \text{irrep}_i} (\text{input}_n(im)c^2) \right)

    Optionally scales the output by :math:`1/\text{irrep}_i\text{.dim}`.

    Args:
        irreps (Irreps): Irreducible representations of the input tensor.
        scaled (bool, optional): If ``True``, scales the output of each ``irrep_i``
                                 by :math:`1/\text{dim}_\text{irrep_i}`.
                                 Defaults to ``True``.
        dim (int, optional): Dimension over which to compute the mean.
                             Allowed values: ``0`` (batch), ``-1`` or ``2`` (channel).
                             Defaults to ``-1``.
    """
    irreps_info: IrrepsInfo

    def __init__(self, irreps: Irreps, scaled: bool = True, dim: Literal[0, -1, 2] = -1):
        super().__init__()
        self.irreps = check_irreps(irreps)
        self.scaled = scaled
        
        # Validate and store dim
        if dim not in [0, -1, 2]:
            raise ValueError(f"Unsupported dim value: {dim}. Allowed values are 0, -1, 2.")
        # Map 2 to -1 for internal consistency
        self.dim = dim if dim != 2 else -1 
            
        self.irreps_dim = self.irreps.dim
        self.irreps_info = irreps_info(self.irreps)

    def forward(self, input_tensor: Tensor) -> Tensor:
        r"""
        Args:
            input_tensor (torch.Tensor): Input tensor of shape ``(N, irreps.dim, C)``.
        Returns:
            torch.Tensor: Output tensor. If ``dim=0``, shape is ``(len(irreps), C)``.
                          If ``dim=-1`` or ``dim=2``, shape is ``(N, len(irreps))``.
        """
        if input_tensor.shape[-2] != self.irreps_dim:
            raise ValueError(
                f"Input tensor's irreps dimension ({input_tensor.shape[-2]}) "
                f"does not match expected irreps dimension ({self.irreps_dim})"
            )
            
        if self.dim == 0:
            # Mean over batch dimension
            return batch_mean_squared_norm(input_tensor, self.irreps_info, self.scaled)
        elif self.dim == -1:
            # Mean over channel dimension
            return channel_mean_squared_norm(input_tensor, self.irreps_info, self.scaled)
        else:
             # Should not happen due to __init__ validation, but added for safety
             raise ValueError(f"Invalid internal dim value: {self.dim}")

    def _apply(self, *args, **kwargs):
        n = super()._apply(*args, **kwargs)
        # Ensure irreps_info is also processed
        if hasattr(n, 'irreps_info') and n.irreps_info is not None: 
            n.irreps_info = n.irreps_info._apply(*args, **kwargs)
        return n

    def __extra_repr__(self):
        # Include dim in the representation
        return f"irreps='{self.irreps}', scaled={self.scaled}, dim={self.dim}"
