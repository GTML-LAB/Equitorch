import torch
from torch import nn, Tensor

from ..irreps import Irreps, check_irreps
from ..structs import IrrepsInfo
from ..utils._structs import irreps_info
from .functional.normalization import batch_rms_norm, layer_rms_norm

class BatchRMSNorm(nn.Module):
    r"""
    Applies Batch Root Mean Square Normalization for equivariant features.

    .. math::
        x'_{nimc} = \gamma_{ic} \cdot (x_{nimc} / \sigma_{ic})

    where

    .. math::
        \sigma_{ic} = \sqrt{E[\text{SquaredNorm}(x_{nic})] + \epsilon}

    The :class:`~equitorch.nn.norm.SquaredNorm` can be scaled by \(1/\text{irrep}_i\text{.dim}\)
    depending on the ``scaled`` argument.
    Running statistics are used during evaluation.

    Args:
        irreps (Irreps): Irreducible representations of the input tensor.
        channels (int): Number of channels in the input tensor (size of the last dimension).
        eps (float, optional): A value added to the denominator for numerical stability. \(\epsilon\)
                               Defaults to 1e-5.
        momentum (float, optional): The value used for the running_mean computation.
                                    Defaults to 0.1.
        affine (bool, optional): If True, this module has learnable affine parameters (weight \(\gamma_{ic}\)).
                                 Defaults to True.
        scaled (bool, optional): If True, the :class:`~equitorch.nn.norm.SquaredNorm` used for calculating statistics
                                 is scaled by \(1/\text{irrep}_i\text{.dim}\). Defaults to True.
    """
    irreps_info: IrrepsInfo

    def __init__(
        self,
        irreps: Irreps,
        channels: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        scaled: bool = True, 
    ):
        super().__init__()
        self.irreps = check_irreps(irreps)
        self.channels = channels
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.scaled = scaled

        self.irreps_dim = self.irreps.dim
        self.num_irreps = len(self.irreps)

        # Store IrrepsInfo. It will be moved to device/dtype by _apply
        self.irreps_info = irreps_info(self.irreps)

        self.register_buffer(
            "running_squared_norm", torch.ones(self.num_irreps, self.channels)
        )
        if self.affine:
            self.weight = nn.Parameter(torch.ones(self.num_irreps, self.channels))
        else:
            self.register_parameter("weight", None)
        
        self.reset_running_stats() # Initialize running_squared_norm

    def reset_running_stats(self) -> None:
        self.running_squared_norm.fill_(1.0)

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine and self.weight is not None:
            nn.init.ones_(self.weight)

    def forward(self, input_tensor: Tensor) -> Tensor:
        r"""
        Args:
            input_tensor (Tensor): Shape (N, irreps_dim, C)
        Returns:
            Tensor: Shape (N, irreps_dim, C)
        """
        if input_tensor.ndim != 3:
            raise ValueError(
                f"Input tensor must be 3D (N, irreps_dim, C), got {input_tensor.ndim}D. Shape: {input_tensor.shape}"
            )
        if input_tensor.shape[-2] != self.irreps_dim:
            raise ValueError(
                f"Input tensor's irreps dimension ({input_tensor.shape[-2]}) "
                f"does not match expected irreps dimension ({self.irreps_dim})"
            )
        if input_tensor.shape[-1] != self.channels:
            raise ValueError(
                f"Input tensor's channel dimension ({input_tensor.shape[-1]}) "
                f"does not match expected channels ({self.channels})"
            )

        # irreps_info is handled by _apply.
        return batch_rms_norm(
            input_tensor,
            self.running_squared_norm,
            self.irreps_info,
            self.weight,
            scaled=self.scaled,
            training=self.training,
            momentum=self.momentum,
            eps=self.eps,
        )

    def _apply(self, *args, **kwargs):
        # This method is called by PyTorch (e.g., when .cuda() or .to() is called).
        # It ensures that IrrepsInfo is also moved to the correct device/dtype.
        n = super()._apply(*args, **kwargs)
        if hasattr(n, 'irreps_info') and n.irreps_info is not None:
            n.irreps_info = n.irreps_info._apply(*args, **kwargs)
        # running_squared_norm (buffer) and weight (parameter) are handled by super()._apply()
        return n

    def __extra_repr__(self):
        return (
            f"irreps='{self.irreps}', channels={self.channels}, "
            f"eps={self.eps}, momentum={self.momentum}, affine={self.affine}, "
            f"scaled={self.scaled}"
        )


class LayerRMSNorm(nn.Module):
    r"""
    Applies Irrep-wise Layer Root Mean Square Normalization.

    Computes statistics independently for each irrep instance within each sample.
    Normalizes using the RMS value calculated across channels and irrep components
    for that specific sample and irrep instance.

    Args:
        irreps (Irreps): Irreducible representations of the input tensor.
        channels (int): Number of channels in the input tensor (size of the last dimension).
        eps (float, optional): A value added to the denominator for numerical stability. \(\epsilon\)
                               Defaults to 1e-5.
        affine (bool, optional): If True, this module has learnable affine parameters (weight \(\gamma_{ic}\)).
                                 Defaults to True.
        scaled (bool, optional): If True, the statistics calculation considers the norm
                                 to be scaled by \(1/\text{irrep}_i\text{.dim}\).
                                 Defaults to True.
    """
    irreps_info: IrrepsInfo

    def __init__(
        self,
        irreps: Irreps,
        channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        scaled: bool = True, # Corresponds to 'scaled' in functional.layer_rms_norm
    ):
        super().__init__()
        self.irreps = check_irreps(irreps)
        self.channels = channels
        self.eps = eps
        self.affine = affine
        self.scaled = scaled

        self.irreps_dim = self.irreps.dim
        self.num_irreps = len(self.irreps)

        # Store IrrepsInfo. It will be moved to device/dtype by _apply
        self.irreps_info = irreps_info(self.irreps)

        if self.affine:
            self.weight = nn.Parameter(torch.ones(self.num_irreps, self.channels))
        else:
            self.register_parameter("weight", None)
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine and self.weight is not None:
            nn.init.ones_(self.weight)

    def forward(self, input_tensor: Tensor) -> Tensor:
        r"""
        Args:
            input_tensor (Tensor): Shape (N, irreps_dim, C)
        Returns:
            Tensor: Shape (N, irreps_dim, C)
        """
        if input_tensor.ndim != 3:
            raise ValueError(
                f"Input tensor must be 3D (N, irreps_dim, C), got {input_tensor.ndim}D. Shape: {input_tensor.shape}"
            )
        if input_tensor.shape[-2] != self.irreps_dim:
            raise ValueError(
                f"Input tensor's irreps dimension ({input_tensor.shape[-2]}) "
                f"does not match expected irreps dimension ({self.irreps_dim})"
            )
        if input_tensor.shape[-1] != self.channels:
            raise ValueError(
                f"Input tensor's channel dimension ({input_tensor.shape[-1]}) "
                f"does not match expected channels ({self.channels})"
            )

        # Prepare weight tensor for functional call
        weight_tensor: Tensor
        if self.affine:
            weight_tensor = self.weight
        else:
            # Functional LayerRMSNorm.forward handles weight is None by setting output = normed.
            # So, if not affine, we should pass None.
            weight_tensor = None

        # irreps_info is handled by _apply.
        return layer_rms_norm( # This calls LayerRMSNorm.apply from functional/normalization.py
            input_tensor,
            self.irreps_info,
            weight_tensor, # Pass the potentially None weight
            scaled=self.scaled,
            eps=self.eps,
        )

    def _apply(self, *args, **kwargs):
        # This method is called by PyTorch (e.g., when .cuda() or .to() is called).
        # It ensures that IrrepsInfo is also moved to the correct device/dtype.
        n = super()._apply(*args, **kwargs)
        if hasattr(n, 'irreps_info') and n.irreps_info is not None:
            n.irreps_info = n.irreps_info._apply(*args, **kwargs)
        # weight (parameter) is handled by super()._apply()
        return n

    def __extra_repr__(self):
        return (
            f"irreps='{self.irreps}', channels={self.channels}, "
            f"eps={self.eps}, affine={self.affine}, "
            f"scaled={self.scaled}"
        )
