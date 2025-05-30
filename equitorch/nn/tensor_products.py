import math
import torch
import torch.nn as nn

from torch.nn import Parameter

from .functional.tensor_products import (
    tensor_product_uuu,
    tensor_product_uvw,
    tensor_dot_uu,
    tensor_dot_uv
)
from ..irreps import check_irreps
from .init import initialize_tensor_product
from ..utils._structs import irreps_info, tp_infos
from ..structs import IrrepsInfo, TensorProductInfo
from ..irreps import Irreps

class TensorProduct(nn.Module):
    r"""Computes the tensor product of two equivariant feature tensors.

    Supports two main modes controlled by ``feature_mode``:

    - ``'uvw'``: Fully connected tensor product.

      - Input1 shape: ``(..., irreps_in1.dim, channels_in1)``
      - Input2 shape: ``(..., irreps_in2.dim, channels_in2)``
      - Weight shape: ``(num_paths, channels_in1, channels_in2, channels_out)``
      - Output shape: ``(..., irreps_out.dim, channels_out)``

    - ``'uuu'``: Depthwise/elementwise tensor product.
      with ``uuu`` instructions (often used for self-interaction).

      - Input1 shape: ``(..., irreps_in1.dim, channels)``
      - Input2 shape: ``(..., irreps_in2.dim, channels)``
      - Weight shape: ``(num_paths, channels_out)`` (where ``channels_out`` usually equals ``channels``)
      - Output shape: ``(..., irreps_out.dim, channels_out)``

    Args:
        irreps_in1 (Irreps): Irreducible representations of the first input tensor.
        irreps_in2 (Irreps): Irreducible representations of the second input tensor.
        irreps_out (Irreps): Irreducible representations of the output tensor.
        channels_in1 (int, optional): Number of channels for the first input.
            Required if ``internal_weights=True`` or ``feature_mode='uvw'``.
        channels_in2 (int, optional): Number of channels for the second input.
            Required if ``internal_weights=True`` or ``feature_mode='uvw'``.
        channels_out (int, optional): Number of channels for the output.
            Required if ``internal_weights=True``.
        internal_weights (bool, default=True): If ``True``, the module manages its own weight parameter.
            If ``False``, weights must be provided during the forward pass.
        feature_mode ({'uuu', 'uvw'}, default='uuu'): Controls the type of tensor product:

            - ``'uuu'``: Depthwise/elementwise product. Assumes ``channels_in1 == channels_in2 == channels_out``.
            - ``'uvw'``: Fully connected product.
        path_norm (bool, default=True): Whether to apply path normalization to the weights.
        channel_norm (bool, default=False): Whether to apply channel normalization (specific to ``'uvw'`` mode).
            Divides weights by :math:`\sqrt{\text{channels_in1} \times \text{channels_in2}}`.
        path (list, optional): Manually specify the coupling paths.
            If ``None``, all allowed paths are used.

    Attributes:
        weight (torch.nn.Parameter or None): The learnable weights of the module if ``internal_weights=True``.
            Shape depends on ``feature_mode``.
        tp_info_forward (TensorProductInfo): Constant information for the forward pass computation.
        tp_info_backward1 (TensorProductInfo): Constant information for the backward pass w.r.t. input1.
        tp_info_backward2 (TensorProductInfo): Constant information for the backward pass w.r.t. input2.
        num_paths (int): Number of coupling paths determined by the irreps.
        weight_numel (int): Total number of elements in the weight tensor.
    """
    tp_info_forward: TensorProductInfo
    tp_info_backward1: TensorProductInfo
    tp_info_backward2: TensorProductInfo

    def __init__(self, 
                 irreps_in1,
                 irreps_in2, 
                 irreps_out, 
                 channels_in1=None,
                 channels_in2=None,
                 channels_out=None,
                 internal_weights=True,
                 feature_mode='uuu',
                 path_norm=True,
                 channel_norm=False, # Add channel_norm parameter
                 path=None):

        super().__init__()

        self.irreps_in1 = check_irreps(irreps_in1)
        self.irreps_in2 = check_irreps(irreps_in2)
        self.irreps_out = check_irreps(irreps_out)
        self.irreps_in1_dim = self.irreps_in1.dim
        self.irreps_in2_dim = self.irreps_in2.dim
        self.irreps_out_dim = self.irreps_out.dim

        assert not internal_weights or (
            channels_in1 is not None and
            channels_in2 is not None and
            channels_out is not None
        )
        self.channels_in1 = channels_in1
        self.channels_in2 = channels_in2
        self.channels_out = channels_out
        self.feature_mode = feature_mode
        self.path_norm = path_norm
        self.channel_norm = channel_norm and (self.feature_mode == 'uvw') # Apply only for uvw

        # Calculate fan_in for channel normalization
        fan_in = self.channels_in1 * self.channels_in2 if self.feature_mode == 'uvw' else 1

        (self.tp_info_forward, 
          self.tp_info_backward1,
          self.tp_info_backward2,
          self.num_paths) = tp_infos(
              self.irreps_out,
              self.irreps_in1,
              self.irreps_in2, path=path, path_norm=path_norm,
              channel_norm=self.channel_norm, channel_scale=fan_in**(-0.5) # Pass channel_norm and fan_in
          )

        if self.feature_mode == 'uvw':
            self.weight_shape = (self.num_paths, self.channels_in1, self.channels_in2, self.channels_out)
        elif self.feature_mode == 'uuu':
            self.weight_shape = (self.num_paths, self.channels_out)
        else:
            raise ValueError(f'Feature_mode should be in ["uvw", "uuu"], got {feature_mode}.')
        self.weight_numel = math.prod(self.weight_shape)

        self.internal_weights = internal_weights
        if internal_weights:
            self.weight = Parameter(torch.empty(*self.weight_shape))
            # Pass channel_norm status to initializer
            initialize_tensor_product(self.weight, self.feature_mode, channel_normed=self.channel_norm)
        else:
            self.weight = None

    def forward(self, input1: torch.Tensor, input2: torch.Tensor, weight: torch.Tensor = None) -> torch.Tensor:
        # Input shape checks
        assert input1.shape[-2] == self.irreps_in1_dim, f"Input1 spherical dim mismatch: expected {self.irreps_in1_dim}, got {input1.shape[-2]}"
        assert input2.shape[-2] == self.irreps_in2_dim, f"Input2 spherical dim mismatch: expected {self.irreps_in2_dim}, got {input2.shape[-2]}"
        if self.feature_mode == 'uvw':
            assert input1.shape[-1] == self.channels_in1, f"Input1 channel dim mismatch: expected {self.channels_in1}, got {input1.shape[-1]}"
            assert input2.shape[-1] == self.channels_in2, f"Input2 channel dim mismatch: expected {self.channels_in2}, got {input2.shape[-1]}"
        elif self.feature_mode == 'uuu':
            assert input1.shape[-1] == input2.shape[-1] == self.channels_in1, f"Input1 channel dim mismatch: expected {self.channels_in1}, got {input1.shape[-1]}"

        if self.internal_weights:
            assert weight is None, 'Do not pass the weight when self.internal_weights is True.'
            weight = self.weight
        else:
            assert weight is not None, 'Please pass the weight when self.internal_weights is False.'
            if weight.numel() > self.weight_numel:
                weight = weight.view(-1, *self.weight_shape)
            else:
                weight = weight.view(*self.weight_shape)

        # Prepare args for functional call
        args = (input1, input2, weight,
                self.tp_info_forward,
                self.tp_info_backward1,
                self.tp_info_backward2)

        # Call functional implementation
        if self.feature_mode == 'uuu':
            output = tensor_product_uuu(*args)
        elif self.feature_mode == 'uvw':
            output = tensor_product_uvw(*args)
        else:
             # This case is already handled in __init__, but added for safety
             raise ValueError(f"Invalid feature_mode: {self.feature_mode}")
        return output

    def _apply(self, *args, **kwargs):
        tp = super()._apply(*args, **kwargs)
        tp.tp_info_forward = self.tp_info_forward._apply(*args, **kwargs)
        tp.tp_info_backward1 = self.tp_info_backward1._apply(*args, **kwargs)
        tp.tp_info_backward2 = self.tp_info_backward2._apply(*args, **kwargs)
        return tp

    def __repr__(self):
        channels_repr = ""
        if self.feature_mode == 'uvw':
            channels_repr = f"channels_in1={self.channels_in1}, channels_in2={self.channels_in2}, channels_out={self.channels_out}"
        elif self.feature_mode == 'uuu':
             # Assuming channels_in1 == channels_in2 == channels_out for uuu mode based on docstring
             channels_repr = f"channels={self.channels_in1}" # Use channels_in1 as the representative channel count
        
        return (f"{self.__class__.__name__}("
                f"irreps_in1={self.irreps_in1.short_repr()}, "
                f"irreps_in2={self.irreps_in2.short_repr()}, "
                f"irreps_out={self.irreps_out.short_repr()}, "
                f"{channels_repr}, "
                f"feature_mode={self.feature_mode}, "
                f"path_norm={self.path_norm}, "
                f"channel_norm={self.channel_norm}, "
                f"internal_weights={self.internal_weights}, "
                f"num_paths={self.num_paths}"
                ")")
class TensorDot(nn.Module):
    r"""Computes the equivariant irrep-wise dot product of two feature tensors.

    Supports two main modes controlled by ``feature_mode``:

    - ``'uv'``: Channel-cartesian dot product.

      - Input1 shape: ``(..., irreps.dim, channels1)``
      - Input2 shape: ``(..., irreps.dim, channels2)``
      - Output shape: ``(..., len(irreps), channels1, channels2)``

    - ``'uu'``: Channel-wise dot product. Sums over the channel dimension after the dot product.

      - Input1 shape: ``(..., irreps.dim, channels)``
      - Input2 shape: ``(..., irreps.dim, channels)``
      - Output shape: ``(..., len(irreps), channels)``

    Args:
        irreps (Irreps or str): Irreducible representations of the input tensors.
            Both inputs must have the same irreps.
        feature_mode ({'uv', 'uu'}): Controls how the channel dimension is handled:

            - ``'uv'``: Channel-cartesian dot product.
            - ``'uu'``: Channel-wise dot product.
        scaled (bool, default=False): If ``True``, scales the dot product by :math:`1 / \sqrt{\text{irrep.dim}}`.

    Attributes:
        irreps_info (IrrepsInfo): Constant information about the input irreps.
    """
    irreps_info: IrrepsInfo

    def __init__(self, irreps, feature_mode, scaled=False):

        super().__init__()

        self.irreps = check_irreps(irreps)
        self.irreps_dim = self.irreps.dim

        self.irreps_info = irreps_info(self.irreps)

        assert feature_mode in ['uv', 'uu']
        self.feature_mode = feature_mode

        self.scaled = scaled

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        # Input shape checks
        assert input1.shape[-2] == self.irreps_dim, f"Input1 spherical dim mismatch: expected {self.irreps_dim}, got {input1.shape[-2]}"
        assert input2.shape[-2] == self.irreps_dim, f"Input2 spherical dim mismatch: expected {self.irreps_dim}, got {input2.shape[-2]}"

        if self.feature_mode == 'uu':
            assert input1.shape[-1] == input2.shape[-1], f"Input channel dims must match in 'uv' mode. Got {input1.shape[-1]} and {input2.shape[-1]}"
            # Inputs should not have channel dim
            output = tensor_dot_uu(input1, input2, self.irreps_info, self.scaled)

        elif self.feature_mode == 'uv':
            # Inputs should have channel dim, and they must match
            output = tensor_dot_uv(input1, input2, self.irreps_info, self.scaled)
            # Output shape check: should have same channel dim, but no spherical dim
        else:
             # This case is already handled in __init__, but added for safety
             raise ValueError(f"Invalid feature_mode: {self.feature_mode}")

        return output

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"irreps={self.irreps.short_repr()}, "
                f"feature_mode={self.feature_mode}, "
                f"scaled={self.scaled}"
                ")")
