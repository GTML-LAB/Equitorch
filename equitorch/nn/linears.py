import math
from typing import Optional
import torch
import torch.nn as nn

from .functional.linears import (
    so3_linear_uu,
    so3_linear_uv,
    irrep_wise_linear,
    irreps_linear
)
from .functional.sparse_product import (
    sparse_mul,
    sparse_vecmat
)
from ..irreps import check_irreps
from .init import initialize_linear, initialize_so3_so2_linear
from ..utils._structs import irreps_info, irreps_linear_infos, tp_infos, so2_linear_infos

from torch import Tensor
from torch.nn import Parameter

from ..structs import IrrepsInfo, IrrepsLinearInfo, TensorProductInfo
from ..irreps import Irreps

class SO3Linear(nn.Module):
    r"""
    SO(3) equivariant linear layer using tensor products.

    Equivalent to a TensorProduct where  ``input2`` does not have a channel dimension.

    Supports two main modes controlled by ``feature_mode``:

    - ``'uv'``: Fully connected in channel dimension.

      - Input1 shape: ``(..., irreps_in1.dim, channels_in)``
      - Input2 shape: ``(..., irreps_in2.dim)``
      - Weight shape: ``(num_paths, channels_in, channels_out)``
      - Output shape: ``(..., irreps_out.dim, channels_out)``

    - ``'uu'``: Depthwise/elementwise in channel dimension.

      - Input1 shape: ``(..., irreps_in1.dim, channels)``
      - Input2 shape: ``(..., irreps_in2.dim)``
      - Weight shape: ``(num_paths, channels)`` 
      - Output shape: ``(..., irreps_out.dim, channels)``

    Args:
        irreps_in1 (Irreps): Irreducible representations of the main input tensor (``input1``).
        irreps_in2 (Irreps): Irreducible representations of the second input tensor (``input2``),
            often representing weights like spherical harmonics.
        irreps_out (Irreps): Irreducible representations of the output tensor.
        channels_in (int, optional): Number of channels for the main input (``input1``).
            Required if ``internal_weights=True``.
        channels_out (int, optional): Number of channels for the output.
            Required if ``internal_weights=True``.
        internal_weights (bool, optional): If ``True``, the module manages its own weight parameter.
            If ``False``, weights must be provided during the forward pass. Defaults to ``True``.
        feature_mode (str, optional): Controls the type of linear operation: ``{'uu', 'uv'}``.
            Defaults to ``'uu'``.

            - ``'uu'``: Depthwise/elementwise linear. Assumes ``channels_in == channels_out``.
            - ``'uv'``: Fully connected linear.
        path_norm (bool, optional): Whether to apply path normalization to the weights.
            Normalizes by the square root of the number of paths to each output irrep. Defaults to ``True``.
        channel_norm (bool, optional): Whether to apply channel normalization (specific to ``'uv'`` mode).
            Divides weights by \(\sqrt{\text{channels_in}}\). Note: This interacts with ``path_norm``.
            Defaults to ``False``.
        path (list, optional): Manually specify the coupling paths.
            If ``None``, all allowed paths are used. Defaults to ``None``.

    Attributes:
        weight (torch.nn.Parameter or None): The learnable weights of the module if ``internal_weights=True``.
            Shape depends on ``feature_mode``.
        tp_info_forward (TensorProductInfo): Constant information for the forward pass computation.
        tp_info_backward1 (TensorProductInfo): Constant information for the backward pass w.r.t. ``input1``.
        tp_info_backward2 (TensorProductInfo): Constant information for the backward pass w.r.t. ``input2``.
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
                 channels_in=None,
                 channels_out=None,
                 internal_weights=True,
                 feature_mode='uu',
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
            channels_in is not None and
            channels_out is not None
        )
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.feature_mode = feature_mode
        self.path_norm = path_norm
        self.channel_norm = channel_norm and (self.feature_mode == 'uv') # Apply only for uv

        # Calculate fan_in for channel normalization
        # For SO3Linear, irreps_in2 is the weight irreps (usually 0e), fan_in is channels_in
        fan_in = self.channels_in if self.feature_mode == 'uv' else 1

        (self.tp_info_forward, 
         self.tp_info_backward1,
         self.tp_info_backward2,
         self.num_paths) = tp_infos(
            self.irreps_out,
            self.irreps_in1,
            self.irreps_in2, path=path, path_norm=path_norm,
            channel_norm=self.channel_norm, channel_scale=fan_in**(-0.5)
        )

        if self.feature_mode == 'uv':
            self.weight_shape = (self.num_paths, self.channels_in, self.channels_out)
        elif self.feature_mode == 'uu':
            self.weight_shape = (self.num_paths, self.channels_out)
        else:
            raise ValueError(f'Feature_mode should be in ["uv", "uu"], got {feature_mode}.')
        self.weight_numel = math.prod(self.weight_shape)
        self.internal_weights = internal_weights
        if internal_weights:
            self.weight = Parameter(torch.empty(*self.weight_shape))
            # Pass channel_norm status to initializer
            initialize_so3_so2_linear(self.weight, self.feature_mode, channel_normed=self.channel_norm)
        else:
            self.weight = None

    def forward(self, input1:Tensor, input2:Tensor, weight:Optional[Tensor]=None):
        # Shape checks
        assert input1.shape[-2] == self.irreps_in1_dim, f"Input1 spherical dim mismatch: expected {self.irreps_in1_dim}, got {input1.shape[-2]}"
        assert input2.shape[-1] == self.irreps_in2_dim, f"Input2 spherical dim mismatch: expected {self.irreps_in2_dim}, got {input2.shape[-1]}"
        # input2 has no channel dimension, check it has one less dim than input1 or weight if provided
        # Check input2 has one less dimension than input1
        assert input2.ndim == input1.ndim - 1, f"Input2 should have one less dimension than Input1 in 'uv' mode. Got input1 ndim={input1.ndim}, input2 ndim={input2.ndim}"
        assert input1.shape[-1] == self.channels_in, f"Input1 channel dim mismatch: expected {self.channels_in}, got {input1.shape[-1]}"

        if self.internal_weights:
            assert weight is None, 'Do not pass the weight when self.internal_weights is True.'
            weight = self.weight
        else:
            assert weight is not None, 'Please pass the weight when self.internal_weights is False.'

            if weight.numel() > self.weight_numel:
                weight = weight.view(-1, *self.weight_shape)
            else:
                weight = weight.view(*self.weight_shape)
        # Call the appropriate functional implementation
        if self.feature_mode == 'uu':
            output = so3_linear_uu(
                input1, input2, weight,
                self.tp_info_forward,
                self.tp_info_backward1,
                self.tp_info_backward2
            )
        elif self.feature_mode == 'uv':
             output = so3_linear_uv(
                input1, input2, weight,
                self.tp_info_forward,
                self.tp_info_backward1,
                self.tp_info_backward2
            )
        else:
             # This case is already handled in __init__, but added for safety
             raise ValueError(f"Invalid feature_mode: {self.feature_mode}")

        # # Check output shape
        # assert output.shape[-2] == self.irreps_out_dim, f"Output spherical dim mismatch: expected {self.irreps_out_dim}, got {output.shape[-2]}"
        # assert output.shape[-1] == self.channels_out, f"Output channel dim mismatch: expected {self.channels_out}, got {output.shape[-1]}"

        return output

    def _apply(self, *args, **kwargs):
        tp = super()._apply(*args, **kwargs)
        tp.tp_info_forward = self.tp_info_forward._apply(*args, **kwargs)
        tp.tp_info_backward1 = self.tp_info_backward1._apply(*args, **kwargs)
        tp.tp_info_backward2 = self.tp_info_backward2._apply(*args, **kwargs)
        return tp

    def __repr__(self):
        channels_repr = f"channels_in={self.channels_in}, channels_out={self.channels_out}"
        if self.feature_mode == 'uu':
             # In 'uu' mode, channels_in and channels_out are expected to be the same
             channels_repr = f"channels={self.channels_in}"

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


class IrrepWiseLinear(nn.Module):
    r"""
    Irrep-wise linear layer (channel mixing).

    Applies a separate linear transformation to the channels associated with each irrep type.
    This operation does not change the spherical tensor structure (the irreps).

    - Input shape: ``(..., irreps.dim, channels_in)``
    - Weight shape: ``(num_paths, channels_in, channels_out)`` where ``num_paths`` is the number of unique irreps in ``irreps``.
    - Output shape: ``(..., irreps.dim, channels_out)``

    Args:
        irreps (Irreps or str): Irreducible representations of the input tensor.
        channels_in (int): Number of input channels.
        channels_out (int): Number of output channels.
        internal_weights (bool, optional): If ``True``, the module manages its own weight parameter.
            If ``False``, weights must be provided during the forward pass. Defaults to ``True``.
        channel_norm (bool, optional): If ``True``, divides the output by \(\sqrt{\text{channels_in}}\).
            Defaults to ``False``.

    Attributes:
        weight (torch.nn.Parameter or None): The learnable weights of the module if ``internal_weights=True``.
        irreps_info (IrrepsInfo): Constant information about the input irreps.
        num_paths (int): Number of unique irreps in the input.
        weight_numel (int): Total number of elements in the weight tensor.
    """
    irreps_info: IrrepsInfo

    def __init__(self, irreps, channels_in, channels_out,
                 internal_weights=True, channel_norm: bool = False):

        super().__init__()

        self.irreps = check_irreps(irreps)
        self.irreps_dim = self.irreps.dim
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.irreps_info = irreps_info(self.irreps)
        self.channel_norm = channel_norm

        self.num_paths = len(self.irreps)
        self.weight_shape = (self.num_paths, self.channels_in, self.channels_out)
        self.weight_numel = math.prod(self.weight_shape)

        self.internal_weights = internal_weights

        if internal_weights:
            self.weight = Parameter(torch.empty(*self.weight_shape))
            initialize_linear(self.weight, channel_normed=self.channel_norm)
        else:
            self.weight = None

    def forward(self, input: torch.Tensor, weight:Optional[torch.Tensor] = None):
        # Shape checks
        assert input.shape[-2] == self.irreps_dim, f"Input spherical dim mismatch: expected {self.irreps_dim}, got {input.shape[-2]}"
        assert input.shape[-1] == self.channels_in, f"Input channel dim mismatch: expected {self.channels_in}, got {input.shape[-1]}"

        if self.internal_weights:
            assert weight is None, 'Do not pass the weight when self.internal_weights is True.'
            weight = self.weight
        else:
            assert weight is not None, 'Please pass the weight when self.internal_weights is False.'
            if weight.numel() > self.weight_numel:
                weight = weight.view(-1, *self.weight_shape)
            else:
                weight = weight.view(*self.weight_shape)

        # Call functional implementation
        output = irrep_wise_linear(input, weight, self.irreps_info)
        if self.channel_norm:
             output = output / (self.channels_in ** 0.5)

        return output

    def _apply(self, *args, **kwargs):
        tp = super()._apply(*args, **kwargs)
        tp.irreps_info = self.irreps_info._apply(*args, **kwargs)
        return tp

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"irreps={self.irreps.short_repr()}, "
                f"channels_in={self.channels_in}, "
                f"channels_out={self.channels_out}, "
                f"channel_norm={self.channel_norm}, "
                f"internal_weights={self.internal_weights}"
                ")")

class IrrepsLinear(nn.Module):
    r"""
    Equivariant linear layer that preserves the spherical tensor structure but mixes channels.

    This layer applies a linear transformation across channels while respecting the
    equivariance constraints imposed by the input and output irreps. It only allows
    paths where the input and output irreps are the same (\(l_{in} = l_{out}\) and (\(p_{in} = p_{out}\) or \(p_{out}=0\))).
    This is often used for channel mixing in equivariant networks.

    - Input shape: ``(..., irreps_in.dim, channels_in)``
    - Weight shape: ``(num_paths, channels_in, channels_out)`` where ``num_paths`` is the number of allowed paths.
    - Output shape: ``(..., irreps_out.dim, channels_out)``

    Args:
        irreps_in (Irreps): Irreducible representations of the input tensor.
        irreps_out (Irreps): Irreducible representations of the output tensor.
        channels_in (int): Number of input channels.
        channels_out (int): Number of output channels.
        internal_weights (bool, optional): If ``True``, the module manages its own weight parameter.
            If ``False``, weights must be provided during the forward pass. Defaults to ``True``.
        path_norm (bool, optional): Whether to apply path normalization to the weights.
            Normalizes by the square root of the number of paths to each output irrep. Defaults to ``True``.
        channel_norm (bool, optional): Whether to apply channel normalization.
            Divides weights by \(\sqrt{\text{channels_in}}\). 
            Defaults to ``False``.
        path (list, optional): Manually specify the coupling paths.
            If ``None``, all allowed paths are used. Defaults to ``None``.

    Attributes:
        weight (torch.nn.Parameter or None): The learnable weights of the module if ``internal_weights=True``.
        forward_info (IrrepsLinearInfo): Constant information for the forward pass computation.
        backward_info (IrrepsLinearInfo): Constant information for the backward pass computation.
        num_paths (int): Number of allowed coupling paths.
        weight_numel (int): Total number of elements in the weight tensor.
    """
    forward_info: IrrepsLinearInfo
    backward_info: IrrepsLinearInfo

    def __init__(self, 
                 irreps_in, irreps_out, 
                 channels_in, channels_out,
                 internal_weights=True,
                 path_norm=True,
                 channel_norm=False,
                 path=None):

        super().__init__()

        self.irreps_in = check_irreps(irreps_in)
        self.irreps_out = check_irreps(irreps_out)
        self.irreps_in_dim = self.irreps_in.dim
        self.irreps_out_dim = self.irreps_out.dim
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.channel_norm = channel_norm

        self.path_norm = path_norm
        
        self.forward_info, self.backward_info, self.num_paths = irreps_linear_infos(
            self.irreps_out, self.irreps_in, path, path_norm, channel_norm, self.channels_in)
        
        self.weight_shape = (self.num_paths, self.channels_in, self.channels_out)
        self.weight_numel = math.prod(self.weight_shape)

        self.internal_weights = internal_weights
        
        if internal_weights:
            self.weight = Parameter(torch.empty(*self.weight_shape))
            initialize_linear(self.weight, channel_normed=channel_norm)
        else:
            self.weight = None

    def forward(self, input: torch.Tensor, weight:Optional[torch.Tensor] = None):
        # Shape checks
        assert input.shape[-2] == self.irreps_in_dim, f"Input spherical dim mismatch: expected {self.irreps_in_dim}, got {input.shape[-2]}"
        assert input.shape[-1] == self.channels_in, f"Input channel dim mismatch: expected {self.channels_in}, got {input.shape[-1]}"

        if self.internal_weights:
            assert weight is None, 'Do not pass the weight when self.internal_weights is True.'
            weight = self.weight
        else:
            assert weight is not None, 'Please pass the weight when self.internal_weights is False.'
            if weight.numel() > self.weight_numel:
                weight = weight.view(-1, *self.weight_shape)
            else:
                weight = weight.view(*self.weight_shape)

        # Call functional implementation
        output = irreps_linear(input, weight,
                             self.forward_info, self.backward_info)
        return output

    def _apply(self, *args, **kwargs):
        tp = super()._apply(*args, **kwargs)
        tp.forward_info = self.forward_info._apply(*args, **kwargs)
        tp.backward_info = self.backward_info._apply(*args, **kwargs)
        return tp

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"irreps_in={self.irreps_in.short_repr()}, "
                f"irreps_out={self.irreps_out.short_repr()}, "
                f"channels_in={self.channels_in}, "
                f"channels_out={self.channels_out}, "
                f"path_norm={self.path_norm}, "
                f"channel_norm={self.channel_norm}, "
                f"internal_weights={self.internal_weights}, "
                f"num_paths={self.num_paths}"
                ")")

class SO2Linear(nn.Module):
    r"""
    SO(2) equivariant linear layer using tensor products.

    This layer applies an SO(2) equivariant linear transformation, as proposed in `Reducing SO(3) Convolutions to SO(2) for Efficient Equivariant GNNs <https://arxiv.org/abs/2302.03655>`_.
    It supports two main modes controlled by ``feature_mode``:

    - ``'uv'``: Fully connected linear layer.

      - Input shape: ``(..., irreps_in.dim, channels_in)``
      - Weight shape: ``(num_weights, channels_in, channels_out)``
      - Output shape: ``(..., irreps_out.dim, channels_out)``

    - ``'uu'``: Depthwise/elementwise linear layer.

      - Input shape: ``(..., irreps_in.dim, channels)``
      - Weight shape: ``(num_weights, channels_out)``
      - Output shape: ``(..., irreps_out.dim, channels_out)``

    Args:
        irreps_in (Irreps or str): Irreducible representations of the input tensor.
        irreps_out (Irreps or str): Irreducible representations of the output tensor.
        channels_in (int, optional): Number of channels for the input.
            Required if ``internal_weights=True``.
        channels_out (int, optional): Number of channels for the output.
            Required if ``internal_weights=True``.
        internal_weights (bool, optional): If ``True``, the module manages its own weight parameter.
            If ``False``, weights must be provided during the forward pass. Defaults to ``True``.
        feature_mode (str, optional): Controls the type of linear operation: ``{'uu', 'uv'}``.
            Defaults to ``'uu'``.

            - ``'uu'``: Depthwise/elementwise linear. Assumes ``channels_in == channels_out``.
            - ``'uv'``: Fully connected linear.
        path_norm (bool, optional): Whether to apply path normalization to the weights.
            Normalizes by the square root of the number of paths to each output irrep. Defaults to ``True``.
        channel_norm (bool, optional): Whether to apply channel normalization (specific to ``'uv'`` mode).
            Divides weights by \(\sqrt{\text{channels_in}}\). Note: This interacts with ``path_norm``.
            Defaults to ``False``.
        path (list, optional): Manually specify the coupling paths.
            If ``None``, all allowed paths are used. Defaults to ``None``.

    Attributes:
        weight (torch.nn.Parameter or None): The learnable weights of the module if ``internal_weights=True``.
            Shape depends on ``feature_mode``.
        info_forward (SparseProductInfo): Constant information for the forward pass computation.
        info_backward1 (SparseProductInfo): Constant information for the first backward pass.
        info_backward2 (SparseProductInfo): Constant information for the second backward pass.
        num_paths (int): Number of coupling paths determined by the irreps.
        weight_numel (int): Total number of elements in the weight tensor.
    """

    def __init__(self,
                 irreps_in, irreps_out,
                 channels_in=None, 
                 channels_out=None,
                 internal_weights=True,
                 feature_mode='uu',
                 path_norm=True,
                 channel_norm=False, 
                 path=None):

        super().__init__()

        self.irreps_in = check_irreps(irreps_in)
        self.irreps_out = check_irreps(irreps_out)
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.internal_weights = internal_weights
        self.feature_mode = feature_mode
        self.path_norm = path_norm
        self.channel_norm = channel_norm

        assert not internal_weights or (
            channels_in is not None and
            channels_out is not None
        )
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.feature_mode = feature_mode
        self.path_norm = path_norm
        self.channel_norm = channel_norm and (self.feature_mode == 'uv') # Apply only for uv

        fan_in = self.channels_in if self.feature_mode == 'uv' else 1
        fan_out = self.channels_out if self.feature_mode == 'uv' else 1

        channel_scale = fan_in**(-0.5) if self.channel_norm and self.feature_mode =='uv' else 1.0

        (self.info_forward, 
         self.info_backward1,
         self.info_backward2,
         self.num_weights) = so2_linear_infos(
            self.irreps_out,
            self.irreps_in, path=path, path_norm=path_norm,
            channel_norm=self.channel_norm, channel_scale=channel_scale
        )

        if self.feature_mode == 'uv':
            self.weight_shape = (self.num_weights, self.channels_in, self.channels_out)
        elif self.feature_mode == 'uu':
            self.weight_shape = (self.num_weights, self.channels_out)
        else:
            raise ValueError(f'Feature_mode should be in ["uv", "uu"], got {feature_mode}.')

        self.weight_numel = math.prod(self.weight_shape)

        if internal_weights:
            self.weight = Parameter(torch.empty(*self.weight_shape))
            # Pass channel_norm status to initializer
            initialize_so3_so2_linear(self.weight, self.feature_mode, channel_normed=self.channel_norm)
        else:
            self.weight = None

    def forward(self, input: torch.Tensor, weight:Optional[torch.Tensor] = None):
        # Shape checks
        assert input.shape[-2] == self.irreps_in.dim, f"Input spherical dim mismatch: expected {self.irreps_in.dim}, got {input.shape[-2]}"
        assert input.shape[-1] == self.channels_in, f"Input channel dim mismatch: expected {self.channels_in}, got {input.shape[-1]}"
        if self.internal_weights:
            assert weight is None, 'Do not pass the weight when self.internal_weights is True.'
            weight = self.weight
        else:
            assert weight is not None, 'Please pass the weight when self.internal_weights is False.'
            if weight.numel() > self.weight_numel:
                weight = weight.view(-1, *self.weight_shape)
            else:
                weight = weight.view(*self.weight_shape)

        if self.feature_mode == 'uu':
            output = sparse_mul(
                input, weight,
                self.info_forward, self.info_backward1, self.info_backward2
            )
        elif self.feature_mode == 'uv':
            output = sparse_vecmat(
                input, weight,
                self.info_forward, self.info_backward1, self.info_backward2
            )

        return output

    def _apply(self, *args, **kwargs):
        lin = super()._apply(*args, **kwargs)
        lin.info_forward = self.info_forward._apply(*args, **kwargs)
        lin.info_backward1 = self.info_backward1._apply(*args, **kwargs)
        lin.info_backward2 = self.info_backward2._apply(*args, **kwargs)
        return lin
    def __repr__(self):
        channels_repr = f"channels_in={self.channels_in}, channels_out={self.channels_out}"
        if self.feature_mode == 'uu':
             channels_repr = f"channels={self.channels_in}"
        return (f"{self.__class__.__name__}("
                f"irreps_in={self.irreps_in.short_repr()}, "
                f"irreps_out={self.irreps_out.short_repr()}, "
                f"{channels_repr}, "
                f"feature_mode={self.feature_mode}, "
                f"path_norm={self.path_norm}, "
                f"channel_norm={self.channel_norm}, "
                f"internal_weights={self.internal_weights}, "
                f"num_weights={self.num_weights}"
                ")")
