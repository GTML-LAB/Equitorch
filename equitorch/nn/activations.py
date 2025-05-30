import torch

from ..irreps import Irreps, check_irreps
from .functional.activations import gating
import torch.nn as nn
from torch import Tensor
from ..utils._structs import irreps_info
from ..structs import IrrepsInfo


class Gate(nn.Module):
    r"""Applies element-wise gates to equivariant features.

    This module implements gating nonlinearities for features represented by :class:`~equitorch.irreps.Irreps`.
    It can operate in two primary modes based on how the gate values are provided:

    1.  **Separate Gates (``gate`` argument provided):**
        The module takes two distinct inputs: ``input`` (the features to be gated) and ``gate`` (the gate scalars).

        - If ``irrep_wise=True`` (default): Each gate scalar in the ``gate`` tensor is applied to its corresponding
          irrep block within the ``input`` features. The ``gate`` tensor should have a shape compatible with
          ``(..., num_gates, channels)``, where ``num_gates`` is the number of irreps in ``irreps``.
        - If ``irrep_wise=False``: A single gate scalar (or a set of scalars broadcastable across irreps)
          is applied to all irrep blocks in the ``input`` features. The ``gate`` tensor should have a shape
          compatible with ``(..., 1, channels)``.

    2.  **Concatenated Input (``gate=None``):**
        The module takes a single ``input`` tensor where the features and their corresponding gate scalars
        are concatenated along the spherical dimension (``dim=-2``). The last ``num_gates`` slices along this
        dimension are interpreted as the gate scalars. The ``input`` tensor shape is expected to be
        ``(..., irreps.dim + num_gates, channels)``.
        The module internally splits this tensor into features and gates, optionally applies an activation
        function to the extracted gates, and then proceeds with the gating operation as described in mode 1.

    An optional activation function can be applied to the gate scalars before they modulate the features.

    Example:
        .. code-block:: python

            import torch
            from equitorch.irreps import Irreps
            from equitorch.nn import Gate

            irreps = Irreps("1x0e + 2x1o") # Example: one scalar, two l=1 odd irreps
            gate_module = Gate(irreps, activation=torch.nn.Tanh())

            batch_size, channels = 4, 8
            num_gates = len(irreps) # This will be 2 for the example irreps

            # Mode 1: Separate input and gate tensors (irrep_wise=True)
            features = torch.randn(batch_size, irreps.dim, channels)
            gates = torch.randn(batch_size, num_gates, channels)
            output_separate = gate_module(features, gates)
            print(f"Output shape (separate gates): {output_separate.shape}")

            # Mode 2: Concatenated input tensor
            # irreps.dim for "1x0e + 2x1o" is 1*1 + 2*3 = 7
            # num_gates is 2
            concatenated_input = torch.randn(batch_size, irreps.dim + num_gates, channels)
            output_concatenated = gate_module(concatenated_input)
            print(f"Output shape (concatenated input): {output_concatenated.shape}")

    Args:
        irreps (Irreps): The irreducible representations of the feature part of the input tensor
            (i.e., the part that will be gated).
        activation (torch.nn.Module, optional): An activation function to be applied to the
            gate scalars before the gating operation. Defaults to ``None`` (no activation).
        irrep_wise (bool, optional): Determines how gates are applied.
            If ``True`` (default), gates are applied irrep-by-irrep. This requires the ``gate``
            tensor (if provided separately) to have a shape like ``(..., num_gates, channels)``.
            If ``False``, a single gate (or a broadcastable set) is applied across all irreps.
            This requires the ``gate`` tensor (if provided separately) to have a shape like
            ``(..., 1, channels)``. ``num_gates`` corresponds to ``len(irreps)``.

    Attributes:
        irreps_info (IrrepsInfo): Cached information about the input feature irreps, used for efficient gating.
        num_gates (int): The number of distinct gate scalars, equal to ``len(irreps)``.
                         This dictates the expected size of the gate dimension in the ``gate`` tensor
                         or the number of gate slices in a concatenated input.
    """
    irreps_info: IrrepsInfo
    def __init__(self, 
                 irreps: Irreps, 
                 activation: nn.Module = None,
                 irrep_wise:bool=True, 
                ):
        super().__init__()

        self.irreps = check_irreps(irreps)
        self.irreps_dim = self.irreps.dim
        self.activation = activation
        self.irrep_wise = irrep_wise

        self.num_gates = len(self.irreps)

        self.irreps_info = irreps_info(self.irreps)

    def forward(self, input: Tensor, gate: Tensor = None) -> Tensor:
        r"""Apply the gating mechanism.

        Args:
            input (torch.Tensor): The input tensor. Its shape depends on whether the ``gate``
                argument is provided separately or if gates are concatenated within ``input``.

                - If ``gate`` is ``None`` (concatenated mode): Expected shape is
                  ``(..., irreps.dim + num_gates, channels)``.
                - If ``gate`` is provided (separate mode): Expected shape is
                  ``(..., irreps.dim, channels)``.
            gate (torch.Tensor, optional): The gate tensor. If ``None``, gate scalars are
                assumed to be concatenated with ``input``. If provided, its shape depends
                on the ``irrep_wise`` setting:

                - If ``irrep_wise=True``: Expected shape is ``(..., num_gates, channels)``.
                - If ``irrep_wise=False``: Expected shape is ``(..., 1, channels)``.

        Returns:
            torch.Tensor: The gated output tensor, with shape ``(..., irreps.dim, channels)``.
        """
        channels = input.shape[-1]
        expected_feature_dim = self.irreps_dim

        if gate is None:
            # Input contains concatenated features and gates
            expected_total_dim = expected_feature_dim + self.num_gates
            assert input.shape[-2] == expected_total_dim, \
                f"Input spherical dim mismatch (concatenated mode): expected {expected_total_dim} (irreps.dim + num_gates), got {input.shape[-2]}"
            
            # Split input into features and gates
            features, gate = torch.split(input, [expected_feature_dim, self.num_gates], dim=-2)
            # Gate shape becomes (..., num_gates, channels) after split
            # No need to flatten here if we handle irrep_wise logic correctly below
        else:
            # Gates are provided separately
            features = input
            assert features.shape[-2] == expected_feature_dim, \
                f"Input feature spherical dim mismatch: expected {expected_feature_dim}, got {features.shape[-2]}"
            if self.irrep_wise:
                assert gate.numel() == self.num_gates * channels or gate.numel() == self.num_gates * channels * input.shape[0],\
                    f"Gate numel mismatch (irrep_wise): expected {self.num_gates * channels} or {self.num_gates * channels * input.shape[0]}, got {gate.numel()}"
            else:
                 # Allow gate dim to be 1 or num_gates if not irrep_wise (will broadcast or use global gate)
                assert gate.numel() == channels or gate.numel() == channels * input.shape[0],\
                     f"Gate numel mismatch (global): expected {channels} or {channels * input.shape[0]}, got {gate.numel()}"
             # Check broadcast compatibility for batch dims (...)

        # Perform gating
        if self.irrep_wise:
            if gate.numel() > self.num_gates * input.shape[-1]:
                gate = gate.view(-1, self.num_gates, input.shape[-1])
            else:
                gate = gate.view(self.num_gates, input.shape[-1])
            if self.activation is not None:
                gate = self.activation(gate)

            out = gating(features, gate, self.irreps_info)
        else:
            gate = gate.view(-1, 1, input.shape[-1])
            if self.activation is not None:
                gate = self.activation(gate)
            out = features * gate
        return out

    def _apply(self, *args, **kwargs):
        tp = super()._apply(*args, **kwargs)
        tp.irreps_info = self.irreps_info._apply(*args, **kwargs)
        return tp

    def __repr__(self):
        activation_repr = f"{self.activation}" if self.activation is not None else "None"
        return (f"{self.__class__.__name__}("
                f"irreps={self.irreps.short_repr()}, "
                f"activation={activation_repr}, "
                f"irrep_wise={self.irrep_wise}, "
                f"num_gates={self.num_gates}"
                ")")
