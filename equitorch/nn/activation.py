from math import log
from typing import Callable, Optional, Union, Tuple
from ..typing import DegreeRange

import torch
from torch import Tensor
import torch.nn as nn
import torch_geometric
from ..utils.indices import check_degree_range, expand_degree_to_order
from ..math import norm, isht, sht


class NormAct(nn.Module):
    r"""The Norm Activation

    - If :obj:`degree_wise`:

    .. math::

        \mathbf{x}'_c = \bigoplus_l \sigma(\|\mathbf{x}^{(l)}_c\|)\mathbf{x}^{(l)}_c

    - otherwise:

    .. math::

        \mathbf{x}'_c = \sigma(\|\mathbf{x}_c\|)\mathbf{x}_c

    for the channel :math:`c`, where :math:`\bigoplus` is the concatenation.

    If :obj:`bias` is true, a bias :math:`b_c` or :math:`b_c^{(l)}` will be added 
    to the corresponding norm before be passed in the activation.

    Parameters
    ----------
    L : :obj:`~equitorch.typing.DegreeRange`
        The degree range.
    activation : Callable[[Tensor], Tensor]
        The activation function to apply to the norm.
    degree_wise : bool, optional
        If True, apply the activation separately for each degree. 
        If False, apply it to the overall norm. Default is True.
    bias : bool, optional
        If True, add a learnable bias before the activation. Default is False.
    num_channels : int, optional
        Number of input channels. Required if bias is True.

    Example
    --------
    >>> activation = torch.nn.SiLU()
    >>> L = (0, 2)
    >>> norm_act = NormAct(L, activation, degree_wise=True, bias=True, num_channels=16)
    >>> x = torch.randn(32, 9, 16)  # batch_size=32, num_orders=9, channels=16
    >>> output = norm_act(x)
    >>> print(output.shape)
    torch.Size([32, 9, 16])
    """

    def __init__(self, 
                 L: DegreeRange, 
                 activation: Callable[[Tensor], Tensor],
                 degree_wise:bool=True, 
                 need_bias:bool=False, 
                 num_channels: Optional[int] = None):
        super().__init__()
        self.activation = activation
        self.degree_wise = degree_wise
        self.L = check_degree_range(L)
        self.num_degrees = self.L[1] - self.L[0] + 1
        self.need_bias = need_bias
        if self.need_bias:
            self.num_channels = num_channels
            if self.degree_wise:
                self.bias = nn.Parameter(
                    torch.randn(self.num_degrees, self.num_channels))
            else:
                self.bias = nn.Parameter(
                    torch.randn(self.num_channels))

    def forward(self, x: Tensor):
        """
        """
        if self.degree_wise:
            n = norm(x, self.L)
            if self.need_bias:
                n = n + self.bias
            n = self.activation(n)
            x = x * expand_degree_to_order(n, self.L, dim=-2)
        else:
            n = torch.norm(x, dim=-2, keepdim=True)
            if self.need_bias:
                n = n + self.bias
            n = self.activation(n)
            x = x * n
        return x

class Gate(nn.Module):
    r""" The Gate Activation

    - If `degree_wise`:

    .. math::
        \text{Gate}_\sigma(\mathbf{x}_c,g_c) = \bigoplus_l\sigma(g_c^{(l)})\mathbf{x}^{(l)},

    - otherwise

    .. math::
        \text{Gate}_\sigma(\mathbf{x}_c,g_c) = \sigma(g_c)\mathbf{x},

    for the channel :math:`c`, where :math:`\bigoplus` is the concatenation, 
    :math:`\sigma` is the activation function and :math:`g_c` or :math:`g_c^{(l)}`
    is the gate variable. 

    Parameters
    ----------
    L : :obj:`~equitorch.typing.DegreeRange`
        The degree range.
    activation: Callable[[Tensor], Tensor], optional
        The activation function applied to gate variables. Default is :obj:`None` for the identity function.
    degree_wise: bool, optional
        If True, apply the gate separately for each degree. 
        If False, apply gate to the whole spherical tensor. Default is True.
    """
    def __init__(self, 
                 L: DegreeRange, 
                 activation: Callable[[Tensor], Tensor] = None,
                 degree_wise:bool=True, 
                ):
        super().__init__()
        self.activation = activation
        self.degree_wise = degree_wise
        self.L = check_degree_range(L)
        self.num_degrees = self.L[1] - self.L[0] + 1

    def forward(self, x: Tensor, gate: Tensor):
        r"""
        Parameters
        ----------
        x : :obj:`~torch.Tensor`
            The input spherical tensor of shape :math:`(N, \text{num_orders}, C)`.
        gate : :obj:`~torch.Tensor`
            The gate variable of shape :math:`(N, \text{num_degrees}, C)` if 
            :obj:`degree_wise` is :obj:`True` or :math:`(N, C)` 
            if :obj:`degree_wise` is :obj:`False`.

        Returns
        -------
        :obj:`~torch.Tensor`
            The gated spherical tensor.
        """
        if self.degree_wise:
            gate = gate.view(-1, self.num_degrees, x.shape[-1])
            if self.activation is not None:
                gate = self.activation(gate)
            x = x * expand_degree_to_order(gate, self.L, dim=-2)
        else:
            gate = gate.view(-1, 1, x.shape[-1])
            if self.activation is not None:
                gate = self.activation(gate)
            x = x * gate
        return x


class S2Act(nn.Module):
    r"""The S2 pointwise activation.
     
    The activation used in 
    `Reducing SO(3) Convolutions to SO(2) for Efficient Equivariant 
    GNNs <https://arxiv.org/abs/2302.03655>`_.

    For a spherical tensor :math:`\mathbf{x}`, the activation transforms it by

    .. math::
        \mathbf{x}'=\mathrm{SHT}_{L}[\sigma(\mathrm{ISHT}[\mathbf{x}])]

    with a scalar activation :math:`\sigma`. 
    
    The :math:`\mathrm{SHT}` and :math:`\mathrm{ISHT}` are spherical harmonic 
    transformation and inverse spherical harmonic transformation, respectively. 
    (See :obj:`~equitorch.math.sht`, :obj:`~equitorch.math.isht`.)

    Warning
    -------
    This operation is **not** exactly equivariant since we use grid summation
    instead of integral when performing ISHT. 

    Parameters
    ----------
    L : :obj:`~equitorch.typing.DegreeRange`
        The degree range of inputs and outputs.
    activation : Callable[[Tensor], Tensor]
        The scalar activation
    resolution : Union[Tuple[int,int],int]
        The grid points of thetas and phis when performing ISHT.

    Example
    -------
    .. code-block:: python

        >>> N, C, L = 20, 2, 3
        >>> x = torch.randn(N,num_orders_in(L),C)
        >>> D = ... # A random Wigner D matrix of degree-range L
        >>> print(x.shape)
        torch.Size([20, 16, 2])
        >>> print(D.shape)
        torch.Size([20, 16, 16])

        >>> act = S2Act(L, torch.nn.SiLU(), (16, 16))
        >>> z = act(x)
        >>> Dz = act(D@x)
        >>> print((D@z - Dz).abs().max())
        tensor(0.0072)

        >>> act = S2Act(L, torch.nn.SiLU(), (16, 32))
        >>> z = act(x)
        >>> Dz = act(D@x)
        >>> print((D@z - Dz).abs().max())
        tensor(3.7670e-05)
    
        >>> act = S2Act(L, torch.nn.SiLU(), (32, 32))
        >>> z = act(x)
        >>> Dz = act(D@x)
        >>> print((D@z - Dz).abs().max())
        tensor(6.4373e-06)
    """
    def __init__(self, L: DegreeRange,
                 activation: Callable[[Tensor], Tensor],
                 resolution:Union[Tuple[int,int],int] = 16, 
                 ):
        super().__init__()

        self.L = check_degree_range(L)
        if isinstance(resolution, int):
            self.resolution = (resolution, resolution)
        else:
            self.resolution = resolution

        self.activation = activation

    def forward(self, x: Tensor):
        """
        """
        sph = isht(x, self.L, *(self.resolution))
        sph = self.activation(sph)
        return sht(sph, self.L, *(self.resolution))

def shifted_softplus(x: Tensor, alpha: float = 0.5, threshold: float = 20.):
    r"""Compute the shifted softplus activation function.

    This function implements a modified version of the softplus activation:

    .. math::

        \text{ssp}(x) = \begin{cases}
            \ln(\alpha \exp(x) + (1-\alpha)) & \text{if } x < \text{threshold} \\
            x + \ln(\alpha) & \text{otherwise}
        \end{cases}

    Parameters
    ----------
    x : :obj:`~torch.Tensor`
        Input tensor.
    alpha : float, optional
        Scaling factor in the exponential term. Default is 0.5.
    threshold : float, optional
        Threshold value for numerical stability. Default is 20.0.

    Returns
    -------
    :obj:`~torch.Tensor`
        Output tensor after applying the shifted softplus activation.

    Note
    ----
    - This function provides a smooth approximation to the ReLU function.
    - The threshold is used to prevent numerical overflow for large input values.
    - When x is large (>= threshold), the function approximates to a linear function.
    """
    return torch.where(x < threshold, 
                       torch.log(alpha*(torch.exp(x)) + (1-alpha)),
                       x + log(alpha))


class ShiftedSoftPlus(nn.Module):
    r'''
    See :obj:`shifted_softplus`
    '''
    def __init__(self, alpha: float = 0.5, threshold: float = 20.):
        super().__init__()
        self.alpha = alpha
        self.threshold = threshold

    def forward(self, x: Tensor):
        return shifted_softplus(x, self.alpha, self.threshold)