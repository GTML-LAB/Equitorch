"""
Some math functions.
"""

from math import sqrt
from typing import Optional

import torch
from torch import Tensor

from torch_geometric.utils import scatter

from e3nn import o3

import functools

from ..utils.indices import check_degree_range, reduce_order_to_degree
from ..typing import DegreeRange
from ..typing import Union, Tuple

from .so3 import *
from .sht import *


def dot(x1: Tensor, x2: Tensor, L: DegreeRange, channel_wise=True):
    r"""
    Compute the degree-wise dot product between spherical features.

    .. math::

        d_c^{(l)} = \sum_{m=-l}^l [{\mathbf{x}_m^{(l)}}]_c [{\mathbf{y}_m^{(l)}}]_c

    if :obj:`channel_wise`, or

    .. math::

        d_{c_1,c_2}^{(l)} = \sum_{m=-l}^l [{\mathbf{x}_m^{(l)}}]_{c_1} [{\mathbf{y}_m^{(l)}}]_{c_2}
        
    otherwise.

    Parameters
    ----------
    x1 : :obj:`~torch.Tensor`
        First input tensor of shape :math:`(N, \text{num_orders_1}, C_1)`.
    x2 : :obj:`~torch.Tensor`
        Second input tensor of shape :math:`(N, \text{num_orders_1}, C_2)`.
    L : :obj:`~equitorch.typing.DegreeRange`
        The degree range of inputs.
    channel_wise : bool, optional
        If True, compute channel-wise dot product. Default is :obj:`True`.

    Returns
    -------
    :obj:`~torch.Tensor`
        The result of the dot product of shape 
        :math:`(N, \text{num_degrees}, C)` if :obj:`channel_wise` is :obj:`True`
        or :math:`(N, \text{num_degrees}, C_1, C_2)` if :obj:`channel_wise` is :obj:`False`.

        
    Where :math:`N` is the batch-size that will automatically broadcast if set to :math:`1`
    and :math:`C_1,C_2,C` are corresponding number of channels.
    If :obj:`channel_wise` is :obj:`True`, :math:`C_1=C_2=C` should be satisfied.  

    Example
    --------
    >>> x1 = torch.randn(32, 16, 64)  # (N, num_orders, C1)
    >>> x2 = torch.randn(32, 16, 64)  # (N, num_orders, C2)
    >>> L = 3
    >>> result = dot(x1, x2, L)
    >>> print(result.shape)
    torch.Size([32, 4, 64])  # (N, num_degrees, C)

    >>> result_not_channel_wise = dot(x1, x2, L, channel_wise=False)
    >>> print(result_not_channel_wise.shape)
    torch.Size([32, 4, 64, 64])  # (N, num_degrees, C1, C2)
    """

    L = check_degree_range(L)
    if not channel_wise:
        x1 = x1.unsqueeze(-1)
        x2 = x2.unsqueeze(-2)
        return reduce_order_to_degree(x1*x2, L, dim=-3)
    else:
        return reduce_order_to_degree(x1*x2, L, dim=-2)

def rms(x: Tensor, L: DegreeRange, additional_dims: int | Tuple[int,...] = -1, 
        degree_scale: bool = False, degree_wise: bool = True, 
        keepdim:bool=True, return_norm:bool=False, scale: Optional[Tensor] = None,
        squared:bool = False):
    r"""
    Compute the root mean square

    When :obj:`degree_wise`, it is computed as 

    .. math::

        \text{RMS}^{(l)}=\sqrt{\mathop{\text{mean}}_{i\text{ over additional dims}}
        \left[\frac{1}{\text{scale}_l}\sum_{m=-l}^{l}
        \big(\mathbf{x}_{i,m}^{(l)}\big)^2\right]},
    
    otherwise as

    .. math::

        \text{RMS}=\sqrt{\mathop{\text{mean}}_{i\text{ over additional dims}}
        \left[\frac{1}{|L|}\sum_{l\in L}\frac{1}{\text{scale}_l}\sum_{m=-l}^{l}
        \big(\mathbf{x}_{i,m}^{(l)}\big)^2\right]},

    where :math:`\text{scale}_l` is :math:`2l+1` if :obj:`degree_scale` or :math:`1` otherwise.

    If the shape of :math:`\mathbf{x}` is :math:`(N,M,C)` for batch, order and channels,
    setting :obj:`degree_wise=True`, :obj:`degree_scale=False` and :obj:`additional_dims=-1`
    will lead to the RMS used in `Equiformer: Equivariant Graph Attention Transformer for 3D Atomistic Graphs <https://arxiv.org/abs/2206.11990>`_,
    while setting :obj:`degree_wise=False`, :obj:`degree_scale=True` will lead to the RMS 
    used in `EquiformerV2: Improved Equivariant Transformer for Scaling to Higher-Degree Representations <https://arxiv.org/abs/2306.12059>`_.

    Parameters
    ----------
    x : :obj:`~torch.Tensor`
        The input tensor.
    L : :obj:`~equitorch.typing.DegreeRange`
        The degree range.
    additional_dims : int | Tuple[int,...], optional
        The additional dimensions to mean over. Default is -1.
    degree_scale : bool, optional
        Whether to divide the RMS of degree :math:`l` by :math:`2l+1`.
        Default is :obj:`False`.
    degree_wise : bool, optional
        Whether the RMS is computed degreewise. Default is :obj:`True`.
    keepdim : bool, optional
        Whether the output tensor has dim retained or not. Default is :obj:`True`.
    return_norm : bool, optional
        Whether to return the norm of the input. Default is :obj:`False`
    scale: Tensor, optional
        Used for degree scale when :obj:`degree_scale` is :obj:`True` if provided.
        Should be of shape :math:`(|L|,1)`
    squared: bool, optional
        Whether to return the squared RMS and norm. Default is :obj:`False`
    Returns
    -------
    RMS: Tensor
        The spherical axis will be 
        If :obj:`degree_wise`, the returned tensor will be of length :math:`L` 
        along the order axis and :math:`1` along the additional dimensions,
        otherwise, it will also be of length :math:`1` along the order axis.
        If :obj:`keepdim` is :obj:`False`, all the dimensions with length :math:`1`
        will be squeeze out.
    norm: Tensor, optional
        Returned only when :obj:`return_norm`.
        If :obj:`degree_wise`, the norm be of length :math:`L` 
        along the order axis, or :math:`1` otherwise.
        If not :obj:`keepdim`, the order axis will be squeezed out.
        Along other axis, the shape of norm will be same as the input.
        The norm will not scaled even if :obj:`degree_scale` is :obj:`True`.

    Example
    --------

    .. code-block:: python

        >>> Ys = spherical_harmonics(torch.randn(2,3,4), 2, True, dim=-2)
        >>> print(Ys.shape)
        torch.Size([2, 9, 4])

        >>> rms1 = rms(Ys, 2, -1, degree_scale=True, degree_wise=True)
        >>> print(rms1.shape)
        torch.Size([2, 3, 1])
        >>> print(rms1)
        tensor([[[0.2821],
                [0.2821],
                [0.2821]],

                [[0.2821],
                [0.2821],
                [0.2821]]])

        >>> rms2 = rms(Ys, 2, -1, degree_scale=True, degree_wise=False)
        >>> print(rms2.shape)
        torch.Size([2, 1, 1])
        >>> print(rms2)
        tensor([[[0.2821]],

                [[0.2821]]])
        
        >>> rms3, norm3 = rms(Ys, 2, -1, degree_scale=False, 
                            degree_wise=True, return_norm=True)
        >>> print(rms3.shape)
        torch.Size([2, 3, 1])
        >>> print(rms3)
        tensor([[[0.2821],
                [0.4886],
                [0.6308]],

                [[0.2821],
                [0.4886],
                [0.6308]]])
        >>> print(norm3.shape)
        torch.Size([2, 3, 4])
        
        >>> rms4, norm4 = rms(Ys, 2, -1, degree_scale=True, degree_wise=False, 
                            keepdim=False, return_norm=True)
        >>> print(rms4.shape)
        torch.Size([2])
        >>> print(rms4)
        tensor([0.2821, 0.2821])
        >>> print(norm4.shape)
        torch.Size([2, 4])
    """

    additional_dims = (additional_dims,) if isinstance(additional_dims, int) else additional_dims
    if not degree_wise:
        mean_dims = (-2,) + additional_dims
    else:
        mean_dims = additional_dims
    n = norm2(x, L)
    ret = n
    if degree_scale: 
        if scale is None:
            scale = (torch.tensor(degrees_in_range(L), dtype=x.dtype, device=x.device)*2+1).unsqueeze(-1)
        ret = ret / scale
    ret = ret.mean(mean_dims, keepdim=keepdim)
    if not squared:
        if return_norm:
            return (ret.relu()).sqrt(), (n.relu()).sqrt() if degree_wise else (n.sum(dim=-2, keepdim=keepdim).relu()).sqrt()
        else:
            return (ret.relu()).sqrt()
    else:
        if return_norm:
            return ret, n if degree_wise else n.sum(dim=-2, keepdim=keepdim)
        else:
            return ret

def norm(x: Tensor, L: DegreeRange, ):
    r"""Compute the degree & channel-wise norm of a spherical feature 

    .. math::

        \|\mathbf{x}^{(l)}_c\| = \sqrt{\sum_{m=-l}^l [{\mathbf{x}_m^{(l)}}]_c^2}, l\in L

    Parameters
    ----------
    x : :obj:`~torch.Tensor`
        Spherical feature tensor of shape :math:`(N, \text{num_orders}, C)`
    L : :obj:`~equitorch.typing.DegreeRange`
        The degree range of input

    Returns
    -------
    :obj:`~torch.Tensor`
        Norm tensor of shape :math:`(N, \text{num_degrees}, C)`

    
    Where :math:`N` is the batch-size and :math:`C` is the number of channels. 
    
    Example
    --------
    >>> x = torch.randn(32, 16, 64)  # (N, num_orders, C1)
    >>> L = 3
    >>> result = norm(x, L)
    >>> print(result.shape)
    torch.Size([32, 4, 64])  # (N, num_degrees, C)
    """

    return (reduce_order_to_degree(x**2, L, dim=-2).relu()).sqrt()

def norm2(x: Tensor, L: DegreeRange):
    r"""Compute the square of degree & channel-wise norm of a spherical feature as

    .. math::

        \|\mathbf{x}^{(l)}_c\|^2 = {\sum_{m=-l}^l [{\mathbf{x}_m^{(l)}}]_c^2}, l\in L

    Parameters
    ----------
    x : :obj:`~torch.Tensor`
        Spherical feature tensor of shape :math:`(N, \text{num_orders}, C)`
    L : :obj:`~equitorch.typing.DegreeRange`
        The degree range of input

    Returns
    -------
    :obj:`~torch.Tensor`
        Square of norm tensor of shape :math:`(N, \text{num_degrees}, C)`

    
    Where :math:`N` is the batch-size and :math:`C` is the number of channels. 
    
    Example
    --------
    >>> x = torch.randn(32, 16, 64)  # (N, num_orders, C1)
    >>> L = 3
    >>> result = norm2(x, L)
    >>> print(result.shape)
    torch.Size([32, 4, 64])  # (N, num_degrees, C)
    """
    return reduce_order_to_degree(x**2, L, dim=-2)

__all__ = [
    'dot',
    'norm',
    'norm2',
    'rms',
    'sht',
    'isht',
    's2_grid'
    'angles_to_xyz',
    'xyz_to_angles',
    'angles_to_matrix',
    'wigner_D',
    'spherical_harmonics',
]