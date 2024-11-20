from typing import Optional, Union
import torch
import torch.nn as nn
from torch import Tensor

class GaussianBasisExpansion(nn.Module):
    r"""
    Gaussian Basis Expansion module.

    This module implements a Gaussian basis expansion of the form:

    .. math::

        \exp[-\gamma_k(r-\mu_k)^2],\ k = 1,2,\dots,\text{num_basis}.

    Parameters
    ----------
    gamma : Union[Tensor, float]
        The gamma parameter for the Gaussian basis function.
    num_basis : int, optional
        The number of basis functions to use. If None, mu must be provided.
    start : float, optional
        The start value for generating mu values. Required if num_basis is provided.
    end : float, optional
        The end value for generating mu values. Required if num_basis is provided.
    mu : :obj:`~torch.Tensor`, optional
        The mu values for the Gaussian basis functions. Required if num_basis is None.
    trainable : bool, optional
        Whether the mu and gamma parameters should be trainable. Default is :obj:`False`.

    Note
    ----
    If :obj:`num_basis` is provided, :math:`\mu_k` are generated evenly in :math:`[\text{start},\text{end}]`.
    If :obj:`gamma` is a float, it is expanded to match the shape of :math:`\mu`.
    If :obj:`trainable` is :obj:`True`, :obj:`mu` and :obj:`gamma` become :obj:`nn.Parameter` objects.
    """
    def __init__(self, 
                 gamma: Union[Tensor,float], 
                 num_basis: float = None, 
                 start: float = None,
                 end: float = None,
                 mu: Tensor = None,
                 trainable: bool = False):
        super().__init__()
        if num_basis is None:
            mu = mu
        else:
            mu = torch.linspace(start, end, num_basis)
        if isinstance(gamma, float) or isinstance(gamma, int):
            gamma = torch.ones_like(mu) * gamma

        if trainable:
            self.mu = nn.Parameter(self.mu)
            self.gamma = nn.Parameter(self.gamma)
        else:
            self.register_buffer('mu', mu)
            self.register_buffer('gamma', gamma)
    def forward(self, x: Tensor):
        r"""
        """
        return torch.exp(-self.gamma * (x.unsqueeze(-1)-self.mu).pow(2))


# Modified from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/dimenet.html#DimeNet
class BesselBasisExpansion(torch.nn.Module):
    r"""
    Bessel Basis Expansion module.

    This module implements a Bessel basis expansion of the form:

    .. math::

        \frac{\sin(\omega_k (\hat r + \epsilon))}{\hat r + \epsilon},\ k = 1,2,\dots,\text{num_basis}.

    where :math:`\omega_k` is the frequency, default to be :math:`k\pi`, 
    :math:`\hat r=r/c` is the cutoff-normalized distance, 
    and :math:`\epsilon` is a small value for stability near zero.

    Parameters
    ----------
    num_basis : int
        The number of basis functions to use.
    cutoff : float, optional
        The cutoff value for the distance. Default is 1.
    trainable : bool, optional
        Whether the frequency parameters should be trainable. Default is :obj:`False`.
    eps : float, optional
        A small value for stability near zero. Default is 1e-6.

    Note
    ----
        If :obj:`trainable` is :obj:`True`, the frequency parameters become :obj:`nn.Parameter` objects.

    """
    def __init__(self, 
                 num_basis: int, 
                 cutoff: float = 1,
                 trainable: bool = False,
                 eps: float = 1e-6):
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff
        self.trainable = trainable

        self.eps = eps
        freq = torch.arange(1, num_basis + 1) * torch.pi

        if trainable:
            self.freq = torch.nn.Parameter(freq)
        else:
            self.register_buffer('freq', freq)


    def forward(self, x: Tensor) -> Tensor:
        r"""
        """
        x = x.unsqueeze(-1) / self.cutoff + self.eps
        return (self.freq * x).sin() / (x)