import torch
import torch.nn as nn

from .functional.sphericals import spherical_harmonics, xyz_to_spherical, spherical_to_xyz, xyz_to_sincos
from typing import Optional, Tuple # Add Optional for type hinting
from torch import Tensor # Add Tensor for type hinting

class SphericalHarmonics(nn.Module):
    r"""
    Computes spherical harmonics from input Cartesian coordinates.
    Wraps the functional :func:`~equitorch.nn.functional.sphericals.spherical_harmonics`.

    Spherical harmonics are a set of orthogonal functions defined on the surface of a sphere.
    They are solutions to Laplace's equation in spherical coordinates.

    If `integral_normalize` is True, the output is scaled by \(1 / \sqrt{4\pi}\).

    Args:
        l_max (int): The maximum degree of the spherical harmonics.
        normalize_input (bool, optional): If True, normalizes the input xyz vector before
                                     computing spherical harmonics. Defaults to True.
        integral_normalize (bool, optional): If True, applies normalization for integration over the sphere.
                                         Defaults to False.
    """
    def __init__(self, l_max: int, normalize_input: bool=True, integral_normalize: bool=False):
        super().__init__()
        self.l_max = l_max
        self.integral_normalize = integral_normalize
        self.normalize_input = normalize_input

    def forward(self, input):
        if self.normalize_input:
            input = torch.nn.functional.normalize(input, dim=-1)
        sh = spherical_harmonics(input, self.l_max, integral_normalize=self.integral_normalize)
        return sh

class XYZToSpherical(nn.Module):
    r"""
    Module to convert Cartesian coordinates (\(x, y, z\)) to spherical (\(\theta, \phi, r\)).
    Wraps the functional :func:`~equitorch.nn.functional.sphericals.xyz_to_spherical`.

    Args:
        normalize_input (bool, optional): If True, normalizes the input `xyz` vector before
                                     computing angles. Defaults to True.
        with_r (bool, optional): If True, returns \(r\) along with \(\theta\) and \(\phi\).
                                 Defaults to False.
        eps (float, optional): Small \(\epsilon\) for numerical stability. Defaults to 1e-14.
        dim (int, optional): Dimension of Cartesian coordinates. Defaults to -1.
    """
    def __init__(self,
                 normalize_input: bool = True,
                 with_r: bool = False,
                 eps: float = 1e-14, # Match default from functional
                 dim: int = -1):
        super().__init__()
        self.normalize_input = normalize_input
        self.with_r = with_r
        self.eps = eps
        self.dim = dim

    def forward(self, xyz: Tensor) -> tuple[Tensor, ...] | Tensor:
        return xyz_to_spherical(
            xyz,
            normalize_input=self.normalize_input,
            with_r=self.with_r,
            eps=self.eps,
            dim=self.dim
        )

    def extra_repr(self) -> str:
        return (f'normalize_input={self.normalize_input}, with_r={self.with_r}, '
                f'eps={self.eps}, dim={self.dim}')

class SphericalToXYZ(nn.Module):
    r"""
    Module to convert spherical coordinates (\(\theta, \phi, r\)) to Cartesian (\(x, y, z\)).
    Wraps the functional :func:`~equitorch.nn.functional.sphericals.spherical_to_xyz`.

    Args:
        dim (int, optional): Dimension along which to stack output \(x,y,z\). Defaults to -1.
    """
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self,
                theta: Tensor,
                phi: Tensor,
                r: Optional[Tensor] = None) -> Tensor:
        return spherical_to_xyz(
            theta,
            phi,
            r=r,
            dim=self.dim
        )

    def extra_repr(self) -> str:
        return f'dim={self.dim}'

class XYZToSinCos(nn.Module):
    r"""
    Module to convert Cartesian coordinates (\(x, y, z\)) to sin/cos embeddings
    of the spherical angles \(\theta\) and \(\phi\).
    Wraps the functional :func:`~equitorch.nn.functional.sphericals.xyz_to_sincos`.

    Args:
        max_m (int): The maximum multiple of the angles to compute \(\sin\) / \(\cos\) for.
        normalize_input (bool, optional): If True, normalizes `xyz` before extracting angles.
                                     Defaults to True.
        component_normalize: (bool, optional): False
        eps (float, optional): Small \(\epsilon\) for numerical stability. Defaults to 1e-14.
        dim (int, optional): Dimension of Cartesian coordinates in `xyz`. Defaults to -1.
    """
    def __init__(self,
                 max_m: int,
                 normalize_input: bool = True,
                 component_normalize: bool = False,
                 eps: float = 1e-14, # Match default from functional
                 dim: int = -1):
        super().__init__()
        self.max_m = max_m
        self.normalize_input = normalize_input
        self.eps = eps
        self.dim = dim
        self.component_normalize = component_normalize

    def forward(self, xyz: Tensor) -> Tuple[Tensor, Tensor]:
        r"""
        Computes \(\text{sincos}(\theta)\) and \(\text{sincos}(\phi)\).

        Args:
            xyz (Tensor): Input Cartesian coordinates.

        Returns:
            Tuple[Tensor, Tensor]: (\(\text{sincos}_\theta\), \(\text{sincos}_\phi\))
        """
        return xyz_to_sincos(
            xyz,
            max_m=self.max_m,
            normalize_input=self.normalize_input,
            component_normalize=self.component_normalize,
            eps=self.eps,
            dim=self.dim
        )

    def extra_repr(self) -> str:
        return (f'max_m={self.max_m}, normalize_input={self.normalize_input}, '
                f'eps={self.eps}, dim={self.dim}')
