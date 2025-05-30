import torch
from torch import Tensor
from torch.autograd import Function
import math
from typing import Optional
from ...ops.spherical_harmonics import spherical_harmonics as _spherical_harmonics
from .angular import sincos

def spherical_harmonics(input: torch.Tensor, l_max:int, dim:int=-1, integral_normalize: bool = False):
    inv_sqrt_4PI = 1 / (2*math.sqrt(math.pi))
    sh = _spherical_harmonics(input, l_max, dim)
    if integral_normalize:
        sh = sh * inv_sqrt_4PI
    return sh


def xyz_to_spherical(
    xyz: Tensor,
    normalize_input: bool = True,
    with_r: bool = False,
    eps: float = 1e-14,
    dim: int = -1
) -> tuple[Tensor, ...] | Tensor:
    """
    Computes spherical coordinates (r, theta, phi) from Cartesian coordinates (x, y, z).

    Args:
        xyz (Tensor): Input tensor with Cartesian coordinates. Assumed to have shape (..., 3).
                      The last dimension (specified by `dim`) should contain x, y, z.
        normalize_input (bool, optional): If True, normalizes the input xyz vector before
                                     computing angles, effectively setting r=1 for angle calculation.
                                     The returned r will still be the original magnitude unless
                                     with_r is False and normalized is True. Defaults to True.
        with_r (bool, optional): If True, returns r along with theta and phi.
                                 If False, returns only theta and phi. Defaults to False.
        eps (float, optional): Small epsilon value to avoid division by zero or acos/atan2
                               instabilities. Defaults to 1e-14.
        dim (int, optional): The dimension along which x, y, z are stored. Defaults to -1.

    Returns:
        tuple[Tensor, ...] | Tensor:
            If with_r is True: (theta, phi, r)
            If with_r is False: (theta, phi)
            - theta: Polar angle (inclination) from z-axis. Range [0, pi]. Shape (...)
            - phi: Azimuthal angle in x-y plane from x-axis. Range [-pi, pi]. Shape (...)
            - r: Radius. Shape (...)
    """
    if xyz.shape[dim] != 3:
        raise ValueError(
            f"Input `xyz` is expected to have 3 components along dimension `dim`={dim}, "
            f"but got {xyz.shape[dim]}."
        )

    # Extract x, y, z components
    x, y, z = torch.unbind(xyz, dim=dim)
    
    # Compute radius if needed
    r = torch.sqrt(xyz.pow(2).sum(dim=-1).clamp_min(eps))
    
    if normalize_input:
        # Normalize coordinates for angle calculations
        x_norm = x / r
        y_norm = y / r
        z_norm = z / r
    else:
        x_norm, y_norm, z_norm = x, y, z
    
    # Compute angles
    theta = torch.acos(torch.clamp(z_norm, -1.0 + eps, 1.0 - eps))
    phi = torch.atan2(y_norm, x_norm + eps)
    
    if with_r:
        return theta, phi, r
    return theta, phi


def spherical_to_xyz(
    theta: Tensor,
    phi: Tensor,
    r: Optional[Tensor] = None,
    dim: int = -1
) -> Tensor:
    """
    Computes Cartesian coordinates (x, y, z) from spherical coordinates (theta, phi, r).

    Args:
        theta (Tensor): Polar angle (inclination) from z-axis. Shape (...).
        phi (Tensor): Azimuthal angle in x-y plane from x-axis. Shape  (...).
        r (Optional[Tensor], optional): Radius. If None, assumed to be 1.
                                        Shape (...). Defaults to None.
        dim (int, optional): The dimension along which the output x, y, z will be stacked.
                             Defaults to -1.

    Returns:
        Tensor: Output tensor with Cartesian coordinates (x, y, z). Shape (..., 3).
    """

    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    sin_phi = torch.sin(phi)
    cos_phi = torch.cos(phi)

    x = sin_theta * cos_phi
    y = sin_theta * sin_phi
    z = cos_theta

    if r is not None:
        x = x * r
        y = y * r
        z = z * r
    

    return torch.stack([x, y, z], dim=dim)



def xyz_to_sincos(
    xyz: Tensor,
    max_m: int,
    normalize_input: bool = True,
    component_normalize: bool = False,
    eps: float = 1e-14, 
    dim: int = -1,
) -> Tensor:
    """
    Computes the sin/cos embedding of the azimuthal angle (phi) derived from
    Cartesian coordinates (x, y, z).

    The sin/cos embedding is typically of the form:
    [1.0, sin(phi), cos(phi), sin(2*phi), cos(2*phi), ..., sin(max_m*phi), cos(max_m*phi)]

    Args:
        xyz (Tensor): Input tensor with Cartesian coordinates. Shape (..., 3).
        max_m (int): The maximum multiple of the angle phi to compute sin/cos for.
        normalize_input (bool, optional): If True, normalizes xyz before extracting angles.
                                     Defaults to True.
        eps (float, optional): Small epsilon for numerical stability in xyz_to_spherical.
                               Defaults to 1e-14.
        dim (int, optional): Dimension of Cartesian coordinates in xyz. Defaults to -1.
        component_normalize(bool): If true, multiply sin and cos by sqrt(2), such that the expectation of each element is 1.

    Returns:
        Tensor: The sin/cos embedding of the phi angle. Shape (..., 1 + 2 * max_m).
    """
    # Get theta and phi. We only need phi for sincos.
    # with_r=False by default in xyz_to_spherical if not specified,
    # but explicitly setting it to False as we don't need r.
    theta, phi = xyz_to_spherical(
        xyz,
        normalize_input=normalize_input,
        with_r=False,
        eps=eps,
        dim=dim
    ) 

    return sincos(theta, max_m, normalized=component_normalize), sincos(phi, max_m,normalized=component_normalize)
