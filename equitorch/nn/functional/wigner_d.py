from typing import Optional
import torch
from torch import Tensor
from torch.autograd import Function

from .sparse_product import sparse_vecsca
from .sparse_scale import sparse_scale
from .sphericals import xyz_to_spherical
from .angular import sincos

from ...structs import WignerRotationInfo
from ...irreps import Irreps # Need Irreps
# No need for Irreps import here
def sparse_wigner_rotation(
    input: Tensor,
    sincos_alpha: Optional[Tensor],
    sincos_beta: Optional[Tensor],
    sincos_gamma: Optional[Tensor],
    info: WignerRotationInfo
) -> Tensor:
    """Applies sparse Wigner D-matrix rotation :math:`D(\alpha, \beta, \gamma) = D_z(\alpha)D_y(\beta)D_z(\gamma)`.

    Functional version of :class:`~equitorch.nn.wigner_d.SparseWignerRotation`.
    See the class for more details.

    Args:
        input (torch.Tensor): Input features.
        sincos_alpha (Optional[torch.Tensor]): Precomputed sin/cos of Euler angle :math:`\alpha`.
        sincos_beta (Optional[torch.Tensor]): Precomputed sin/cos of Euler angle :math:`\beta`.
        sincos_gamma (Optional[torch.Tensor]): Precomputed sin/cos of Euler angle :math:`\gamma`.
        info (WignerRotationInfo): Precomputed sparse rotation information.

    Returns:
        torch.Tensor: Rotated features.
    """

    x = input
    if sincos_gamma is not None:
        x = sparse_vecsca(x, sincos_gamma, 
                           info.rotate_z_info_fwd, 
                           info.rotate_z_info_bwd_input,
                           info.rotate_z_info_bwd_cs)
    if sincos_beta is not None:
        x = sparse_scale(x, info.j_matrix_info, info.j_matrix_info)
        x = sparse_vecsca(x, sincos_beta,
                        info.rotate_z_info_fwd, 
                        info.rotate_z_info_bwd_input,
                        info.rotate_z_info_bwd_cs)
        x = sparse_scale(x, info.j_matrix_info, info.j_matrix_info)
    if sincos_alpha is not None:
        x = sparse_vecsca(x, sincos_alpha,
                          info.rotate_z_info_fwd, 
                          info.rotate_z_info_bwd_input,
                          info.rotate_z_info_bwd_cs)

    return x

def dense_wigner_rotation(input: Tensor, wigner_d: Tensor):
    r"""Applies a precomputed dense Wigner D-matrix to input features.

    Functional version of :class:`~equitorch.nn.wigner_d.DenseWignerRotation`.
    See the class for more details.

    Args:
        input (torch.Tensor): Input features of shape ``(batch_size, irreps.dim, channels)``.
        wigner_d (torch.Tensor): Dense Wigner D-matrix of shape ``(batch_size, irreps.dim, irreps.dim)``.

    Returns:
        torch.Tensor: Rotated features.
    """
    return wigner_d @ input


def wigner_d_matrix(
    eye: Tensor, # Input identity matrix (dim, dim)
    alpha: Optional[Tensor] = None,
    beta: Optional[Tensor] = None,
    gamma: Optional[Tensor] = None,
    sincos_alpha: Optional[Tensor] = None,
    sincos_beta: Optional[Tensor] = None,
    sincos_gamma: Optional[Tensor] = None,
    info: Optional[WignerRotationInfo] = None
) -> Tensor:
    r"""
    Computes the dense Wigner D-matrix by applying sparse rotation operations
    to a provided identity matrix.

    Args:
        eye (Tensor): Identity matrix of shape (dim, dim). Device and dtype are inferred.
        alpha (Optional[Tensor]): Alpha Euler angle.
        beta (Optional[Tensor]): Beta Euler angle.
        gamma (Optional[Tensor]): Gamma Euler angle.
        sincos_alpha (Optional[Tensor]): Precomputed sin/cos for alpha angle.
        sincos_beta (Optional[Tensor]): Precomputed sin/cos for beta angle.
        sincos_gamma (Optional[Tensor]): Precomputed sin/cos for gamma angle.
        info (WignerRotationInfo): Precomputed sparse rotation info.

    Returns:
        Tensor: The dense Wigner D-matrix of shape (batch, dim, dim)
                or (dim, dim) if angles are not batched.
    """
    assert info is not None
    # Precompute sincos if angles are given
    if sincos_alpha is None and alpha is not None:
        sincos_alpha = sincos(alpha, info.max_m)
    if sincos_beta is None and beta is not None:
        sincos_beta = sincos(beta, info.max_m)
    if sincos_gamma is None and gamma is not None:
        sincos_gamma = sincos(gamma, info.max_m)

    dim = eye.shape[0]
    if dim == 0:
        # Handle empty irreps case based on eye matrix
        batch_size = 1 # Default if no angles
        if sincos_alpha is not None: batch_size = sincos_alpha.shape[0]
        elif sincos_beta is not None: batch_size = sincos_beta.shape[0]
        elif sincos_gamma is not None: batch_size = sincos_gamma.shape[0]

        return torch.empty((batch_size, 0, 0), device=eye.device, dtype=eye.dtype)

    # Reshape eye to (N, M, C) where N=1, M=dim, C=dim
    # The rotation acts on the M dimension, treating C as channels.

    # Apply sparse rotation
    # Assume info is already on the correct device (handled by the Module wrapper)
    rotated_eye = sparse_wigner_rotation(
        eye, sincos_alpha, sincos_beta, sincos_gamma, info
    ) # Output shape (N, dim, dim) where N is batch size from angles or 1

    # Determine if the operation was batched based on angle inputs
    is_batched = (sincos_alpha is not None and sincos_alpha.ndim > 1 and sincos_alpha.shape[0] > 1) or \
                 (sincos_beta is not None and sincos_beta.ndim > 1 and sincos_beta.shape[0] > 1) or \
                 (sincos_gamma is not None and sincos_gamma.ndim > 1 and sincos_gamma.shape[0] > 1)

    # If not batched and output has batch dim 1, squeeze it
    if not is_batched and rotated_eye.shape[0] == 1:
        return rotated_eye.squeeze(0) # Shape (dim, dim)
    else:
        return rotated_eye # Shape (batch, dim, dim)


def align_to_z_wigner_d(
    eye: Tensor,
    xyz: Tensor,
    max_m: int,
    info: WignerRotationInfo,
    normalized: bool = True,
    eps: float = 1e-14
) -> Tensor:
    r"""
    Computes Wigner D-matrix :math:`D(R_{align})` that rotates vector :math:`\vec{v}` to z-axis.

    Functional version of :class:`~equitorch.nn.wigner_d.AlignToZWignerD`.

    See the class for more details.

    The rotation :math:`R_{align}` is :math:`(0, -\theta, -\phi)` where :math:`\theta, \phi` are polar and azimuthal angles of :math:`\vec{v}`.

    Args:
        eye (torch.Tensor): Identity matrix of shape ``(dim, dim)``.
        xyz (torch.Tensor): Input Cartesian coordinates, shape ``(..., 3)``.
        max_m (int): Maximum m value for sincos calculation, derived from irreps.
        info (WignerRotationInfo): Precomputed sparse rotation info for the irreps.
        normalized (bool, optional): Whether ``xyz`` is already normalized. Defaults to ``True``.
        eps (float, optional): Epsilon for numerical stability. Defaults to ``1e-14``.

    Returns:
        torch.Tensor: Dense Wigner D-matrix for alignment, shape ``(..., dim, dim)``.
    """
    # 1. Calculate theta and phi from xyz
    theta, phi = xyz_to_spherical(
        xyz,
        input_normalized=normalized,
        with_r=False, # We don't need r
        eps=eps,
        dim=-1 # Assume xyz is in the last dimension
    ) # Output shapes (..., 1)

    # 2. Squeeze angles and calculate sincos embeddings
    theta_squeezed = theta.squeeze(-1)
    phi_squeezed = phi.squeeze(-1)

    sincos_neg_theta = sincos(-theta_squeezed, max_m)
    sincos_neg_phi = sincos(-phi_squeezed, max_m)

    # Infer batch shape from xyz (excluding the last dim)
    batch_shape = xyz.shape[:-1]
    # Create identity matrix without batch dim first
    # Expand eye matrix to match the batch shape of xyz/angles
    # Example: if xyz is (B, N, 3), batch_shape is (B, N)
    # We need eye to be broadcastable to (B, N, dim, dim) for wigner_d_matrix if angles are batched.
    # However, wigner_d_matrix expects eye (dim, dim) and handles batching internally based on angles.

    # 4. Compute Wigner D matrix D(alpha=phi, beta=theta, gamma=0)
    # gamma=0 means sincos_gamma=None
    wigner_d = wigner_d_matrix(
        eye=eye,
        sincos_alpha=None, # Gamma rotation is zero
        sincos_beta=sincos_neg_theta,
        sincos_gamma=sincos_neg_phi,
        info=info
    )

    return wigner_d
