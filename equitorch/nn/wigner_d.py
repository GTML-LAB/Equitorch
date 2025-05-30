from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn

from ..irreps import check_irreps, Irreps
# Import the functional wigner_d_matrix and prepare_sincos (if needed, or assume angles are precomputed)
from .functional.wigner_d import sparse_wigner_rotation, wigner_d_matrix
from .functional.angular import sincos
from .functional.wigner_d import align_to_z_wigner_d
from ..structs import WignerRotationInfo
# Assuming wigner_d_info exists and prepares WignerRotationInfo correctly
from ..utils._structs import wigner_d_info

class SparseWignerRotation(nn.Module):
    r"""
    Applies a sparse Wigner D-matrix rotation to input features.

    This module computes the rotation based on Euler angles (:math:`\alpha, \beta, \gamma`)
    provided as precomputed sin/cos tensors. It utilizes sparse matrix operations for the rotation.

    .. warning::
        It is currently suggested to use :class:`DenseWignerRotation` or :class:`WignerD` 
        for applying rotations, at least when gradients with respect to angles are not required.

    The Wigner D-matrix :math:`D^l_{m'm}(R)` transforms spherical tensors under rotation :math:`R`.
    This module applies such transformations for a given :class:`~equitorch.irreps.Irreps`.

    Args:
        irreps (Irreps): The irreducible representations defining the input and output feature space.
    """

    info: WignerRotationInfo

    def __init__(self, irreps: Irreps):
        super().__init__()

        self.irreps = check_irreps(irreps)
        self.info = wigner_d_info(self.irreps)
        self.max_m = max(ir.l for ir in irreps)

    def forward(self, input: Tensor, 
                sincos_alpha: Optional[Tensor], 
                sincos_beta: Optional[Tensor],
                sincos_gamma: Optional[Tensor]) -> Tensor:
        r"""
        Applies the sparse Wigner rotation.

        Args:
            input (torch.Tensor): Input features of shape ``(batch_size, irreps.dim, channels)``.
            sincos_alpha (Optional[torch.Tensor]): Precomputed :math:`\sin` and :math:`\cos` of Euler angle :math:`\alpha`.
            sincos_beta (Optional[torch.Tensor]): Precomputed :math:`\sin` and :math:`\cos` of Euler angle :math:`\beta`.
            sincos_gamma (Optional[torch.Tensor]): Precomputed :math:`\sin` and :math:`\cos` of Euler angle :math:`\gamma`.

        Returns:
            torch.Tensor: Rotated features of shape ``(batch_size, irreps.dim, channels)``.
        """
        return sparse_wigner_rotation(input, sincos_alpha,
                        sincos_beta, sincos_gamma,
                        self.info)
    
    
    def _apply(self, *args, **kwargs):
        # Ensure info objects are moved to the correct device/dtype
        wig = super()._apply(*args, **kwargs)
        # Apply to the WignerDRotationInfo NamedTuple fields
        wig.info = wig.info._apply(*args, **kwargs)
        return wig

    def extra_repr(self) -> str:
        return f'irreps={self.irreps}, max_m={self.max_m}'

class DenseWignerRotation(nn.Module):
    r"""
    Applies a dense Wigner D-matrix rotation to input features.

    This module takes a precomputed dense Wigner D-matrix and applies it to the input features.
    The Wigner D-matrix :math:`D(R)` itself should be computed separately, for example, using the :class:`WignerD` module.

    Args:
        irreps (Irreps): The irreducible representations defining the input and output feature space.
                         This is used for validation and representation purposes.
    """
    def __init__(self, irreps: Irreps):
        super().__init__()

        self.irreps = check_irreps(irreps)

    def forward(self, input: Tensor, wigner_d: Tensor) -> Tensor:
        r"""
        Applies the dense Wigner rotation.

        The operation performed is effectively a batched matrix multiplication:
        `output = wigner_d @ input`

        Args:
            input (Tensor): Input features of shape ``(batch_size, irreps.dim, channels)``.
            wigner_d (Tensor): Dense Wigner D-matrix of shape ``(batch_size, irreps.dim, irreps.dim)``.

        Returns:
            Tensor: Rotated features of shape ``(batch_size, irreps.dim, channels)``.
        """
        # Using einsum for clarity on dimensions, equivalent to batched matmul
        # D_bij, F_bjk -> R_bik (b=batch, i=output_dim, j=input_dim, k=channels)
        # For Wigner D, input_dim and output_dim are the same (self.irreps.dim)
        return wigner_d @ input # [N, dim, dim] @ [N, dim, channels] -> [N, dim, channels]

    
    def extra_repr(self) -> str:
        return f'irreps={self.irreps}'


class WignerD(nn.Module):
    r"""
    Computes the dense Wigner D-matrix :math:`D(R)` for given :class:`~equitorch.irreps.Irreps` and Euler angles :math:`(\alpha, \beta, \gamma)`.

    The Wigner D-matrix is constructed based on the ZYZ Euler angle convention:

    .. math::

        D(\alpha, \beta, \gamma) = D_z(\alpha) D_y(\beta) D_z(\gamma)

    This module caches the necessary sparse rotation information and an identity matrix
    to efficiently compute the dense D-matrix using the :func:`~equitorch.nn.functional.wigner_d.wigner_d_matrix` functional.

    Args:
        irreps (Irreps): The irreducible representations for which to compute the D-matrix.
                         The resulting D-matrix will have dimensions ``(irreps.dim, irreps.dim)``.
    """
    info: WignerRotationInfo

    def __init__(self, irreps: Irreps):
        super().__init__()
        self.irreps = check_irreps(irreps)
        self.max_m = max((ir.l for ir in self.irreps), default=0)

        # Prepare and store sparse rotation info
        self.info = wigner_d_info(self.irreps)

        # Prepare and register identity matrix as a buffer
        dim = self.irreps.dim
        self.register_buffer('identity', torch.eye(dim))

    def forward(self,
                alpha: Optional[Tensor] = None,
                beta: Optional[Tensor] = None,
                gamma: Optional[Tensor] = None,
                sincos_alpha: Optional[Tensor] = None,
                sincos_beta: Optional[Tensor] = None,
                sincos_gamma: Optional[Tensor] = None) -> Tensor:
        r"""
        Computes the Wigner D-matrix.

        Provide either the angles (alpha, beta, gamma) or the precomputed
        sin/cos tensors (sincos_alpha, sincos_beta, sincos_gamma).

        Args:
            alpha (Optional[torch.Tensor]): Euler angle alpha.
            beta (Optional[torch.Tensor]): Euler angle beta.
            gamma (Optional[torch.Tensor]): Euler angle gamma.
            sincos_alpha (Optional[torch.Tensor]): Precomputed sin/cos for alpha.
            sincos_beta (Optional[torch.Tensor]): Precomputed sin/cos for beta.
            sincos_gamma (Optional[torch.Tensor]): Precomputed :math:`\sin` and :math:`\cos` of Euler angle :math:`\gamma`.

        Returns:
            torch.Tensor: The dense Wigner D-matrix of shape ``(batch_size, irreps.dim, irreps.dim)``
                          if batch size is 1 and input angles are unbatched.
        """
        return wigner_d_matrix(
            self.identity, # Use cached identity matrix
            alpha,
            beta,
            gamma,
            sincos_alpha,
            sincos_beta,
            sincos_gamma,
            self.info # Use cached info
        )

    def _apply(self, *args, **kwargs):
        # Ensure info objects are moved to the correct device/dtype
        wig = super()._apply(*args, **kwargs)
        # Apply to the WignerDRotationInfo NamedTuple fields
        wig.info = wig.info._apply(*args, **kwargs)
        # Identity matrix buffer is handled automatically by PyTorch
        return wig

    def extra_repr(self) -> str:
        return f'irreps={self.irreps}'


class AlignToZWignerD(nn.Module):
    r"""
    Computes the Wigner D-matrix :math:`D(R_{align})` that rotates a given vector :math:`\vec{v} = (x, y, z)` onto the z-axis.

    The rotation :math:`R_{align}` is defined by Euler angles :math:`(0, -\theta, -\phi)`, where :math:`\theta` and :math:`\phi` are the
    polar and azimuthal angles of the vector :math:`\vec{v}`, respectively. This means:

    .. math::

        R_{align} \vec{v} = ||\vec{v}|| \hat{z}

    The Wigner D-matrix is then :math:`D(0, -\theta, -\phi)`.

    This module caches the necessary sparse rotation information and an identity matrix.
    It utilizes the :func:`~equitorch.nn.functional.wigner_d.align_to_z_wigner_d` functional.

    Args:
        irreps (Irreps): The irreducible representations for which to compute the D-matrix.
        normalized (bool, optional): Whether to normalize the input ``xyz`` vector
            before calculating angles for rotation. If ``True``, effectively rotates :math:`\hat{v}`.
            Defaults to ``True``.
        eps (float, optional): Small :math:`\epsilon` value for numerical stability in angle calculation.
            Defaults to ``1e-14``.
    """
    info: WignerRotationInfo

    def __init__(self,
                 irreps: Irreps,
                 normalized: bool = True,
                 eps: float = 1e-14):
        super().__init__()
        self.irreps = check_irreps(irreps)
        self.normalized = normalized
        self.eps = eps
        self.max_m = max((ir.l for ir in self.irreps), default=0)

        # Prepare and store sparse rotation info
        self.info = wigner_d_info(self.irreps)

        # Prepare and register identity matrix as a buffer
        dim = self.irreps.dim
        self.register_buffer('identity', torch.eye(dim))

    def forward(self, xyz: Tensor) -> Tensor:
        r"""
        Computes the alignment Wigner D-matrix.

        Args:
            xyz (Tensor): Input Cartesian coordinates, shape (..., 3).

        Returns:
            Tensor: The dense Wigner D-matrix for alignment.
                    Shape (..., irreps.dim, irreps.dim).
        """
        # Import functional align_to_z_wigner_d here to avoid circular dependency at top level
        # if functional layer imports from nn layer (though unlikely)
        
        return align_to_z_wigner_d(
            eye=self.identity,
            xyz=xyz,
            max_m=self.max_m,
            info=self.info,
            normalized=self.normalized,
            eps=self.eps
        )

    def _apply(self, *args, **kwargs):
        # Ensure info objects are moved to the correct device/dtype
        wig = super()._apply(*args, **kwargs)
        # Apply to the WignerDRotationInfo NamedTuple fields
        wig.info = wig.info._apply(*args, **kwargs)
        # Identity matrix buffer is handled automatically by PyTorch
        return wig

    def extra_repr(self) -> str:
        return (f'irreps={self.irreps}, normalized={self.normalized}, '
                f'eps={self.eps}')
