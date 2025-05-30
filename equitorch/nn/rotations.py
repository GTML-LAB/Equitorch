import torch
from torch import nn
from typing import Optional

from .functional.rotations import angles_to_matrix


class AnglesToMatrix(nn.Module):
    r"""Module to convert Euler angles (ZYZ convention) to rotation matrices.

    The ZYZ Euler angles \(\alpha, \beta, \gamma\) correspond to the rotation matrix:

    .. math::
        R(\alpha, \beta, \gamma) = R_z(\alpha) R_y(\beta) R_z(\gamma)

    which is explicitly:

    .. math::
        \begin{pmatrix}
        -\sin(\alpha)\sin(\gamma) + \cos(\alpha)\cos(\beta)\cos(\gamma) & -\sin(\alpha)\cos(\beta)\cos(\gamma) - \sin(\gamma)\cos(\alpha) & \sin(\beta)\cos(\gamma) \\
        \sin(\alpha)\cos(\gamma) + \sin(\gamma)\cos(\alpha)\cos(\beta) & -\sin(\alpha)\sin(\gamma)\cos(\beta) + \cos(\alpha)\cos(\gamma) & \sin(\beta)\sin(\gamma) \\
        -\sin(\beta)\cos(\alpha) & \sin(\alpha)\sin(\beta) & \cos(\beta)
        \end{pmatrix}

    Wraps the functional version :func:`~equitorch.nn.functional.rotations.angles_to_matrix`.
    """
    def __init__(self):
        super().__init__()

    def forward(
        self,
        alpha: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
        gamma: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        r"""
        Args:
            alpha: First rotation angle about z-axis (radians). Shape (...)
            beta: Second rotation angle about y-axis (radians). Shape (...)
            gamma: Third rotation angle about z-axis (radians). Shape (...)
            
        Returns:
            Rotation matrices of shape (..., 3, 3)
        """
        return angles_to_matrix(alpha=alpha, beta=beta, gamma=gamma)

    def extra_repr(self) -> str:
        # This module has no parameters to display.
        return ""
