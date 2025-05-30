import torch
import math
from typing import Tuple

def rand_spherical_xyz(shape: Tuple[int, ...], device=None, dtype=None) -> torch.Tensor:
    """Generate random points uniformly distributed on a unit sphere.
    
    Args:
        shape: Tuple defining the batch dimensions (e.g., (10,) or (5,5))
        device: Torch device for the output tensor
        dtype: Torch dtype for the output tensor
        
    Returns:
        Tensor of shape ``(*shape, 3)`` where each vector has unit norm
    """
    # Generate random points and normalize
    xyz = torch.randn(*shape, 3, device=device, dtype=dtype)
    return xyz / xyz.norm(dim=-1, keepdim=True)

def rand_spherical_angles(shape: Tuple[int, ...], device=None, dtype=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate random spherical angles with uniform distribution.
    
    Args:
        shape: Tuple defining the batch dimensions
        device: Torch device for the output tensors
        dtype: Torch dtype for the output tensors
        
    Returns:
        Tuple of (theta, phi) where:
        - theta is in [0, π) (polar angle from +z axis)
        - phi is in [0, 2π) (azimuthal angle from +x axis)
    """
    # Uniform theta using inverse transform sampling
    theta = torch.acos(2 * torch.rand(shape, device=device, dtype=dtype) - 1)
    # Uniform phi
    phi = 2 * math.pi * torch.rand(shape, device=device, dtype=dtype)
    return theta, phi

def rand_rotation_angles(shape: Tuple[int, ...], device=None, dtype=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate random Euler angles (ZYZ convention) with uniform distribution.
    
    Args:
        shape: Tuple defining the batch dimensions
        device: Torch device for the output tensors
        dtype: Torch dtype for the output tensors
        
    Returns:
        Tuple of (alpha, beta, gamma) where:
        - alpha is in [0, 2π) (first rotation about z-axis)
        - beta is in [0, π) (rotation about y-axis)
        - gamma is in [0, 2π) (second rotation about z-axis)
    """
    beta = torch.acos(2 * torch.rand(shape, device=device, dtype=dtype) - 1)
    alpha = 2 * math.pi * torch.rand(shape, device=device, dtype=dtype)
    gamma = 2 * math.pi * torch.rand(shape, device=device, dtype=dtype)
    return alpha, beta, gamma

def rand_rotation_matrices(shape: Tuple[int, ...], device=None, dtype=None) -> torch.Tensor:
    """Generate random rotation matrices using Rodrigues' rotation formula.
    
    Args:
        shape: Tuple defining the batch dimensions
        device: Torch device for the output tensor
        dtype: Torch dtype for the output tensor
        
    Returns:
        Tensor of shape (*shape, 3, 3) containing valid rotation matrices
    """
    # Generate random rotation axis (unit vector) and angle
    axis = rand_spherical_xyz(shape, device, dtype)
    angle = 2 * math.pi * torch.rand(shape, device=device, dtype=dtype)
    
    # Rodrigues' rotation formula components
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one_minus_cos = 1 - cos
    
    # Cross product matrix [a]×
    a_x = torch.zeros(*shape, 3, 3, device=device, dtype=dtype)
    a_x[..., 0, 1] = -axis[..., 2]
    a_x[..., 0, 2] = axis[..., 1]
    a_x[..., 1, 0] = axis[..., 2]
    a_x[..., 1, 2] = -axis[..., 0]
    a_x[..., 2, 0] = -axis[..., 1]
    a_x[..., 2, 1] = axis[..., 0]
    
    # Outer product a ⊗ a
    a_outer = axis.unsqueeze(-1) * axis.unsqueeze(-2)
    
    # Rodrigues' formula: R = cosθ I + sinθ [a]× + (1-cosθ) a ⊗ a
    eye = torch.eye(3, device=device, dtype=dtype).expand(*shape, 3, 3)
    return (cos[..., None, None] * eye + 
            sin[..., None, None] * a_x + 
            one_minus_cos[..., None, None] * a_outer)
