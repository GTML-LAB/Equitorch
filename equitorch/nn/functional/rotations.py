import torch
from typing import Optional, Tuple

def angles_to_matrix(
    alpha: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    gamma: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Convert Euler angles (ZYZ convention) to rotation matrices.
    
    Args:
        alpha: First rotation angle about z-axis (radians)
        beta: Second rotation angle about y-axis (radians)
        gamma: Third rotation angle about z-axis (radians)
        
    Returns:
        Rotation matrices of shape (..., 3, 3)
    """
    if alpha is None and beta is None and gamma is None:
        raise ValueError("At least one of alpha, beta, or gamma must be provided.")

    # Determine broadcast shape
    shapes = []
    if alpha is not None:
        shapes.append(alpha.shape)
    if beta is not None:
        shapes.append(beta.shape)
    if gamma is not None:
        shapes.append(gamma.shape)
    
    if not shapes: # Should be caught by the None check above, but as a safeguard
        return torch.eye(3)

    # Find a non-None tensor to get dtype and device
    ref_tensor = alpha if alpha is not None else beta if beta is not None else gamma
    assert ref_tensor is not None # Ensured by the first check

    # Broadcast all angles to the same shape
    # We need to handle the case where an angle is None by creating a zero tensor of the broadcasted shape
    # First, let's find the target broadcast shape using torch.broadcast_shapes
    # To do this, we need at least one tensor. If an angle is None, we can't directly use it.
    # Instead, we'll create dummy tensors of shape (1,) for None angles to participate in broadcasting shape calculation.
    
    dummy_alpha = alpha if alpha is not None else torch.zeros((1,), dtype=ref_tensor.dtype, device=ref_tensor.device)
    dummy_beta = beta if beta is not None else torch.zeros((1,), dtype=ref_tensor.dtype, device=ref_tensor.device)
    dummy_gamma = gamma if gamma is not None else torch.zeros((1,), dtype=ref_tensor.dtype, device=ref_tensor.device)

    try:
        broadcast_shape = torch.broadcast_shapes(dummy_alpha.shape, dummy_beta.shape, dummy_gamma.shape)
    except RuntimeError as e:
        raise ValueError(f"Could not broadcast shapes of alpha, beta, gamma: {shapes}. Error: {e}")


    zeros = torch.zeros(broadcast_shape, dtype=ref_tensor.dtype, device=ref_tensor.device)
    ones = torch.ones(broadcast_shape, dtype=ref_tensor.dtype, device=ref_tensor.device)

    # Rotation around Z-axis by alpha
    if alpha is not None:
        expanded_alpha = alpha.expand_as(zeros)
        sin_a = torch.sin(expanded_alpha)
        cos_a = torch.cos(expanded_alpha)
    else:
        sin_a, cos_a = zeros, ones # No rotation
    
    # R_z(alpha)
    # yapf: disable
    R_alpha = torch.stack([
        cos_a, -sin_a, zeros,
        sin_a,  cos_a, zeros,
        zeros,  zeros,  ones
    ], dim=-1).reshape(broadcast_shape + (3, 3))
    # yapf: enable

    # Rotation around Y-axis by beta
    if beta is not None:
        expanded_beta = beta.expand_as(zeros)
        sin_b = torch.sin(expanded_beta)
        cos_b = torch.cos(expanded_beta)
    else:
        sin_b, cos_b = zeros, ones # No rotation

    # R_y(beta)
    # yapf: disable
    R_beta = torch.stack([
        cos_b,  zeros, sin_b,
        zeros,   ones, zeros,
       -sin_b,  zeros, cos_b
    ], dim=-1).reshape(broadcast_shape + (3, 3))
    # yapf: enable

    # Rotation around Z-axis by gamma
    if gamma is not None:
        expanded_gamma = gamma.expand_as(zeros)
        sin_g = torch.sin(expanded_gamma)
        cos_g = torch.cos(expanded_gamma)
    else:
        sin_g, cos_g = zeros, ones # No rotation

    # R_z(gamma)
    # yapf: disable
    R_gamma = torch.stack([
        cos_g, -sin_g, zeros,
        sin_g,  cos_g, zeros,
        zeros,  zeros,  ones
    ], dim=-1).reshape(broadcast_shape + (3, 3))
    # yapf: enable
    
    R_matrix = R_gamma @ R_beta @ R_alpha # More concise

    return R_matrix
