import torch

@torch.jit.script
def radial_standarize(input: torch.Tensor, range: float, r_min: float):
    r"""Standardize radial distances for cutoff functions.

    Transforms input distances :math:`r` to a normalized range :math:`u \in [0, 1]` using:

    .. math::
        u = \text{clamp}\left(\frac{r - r_{\text{min}}}{r_{\text{max}} - r_{\text{min}}}, 0, 1\right)

    where ``range`` = :math:`r_{\text{max}} - r_{\text{min}}`.

    Args:
        input (torch.Tensor): Input distance tensor.
        range (float): The difference :math:`r_{\text{max}} - r_{\text{min}}`.
        r_min (float): The minimum cutoff distance :math:`r_{\text{min}}`.

    Returns:
        torch.Tensor: Standardized distances.
    """
    if r_min != 0:
        input = input - r_min
    input = (input / range).clamp(0,1)
    return input


# adapted from https://github.com/mir-group/nequip/blob/v0.6.2/nequip/nn/cutoffs.py
@torch.jit.script
def polynomial_cutoff(input: torch.Tensor, p: float = 6.0) -> torch.Tensor:
    r"""Polynomial cutoff function.

    This is the functional version of the :class:`~equitorch.nn.cutoffs.PolynomialCutoff` module.
    See :class:`~equitorch.nn.cutoffs.PolynomialCutoff` for the mathematical formula and more details.

    Args:
        input (torch.Tensor): Standardized distance tensor, :math:`u \in [0, 1]`.
        p (float, optional): Power parameter. Defaults to ``6.0``.

    Returns:
        torch.Tensor: Cutoff values.
    """
    
    out = 1.0
    out = out - (((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(input, p))
    out = out + (p * (p + 2.0) * torch.pow(input, p + 1.0))
    out = out - ((p * (p + 1.0) / 2) * torch.pow(input, p + 2.0))

    return out

@torch.jit.script
def cosine_cutoff(input: torch.Tensor) -> torch.Tensor:
    r"""Cosine cutoff function.

    This is the functional version of the :class:`~equitorch.nn.cutoffs.CosineCutoff` module.
    See :class:`~equitorch.nn.cutoffs.CosineCutoff` for the mathematical formula and more details.

    Args:
        input (torch.Tensor): Standardized distance tensor, :math:`u \in [0, 1]`.

    Returns:
        torch.Tensor: Cutoff values.
    """
    return 0.5 * (1.0 + torch.cos(torch.pi * input))

@torch.jit.script
def mollifier_cutoff(input: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    r"""Mollifier cutoff function.

    This is the functional version of the :class:`~equitorch.nn.cutoffs.MollifierCutoff` module.
    See :class:`~equitorch.nn.cutoffs.MollifierCutoff` for the mathematical formula and more details.

    Args:
        input (torch.Tensor): Standardized distance tensor, :math:`u \in [0, 1]`.
        eps (float, optional): Small epsilon for numerical stability. Defaults to ``1e-12``.

    Returns:
        torch.Tensor: Cutoff values.
    """
    return torch.exp(1-1/(1-input**2+eps))