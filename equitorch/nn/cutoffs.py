import torch
import torch.nn as nn

from .functional.cutoffs import radial_standarize, polynomial_cutoff, cosine_cutoff, mollifier_cutoff

# adapted from https://github.com/mir-group/nequip/blob/v0.6.2/nequip/nn/cutoffs.py
class PolynomialCutoff(torch.nn.Module):
    r"""Polynomial cutoff, as proposed in `DimeNet <https://arxiv.org/abs/2003.03123>`_.

    The polynomial cutoff function is defined as:

    .. math::
        f(r) = \begin{cases}
        1, & r < r_{\text{min}} \\
        1 - \frac{(p+1)(p+2)}{2}u^p + p(p+2)u^{p+1} - \frac{p(p+1)}{2}u^{p+2}, & r_{\text{min}} \leq r \leq r_{\text{max}} \\
        0, & r > r_{\text{max}}
        \end{cases}

    where \(u = \frac{r - r_{\text{min}}}{r_{\text{max}} - r_{\text{min}}}\) and \(r\) is the input distance.

    Args:
        r_max (float): The cutoff distance \(r_{\text{max}}\) where the function reaches zero.
        r_min (float, optional): The starting distance \(r_{\text{min}}\) where the function begins to decrease from 1.
            Must be less than or equal to ``r_max``. Defaults to ``0.``.
        p (float, optional): The power parameter \(p\) controlling the smoothness of the cutoff.
            Must be greater than or equal to ``2.0``. Defaults to ``6.``.
    """
    def __init__(self, r_max: float, r_min: float =0., p: float = 6):
        super().__init__()
        assert r_max >= r_min
        assert p >= 2.0
        self.p = float(p)
        self.r_max = r_max
        self.r_min = r_min
        self.range = float(r_max-r_min)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""Evaluate cutoff function.

        Args:
            input (torch.Tensor): Input distance tensor \(r\).

        Returns:
            torch.Tensor: Cutoff values for the input distances \(f(r)\).
        """
        input = radial_standarize(input, self.range, self.r_min) 
        return polynomial_cutoff(input, p=self.p)

class CosineCutoff(nn.Module):
    r"""The cosine cutoff function.

    The cosine cutoff function is defined as:

    .. math::
        f(r) = \begin{cases}
        1, & r < r_{\text{min}} \\
        \frac{1}{2}\left[1 + \cos\left(\pi \cdot u\right)\right], & r_{\text{min}} \leq r \leq r_{\text{max}} \\
        0, & r > r_{\text{max}}
        \end{cases}

    where \(u = \frac{r - r_{\text{min}}}{r_{\text{max}} - r_{\text{min}}}\) and \(r\) is the input distance.

    This cutoff function smoothly decreases from 1 to 0 in the range
    \[r_{\text{min}}, r_{\text{max}}\] using a cosine function.

    Args:
        r_max (float): The cutoff distance \(r_{\text{max}}\) where the function reaches zero.
        r_min (float, optional): The starting distance \(r_{\text{min}}\) where the function begins to decrease from 1.
            Must be less than ``r_max``. Defaults to ``0.``.
    """
    def __init__(self, r_max: float, r_min: float = 0):
        super().__init__()        
        assert r_min < r_max
        self.r_max = r_max
        self.r_min = r_min
        self.range = float(r_max - r_min)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Evaluate cutoff function.

        Args:
            input (torch.Tensor): Input distance tensor \(r\).

        Returns:
            torch.Tensor: Cutoff values for the input distances \(f(r)\).
        """
        input = radial_standarize(input, self.range, self.r_min)
        return cosine_cutoff(input)

class MollifierCutoff(nn.Module):
    r"""The mollifier cutoff function.

    The mollifier cutoff function is defined as:

    .. math::
        f(r) = \begin{cases}
        1, & r < r_{\text{min}} \\
        \exp\left(1 - \frac{1}{1 - u^2 + \epsilon}\right), & r_{\text{min}} \leq r \leq r_{\text{max}} \\
        0, & r > r_{\text{max}}
        \end{cases}

    where \(u = \frac{r - r_{\text{min}}}{r_{\text{max}} - r_{\text{min}}}\) and \(r\) is the input distance.

    This cutoff function smoothly decreases from 1 to 0 in the range
    \[r_{\text{min}}, r_{\text{max}}\] using a mollifier (bump) function.

    Args:
        r_max (float): The cutoff distance \(r_{\text{max}}\) where the function reaches zero.
        r_min (float, optional): The starting distance \(r_{\text{min}}\) where the function begins to decrease from 1.
            Must be less than ``r_max``. Defaults to ``0.``.
        eps (float, optional): Small epsilon value \(\epsilon\) to prevent division by zero.
            Defaults to ``1e-7``.
    """
    def __init__(self, r_max: float, r_min: float = 0, eps: float = 1e-7):
        super().__init__()
        assert r_min < r_max
        self.r_max = r_max
        self.r_min = r_min
        self.range = float(r_max - r_min)
        self.eps = eps
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Evaluate cutoff function.

        Args:
            input (torch.Tensor): Input distance tensor \(r\).

        Returns:
            torch.Tensor: Cutoff values for the input distances \(f(r)\).
        """
        input = radial_standarize(input, self.range, self.r_min)
        return mollifier_cutoff(input, eps=self.eps)