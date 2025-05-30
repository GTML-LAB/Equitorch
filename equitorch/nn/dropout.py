import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from ..utils._structs import irreps_info

from ..irreps import Irreps
from .functional.dropout import irrep_wise_dropout

class Dropout(nn.Module):
    r"""
    Apply dropout to equivariant features.

    Can operate irrep-wise or on the entire feature vector (channel-wise).

    Args:
        p (float, optional): Probability of an element to be zeroed.
                             Default: 0.5
        irreps (Irreps, optional): Irreps of the input tensor.
                                   Required if `irrep_wise` is True.
                                   Default: None
        irrep_wise (bool, optional): If True, applies dropout independently
                                     for each (irrep_instance, channel).
                                     If False, applies standard 1D dropout
                                     treating (irreps_dim, channels) as a
                                     single feature dimension for dropout.
                                     Default: True
        work_on_eval (bool, optional): If True, dropout is applied even during
                                       evaluation. Default: False
    """
    def __init__(self, p: float = 0.5, 
                 irreps: Irreps = None, 
                 irrep_wise: bool = True, 
                 work_on_eval: bool = False):
        super().__init__()
        if p < 0.0 or p > 1.0:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
        if irrep_wise and irreps is None:
            raise ValueError("irreps must be provided if irrep_wise is True")

        self.p = p
        self.irrep_wise = irrep_wise
        self.work_on_eval = work_on_eval
        
        if irrep_wise:
            # We need IrrepsInfo for irrep_wise_dropout
            # Construct it once and store it if irreps are provided
            self.irreps_info = irreps_info(irreps) if irreps is not None else None
        else:
            self.irreps_info = None # Not needed for non-irrep-wise

    def forward(self, input: Tensor) -> Tensor:
        r"""
        Args:
            input (Tensor): Input tensor of shape (N, irreps_dim, C)
        Returns:
            Tensor: Output tensor with dropout applied.
        """
        assert input.ndim >= 2, "Input tensor must have at least 2 dimensions (irreps_dim, channels)"


        if not self.work_on_eval and not self.training:
            return input
        
        if self.p == 0.0: # No dropout if p is 0
            return input

        if self.irrep_wise:            
            return irrep_wise_dropout(input, self.p, self.training or self.work_on_eval, self.irreps_info)
        else:
            
            x = input.transpose(-1, -2)
            x_dropout = F.dropout1d(x, self.p, self.training or self.work_on_eval)
            output = x_dropout.transpose(-1, -2)
            
            return output.contiguous() # Ensure contiguous after transpose

    def extra_repr(self) -> str:
        return f'p={self.p}, irrep_wise={self.irrep_wise}, work_on_eval={self.work_on_eval}'

    def _apply(self, *args, **kwargs):
        d = super()._apply(*args, **kwargs)
        if d.irreps_info is not None:
            d.irreps_info = self.irreps_info._apply(*args, **kwargs)
        return d