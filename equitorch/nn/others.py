import torch
from torch import nn

from ..irreps import Irreps, check_irreps

import torch
from torch import Tensor
from torch.nn import ModuleList 
from typing import Callable, Iterable, List, Union


class SplitIrreps(torch.nn.Module):
    r"""
    A module that splits input tensors according to specified irrep segments.

    The splitting is done based on the dimensions of the irreducible representations.

    Args:
        irreps (Irreps): The irreducible representations specification.
        split_num_irreps (Iterable[int]): Number of irreps in each split segment.
            Must sum to total irreps. May contain at most one -1 or ``...``
            to represent the remaining irreps.
        dim (int, optional): The dimension along which to split the input tensor.
            Defaults to -2.

    Examples:
        >>> # Split 3 scalar irreps (dim=1 each) and 2 vector irreps (dim=3 each)
        >>> irreps = Irreps("3x0e + 2x1o")  # 3 scalars + 2 vectors
        >>> split = SplitIrreps(irreps, [2, -1])  # Split first 2 irreps, then remaining
        >>> x = torch.randn(5, irreps.dim, 10)  # batch=5, dim=9 (3*1 + 2*3), channels=10
        >>> splits = split(x)  # Returns list of 2 tensors
        >>> splits[0].shape
        torch.Size([5, 2, 10])
        >>> splits[1].shape
        torch.Size([5, 7, 10])

        >>> # Using ... for automatic size calculation
        >>> split = SplitIrreps(irreps, [1, ...])  # First irrep, then remaining
        >>> splits = split(x)
        >>> splits[0].shape
        torch.Size([5, 1, 10])
        >>> splits[1].shape
        torch.Size([5, 8, 10])
    """
    
    def __init__(self, irreps: Irreps, split_num_irreps: Iterable[int], dim: int = -2):
        super().__init__()
        self.irreps = check_irreps(irreps)
        self.dim = dim
        
        # Handle -1 or ... in split_num_irreps
        if sum(1 for x in split_num_irreps if x in (-1, ...)) > 1:
            raise ValueError("split_num_irreps can contain at most one -1 or ...")
            
        # Calculate the remaining count if -1 or ... is present
        if -1 in split_num_irreps or ... in split_num_irreps:
            total_irreps = len(self.irreps)
            specified = sum(x for x in split_num_irreps if x not in (-1, ...))
            remaining = total_irreps - specified
            self.split_num_irreps = [
                remaining if x in (-1, ...) else x 
                for x in split_num_irreps
            ]
        else:
            self.split_num_irreps = list(split_num_irreps)

        # Validate the split sizes
        if sum(self.split_num_irreps) != len(self.irreps):
            raise ValueError(
                f"Sum of split_num_irreps ({sum(self.split_num_irreps)}) "
                f"must equal to the number of total irreps ({len(self.irreps)})"
            )

        # Precompute the split sizes based on irrep dimensions
        self.input_dim = irreps.dim
        self.split_sizes = []
        acc = 0
        self.irreps_out = []
        for num in self.split_num_irreps:
            self.irreps_out.append([])
            current_size = 0
            for i in range(num):
                current_size += irreps[acc + i].dim
                self.irreps_out[-1].append(irreps[acc+i])
            self.split_sizes.append(current_size)
            self.irreps_out[-1] = Irreps(self.irreps_out[-1]).merged()
            acc += num

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Shape check for input dimension
        if x.shape[self.dim] != self.input_dim:
             raise ValueError(f"Input tensor dimension {x.shape[self.dim]} does not match expected irreps dimension {self.input_dim}")
        return torch.split(x, self.split_sizes, dim=self.dim) # Split along the specified dimension

    def __repr__(self):
        # Use the short_repr for a concise irreps representation
        # Format the list of output irreps using their short_repr
        irreps_out_repr = ', '.join([f"{ir.short_repr()}" for ir in self.irreps_out])
        return (f"{self.__class__.__name__}("
                f"irreps_in={self.irreps.short_repr()}, " # Renamed for clarity
                f"irreps_out=[{irreps_out_repr}]" # Show the list of resulting Irreps objects
                ")")
    

class Separable(nn.Module):
    r"""
    A module that applies different transformations to different parts of input tensor
    according to irreducible representations (irreps), with optional concatenation.

    Args:
        irreps (Irreps): The irreducible representations specification for the input tensor.
        split_num_irreps (Iterable[int]): Number of irreps in each split segment.
            May contain at most one -1 or ``...`` to represent remaining irreps.
            Length must match length of ``sub_modules``.
        sub_modules (Iterable[Callable]): Transformation modules for each split segment.
            Use ``None`` for identity operation.
        cat_after (bool, optional): Whether to concatenate results after transformation.
            Defaults to True.
        dim (int, optional): The dimension along which to split and concatenate tensors.
            Defaults to -2.

    Raises:
        ValueError:
            - If lengths of ``split_num_irreps`` and ``sub_modules`` don't match.
            - If sum of ``split_num_irreps`` doesn't match total irreps.
            - If invalid ``split_num_irreps`` specification.
    """
    
    def __init__(self, 
                 irreps: Irreps, 
                 split_num_irreps: Iterable[int],
                 sub_modules: Iterable[nn.Module],
                 cat_after: bool = True,
                 dim: int = -2):
        super().__init__()

        self.irreps = check_irreps(irreps)
        self.dim = dim
        self.split = SplitIrreps(self.irreps, split_num_irreps, dim=self.dim)
        self.cat_after = cat_after

        # Convert to lists for length checking
        split_num_irreps = list(split_num_irreps)
        sub_modules = list(sub_modules)
        
        if len(split_num_irreps) != len(sub_modules):
            raise ValueError(
                f"Length of split_num_irreps ({len(split_num_irreps)}) must match "
                f"length of sub_modules ({len(sub_modules)})"
            )

        self.sub_modules = nn.ModuleList(sub_modules)

    def forward(self, x: Tensor) -> Union[Tensor, List[Tensor]]:
        r"""
        Apply separate transformations to different irrep segments of input.

        Args:
            x (Tensor): Input tensor. The size of the dimension specified by the ``dim``
                parameter (default: -2) should match the total dimension of all
                irreps (``irreps.dim``). For example, if ``dim=-2``, ``x`` could be of
                shape ``[batch, irreps.dim, channels]``.

        Returns:
            Union[Tensor, List[Tensor]]: Transformed tensors concatenated along ``dim``
            if ``cat_after`` is True, otherwise a list of transformed tensors.

        Raises:
            RuntimeError: If input dimension doesn't match ``self.split.input_dim``.
        """
        if x.shape[self.dim] != self.split.input_dim:
            raise RuntimeError(
                f"Input dimension ({x.shape[self.dim]}) must match "
                f"irreps dimension ({self.split.input_dim})"
            )

        x_splited = self.split(x)
        
        # Fixed typo: self.module -> self.sub_modules
        x_transformed = [
            m(x_split) if m is not None else x_split 
            for m, x_split in zip(self.sub_modules, x_splited)
        ]

        if self.cat_after:
            return torch.cat(x_transformed, dim=self.dim)
        return x_transformed

    def __repr__(self):
        # Represent modules by their class names or 'Identity' if None
        module_reprs = [
            m if m is not None else "None" 
            for m in self.sub_modules
        ]
        # Format the list of output irreps from the internal SplitIrreps module
        irreps_out_repr = ', '.join([f"{ir.short_repr()}" for ir in self.split.irreps_out])
        return (f"{self.__class__.__name__}("
                f"irreps_in={self.irreps.short_repr()}, " # Renamed for clarity
                f"irreps_out=[{irreps_out_repr}], " # Show the split irreps
                f"sub_modules=[{', '.join(module_reprs)}], "
                f"cat_after={self.cat_after}"
                ")")
