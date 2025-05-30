# Equitorch "Others" Module Documentation

This document provides a detailed explanation of utility modules found in `equitorch/nn/others.py`. These modules offer functionalities for manipulating and processing equivariant tensors in flexible ways.

## Table of Contents

1. [SplitIrreps Module](#1-splitirreps-module)
   - [Overview](#overview-splitirreps)
   - [Parameters](#parameters-splitirreps)
   - [Functionality](#functionality-splitirreps)
   - [Usage Examples](#usage-examples-splitirreps)
2. [Separable Module](#2-separable-module)
   - [Overview](#overview-separable)
   - [Parameters](#parameters-separable)
   - [Functionality](#functionality-separable)
   - [Usage Examples](#usage-examples-separable)

## 1. SplitIrreps Module

### Overview (SplitIrreps)

The `SplitIrreps` module is designed to split an input tensor into multiple segments based on the dimensions of its irreducible representations (irreps). This is useful when you need to process different parts of an equivariant feature tensor separately.

### Parameters (SplitIrreps)

- `irreps` (Irreps): The irreducible representations specification of the input tensor.
- `split_num_irreps` (List[int]): A list specifying the number of irreps in each output segment.
    - The sum of elements in this list must equal the total number of irreps in the input `irreps`.
    - It can contain at most one `-1` or `...` entry, which will automatically be calculated to include all remaining irreps.
- `dim` (int, optional, default=-2): The dimension along which the input tensor will be split. By default, this is the `irreps_dim` (second to last dimension).

### Functionality (SplitIrreps)

The `SplitIrreps` module calculates the actual dimension sizes for each split based on the `irreps` and `split_num_irreps`. When an input tensor is passed through its `forward` method, it uses `torch.split` to divide the tensor along the specified `dim` into a list of tensors.

The module also stores the `Irreps` object corresponding to each output split in `self.irreps_out`.

### Usage Examples (SplitIrreps)

```python
import torch
from equitorch.irreps import Irreps
from equitorch.nn.others import SplitIrreps

# Define input irreps: 3 scalars (0e) and 2 vectors (1o)
irreps_in = Irreps("3x0e + 2x1o")
# Total dimension = (3 * 1) + (2 * 3) = 3 + 6 = 9
# Total number of irreps = 3 + 2 = 5

# Create an input tensor
batch_size = 5
channels = 10
x = torch.randn(batch_size, irreps_in.dim, channels)  # Shape: [5, 9, 10]

# Example 1: Split into two segments [first 2 irreps, remaining 3 irreps]
# First 2 irreps: 0e, 0e (dims: 1, 1) -> total dim = 2
# Remaining 3 irreps: 0e, 1o, 1o (dims: 1, 3, 3) -> total dim = 7
split_module1 = SplitIrreps(irreps_in, split_num_irreps=[2, -1])
# or split_module1 = SplitIrreps(irreps_in, split_num_irreps=[2, ...])

splits1 = split_module1(x)
print(f"Split 1 - Output Irreps: {split_module1.irreps_out}")
# Output Irreps: [Irreps(2x0e), Irreps(0e+2x1o)]
print(f"Split 1 - Tensor 0 shape: {splits1[0].shape}") # Expected: [5, 2, 10]
print(f"Split 1 - Tensor 1 shape: {splits1[1].shape}") # Expected: [5, 7, 10]

# Example 2: Split into three segments [1 irrep, 3 irreps, 1 irrep]
# 1st irrep: 0e (dim: 1)
# Next 3 irreps: 0e, 0e, 1o (dims: 1, 1, 3) -> total dim = 5
# Last 1 irrep: 1o (dim: 3)
split_module2 = SplitIrreps(irreps_in, split_num_irreps=[1, 3, 1])

splits2 = split_module2(x)
print(f"\nSplit 2 - Output Irreps: {split_module2.irreps_out}")
# Output Irreps: [Irreps(0e), Irreps(2x0e+1o), Irreps(1o)]
print(f"Split 2 - Tensor 0 shape: {splits2[0].shape}") # Expected: [5, 1, 10]
print(f"Split 2 - Tensor 1 shape: {splits2[1].shape}") # Expected: [5, 5, 10]
print(f"Split 2 - Tensor 2 shape: {splits2[2].shape}") # Expected: [5, 3, 10]

# Example of invalid usage (multiple -1)
try:
    SplitIrreps(irreps_in, [1, -1, -1])
except ValueError as e:
    print(f"\nError (as expected): {e}")

# Example of invalid usage (sum of splits doesn't match total irreps)
try:
    SplitIrreps(irreps_in, [1, 1]) # Sum is 2, total irreps is 5
except ValueError as e:
    print(f"Error (as expected): {e}")
```

## 2. Separable Module

### Overview (Separable)

The `Separable` module allows you to apply different neural network modules (transformations) to different segments of an input equivariant tensor. It leverages the `SplitIrreps` module internally to divide the input tensor based on its irreps structure.

### Parameters (Separable)

- `irreps` (Irreps): The irreducible representations specification for the input tensor.
- `split_num_irreps` (Iterable[int]): A list specifying the number of irreps in each segment to be processed by a corresponding sub-module. The length of this list must match the length of `sub_modules`. It follows the same rules as `SplitIrreps` for `-1` or `...`.
- `sub_modules` (Iterable[torch.nn.Module]): A list of PyTorch modules (or `None` for an identity operation). Each module in this list will be applied to the corresponding segment of the input tensor created by `SplitIrreps`.
- `cat_after` (bool, optional, default=True): If `True`, the outputs from all sub-modules are concatenated along the `dim` to form a single output tensor. If `False`, a list of output tensors is returned.
- `dim` (int, optional, default=-2): The dimension along which the input tensor is split and (if `cat_after=True`) concatenated.

### Functionality (Separable)

1. The input tensor `x` is first split into segments using an internal `SplitIrreps` instance, configured with `irreps` and `split_num_irreps`.
2. Each resulting segment is then passed through the corresponding module in the `sub_modules` list. If a sub-module is `None`, the segment is passed through unchanged (identity operation).
3. If `cat_after` is `True`, the transformed segments are concatenated along the `dim`. Otherwise, they are returned as a list.

This module is particularly useful for creating architectures where different types of features (e.g., scalar, vector, tensor) need to undergo different processing paths.

### Usage Examples (Separable)

```python
import torch
from torch import nn
from equitorch.irreps import Irreps
from equitorch.nn.others import Separable
from equitorch.nn.linears import IrrepWiseLinear # Example sub-module

# Define input irreps: 2 scalars (0e), 1 vector (1o), 1 rank-2 tensor (2e)
irreps_in = Irreps("2x0e + 1x1o + 1x2e")
# Total dimension = (2*1) + (1*3) + (1*5) = 2 + 3 + 5 = 10
# Total number of irreps = 2 + 1 + 1 = 4

# Create an input tensor
batch_size = 5
channels_in = 16
x = torch.randn(batch_size, irreps_in.dim, channels_in) # Shape: [5, 10, 16]

# Define sub-modules for different parts
# Let's say we want to:
# 1. Process the first scalar (1x0e) with a linear layer.
# 2. Keep the second scalar (1x0e) and the vector (1x1o) as is (identity).
# 3. Process the rank-2 tensor (1x2e) with another linear layer.

# Define split_num_irreps: [1 (for 1st 0e), 2 (for 2nd 0e and 1x1o), 1 (for 1x2e)]
split_config = [1, 2, 1]

# Corresponding Irreps for each split:
# Split 0: Irreps("0e")
# Split 1: Irreps("0e + 1o")
# Split 2: Irreps("2e")

channels_out_linear1 = 32
channels_out_linear2 = 64

sub_module_list = [
    IrrepWiseLinear(Irreps("0e"), channels_in=channels_in, channels_out=channels_out_linear1), # For the first 0e
    None,  # Identity for the "0e + 1o" segment
    IrrepWiseLinear(Irreps("2e"), channels_in=channels_in, channels_out=channels_out_linear2)  # For the 2e
]

# Create Separable module
separable_module = Separable(
    irreps=irreps_in,
    split_num_irreps=split_config,
    sub_modules=sub_module_list,
    cat_after=True # Concatenate outputs
)
print(f"Separable Module: {separable_module}")

# Forward pass
output_cat = separable_module(x)

# Expected output dimension if cat_after=True:
# Dim from module1 (0e): 1 (Irrep("0e").dim)
# Dim from module2 (0e+1o): 1 + 3 = 4 (Irreps("0e+1o").dim)
# Dim from module3 (2e): 5 (Irreps("2e").dim)
# Total output dim = 1 + 4 + 5 = 10
# Output channels will vary per segment if sub-modules change them.
# If IrrepWiseLinear is used, it changes channel dim, not irrep dim.
# So, the irreps_dim of the concatenated output will be the same as input.
# The channel dimension will be determined by the sub_modules.
# However, IrrepWiseLinear preserves the irreps_dim and changes the channel_dim.
# For concatenation along dim=-2 (irreps_dim), all output segments must have the same channel_dim.
# This example needs adjustment if sub_modules change channel dimensions and cat_after=True along irreps_dim.

# Let's adjust the example for cat_after=True, assuming sub_modules preserve channel count or output a common one.
# For simplicity, let's assume all sub_modules output `channels_in` for now.
# Or, more realistically, if cat_after=True, the sub_modules should be designed such that
# their output channel dimensions are compatible for concatenation or the `Separable` module
# would need to handle channel alignment if concatenating along the channel dimension.
# The current `Separable` concatenates along `dim` (default -2, irreps_dim).
# This means all output segments from sub_modules must have the SAME channel dimension.

# Re-defining sub_modules to output the same channel dimension for concatenation
common_channels_out = 32
sub_module_list_cat = [
    IrrepWiseLinear(Irreps("0e"), channels_in=channels_in, channels_out=common_channels_out),
    IrrepWiseLinear(Irreps("0e+1o"), channels_in=channels_in, channels_out=common_channels_out), # Apply linear to make channels match
    IrrepWiseLinear(Irreps("2e"), channels_in=channels_in, channels_out=common_channels_out),
]

separable_module_cat = Separable(
    irreps=irreps_in,
    split_num_irreps=split_config,
    sub_modules=sub_module_list_cat,
    cat_after=True
)
output_cat_final = separable_module_cat(x)
print(f"\nOutput shape (cat_after=True): {output_cat_final.shape}")
# Expected: [batch_size, irreps_in.dim, common_channels_out] -> [5, 10, 32]

# Example with cat_after=False
separable_module_list_out = Separable(
    irreps=irreps_in,
    split_num_irreps=split_config,
    sub_modules=sub_module_list, # Original sub_modules with varying output channels
    cat_after=False
)
output_list = separable_module_list_out(x)
print(f"\nOutput (cat_after=False):")
print(f"  Tensor 0 shape: {output_list[0].shape}") # Expected: [5, 1, 32] (Irreps("0e").dim, channels_out_linear1)
print(f"  Tensor 1 shape: {output_list[1].shape}") # Expected: [5, 4, 16] (Irreps("0e+1o").dim, channels_in)
print(f"  Tensor 2 shape: {output_list[2].shape}") # Expected: [5, 5, 64] (Irreps("2e").dim, channels_out_linear2)

```

These modules provide powerful tools for building complex equivariant architectures by allowing fine-grained control over how different parts of feature tensors are processed and combined.
