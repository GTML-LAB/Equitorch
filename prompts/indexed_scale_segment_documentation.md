# Indexed Scale Segment Operation Documentation

This document describes the Triton kernel and Python wrapper functions used for performing an indexed, scaled segment reduction operation. This operation is effectively a sparse matrix multiplication applied along the second-to-last dimension of an input tensor, often used in equivariant neural networks and related fields.

The implementation is primarily located in:
- `equitorch/ops/indexed_scale_segment_op.py`: Contains the Triton kernel (`indexed_scale_segment_kernel`) and its Python wrapper (`indexed_scale_segment`).
- `equitorch/nn/functional/sparse_scale.py`: Provides the `torch.autograd.Function` wrapper (`SparseScale`) to make the operation differentiable.

## `indexed_scale_segment_op.py`: Kernel and Wrapper

### `indexed_scale_segment_kernel(...)` (Triton Kernel)

- **Purpose:** Performs the core computation: segment-wise summation of scaled, indexed elements from an input tensor.
- **Functionality:**
    - **Grid:** Launched with a 3D grid:
        - `pid_n`: Blocks over the batch dimension (N).
        - `pid_seg`: Iterates directly over the output segments.
        - `pid_c`: Blocks over the channel dimension (C).
    - **Offsets and Masks:** Calculates offsets and masks for the batch (`n_offsets`, `n_mask`) and channel (`c_offsets`, `c_mask`) dimensions within the current block.
    - **Segment Range:** Loads the start (`loop_start`) and end (`loop_end`) indices for the current output segment (`pid_seg`) from `seg_ptr`. This defines the range of sparse indices `t` contributing to this output segment.
    - **Accumulation:** Initializes an `accumulator` tensor (size `BLOCK_SIZE_N x BLOCK_SIZE_C`) with zeros for the current block.
    - **Main Loop:** Iterates from `loop_start` to `loop_end` (`loop_idx` corresponds to the sparse index `t`).
        1.  **Load Index & Scale:** Loads the input index (`idx = index[loop_idx]`) and the scaling factor (`scale_val = scale[loop_idx]`) for the current sparse index `t`.
        2.  **Calculate Input Pointers:** Computes the memory addresses (`input_ptrs`) for the input elements `input[n, idx, c]` corresponding to the current block's batch (`n_offsets`) and channel (`c_offsets`) offsets, using the loaded `idx`.
        3.  **Load & Scale Input:** Loads the input values (`input_vals`) using `input_ptrs` and the calculated masks. Multiplies the loaded values by `scale_val`.
        4.  **Accumulate:** Adds the `scaled_vals` to the `accumulator`.
    - **Write Back:** Calculates the output pointers (`output_ptrs`) for the current block (`output[n, pid_seg, c]`) and stores the final `accumulator` values, applying masks.

### `indexed_scale_segment(...)` (Python Wrapper)

- **Purpose:** Provides a user-friendly Python interface to the `indexed_scale_segment_kernel`.
- **Functionality:**
    - **Argument Handling:** Accepts PyTorch tensors for `input`, `scale`, `index`, and `seg`. Handles optional `out_size` and `out` tensor.
    - **Shape Calculation:** Determines the input dimensions (N, _, C) and the number of output segments. Calculates the required output shape `(N, out_size, C)`.
    - **Output Allocation:** If the `out` tensor is not provided, it allocates a new tensor with the correct shape, dtype, and device, initialized appropriately (implicitly zeros, suitable for accumulation).
    - **Grid Calculation:** Defines the Triton launch grid based on N, C, and the chosen block sizes (`block_size_n`, `block_size_c`).
    - **Kernel Launch:** Calls the `indexed_scale_segment_kernel` with the prepared tensor pointers, dimensions, strides, block sizes, and optimization parameters (`NUM_STAGES`).
    - **Return Value:** Returns the output tensor `out`.

## `sparse_scale.py`: Autograd Wrapper

### `SparseScale(Function)`

- **Purpose:** Wraps the `indexed_scale_segment` operation in a `torch.autograd.Function` to enable automatic differentiation.
- **Structure:**
    - **`forward(ctx, input, info_fwd, info_bwd)`:**
        1.  Calls `indexed_scale_segment` using the input tensor and parameters extracted from `info_fwd` (`info_fwd.scale`, `info_fwd.index`, `info_fwd.seg_out`, `info_fwd.out_size`).
        2.  Saves the original `input` tensor for the backward pass using `ctx.save_for_backward`.
        3.  Stores the forward (`info_fwd`) and backward (`info_bwd`) `SparseScaleInfo` objects in the context `ctx`.
        4.  Returns the result of the forward computation.
    - **`backward(ctx, grad_output)`:**
        1.  Retrieves the saved `input` tensor and the `info_fwd`, `info_bwd` objects from `ctx`.
        2.  Receives the incoming gradient `grad_output` with respect to the forward pass's output.
        3.  If the gradient with respect to the original `input` is required (`ctx.needs_input_grad[0]`):
            - It calls `SparseScale.apply` recursively. This is the key insight: the backward pass for this operation has the same mathematical structure as the forward pass.
            - The inputs to the backward call are:
                - `input`: The incoming gradient `grad_output`.
                - `info_fwd`: The *backward* information `info_bwd` (contains scale, index, seg for the backward pass).
                - `info_bwd`: The *forward* information `info_fwd` (contains scale, index, seg for the *gradient* of the backward pass, which isn't needed here but satisfies the function signature).
            - This computes the gradient `grad_in`.
        4.  Returns `grad_in` (or `None`), and `None` placeholders for the gradients of `info_fwd` and `info_bwd` (as they are non-tensor inputs containing configuration).

### `sparse_scale(...)`

- A simple alias for `SparseScale.apply` for convenience.

### `SparseScaleInfo` Structure (`structs/__init__.py`)

The configuration for the operation is passed via `SparseScaleInfo` objects (a NamedTuple likely defined in `equitorch/structs/__init__.py`). This structure typically holds:

```python
# Likely definition based on usage
class SparseScaleInfo(NamedTuple):
    scale: Tensor # (num_t,), scaling factors
    index: Tensor # (num_t,), indices into input's second-to-last dim
    seg_out: Tensor # (num_M+1,), segment boundaries for output
    out_size: int # Size of the output's second-to-last dim (num_M)
    # Potentially other fields if needed for more complex variants
```

- **`scale`**: Tensor containing scaling factors $s_t$.
- **`index`**: Tensor mapping sparse index $t$ to the index in the input tensor's second-to-last dimension.
- **`seg_out`**: Tensor defining the segments for the output dimension. `seg_out[m]` is the starting sparse index $t$ for output segment $m$.
- **`out_size`**: The size of the output's second-to-last dimension ($M$).

Helper functions (likely in `equitorch/utils/_structs.py`) are probably used to construct these `SparseScaleInfo` objects from user-provided lists or arrays, handling sorting and segment creation.

## Relation to Mathematical Formulation

The `indexed_scale_segment` operation computes the following:

$\text{output}_{nmc} = \sum_{t \text{ s.t. } \text{seg\_out}[m] \le t < \text{seg\_out}[m+1]} \text{scale}_t \times \text{input}_{n, \text{index}[t], c}$

Where:
- $\text{output}_{nmc}$: The output tensor element for batch item $n$, output segment index $m$, and channel $c$.
- $\text{input}_{n, i, c}$: The input tensor element for batch item $n$, input index $i$, and channel $c$.
- $\text{scale}_t$: The scaling factor for sparse index $t$.
- $\text{index}_t$: The input index $i$ corresponding to sparse index $t$.
- $\text{seg\_out}_m$: The starting sparse index $t$ for output segment $m$. The summation runs over all $t$ belonging to segment $m$.
- $n$: Batch index.
- $m$: Output segment index (ranges from 0 to `out_size - 1`).
- $c$: Channel index.
- $t$: Sparse index iterating through the `scale` and `index` tensors.

**Connection to Sparse Matrix Multiplication:**

This operation can be viewed as a batched multiplication by a sparse matrix along the second-to-last dimension. Consider a fixed batch $n$ and channel $c$. Let $\mathbf{x}_n = \text{input}_{n, :, c}$ be a vector of size `input.shape[-2]` and $\mathbf{y}_n = \text{output}_{n, :, c}$ be a vector of size `out_size`. The operation computes $\mathbf{y}_n = \mathbf{S} \mathbf{x}_n$, where $\mathbf{S}$ is a sparse matrix of shape (`out_size`, `input.shape[-2]`).

The non-zero entries of $\mathbf{S}$ are defined by `scale`, `index`, and `seg_out`. Specifically, the entry $S_{mi}$ is non-zero if there exists a sparse index $t$ such that $m$ is the segment containing $t$ (i.e., $\text{seg\_out}[m] \le t < \text{seg\_out}[m+1]$) and $i = \text{index}[t]$. The value of this non-zero entry is $S_{m, \text{index}[t]} = \text{scale}_t$. The summation $\sum_t$ in the formula corresponds to accumulating the contributions for each output row $m$.

The backward pass computes $\text{grad\_in} = \mathbf{S}^T \text{grad\_output}$, which involves multiplication by the transpose of the same sparse matrix structure, explaining why the backward call reuses `SparseScale.apply` with swapped info objects.

## Tensor Shape Conventions

- **`input`**: `(N, M_in, C)` where `N` is batch size, `M_in` is the size of the indexed dimension, `C` is the channel size.
- **`scale`**: `(T,)` where `T` is the total number of sparse entries.
- **`index`**: `(T,)` containing indices into the `M_in` dimension of `input`. Values must be in `[0, M_in)`.
- **`seg_out`**: `(M_out + 1,)` defining segment boundaries in the range `[0, T]`. `M_out` is the size of the output's second-to-last dimension (`out_size`).
- **`output`**: `(N, M_out, C)`.

The operation effectively transforms the `M_in` dimension of the input to the `M_out` dimension of the output via a sparse transformation defined by `scale`, `index`, and `seg_out`.
