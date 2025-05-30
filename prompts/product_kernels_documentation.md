# Product Kernels Documentation

This document describes the Triton kernels and Python wrapper functions used for performing various types of products, focusing on batched operations with sparse/indexed inputs, optional scaling, and segment gathering/accumulation capabilities. These operations are fundamental building blocks in equivariant neural networks.

The implementation is split across three main files:
- `equitorch/ops/kernel_utils.py`: Provides low-level utility functions for Triton kernels, primarily for memory access (pointer calculation, loading, storing).
- `equitorch/ops/kernel_dense.py`: Implements the core dense product operations (multiplication, outer product, inner product, etc.) as Triton kernels.
- `equitorch/ops/batched_sparse_dense_op.py`: Wraps the dense kernels to handle batching, sparse indexing, optional scaling, and segment-based gathering/accumulation. It contains both Triton kernels and Python interface functions.

## `kernel_utils.py`: Kernel Utilities

These Triton JIT functions provide helpers for memory operations within other kernels.

### `indexed_ptr(...)`
- **Purpose:** Calculates base memory addresses for accessing elements in potentially indexed tensors.
- **Functionality:** Takes a base pointer, an optional index pointer, an index offset, batch offsets (`n_offsets`), and strides. If `index_ptr` is provided, it loads the specific index for the current sparse element; otherwise, it uses the `index_offset`. It computes the final row base addresses, handling cases where an input might be shared across the batch dimension (`SHARED` flag).

### `load_block_scalar(...)`
- **Purpose:** Loads a block of scalar values.
- **Functionality:** Loads data from `base_offsets`, applying `base_mask`. Handles shared memory (`SHARED` flag).

### `load_block_vec(...)`
- **Purpose:** Loads a block of vector values.
- **Functionality:** Loads data using `base_offsets` and `c_offsets` (channel offsets), applying masks (`base_mask`, `c_mask`) and stride. Handles shared memory (`SHARED` flag).

### `load_block_mat(...)`
- **Purpose:** Loads a block of matrix values.
- **Functionality:** Loads data using `base_offsets`, `c_in_offsets`, and `c_out_offsets`, applying masks and strides for both input and output channels. Handles shared memory (`SHARED` flag).

### `store_block_scalar(...)`
- **Purpose:** Stores a block of scalar values.
- **Functionality:** Stores `value` to `base_offsets`, applying `base_mask`. If `ACCUMULATED` is true, performs an atomic add instead of a direct store.

### `store_block_vec(...)`
- **Purpose:** Stores a block of vector values.
- **Functionality:** Stores `value` using `base_offsets` and `c_offsets`, applying masks and stride. If `ACCUMULATED` is true, performs atomic adds.

### `store_block_mat(...)`
- **Purpose:** Stores a block of matrix values.
- **Functionality:** Stores `value` using `base_offsets`, `c_in_offsets`, and `c_out_offsets`, applying masks and strides. If `ACCUMULATED` is true, performs atomic adds.

### `store_block_indexed_vec(...)`
- **Purpose:** Combines indexing and storing a block of vector values.
- **Functionality:** Calculates the target index using `index_ptr` and `index_offset`, then stores `value` similar to `store_block_vec`, handling accumulation.

## `kernel_dense.py`: Core Dense Kernels

These Triton JIT functions implement the fundamental dense product operations. They operate on blocks of data provided by the calling kernel and utilize functions from `kernel_utils.py` for memory access.

### `kernel_mul(...)`
- **Purpose:** Performs element-wise multiplication of two input blocks.
- **Functionality:** Loads `input1` and `input2` blocks (vectors) using `load_block_vec`. Returns `input1 * input2`. If `OUT_ACCUMULATED` is true, it returns the sum along the batch dimension (`axis=0`). Handles shared inputs (`SHARED_INPUT1`, `SHARED_INPUT2`).

### `kernel_outer(...)`
- **Purpose:** Performs the outer product of two input blocks (vectors).
- **Functionality:** Loads `input1` and `input2` blocks using `load_block_vec`, expanding dimensions appropriately (`.expand_dims(-1)` and `.expand_dims(-2)`). Returns `input1 * input2`. If `OUT_ACCUMULATED` is true, it returns the sum along the batch dimension (`axis=0`). Handles shared inputs.

### `kernel_inner(...)`
- **Purpose:** Performs the inner product (dot product) of two input blocks (vectors).
- **Functionality:** Iterates over the channel dimension (`c_in`) in blocks (`BLOCK_SIZE_C`). In each iteration, loads parts of `input1` and `input2` using `load_block_vec` and accumulates their element-wise product (`input1 * input2`). Returns the accumulated sum. If `OUT_ACCUMULATED` is false, the accumulation happens per batch element; otherwise, it sums across the batch dimension as well. Handles shared inputs, staging (`NUM_STAGES`), and loop unrolling (`LOOP_UNROLL_FACTOR`).

### `kernel_vecmat(...)`
- **Purpose:** Performs vector-matrix multiplication.
- **Functionality:** Iterates over the input channel dimension (`c_in`) in blocks (`BLOCK_SIZE_C_IN`). In each iteration, loads a part of the input vector (`input1`) using `load_block_vec` and a part of the input matrix (`input2`) using `load_block_mat`. Accumulates the product (`input2 * input1.expand_dims(-1)`), summing over the input channel dimension (`axis=-2`). Returns the accumulated result. Handles accumulation across the batch dimension (`OUT_ACCUMULATED`), shared inputs, staging, and loop unrolling.

### `kernel_vecsca(...)`
- **Purpose:** Performs element-wise multiplication of a vector block by a scalar block (scaling).
- **Functionality:** Loads the vector block (`input1`) using `load_block_vec` and the scalar block (`input2`) using `load_block_scalar`, expanding the scalar's dimensions. Returns `input1 * input2`. Handles accumulation across the batch dimension (`OUT_ACCUMULATED`) and shared inputs.

## `batched_sparse_dense_op.py`: Batched Sparse/Indexed Operations

This file provides the main interface for performing products with support for sparse indexing, optional scaling, and segment-based operations (gathering/accumulation).

### Intermediate Triton Kernels (`batched_single_*`)

These kernels are called internally by the main `indexed_*_scale_gather_kernel` functions. They handle a single sparse index lookup and dispatch to the corresponding dense kernel from `kernel_dense.py`.

- **`batched_single_mul(...)`**: Handles one sparse index, calls `kernel_mul`, applies optional scaling.
- **`batched_single_outer(...)`**: Handles one sparse index, calls `kernel_outer`, applies optional scaling.
- **`batched_single_inner(...)`**: Handles one sparse index, calls `kernel_inner`, applies optional scaling.
- **`batched_single_vecmat(...)`**: Handles one sparse index, calls `kernel_vecmat`, applies optional scaling. Computes `vector . matrix`.
- **`batched_single_vecsca(...)`**: Handles one sparse index, calls `kernel_vecsca`, applies optional scaling. Computes `vector * scalar`.
- **`batched_single_scavec(...)`**: Handles one sparse index, calls `kernel_vecsca` with swapped inputs, applies optional scaling. Computes `scalar * vector`.
- **`batched_single_mat_t_vec(...)`**: Handles one sparse index, calls `kernel_vecmat` with swapped inputs, applies optional scaling. Computes `matrix^T . vector`.

### Main Triton Kernels (`indexed_*_scale_gather_kernel`)

These are the kernels launched directly from the Python wrappers. They manage the overall computation grid and looping logic.

- **Purpose:** Implement the batched, indexed, optionally scaled, and optionally gathered/accumulated product operations for different dense patterns.
- **Functionality:**
    - Each kernel corresponds to a product type (`mul`, `outer`, `inner`, `vecmat`, `vecsca`, `scavec`, `mat_t_vec`).
    - They receive pointers to inputs (`input1_ptr`, `input2_ptr`), optional scale (`scale_ptr`), indices (`index1_ptr`, `index2_ptr`), segment information (`seg_ptr`), gather indices (`gather_index_ptr`), output index (`index_out_ptr`), and the output tensor (`output_ptr`).
    - They calculate offsets and masks based on program IDs and block sizes appropriate for the operation type.
    - **Gathering/Accumulation Logic:** If `seg_ptr` is provided (`OUT_GATHERED` mode), they loop from `loop_start` to `loop_end` (defined by `seg_ptr`). Inside the loop:
        - They determine the `sparse_idx` either directly from the loop (`gather_index_ptr` is None) or by loading from `gather_index_ptr`.
        - They call the corresponding `batched_single_*` kernel to compute the product for the current `sparse_idx`.
        - They accumulate the `product` into an `accumulator` tensor.
    - **Direct Indexed Logic:** If `seg_ptr` is None (not `OUT_GATHERED`), they compute the product for a single `sparse_idx` (determined by `pid_seg`) using the `batched_single_*` kernel.
    - **Output:** They calculate the output base pointers using `indexed_ptr` (handling `OUT_ACCUMULATED` and `index_out_ptr`).
    - They store the final `accumulator` result to the output tensor using the appropriate `store_block_*` function from `kernel_utils.py` (e.g., `store_block_vec`, `store_block_mat`, `store_block_scalar`).

### Python Wrappers (`indexed_*_scale_gather`)

These functions provide the user-facing API.

- **Purpose:** Offer a convenient Python interface to the underlying Triton kernels for each product type.
- **Functionality:**
    - Each function corresponds to a product type (`mul`, `outer`, `inner`, `vecmat`, `vecsca`, `scavec`, `mat_t_vec`).
    - **Argument Handling:** Accept PyTorch tensors for inputs, scale, indices, segments, etc. Handle `None` values for optional arguments. Validate input shapes based on the operation type.
    - **Mode Detection:** Determine operational modes (e.g., `left_shared`, `right_shared`, `out_gathered`, `out_accumulated`, `out_indexed`) based on the provided arguments and input tensor dimensions.
    - **Input Preparation:** Ensure inputs are contiguous and unsqueeze dimensions if necessary (e.g., converting a 2D shared input to 3D).
    - **Output Allocation:** Calculate the required output shape based on input sizes, `out_size`, and modes. Allocate the output tensor (`out`) with the correct shape, dtype, and device if not provided, initializing with zeros for accumulation modes.
    - **Grid Calculation:** Define the Triton launch grid (`grid = lambda meta: ...`) based on input dimensions and block sizes specified in `meta`.
    - **Kernel Launch:** Call the corresponding `indexed_*_scale_gather_kernel` (e.g., `indexed_mul_scale_gather_kernel`, `indexed_scavec_scale_gather_kernel`, `indexed_vecmat_scale_gather`) with the prepared arguments, strides, sizes, block sizes, and mode flags.
    - **Return Value:** Return the output tensor `out`.

## Example Usage

The functions defined in `batched_sparse_dense_op.py` are called by higher-level operation files to implement specific product variations. Examples include:

- `equitorch/ops/accumulated_indexed_product_segment_op.py`: Implements operations that *require* segment accumulation (e.g., `accumulated_indexed_mul_segment`).
- `equitorch/ops/indexed_product_op.py`: Implements basic indexed products without scaling or segmentation (e.g., `indexed_mul`).
- `equitorch/ops/indexed_product_scale_segment_op.py`: Implements operations with indexing, scaling, *and* segmentation (e.g., `indexed_mul_scale_segment`).
- `equitorch/ops/product_segment_op.py`: Implements segmented products *without* requiring explicit indexing (though indexing might be handled internally if inputs are structured that way) (e.g., `mul_segment`).

These examples demonstrate how the flexible `indexed_*_scale_gather` functions serve as a unified backend for various product scenarios encountered in the library.

## Autograd Wrappers (`sparse_product.py`)

The file `equitorch/nn/functional/sparse_product.py` defines `torch.autograd.Function` subclasses that wrap the `indexed_*_scale_gather` functions described above. These wrappers make the sparse product operations differentiable, allowing gradients to be automatically computed during backpropagation.

Each core product operation has a corresponding `autograd.Function` class:
- `SparseMul` (for `indexed_mul_scale_gather`)
- `SparseOuter` (for `indexed_outer_scale_gather`)
- `SparseInner` (for `indexed_inner_scale_gather`)
- `SparseVecMat` (for `indexed_vecmat_scale_gather`)
- `SparseVecSca` (for `indexed_vecsca_scale_gather`)
- `SparseScaVec` (for `indexed_scavec_scale_gather`)
- `SparseMatTVec` (for `indexed_mat_t_vec_scale_gather`)

Convenience aliases are provided (e.g., `sparse_mul = SparseMul.apply`).

### `forward` Method Structure

The `forward` static method of each class:
1.  Calls the corresponding `indexed_*_scale_gather` function with the input tensors (`input1`, `input2`) and the forward pass configuration (`info_fwd`).
2.  Accepts three `SparseProductInfo` objects:
    - `info_fwd`: Contains indexing, scaling, segmentation, and output size information for the forward pass.
    - `info_bwd1`: Contains the necessary information to compute the gradient with respect to `input1`.
    - `info_bwd2`: Contains the necessary information to compute the gradient with respect to `input2`.
3.  Saves `input1` and `input2` for the backward pass using `ctx.save_for_backward`.
4.  Stores the `SparseProductInfo` objects and relevant flags (`shared1`, `shared2`, `out_accumulated`) in the context `ctx` for use in the backward pass.

### `backward` Method Structure

The `backward` static method:
1.  Retrieves the saved input tensors and context information (`infos`, flags) from `ctx`.
2.  Receives the incoming gradient (`grad_output`) with respect to the forward pass's output.
3.  If gradients are needed for `input1` (`ctx.needs_input_grad[0]`):
    - Determines the correct backward operation based on the "Dense Operation Relationships" table (e.g., for `SparseOuter.forward`, the backward op for `input1` is `vecmat`).
    - Determines the correct `out_accumulated` flag for this backward call based on the "Relationship Between Forward and Backward Flags" table (e.g., `out_accumulated_bx = left_shared_fwd`).
    - Calls the corresponding `Sparse*.apply` function (e.g., `SparseVecMat.apply`) with the appropriate inputs (`input2`, `grad_output`, potentially transposed) and the correct backward info (`info_bwd1`).
4.  If gradients are needed for `input2` (`ctx.needs_input_grad[1]`):
    - Performs similar steps as above, using the appropriate backward operation (e.g., for `SparseOuter.forward`, the backward op for `input2` is `mat_t_vec`), `out_accumulated` flag (`out_accumulated_by = right_shared_fwd`), inputs (`grad_output`, `input1`), and backward info (`info_bwd2`).
5.  Handles necessary tensor transpositions (e.g., `grad.transpose(-1,-2).contiguous()`) to match the input requirements of the backward operations.
6.  Returns the computed gradients `grad1` and `grad2` (or `None` if not needed), followed by `None` placeholders for the non-tensor inputs of the `forward` method.

This structure ensures that the chain rule is correctly applied, leveraging the same underlying sparse product kernels for both forward and backward computations, while correctly managing indexing, segmentation, scaling, and batch dimension handling via the `SparseProductInfo` objects and context flags.

### `SparseProductInfo` Structure (`structs/__init__.py`)

The configuration for the sparse indexing, scaling, segmentation, and output mapping is encapsulated in the `SparseProductInfo` NamedTuple, defined in `equitorch/structs/__init__.py`. This structure holds the necessary tensors and metadata required by both the low-level `indexed_*_scale_gather` kernels and the `autograd.Function` wrappers.

```python
@add_operation_methods
class SparseProductInfo(NamedTuple):
    '''
        z_M = sum_{t in Ind*[M]} s_t * x_Ind1[t] * y_Ind2[t]

        or

        z_M = sum_{M1M2} s_{MM1M2} x_M1 * y_M2
    '''
    scale: Optional[Tensor] = None # (num_t,), floating
    index1: Optional[Tensor] = None # (num_t,), int in [0, num_M1)
    index2: Optional[Tensor] = None # (num_t,), int in [0, num_M2)
    seg_out: Optional[Tensor] = None # (num_M_nonzero+1,), increasing int in [0, num_t]
    gather_index: Optional[Tensor] = None # (num_M_nonzero,) int in [0, num_t) # Note: This field seems missing in the provided code but is used in kernels. Assuming it should be here.
    index_out: Optional[Tensor] = None # (num_M_nonzero,), int in [0, num_M)
    out_size: Optional[int] = None # num_M
```

- **`scale`**: Optional tensor containing scaling factors $s_t$.
- **`index1`**: Optional tensor mapping sparse index $t$ to the index in `input1`. If `None`, assumes identity mapping ($t \rightarrow t$).
- **`index2`**: Optional tensor mapping sparse index $t$ to the index in `input2`. If `None`, assumes identity mapping ($t \rightarrow t$).
- **`seg_out`**: Optional tensor defining the segments for gathering/accumulation. If present, indicates `OUT_GATHERED` mode. Defines the start and end points for the loop over $t$ for each output segment $M$.
- **`gather_index`**: Optional tensor used with `seg_out` to gather non-contiguous sparse indices $t$ into segments.
- **`index_out`**: Optional tensor mapping the output segment index $M$ (or sparse index $t$ if not gathered) to the final storage index in the output tensor. If present, indicates `OUT_INDEXED` mode.
- **`out_size`**: The total size of the output sparse dimension ($M$).

### Helper Functions (`utils/_structs.py`)

The file `equitorch/utils/_structs.py` provides helper functions to construct `SparseProductInfo` objects:

- **`sparse_product_info(...)`**:
    - Takes Python lists or similar iterables for `index1`, `index2`, `index` (used for `index_out`), and `scale`.
    - Performs necessary sorting (`sort_by_column_key`) and segment extraction (`extract_batch_segments`) based on the output index (`index`).
    - Converts the processed lists into PyTorch tensors.
    - Optimizes storage by setting index tensors to `None` if they represent a simple range (identity mapping).
    - Returns a single `SparseProductInfo` object configured for a specific operation (e.g., forward pass).

- **`sparse_product_infos(...)`**:
    - A convenience function that calls `sparse_product_info` three times to generate the configurations needed for a forward pass and the two corresponding backward passes (gradient w.r.t. `input1` and gradient w.r.t. `input2`).
    - It correctly permutes the input indices (`index1`, `index2`, `index`) and sizes (`out_size`, `in1_size`, `in2_size`) for each of the three calls to reflect the data flow in the forward and backward computations.
    - Returns a tuple containing three `SparseProductInfo` objects: `(info_fwd, info_bwd1, info_bwd2)`.

These helper functions simplify the process of preparing the complex indexing and segmentation information required by the sparse product operations and their gradients.


## Relation to Mathematical Formulation

The core functionality implemented by the `indexed_*_scale_gather` functions (and their wrappers) can be related to the following general formula for a segmented, indexed, and scaled product:

$\mathbf{z}_{nM}=\sum_{t\in \text{Ind}^*[M]}s_{t}\ \mathbf{x}_{n,\text{Ind}_1[t]}\circ \mathbf{y}_{n,\text{Ind}_2[t]}$

Where:
- $\mathbf{z}_{nM}$: The output tensor element for batch item $n$ and output segment index $M$. The bold font indicates it represents a tensor (scalar, vector, or matrix) over the "dense" channel dimensions, which are omitted for brevity.
- $\mathbf{x}_{n, \text{Ind}_1[t]}$: The first input tensor element for batch item $n$ and input index $\text{Ind}_1[t]$.
- $\mathbf{y}_{n, \text{Ind}_2[t]}$: The second input tensor element for batch item $n$ and input index $\text{Ind}_2[t]$.
- $\circ$: Represents the specific product operation (element-wise multiplication `mul`, outer product `outer`, inner product `inner`, vector-matrix product `vecmat`, vector-scalar product `vecsca`, scalar-vector product `scavec`, or matrix^T-vector product `mat_t_vec`). The specific operation is determined by the shapes of $\mathbf{x}$ and $\mathbf{y}$.
- $s_t$: An optional scaling factor associated with index $t$.
- $t$: An index iterating over a "sparse" dimension.
- $\text{Ind}_1[t]$: A function (or lookup table) mapping the sparse index $t$ to the corresponding index in the first input tensor's sparse dimension.
- $\text{Ind}_2[t]$: A function (or lookup table) mapping the sparse index $t$ to the corresponding index in the second input tensor's sparse dimension.
- $\text{Ind}^*[M]$: The set of sparse indices $t$ that belong to the output segment $M$. This defines the summation range for each output element $M$.

Let's analyze how the function arguments map to this formula:

- **`input1`, `input2`**: Correspond to $\mathbf{x}$ and $\mathbf{y}$ respectively. Their shapes determine the nature of the $\circ$ operation (e.g., two vectors for `mul`, `outer`, `inner`; vector and matrix for `vecmat`; vector and scalar for `vecsca`).
- **`scale`**: Corresponds to $s_t$.
    - If `scale` is `None`: $s_t = 1$ for all $t$. The formula becomes $\mathbf{z}_{nM}=\sum_{t\in \text{Ind}^*[M]} \mathbf{x}_{n,\text{Ind}_1[t]}\circ \mathbf{y}_{n,\text{Ind}_2[t]}$.
    - If `scale` is provided: It's a tensor containing the $s_t$ values, indexed by $t$.
- **`index1`**: Corresponds to the mapping $\text{Ind}_1[t]$.
    - If `index1` is `None`: $\text{Ind}_1[t] = t$. Input $\mathbf{x}$ is accessed directly using the sparse index $t$. This assumes `input1` has a sparse dimension compatible with $t$.
    - If `index1` is provided: It's a tensor mapping $t$ to the actual index used to access $\mathbf{x}$.
- **`index2`**: Corresponds to the mapping $\text{Ind}_2[t]$.
    - If `index2` is `None`: $\text{Ind}_2[t] = t$. Input $\mathbf{y}$ is accessed directly using the sparse index $t$.
    - If `index2` is provided: It's a tensor mapping $t$ to the actual index used to access $\mathbf{y}$.
- **`seg`**: Defines the segments and thus the summation sets $\text{Ind}^*[M]$.
    - If `seg` is `None`: The operation is not segmented (or gathered). The summation $\sum_{t\in \text{Ind}^*[M]}$ is removed, and the output corresponds directly to the indexed product for each $t$. The output sparse dimension matches the input sparse dimension $t$. The formula simplifies to $\mathbf{z}_{nt} = s_t \mathbf{x}_{n,\text{Ind}_1[t]} \circ \mathbf{y}_{n,\text{Ind}_2[t]}$. The output index $M$ becomes equivalent to the input index $t$.
    - If `seg` is provided: It's a tensor defining the start and end points for each segment $M$. The kernel loops from `seg[M]` to `seg[M+1]-1`, effectively performing the summation $\sum_{t\in \text{Ind}^*[M]}$. This gathers contributions based on $t$ into the output index $M$.
- **`gather_index`**: Modifies how $t$ is interpreted within the summation when `seg` is provided.
    - If `gather_index` is `None` (and `seg` is not `None`): The loop variable within the segment *is* the sparse index $t$. $\text{Ind}^*[M] = \{t \mid \text{seg}[M] \le t < \text{seg}[M+1]\}$.
    - If `gather_index` is provided (and `seg` is not `None`): The loop variable `loop_idx` iterates from `seg[M]` to `seg[M+1]-1`, but the actual sparse index used is $t = \text{gather\_index}[\text{loop\_idx}]$. This allows gathering non-contiguous sparse indices into segments. $\text{Ind}^*[M] = \{t=\text{gather\_index}[k] \mid \text{seg}[M] \le k < \text{seg}[M+1]\}$.
- **`index_out`**: Determines if the output $\mathbf{z}$ is indexed.
    - If `index_out` is `None`: The output $\mathbf{z}_{nM}$ is stored densely according to the segment index $M$ (if `seg` is provided) or the sparse index $t$ (if `seg` is `None`).
    - If `index_out` is provided: The output is stored sparsely. Instead of storing at index $M$ (or $t$), it stores at index $\text{index\_out}[M]$ (or $\text{index\_out}[t]$). The formula doesn't explicitly show this, but it affects the memory location where $\mathbf{z}_{nM}$ (or $\mathbf{z}_{nt}$) is written.
- **`out_accumulated`**: Controls whether the batch dimension (`n`) is present in the output or if results are summed across it.
    - If `out_accumulated` is `False`: The output tensor `out` has shape `(N, M_out, C_out...)` (or `(N, T_out, C_out...)` if `seg` is None), retaining the batch dimension $n$. This corresponds directly to the formula $\mathbf{z}_{nM}=\sum_{t\in \text{Ind}^*[M]}s_{t}\ \mathbf{x}_{n,\text{Ind}_1[t]}\circ \mathbf{y}_{n,\text{Ind}_2[t]}$.
    - If `out_accumulated` is `True`: The output tensor `out` has shape `(M_out, C_out...)` (or `(T_out, C_out...)` if `seg` is None), effectively summing over the batch dimension $n$. The formula becomes $\mathbf{z}_{M}=\sum_{n} \sum_{t\in \text{Ind}^*[M]}s_{t}\ \mathbf{x}_{n,\text{Ind}_1[t]}\circ \mathbf{y}_{n,\text{Ind}_2[t]}$. This requires inputs to be compatible for summation across $n$.
- **`left_shared`, `right_shared`**: These flags indicate whether `input1` or `input2` lack the batch dimension $n$.
    - If `left_shared` is `True`: $\mathbf{x}_{n, \text{Ind}_1[t]}$ becomes $\mathbf{x}_{\text{Ind}_1[t]}$ (independent of $n$). The kernel loads this shared value for all batch items.
    - If `right_shared` is `True`: $\mathbf{y}_{n, \text{Ind}_2[t]}$ becomes $\mathbf{y}_{\text{Ind}_2[t]}$ (independent of $n$).
    - If both are `True` and `out_accumulated` is `True`: $\mathbf{z}_{M}= N \sum_{t\in \text{Ind}^*[M]}s_{t}\ \mathbf{x}_{\text{Ind}_1[t]}\circ \mathbf{y}_{\text{Ind}_2[t]}$. The batch dimension $n$ disappears, and the result is implicitly scaled by the batch size $N$ due to the summation over $n$. If `out_accumulated` is `False`, the batch dimension is still removed, but the summation over $n$ doesn't happen: $\mathbf{z}_{M}=\sum_{t\in \text{Ind}^*[M]}s_{t}\ \mathbf{x}_{\text{Ind}_1[t]}\circ \mathbf{y}_{\text{Ind}_2[t]}$.
- **`out_size`**: Specifies the size of the output sparse dimension $M$. If not provided, it's inferred.

## Relationship Between Forward and Backward Flags

The general product formula $z_{M}=\sum_{t\in \text{Ind}^*[M]}s_{t}\ \mathbf{x}_{n,\text{Ind}_1[t]}\circ \mathbf{y}_{n,\text{Ind}_2[t]}$ involves inputs $\mathbf{x}$ (input1) and $\mathbf{y}$ (input2) and produces output $\mathbf{z}$ (output). The backward pass involves computing gradients $g^x$ and $g^y$. The computation for these gradients can also be expressed as products, following the structure described by the chain rule:

- **Gradient w.r.t. x ($g^x$):** This involves a product between $\mathbf{y}$ and the incoming gradient $g^z$. Schematically, $g^x = \text{op}_{bx}(\mathbf{y}, g^z)$.
- **Gradient w.r.t. y ($g^y$):** This involves a product between $g^z$ and $\mathbf{x}$. Schematically, $g^y = \text{op}_{by}(g^z, \mathbf{x})$.

The boolean flags `LEFT_SHARED`, `RIGHT_SHARED`, and `OUT_ACCUMULATED` control how the batch dimension (`n`) is handled for input1, input2, and the output, respectively. These flags need to be mapped correctly when transitioning from the forward pass operation to the backward pass operations.

Let the flags for the forward operation $z = \text{op}(x, y)$ be:
- `LEFT_SHARED` (fwd): True if $x$ is shared across the batch.
- `RIGHT_SHARED` (fwd): True if $y$ is shared across the batch.
- `OUT_ACCUMULATED` (fwd): True if $z$ is accumulated (summed) across the batch.

The flags for the backward operation $g^x = \text{op}_{bx}(y, g^z)$ are determined as follows:
- `LEFT_SHARED` (bx): Corresponds to the properties of the first input, $y$. Since $y$ was input2 in the forward pass, `LEFT_SHARED` (bx) = `RIGHT_SHARED` (fwd).
- `RIGHT_SHARED` (bx): Corresponds to the properties of the second input, $g^z$. $g^z$ inherits properties from $z$, which was the output. So, `RIGHT_SHARED` (bx) = `OUT_ACCUMULATED` (fwd).
- `OUT_ACCUMULATED` (bx): Corresponds to the properties of the output, $g^x$. $g^x$ inherits properties from $x$, which was input1. So, `OUT_ACCUMULATED` (bx) = `LEFT_SHARED` (fwd).

Similarly, the flags for the backward operation $g^y = \text{op}_{by}(g^z, x)$ are:
- `LEFT_SHARED` (by): Corresponds to the properties of the first input, $g^z$. $g^z$ inherits properties from $z$ (output). So, `LEFT_SHARED` (by) = `OUT_ACCUMULATED` (fwd).
- `RIGHT_SHARED` (by): Corresponds to the properties of the second input, $x$. $x$ was input1. So, `RIGHT_SHARED` (by) = `LEFT_SHARED` (fwd).
- `OUT_ACCUMULATED` (by): Corresponds to the properties of the output, $g^y$. $g^y$ inherits properties from $y$, which was input2. So, `OUT_ACCUMULATED` (by) = `RIGHT_SHARED` (fwd).

**Summary Table:**

| Backward Op        | Input 1 (`arg1`) | Input 2 (`arg2`) | Output (`out`) | `LEFT_SHARED` (`arg1`) | `RIGHT_SHARED` (`arg2`) | `OUT_ACCUMULATED` (`out`) |
|--------------------|------------------|------------------|----------------|------------------------|-------------------------|---------------------------|
| $g^x = op_{bx}(y, g^z)$ | `y`              | `gz`             | `gx`           | `RIGHT_SHARED` (fwd)   | `OUT_ACCUMULATED` (fwd) | `LEFT_SHARED` (fwd)       |
| $g^y = op_{by}(g^z, x)$ | `gz`             | `x`              | `gy`           | `OUT_ACCUMULATED` (fwd) | `LEFT_SHARED` (fwd)     | `RIGHT_SHARED` (fwd)      |

This mapping ensures that the batch dimension handling (shared inputs, accumulated outputs) is consistent between the forward and backward passes, reflecting the structure imposed by the chain rule. Note that index mappings (`index1`, `index2`, `index_out`) also transform similarly based on which tensor takes which role (input1, input2, output) in the backward pass operations.

## Tensor Shape Conventions

The functions in `batched_sparse_dense_op.py` handle tensors with potentially multiple dimensions representing different aspects of the data. Understanding these conventions is crucial for using the functions correctly. The typical dimensions are:

1.  **Batch Dimension (N):**
    *   Represents independent samples processed in parallel.
    *   Usually the *first* dimension.
    *   Presence is often optional. If an input tensor lacks this dimension (e.g., shape `(M, C)` instead of `(N, M, C)`), the corresponding `_SHARED` flag (e.g., `LEFT_SHARED`, `RIGHT_SHARED`) should be `True`, indicating the tensor's data is shared across all batch items.
    *   The `OUT_ACCUMULATED` flag controls whether this dimension is present in the output or summed over.

2.  **Sparse/Indexed Dimension (M):**
    *   Represents the "sparse" elements being operated on. This could correspond to atoms in a molecule, points in a point cloud, or other indexed entities.
    *   Usually the *second* dimension (when the batch dimension N is present) or the *first* dimension (when N is absent).
    *   The size of this dimension in the *input* tensors (`M_in`) might differ from the *output* tensor (`M_out`).
    *   Arguments like `index1`, `index2`, `seg`, `gather_index`, and `index_out` control how elements along this dimension are accessed, grouped (segmented/gathered), and written to the output.
    *   The `sparse_grid_size` calculated within the Python wrappers corresponds to the size of this dimension in the *output* (`M_out`).

3.  **Dense/Channel Dimensions (C, C1, C2, Cin, Cout):**
    *   Represent the features or channels associated with each sparse element. These are the dimensions involved in the core product operation (`mul`, `outer`, `inner`, etc.).
    *   Usually the *last* one or two dimensions.
    *   Examples:
        *   `mul`, `inner`, `vecsca`, `scavec`: Often involve vectors of shape `(..., C)`.
        *   `outer`: Involves vectors `(..., C1)` and `(..., C2)`, producing `(..., C1, C2)`.
        *   `vecmat`, `mat_t_vec`: Involve vectors `(..., Cin)` or `(..., Cout)` and matrices `(..., Cin, Cout)` or `(..., Cout, Cin)`.

**Example Shapes:**

*   `input1` for `indexed_mul_scale_gather`: `(N, M1, C)` or `(M1, C)` (if `LEFT_SHARED`)
*   `input2` for `indexed_mul_scale_gather`: `(N, M2, C)` or `(M2, C)` (if `RIGHT_SHARED`)
*   `output` for `indexed_mul_scale_gather` (non-accumulated): `(N, M_out, C)`
*   `output` for `indexed_mul_scale_gather` (accumulated): `(M_out, C)`
*   `input1` for `indexed_outer_scale_gather`: `(N, M1, C1)` or `(M1, C1)`
*   `input2` for `indexed_outer_scale_gather`: `(N, M2, C2)` or `(M2, C2)`
*   `output` for `indexed_outer_scale_gather` (non-accumulated): `(N, M_out, C1, C2)`
*   `input1` for `indexed_vecmat_scale_gather`: `(N, M1, Cin)` or `(M1, Cin)`
*   `input2` for `indexed_vecmat_scale_gather`: `(N, M2, Cin, Cout)` or `(M2, Cin, Cout)`
*   `output` for `indexed_vecmat_scale_gather` (non-accumulated): `(N, M_out, Cout)`

The specific requirements for `M1`, `M2`, and `M_out` depend on the indexing (`index1`, `index2`, `index_out`) and segmentation (`seg`, `gather_index`) arguments used.

## Dense Operation Relationships (Forward and Backward)

This section outlines the relationships between the forward dense product operations and the operations required for their backward passes (gradients), **focusing specifically on the dense/channel dimensions (C, C1, C2, Cin, Cout). The batch (N) and sparse/indexed (M) dimensions are omitted here for clarity, as the core mathematical relationship applies independently to each element along those dimensions.** The convention followed is:
- Forward: `z = op(x, y)` (representing `z = x * y`)
- Backward w.r.t. x: `gx = op_bx(y, gz)` (representing `gx = y * gz`)
- Backward w.r.t. y: `gy = op_by(gz, x)` (representing `gy = gz * x`)

Here, `x` corresponds to `input1`, `y` to `input2`, `z` to the output, `gz` to the incoming gradient w.r.t. the output, `gx` to the gradient w.r.t. `input1`, and `gy` to the gradient w.r.t. `input2`. The operation names (`mul`, `outer`, etc.) refer to the dense kernels. Transposition (`^T`) is applied to inputs where necessary to match the required input shapes for the backward operation, ensuring the input order `(y, gz)` or `(gz, x)` is maintained. Transposition is applied to the *output* of a backward operation if its natural result shape is the transpose of the target gradient shape.

| Forward Operation (`z = x * y`) | Input Shapes (`x`, `y`) | Output Shape (`z`) | Backward for x (`gx = y * gz`) | Backward for y (`gy = gz * x`) | Notes                                                                                             |
| :------------------------------ | :---------------------- | :----------------- | :----------------------------- | :----------------------------- | :------------------------------------------------------------------------------------------------ |
| `mul`                           | `(C)`, `(C)`            | `(C)`              | `mul(y, gz)`                   | `mul(gz, x)`                   | Element-wise multiplication.                                                                      |
| `outer`                         | `(C1)`, `(C2)`          | `(C1, C2)`         | `vecmat(y, gz^T)`              | `mat_t_vec(gz, x)`             | `gz` shape is `(C1, C2)`. `gz^T` is `(C2, C1)`.                                                   |
| `inner`                         | `(C)`, `(C)`            | `()`               | `vecsca(y, gz)`              | `scavec(gz, x)`              | `gz` is scalar `()`.                                                                              |
| `vecmat`                        | `(Cin)`, `(Cin, Cout)`  | `(Cout)`           | `mat_t_vec(y^T, gz)`           | `outer(gz, x)^T`               | `y^T` shape is `(Cout, Cin)`. `gz` shape is `(Cout)`. `outer` output is `(Cout, Cin)`, needs `^T` for `gy:(Cin, Cout)`. |
| `vecsca`                      | `(C)`, `()`             | `(C)`              | `scavec(y, gz)`              | `inner(gz, x)`                 | `y` is scalar `()`. `gz` shape is `(C)`. `gy` sums contributions.                                 |
| `scavec`                      | `()`, `(C)`             | `(C)`              | `inner(y, gz)`                 | `vecsca(gz, x)`              | `x` is scalar `()`. `gz` shape is `(C)`. `gx` sums contributions.                                 |
| `mat_t_vec`                     | `(Cin, Cout)`, `(Cin)`  | `(Cout)`           | `outer(y, gz)`               | `vecmat(gz, x^T)`                | `x` is matrix. `gz` shape is `(Cout)`. `outer` output is `(Cin, Cout)`. |

**Explanation of Backward Operations (with Shape Considerations):**

-   **`mul`**: $z_c = x_c y_c$. (`x:(C), y:(C) -> z:(C)`)
    -   $g^x_c = y_c g^z_c$. (`y:(C), gz:(C) -> gx:(C)`). Operation: `mul(y, gz)`.
    -   $g^y_c = g^z_c x_c$. (`gz:(C), x:(C) -> gy:(C)`). Operation: `mul(gz, x)`.
-   **`outer`**: $z_{c_1 c_2} = x_{c_1} y_{c_2}$. (`x:(C1), y:(C2) -> z:(C1, C2)`)
    -   $g^{x}_{c_1} = \sum_{c_2} y_{c_2} g^{z}_{c_1 c_2} = y \cdot (g^z)^T$. (`y:(C2), gz:(C1, C2)`). Requires `vecmat` with `gz` transposed (`gz^T:(C2, C1)`). Operation: `vecmat(y, gz^T)`. Output `gx:(C1)`.
    -   $g^{y}_{c_2} = \sum_{c_1} g^{z}_{c_1 c_2} x_{c_1} = g^z \cdot x$. (`gz:(C1, C2), x:(C1)`). Requires `mat_t_vec` (computes `matrix . vector`). Operation: `mat_t_vec(gz, x)`. Output `gy:(C2)`.
-   **`inner`**: $z = \sum_c x_c y_c$. (`x:(C), y:(C) -> z:()`)
    -   $g^x_c = y_c g^z$. (`y:(C), gz:()`). Requires `vecsca`. Operation: `vecsca(y, gz)`. Output `gx:(C)`.
    -   $g^y_c = g^z x_c$. (`gz:(), x:(C)`). Requires `scavec`. Operation: `scavec(gz, x)`. Output `gy:(C)`.
-   **`vecmat`**: $z_{c_{out}} = \sum_{c_{in}} x_{c_{in}} y_{c_{in} c_{out}}$. (`x:(Cin), y:(Cin, Cout) -> z:(Cout)`)
    -   $g^x_{c_{in}} = \sum_{c_{out}} y_{c_{in} c_{out}} g^z_{c_{out}} = y \cdot g^z$. (`y:(Cin, Cout), gz:(Cout)`). Requires `mat_t_vec` (computes `matrix . vector`) with `y` transposed (`y^T:(Cout, Cin)`). Operation: `mat_t_vec(y^T, gz)`. Output `gx:(Cin)`.
    -   $g^y_{c_{in} c_{out}} = g^z_{c_{out}} x_{c_{in}}$. (`gz:(Cout), x:(Cin)`). Requires `outer`. The natural output shape of `outer(gz, x)` is `(Cout, Cin)`. To match `gy`'s target shape `(Cin, Cout)`, we need to transpose the result. Operation: `outer(gz, x)^T`.
-   **`vecsca`**: $z_c = x_c y$. (`x:(C), y:() -> z:(C)`)
    -   $g^x_c = y g^z_c$. (`y:(), gz:(C)`). Requires `scavec`. Operation: `scavec(y, gz)`. Output `gx:(C)`.
    -   $g^y = \sum_c g^z_c x_c$. (`gz:(C), x:(C)`). Requires `inner`. Operation: `inner(gz, x)`. Output `gy:()`.
-   **`scavec`**: $z_c = x y_c$. (`x:(), y:(C) -> z:(C)`)
    -   $g^x = \sum_c y_c g^z_c$. (`y:(C), gz:(C)`). Requires `inner`. Operation: `inner(y, gz)`. Output `gx:()`.
    -   $g^y_c = g^z_c x$. (`gz:(C), x:()`). Requires `vecsca`. Operation: `vecsca(gz, x)`. Output `gy:(C)`.
-   **`mat_t_vec`**: $z_{c_{out}} = \sum_{c_{in}} x_{c_{in} c_{out}} y_{c_{in}}$. (`x:(Cin, Cout), y:(Cin) -> z:(Cout)`)
    -   $g^x_{c_{in} c_{out}} = y_{c_{in}} g^z_{c_{out}}$. (`y:(Cin), gz:(Cout)`). Requires `outer`. The output shape of `outer(y, gz)` is `(Cin, Cout)`, which directly matches `gx`'s target shape `(Cin, Cout)`. Operation: `outer(y, gz)`.
    -   $g^y_{c_{in}} = \sum_{c_{out}} g^z_{c_{out}} x_{c_{in} c_{out}} = g^z \cdot x^T$. (`gz:(Cout), x:(Cin, Cout)`). Requires `vecmat` with `x^T` (shape `(Cout, Cin)`). Operation: `vecmat(gz, x^T)`. Output `gy:(Cin)`.

This table summarizes the dual relationships between the forward and backward operations within this set of dense products, explicitly considering necessary transpositions to maintain the `(y, gz)` and `(gz, x)` input order convention and ensure correct output gradient shapes.

## Examples of Functional Layers and `sparse_*` Operations Usage

This section illustrates how key functional layers from `equitorch.nn.functional` leverage the `sparse_*` operations discussed previously. Tensor shapes are described assuming `N` is the batch size, `irreps_dim` is the dimension corresponding to the geometric features, `channels` (or `channels_in`/`channels_out`) is the feature channel dimension. Other specific dimension names like `num_irrep_instances` or `num_gate_features` are explained in context.

### Example 1: `TensorDotUU` (from `equitorch/nn/functional/tensor_products.py`)

*   **Purpose:** Computes a channel-wise, irrep-wise dot product between two input tensors, `input1` and `input2`. These inputs are assumed to share the same irreps structure and channel dimension.
*   **Key Inputs:**
    *   `input1`, `input2` (Tensors): Input tensors, typically `(N, irreps_dim, channels)`.
    *   `irreps_info` (IrrepsInfo): Contains irreps structure details, especially `irreps_info.irrep_seg` (defining segments for each irrep instance) and `irreps_info.rsqrt_dims` (for scaling).
    *   `scaled` (bool): If true, scales the result.
*   **Core `sparse_*` Usage (`sparse_mul`):**
    1.  A `SparseProductInfo` object (`info_fwd`) is created with `seg_out=irreps_info.irrep_seg`.
    2.  `sparse_mul(input1, input2, info_fwd, ...)` is called.
        *   Internally, this performs an element-wise multiplication of `input1` and `input2`.
        *   The `seg_out` argument in `info_fwd` then instructs the underlying `indexed_mul_scale_gather` kernel to **sum** these products over the segments defined by `irreps_info.irrep_seg`. This effectively computes a dot product for each irrep instance within each channel.
    3.  If `scaled` is true, the result is further scaled.
*   **Output:** A tensor where `irreps_dim` is replaced by a dimension representing `num_irrep_instances` (the total number of irrep instances after considering multiplicities). Shape: `(N, num_irrep_instances, channels)`.
*   **Backward Pass:** Also uses `sparse_mul` with appropriately configured `SparseProductInfo` for gradients.

### Example 2: `Gating` (from `equitorch/nn/functional/activations.py`)

*   **Purpose:** Implements an equivariant gating mechanism, modulating an `input` tensor using a `gates` tensor.
*   **Key Inputs:**
    *   `input` (Tensor): Tensor to be gated, `(N, irreps_dim, channels)`.
    *   `gates` (Tensor): Gating values, e.g., `(N, num_gate_features, channels)`. `num_gate_features` is the size of the dimension indexed by `irreps_info.irrep_index`.
    *   `irreps_info` (IrrepsInfo): Contains `irreps_info.irrep_index`, which maps each component of `input`'s `irreps_dim` to an index in `gates`' `num_gate_features` dimension.
*   **Core `sparse_*` Usage (`sparse_mul`):**
    1.  A `SparseProductInfo` object (`info_fwd`) is created with `index2=irreps_info.irrep_index`.
    2.  `sparse_mul(input, gates, info_fwd)` is called.
        *   The `index2` argument ensures that each element `input[:, k, c]` is multiplied by the gate value `gates[:, irreps_info.irrep_index[k], c]`. This is an **indexed element-wise multiplication**.
*   **Output:** Tensor with the same shape as `input`: `(N, irreps_dim, channels)`, now gated.
*   **Backward Pass:** Uses `sparse_mul` for gradients.

### Example 3: `IrrepWiseLinear` (from `equitorch/nn/functional/linears.py`)

*   **Purpose:** Performs an irrep-wise linear transformation, applying a separate linear map (matrix multiplication on channels) to each irrep block.
*   **Key Inputs:**
    *   `input` (Tensor): Input tensor, `(N, irreps_dim, channels_in)`.
    *   `weight` (Tensor): Weight tensor. Its shape determines if weights are shared across the batch.
        *   If `weight` shape is `(num_irrep_types, channels_in, channels_out)`, the weights are shared across all items in the batch.
        *   If `weight` shape is `(N, num_irrep_types, channels_in, channels_out)`, each batch item uses a distinct set of weights.
        *   The underlying `sparse_vecmat` operation automatically handles this based on the `weight` tensor's dimensionality. `num_irrep_types` is the number of unique irrep types.
    *   `irreps_info` (IrrepsInfo): Contains `irreps_info.irrep_index` to map `input` components to their irrep type, selecting the correct weight matrix.
*   **Core `sparse_*` Usage (`sparse_vecmat`):**
    1.  A `SparseProductInfo` object (`info_fwd`) is created with `index2=irreps_info.irrep_index`.
    2.  `sparse_vecmat(input, weight, info_fwd)` is called.
        *   For each component `k` along `input`'s `irreps_dim`, the channel vector `input[:, k, :]` is multiplied by the weight matrix `weight[irreps_info.irrep_index[k], :, :]`. This is an **indexed vector-matrix multiplication**.
*   **Output:** Tensor of shape `(N, irreps_dim, channels_out)`.
*   **Backward Pass:** Uses `sparse_mat_t_vec` and `sparse_outer` for gradients.

### General Role of `sparse_*` Operations in these Examples

The `sparse_*` operations (like `sparse_mul`, `sparse_vecmat`) are the workhorses enabling these functional layers. Their key characteristics are leveraged as follows:

1.  **Core Computational Engine:** They provide the underlying product logic (element-wise, vector-matrix, etc.).
2.  **Handling Sparsity and Indexing:**
    *   The `index1` and `index2` arguments in `SparseProductInfo` are crucial for selecting specific elements or weights.
    *   In `Gating` and `IrrepWiseLinear`, `index2=irreps_info.irrep_index` allows different parts of the `input` tensor to interact with different parts of the `gates` or `weight` tensors, based on their irrep type.
3.  **Segmentation and Gathering/Accumulation:**
    *   The `seg_out` argument in `SparseProductInfo` is used by `TensorDotUU` (via `sparse_mul`) to sum results over specific segments (irrep blocks), thereby achieving the dot product functionality.
4.  **Autograd Support:** As `torch.autograd.Function` instances, they seamlessly integrate into PyTorch's differentiation framework, allowing gradients to be computed through these complex indexed and segmented operations.

These examples highlight how the `sparse_*` operations, through the flexible `SparseProductInfo` configuration, provide a powerful and unified way to implement diverse equivariant operations by composing fundamental products with sophisticated indexing and aggregation schemes.
