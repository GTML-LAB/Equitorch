# Autograd Functions for Sparse Operations: Mathematical Overview

This document provides a high-level mathematical and operational overview of `torch.autograd.Function` subclasses and their associated `...Info` configuration structures, as used in `equitorch`. These components facilitate complex tensor operations involving sparse indexing, optional scaling, and segment-based aggregation, all while supporting automatic differentiation.

The primary focus is on two families of operations:
1.  **Sparse Tensor Products**: Operations like element-wise multiplication, outer products, inner products, etc., applied to sparsely indexed and potentially batched tensors.
2.  **Sparse Scale and Segment Operations**: Operations involving scaling elements of a tensor and then summing them over specified segments.

This document will not delve into the low-level Triton kernel implementations but will focus on the mathematical definitions, the role of configuration objects, and the mechanics of forward and backward passes, emphasizing the connection between the mathematical formulas and the operations.

## 1. Configuration Structures (`...Info`)

These `NamedTuple` structures hold the necessary metadata and tensor indices to configure the sparse operations.

### 1.1. `SparseProductInfo`

Used by the `SparseProduct*` family of `autograd.Function` classes (e.g., `SparseMul`, `SparseOuter`).

**Purpose:** Configures sparse tensor product operations, defining how input elements are selected, scaled, combined, and how results are aggregated and stored.

**General Mathematical Formulation:**
The operations generally compute:
$\mathbf{z}_{nM}=\sum_{t\in \text{Ind}^*[M]}s_{t}\ \mathbf{x}_{n,\text{Ind}_1[t]}\circ \mathbf{y}_{n,\text{Ind}_2[t]}$

Where:
- $\mathbf{z}_{nM}$: The output tensor element for batch item $n$ and output segment index $M$. The bold font indicates it represents a tensor (scalar, vector, or matrix) over the "dense" channel dimensions.
- $\mathbf{x}_{n, \text{Ind}_1[t]}$: The first input tensor element for batch item $n$ and input index $\text{Ind}_1[t]$.
- $\mathbf{y}_{n, \text{Ind}_2[t]}$: The second input tensor element for batch item $n$ and input index $\text{Ind}_2[t]$.
- $\circ$: Represents the specific tensor product operation (element-wise multiplication `mul`, outer product `outer`, inner product `inner`, vector-matrix product `vecmat`, etc.). This is the "dense" part of the operation.
- $s_t$: An optional scaling factor associated with the sparse interaction index $t$.
- $t$: An index iterating over a "sparse" dimension of interactions.
- $\text{Ind}_1[t]$: A function (or lookup table) mapping the sparse index $t$ to the corresponding index in the first input tensor's sparse dimension.
- $\text{Ind}_2[t]$: A function (or lookup table) mapping the sparse index $t$ to the corresponding index in the second input tensor's sparse dimension.
- $\text{Ind}^*[M]$: The set of sparse indices $t$ that belong to the output segment $M$. This defines the summation range for each output element $M$.

**Fields and their Mathematical Roles:**

-   `scale: Optional[Tensor]`
    -   Role: Corresponds to $s_t$ in the formula. These are the scaling factors applied to each product term $\mathbf{x} \circ \mathbf{y}$ before summation. Shape `(num_t,)`. If `None`, $s_t = 1$ for all $t$.
-   `index1: Optional[Tensor]`
    -   Role: Implements the mapping $\text{Ind}_1[t]$. It provides the indices used to gather elements from `input1`. Shape `(num_t,)`. If `None`, $\text{Ind}_1[t] = t$ (identity mapping).
-   `index2: Optional[Tensor]`
    -   Role: Implements the mapping $\text{Ind}_2[t]$. It provides the indices used to gather elements from `input2`. Shape `(num_t,)`. If `None`, $\text{Ind}_2[t] = t$ (identity mapping).
-   `seg_out: Optional[Tensor]`
    -   Role: Defines the sets $\text{Ind}^*[M]$ for the summation $\sum_{t\in \text{Ind}^*[M]}$. If present, `seg_out[M]` gives the starting $t$ (or an index into `gather_index`) for segment $M$, and `seg_out[M+1]-1` gives the end. This enables segment-wise aggregation (gathering). Shape `(num_M_nonzero+1,)`. If `None`, the summation $\sum_{t\in \text{Ind}^*[M]}$ is effectively removed (or each $t$ forms its own segment), and the output sparse dimension usually corresponds to $t$.
-   `gather_index: Optional[Tensor]`
    -   Role: Modifies how $t$ is obtained when `seg_out` is used. If `gather_index` is provided, the actual sparse index $t$ for an iteration `k` within a segment is $t = \text{gather\_index}[k]$. This allows gathering non-contiguous $t$ values into segments. Shape `(num_t_gathered,)`. If `None` (and `seg_out` is present), $t$ iterates contiguously from `seg_out[M]` to `seg_out[M+1]-1`.
-   `index_out: Optional[Tensor]`
    -   Role: Specifies the final storage location for $\mathbf{z}_{nM}$. If provided, the result for segment $M$ (or sparse index $t$ if not segmented) is written to $\text{output}[\text{index\_out}[M]]$. This enables sparse output writing. Shape `(num_M_nonzero,)`. If `None`, output is stored densely based on $M$ or $t$.
-   `out_size: Optional[int]`
    -   Role: Defines $N_M$, the total size of the output sparse dimension (the range of $M$).

### 1.2. `SparseScaleInfo`

Used by the `SparseScale` `autograd.Function` class.

**Purpose:** Configures an operation that scales elements of an input tensor and then sums them over specified segments.

**Mathematical Formulation:**
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

**Fields and their Mathematical Roles:**

-   `scale: Tensor`
    -   Role: Corresponds to $\text{scale}_t$ in the formula. These are the scaling factors applied to each selected input element. Shape `(num_t,)`.
-   `index: Tensor`
    -   Role: Corresponds to $\text{index}[t]$. It provides the indices used to gather elements from the input tensor's second-to-last dimension. Shape `(num_t,)`.
-   `seg_out: Tensor`
    -   Role: Defines the summation segments. `seg_out[m]` is the starting sparse index $t$ for output segment $m$, and the sum runs for $t$ from `seg_out[m]` to `seg_out[m+1]-1`. Shape `(num_M+1,)`.
-   `out_size: int`
    -   Role: Defines $N_M$, the size of the output's segmented dimension (the range of $m$).

## 2. Sparse Tensor Product Operations (`SparseProduct*`)

These classes (`SparseMul`, `SparseOuter`, `SparseInner`, `SparseVecMat`, `SparseVecSca`, `SparseScaVec`, `SparseMatTVec`) are defined in `equitorch/nn/functional/sparse_product.py`. They wrap underlying `indexed_*_scale_gather` functions to make them differentiable.

### 2.1. General `forward` Method Structure

```python
@staticmethod
def forward(ctx, input1: Tensor, input2: Tensor, 
            info_fwd: SparseProductInfo, 
            info_bwd1: Optional[SparseProductInfo] = None, 
            info_bwd2: Optional[SparseProductInfo] = None, 
            out_accumulated: bool = False) -> Tensor:
    # 1. Calls the corresponding indexed_*_scale_gather function
    #    (e.g., indexed_mul_scale_gather for SparseMul).
    #    This function implements the mathematical formula using input1, input2,
    #    and parameters from info_fwd (scale, index1, index2, seg_out, etc.).
    #    The out_accumulated flag controls if a sum over the batch dimension 'n' is performed.
    # 2. Saves input1, input2 for backward pass using ctx.save_for_backward.
    # 3. Stores info_fwd, info_bwd1, info_bwd2 in ctx.infos.
    #    info_bwd1/info_bwd2 are pre-configured for the gradient calculations.
    # 4. Determines shared1, shared2 (if inputs lack batch dim) and stores them with out_accumulated in ctx.
    #    These flags are crucial for determining batch handling in the backward pass.
    # 5. Returns the computed result z.
```
-   `info_fwd`: Configures the forward pass according to the mathematical formula.
-   `info_bwd1`, `info_bwd2`: Configurations for computing gradients $\frac{\partial L}{\partial \mathbf{x}}$ and $\frac{\partial L}{\partial \mathbf{y}}$, respectively. These infos define the structure of the backward operations.
-   `out_accumulated`: If `True`, the output $\mathbf{z}$ is summed over the batch dimension $n$. If $\mathbf{z}_{nM}$ is the formula without batch sum, then with `out_accumulated=True`, the output is $\mathbf{z}_{M} = \sum_n \mathbf{z}_{nM}$.

### 2.2. Dense Operation Relationships (Forward and Backward)

The core of each sparse product is a "dense" tensor operation ($\circ$) applied element-wise after indexing and scaling. The backward pass for these operations relies on the chain rule, which often transforms the original dense operation into a related one. The following table (from `tensor_product_kernels_documentation.md`) summarizes these relationships, crucial for understanding the `backward` methods of the `SparseProduct*` classes.

Convention:
- Forward: `z = op(x, y)` (representing the dense part $\mathbf{z} = \mathbf{x} \circ \mathbf{y}$)
- Backward w.r.t. x: `gx = op_bx(y, gz)` (representing $\frac{\partial L}{\partial \mathbf{x}} = \mathbf{y} \diamond \frac{\partial L}{\partial \mathbf{z}}$)
- Backward w.r.t. y: `gy = op_by(gz, x)` (representing $\frac{\partial L}{\partial \mathbf{y}} = \frac{\partial L}{\partial \mathbf{z}} \star \mathbf{x}$)

| Forward Operation (`z = x * y`) | Input Shapes (`x`, `y`) | Output Shape (`z`) | Backward for x (`gx = y * gz`) | Backward for y (`gy = gz * x`) | Notes                                                                                             |
| :------------------------------ | :---------------------- | :----------------- | :----------------------------- | :----------------------------- | :------------------------------------------------------------------------------------------------ |
| `mul`                           | `(C)`, `(C)`            | `(C)`              | `mul(y, gz)`                   | `mul(gz, x)`                   | Element-wise multiplication.                                                                      |
| `outer`                         | `(C1)`, `(C2)`          | `(C1, C2)`         | `vecmat(y, gz^T)`              | `mat_t_vec(gz, x)`             | `gz` shape is `(C1, C2)`. `gz^T` is `(C2, C1)`.                                                   |
| `inner`                         | `(C)`, `(C)`            | `()`               | `vecsca(y, gz)`              | `scavec(gz, x)`              | `gz` is scalar `()`.                                                                              |
| `vecmat`                        | `(Cin)`, `(Cin, Cout)`  | `(Cout)`           | `mat_t_vec(y^T, gz)`           | `outer(gz, x)^T`               | `y^T` shape is `(Cout, Cin)`. `gz` shape is `(Cout)`. `outer` output is `(Cout, Cin)`, needs `^T` for `gy:(Cin, Cout)`. |
| `vecsca`                      | `(C)`, `()`             | `(C)`              | `scavec(y, gz)`              | `inner(gz, x)`                 | `y` is scalar `()`. `gz` shape is `(C)`. `gy` sums contributions.                                 |
| `scavec`                      | `()`, `(C)`             | `(C)`              | `inner(y, gz)`                 | `vecsca(gz, x)`              | `x` is scalar `()`. `gz` shape is `(C)`. `gx` sums contributions.                                 |
| `mat_t_vec`                     | `(Cin, Cout)`, `(Cin)`  | `(Cout)`           | `outer(y, gz)`               | `vecmat(gz, x^T)`                | `x` is matrix. `gz` shape is `(Cout)`. `outer` output is `(Cin, Cout)`. |

### 2.3. General `backward` Method Structure

```python
@staticmethod
def backward(ctx, grad_output: Tensor): # grad_output is dL/dz
    # 1. Retrieves input1 (x), input2 (y), infos (info_fwd, info_bwd1, info_bwd2), 
    #    shared1 (left_shared_fwd), shared2 (right_shared_fwd), out_accumulated_fwd from ctx.
    # 2. Initializes grad1 (dL/dx), grad2 (dL/dy) to None.
    # 3. If gradient w.r.t. input1 is needed (ctx.needs_input_grad[0]):
    #    a. Determine out_accumulated_bwd1 = left_shared_fwd. This flag dictates if dL/dx
    #       should be summed over the batch dimension.
    #    b. Call the appropriate Sparse*.apply function for op_bx (from the table above).
    #       Example: For SparseOuter, op_bx is vecmat. So, call SparseVecMat.apply.
    #       The arguments to this nested .apply call for infos are typically (info_bwd1, info_bwd2, info_fwd),
    #       where info_bwd1 serves as the 'forward info' for this specific gradient calculation.
    #       Inputs are (input2, grad_output_possibly_transposed, info_bwd1, info_bwd2, info_fwd, out_accumulated_bwd1).
    #       info_bwd1 configures the sparse aspects of this op_bx.
    # 4. If gradient w.r.t. input2 is needed (ctx.needs_input_grad[1]):
    #    a. Determine out_accumulated_bwd2 = right_shared_fwd.
    #    b. Call the appropriate Sparse*.apply function for op_by.
    #       Example: For SparseOuter, op_by is mat_t_vec. So, call SparseMatTVec.apply.
    #       The arguments to this nested .apply call for infos are typically (info_bwd2, info_fwd, info_bwd1).
    #       Inputs are (grad_output_possibly_transposed, input1, info_bwd2, info_fwd, info_bwd1, out_accumulated_bwd2).
    #       info_bwd2 configures the sparse aspects of this op_by.
    # 5. Returns (grad1, grad2, None, None, None, None) for the original forward inputs.
```
The `info_bwd1` and `info_bwd2` objects are crucial. They are constructed such that their `scale`, `index1`, `index2`, `seg_out`, `index_out`, and `out_size` fields correctly implement the sparse indexing and aggregation for the *backward* operations. For example, if the forward operation was $z_M = \sum_t s_t x_{\text{Ind1}[t]} y_{\text{Ind2}[t]}$, then $g^x_K = \sum_u s'_u y_{\text{IndA}[u]} (g^z)_{\text{IndB}[u]}$. `info_bwd1` would contain $s'$, $\text{IndA}$, $\text{IndB}$, and the segmentation for $K$ and $u$.

A key pattern in the `backward` pass is the cyclical use of the `SparseProductInfo` objects. When `Sparse*.apply` is called to compute a gradient:
- For `grad1` (gradient w.r.t. `input1`): The `info` arguments passed are `(info_bwd1, info_bwd2, info_fwd)`. Here, `info_bwd1` acts as the "forward" configuration for this specific gradient sub-problem.
- For `grad2` (gradient w.r.t. `input2`): The `info` arguments passed are `(info_bwd2, info_fwd, info_bwd1)`. Here, `info_bwd2` acts as the "forward" configuration.
This cyclical arrangement ensures that the correct indexing, scaling, and segmentation rules are applied for each part of the chain rule.

### 2.4. Specific Sparse Product Functions

All functions take `input1`, `input2`, `info_fwd`, `info_bwd1`, `info_bwd2`, and `out_accumulated` as arguments. Their backward passes use the operations from the table:

-   **`SparseMul`**:
    -   Forward $\circ$: Element-wise multiplication.
    -   Backward for `input1` ($g^x$): `SparseMul.apply(input2, grad_output, info_bwd1, ...)`
    -   Backward for `input2` ($g^y$): `SparseMul.apply(grad_output, input1, info_bwd2, ...)`

-   **`SparseOuter`**:
    -   Forward $\circ$: Outer product.
    -   Backward for `input1` ($g^x$): `SparseVecMat.apply(input2, grad_output.transpose(-1,-2).contiguous(), info_bwd1, ...)` (uses `vecmat(y, gz^T)`)
    -   Backward for `input2` ($g^y$): `SparseMatTVec.apply(grad_output, input1, info_bwd2, ...)` (uses `mat_t_vec(gz, x)`)

-   **`SparseInner`**:
    -   Forward $\circ$: Inner product.
    -   Backward for `input1` ($g^x$): `SparseVecSca.apply(input2, grad_output, info_bwd1, ...)` (uses `vecsca(y, gz)`)
    -   Backward for `input2` ($g^y$): `SparseScaVec.apply(grad_output, input1, info_bwd2, ...)` (uses `scavec(gz, x)`)

-   **`SparseVecMat`**:
    -   Forward $\circ$: Vector-matrix multiplication.
    -   Backward for `input1` ($g^x$): `SparseMatTVec.apply(input2.transpose(-1,-2).contiguous(), grad_output, info_bwd1, ...)` (uses `mat_t_vec(y^T, gz)`)
    -   Backward for `input2` ($g^y$): `SparseOuter.apply(grad_output, input1, info_bwd2, ...).transpose(-1,-2).contiguous()` (uses `outer(gz, x)^T`)

-   **`SparseVecSca`**:
    -   Forward $\circ$: Vector-scalar multiplication.
    -   Backward for `input1` ($g^x$): `SparseScaVec.apply(input2, grad_output, info_bwd1, ...)` (uses `scavec(y, gz)`)
    -   Backward for `input2` ($g^y$): `SparseInner.apply(grad_output, input1, info_bwd2, ...)` (uses `inner(gz, x)`)

-   **`SparseScaVec`**:
    -   Forward $\circ$: Scalar-vector multiplication.
    -   Backward for `input1` ($g^x$): `SparseInner.apply(input2, grad_output, info_bwd1, ...)` (uses `inner(y, gz)`)
    -   Backward for `input2` ($g^y$): `SparseVecSca.apply(grad_output, input1, info_bwd2, ...)` (uses `vecsca(gz, x)`)

-   **`SparseMatTVec`**:
    -   Forward $\circ$: Matrix transpose-vector multiplication.
    -   Backward for `input1` ($g^x$): `SparseOuter.apply(input2, grad_output, info_bwd1, ...)` (uses `outer(y, gz)`)
    -   Backward for `input2` ($g^y$): `SparseVecMat.apply(grad_output, input1.transpose(-1,-2).contiguous(), info_bwd2, ...)` (uses `vecmat(gz, x^T)`)

## 3. Sparse Scale and Segment Operation (`SparseScale`)

Defined in `equitorch/nn/functional/sparse_scale.py`. This class wraps the `indexed_scale_segment` operation.

**Mathematical Formulation (recap):**
$\text{output}_{nmc} = \sum_{t \text{ s.t. } \text{seg\_out}[m] \le t < \text{seg\_out}[m+1]} \text{scale}_t \times \text{input}_{n, \text{index}[t], c}$

This operation can be viewed as a batched multiplication by a sparse matrix $\mathbf{S}$: $\mathbf{y}_n = \mathbf{S} \mathbf{x}_n$, where $\mathbf{x}_n = \text{input}_{n, :, c}$ and $\mathbf{y}_n = \text{output}_{n, :, c}$. The matrix $\mathbf{S}$ (shape `(out_size, input.shape[-2])`) has non-zero entries $S_{m, \text{index}[t]} = \text{scale}_t$ for $t$ within segment $m$.

### 3.1. `forward` Method

```python
@staticmethod
def forward(ctx, input: Tensor, 
            info_fwd: SparseScaleInfo, 
            info_bwd: SparseScaleInfo) -> Tensor:
    # 1. Calls indexed_scale_segment. This function implements the mathematical formula
    #    using 'input' and parameters from info_fwd (scale, index, seg_out, out_size).
    # 2. Saves 'input' for backward pass.
    # 3. Stores info_fwd (defining S) and info_bwd (defining S^T) in ctx.
    # 4. Returns the computed result 'output'.
```
-   `info_fwd`: Configures the forward operation $\mathbf{S}$.
-   `info_bwd`: Configures the backward operation, which will use $\mathbf{S}^T$.

### 3.2. `backward` Method

```python
@staticmethod
def backward(ctx, grad_output: Tensor): # grad_output is dL/d(output)
    # 1. Retrieves saved 'input', info_fwd, info_bwd from ctx.
    # 2. If gradient w.r.t. 'input' is needed (ctx.needs_input_grad[0]):
    #    a. The gradient dL/d(input) is computed as S^T * dL/d(output).
    #       This is achieved by reusing SparseScale.apply:
    #       grad_in = SparseScale.apply(grad_output, info_bwd, info_fwd)
    #       - The 'input' to this backward call is grad_output.
    #       - The 'info_fwd' for this call is the original info_bwd (this configures the S^T operation).
    #       - The 'info_bwd' for this call is the original info_fwd (not strictly needed for S^T's gradient, but satisfies signature).
    # 3. Returns (grad_in, None, None) for the original forward inputs.
```
The backward pass for $\mathbf{y} = \mathbf{S} \mathbf{x}$ is $\frac{\partial L}{\partial \mathbf{x}} = \mathbf{S}^T \frac{\partial L}{\partial \mathbf{y}}$. The `info_bwd` provided to the forward pass is specifically constructed to represent this $\mathbf{S}^T$ operation when used as `info_fwd` in a `SparseScale.apply` call.

## 4. Tensor Shape Conventions and Batch Handling

-   **Typical Dimensions:**
    -   `N`: Batch dimension (often first).
    -   `M` (or `M_in`, `M_out`, `T`): Sparse/indexed dimension. This is the dimension over which indexing (`index1`, `index2`, `index`) and segmentation (`seg_out`) primarily operate.
    -   `C` (or `C1`, `C2`, `Cin`, `Cout`): Dense/channel dimensions (usually last). These are the dimensions involved in the $\circ$ operation.

-   **Batch Handling Flags (in `SparseProduct*` context `ctx`):**
    -   `shared1` / `shared2`: Boolean. `True` if `input1` / `input2` respectively lacks the batch dimension $N$. This means the data from that input is broadcasted or shared across all batch items.
        - Mathematically, if `shared1` is true, $\mathbf{x}_{n, \text{Ind}_1[t]}$ becomes $\mathbf{x}_{\text{Ind}_1[t]}$.
    -   `out_accumulated` (forward): Boolean. If `True`, the forward output $\mathbf{z}$ is summed over $n$: $\mathbf{z}_{M} = \sum_n \sum_{t\in \text{Ind}^*[M]}s_{t}\ \mathbf{x}_{n,\text{Ind}_1[t]}\circ \mathbf{y}_{n,\text{Ind}_2[t]}$. The output tensor lacks the batch dimension.

-   **Influence on Backward Pass Batch Handling:**
    -   The `out_accumulated` flag for a backward gradient computation (e.g., for $g^x = \frac{\partial L}{\partial \mathbf{x}}$) is determined by the `shared` status of the corresponding forward input.
        -   `out_accumulated_bwd1 = ctx.shared1` (i.e., `left_shared_fwd`): If `input1` was shared across the batch in the forward pass, its gradient `grad1` will be accumulated (summed) over the batch dimension.
        -   `out_accumulated_bwd2 = ctx.shared2` (i.e., `right_shared_fwd`): Similarly for `grad2`.
    -   This ensures consistency with the chain rule: if an input variable is used multiple times by being broadcast across a batch, its gradient contributions from each batch item must be summed up.

## 5. Autodifferentiation Summary

The `autograd.Function` classes (`SparseProduct*` and `SparseScale`) are pivotal for integrating complex sparse tensor operations into PyTorch's automatic differentiation framework.
-   They define the precise forward computation based on mathematical formulas, parameterized by `...Info` objects.
-   Their `backward` methods implement the chain rule, calculating gradients with respect to inputs. This often involves reusing the same family of sparse operations but with inputs and `...Info` configurations transformed according to the rules of differentiation (e.g., using related dense operations from the table, or using a "transposed" sparse structure like in `SparseScale`).
-   The `...Info` structures are central, as they encode the specific indexing, scaling, and aggregation patterns for both forward and backward passes, ensuring mathematical correctness.
-   This system allows developers to construct sophisticated neural network layers using these sparse primitives, with PyTorch handling the gradient computations seamlessly.
