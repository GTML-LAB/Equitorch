# Prompts

This folder provides some AI-generated backgrounds for related operations.
The backgrounds are not carefully revised and may contain some errors, but it may provide a starting point for LLMs to understand basic ideas.


## File Summaries:

### 1. `irreps_introduction.md`
*   **Keywords:** Irreps, O(3), SO(3), tensor shape, `Irrep` class, `Irreps` class, multiplicity, channels, e3nn comparison, Schur's Lemma, Clebsch-Gordan coefficients.
*   **Introduction:** This document details the conventions for representing irreducible representations (irreps) of O(3)/SO(3) and the standard tensor shapes within the `equitorch` library. It focuses on the `Irrep` and `Irreps` classes, their attributes, initialization, methods (like tensor product), and the distinction between multiplicity and channels. It also covers tensor shape conventions, compares them with `e3nn`, discusses irrep interactions based on Schur's Lemma, and provides examples of module input/output shapes.

### 2. `equitorch_others.md`
*   **Keywords:** `SplitIrreps`, `Separable` module, equivariant tensors, `torch.split`, sub-modules, tensor manipulation.
*   **Introduction:** This document explains utility modules in `equitorch/nn/others.py`. It covers `SplitIrreps` for dividing tensors based on irrep dimensions and `Separable` for applying different transformations to these segments. It includes parameters, functionality, and usage examples for both modules.

### 3. `sparse_autograd_functions_documentation.md`

*   **Keywords:** Sparse Tensor Operations, Autograd Functions, SparseProductInfo, SparseScaleInfo, Segment Aggregation, Automatic Differentiation, Triton Kernels, Tensor Products, Sparse Indexing, Forward Pass, Backward Pass, Chain Rule.
*   **Introduction:** This document provides a mathematical and operational overview of `torch.autograd.Function` subclasses and `...Info` configuration structures in `equitorch`. It covers sparse tensor products and sparse scale/segment operations, detailing their mathematical formulations, configuration objects (`SparseProductInfo`, `SparseScaleInfo`), and the mechanics of their forward and backward passes for automatic differentiation, without delving into low-level Triton kernel implementations.

### 4. `product_kernels_documentation.md`

*   **Keywords:** Triton Kernels, Sparse Operations, Indexed Operations, Batched Operations, Segment Gathering, Kernel Utilities, Dense Product Kernels, `SparseProductInfo`, Autograd Wrappers, Tensor Shape Conventions, Forward/Backward Pass Relationships.
*   **Introduction:** This document details the Triton kernels and Python wrapper functions in `equitorch` for various product operations (e.g., mul, outer, inner, vecmat). It covers low-level kernel utilities (`kernel_utils.py`), core dense product kernels (`kernel_dense.py`), and batched sparse/indexed operations with scaling and segment gathering (`batched_sparse_dense_op.py`). It also explains the `SparseProductInfo` structure, autograd wrappers for differentiability (`sparse_product.py`), tensor shape conventions, and the mathematical relationships between forward and backward passes for these operations.

### 5. `indexed_scale_segment_documentation.md`

*   **Keywords:** Triton Kernel, Indexed Operation, Scaled Operation, Segment Reduction, Sparse Matrix Multiplication, Autograd Wrapper, `SparseScaleInfo`, Tensor Shapes.
*   **Introduction:** This document describes the Triton kernel (`indexed_scale_segment_kernel`) and Python wrapper (`indexed_scale_segment`) in `equitorch/ops/indexed_scale_segment_op.py` for performing an indexed, scaled segment reduction. It also covers the `SparseScale` autograd function from `equitorch/nn/functional/sparse_scale.py` for differentiability, the `SparseScaleInfo` configuration object, the mathematical formulation (relating to sparse matrix multiplication), and tensor shape conventions.

