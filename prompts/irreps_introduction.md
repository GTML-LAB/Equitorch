# Equitorch Irreps and Tensor Shape Conventions

This document details the conventions for representing irreducible representations (irreps) of O(3)/SO(3) and the standard tensor shapes within the `equitorch` library, focusing on the `Irrep` and `Irreps` classes and their usage in defining tensor structures.

## 1. Irreducible Representation (Irrep) Definition

Irreps are fundamental building blocks for equivariant neural networks. They are handled by the `Irrep` and `Irreps` classes in `equitorch/irreps/__init__.py`.

### `Irrep` Class

Represents a *single* irreducible representation of O(3) (with parity) or SO(3) (without parity).

*   **Key Attributes:**
    *   `l` (int): The degree or order of the representation (e.g., 0 for scalar, 1 for vector, 2 for rank-2 tensor). Corresponds to the angular momentum quantum number.
    *   `p` (int): The parity.
        *   `+1`: Even parity (denoted by 'e'). Behaves like `Y_lm` where `l` is even.
        *   `-1`: Odd parity (denoted by 'o'). Behaves like `Y_lm` where `l` is odd.
        *   `0`: No specific parity (used for SO(3) irreps).
    *   `dim` (int): The dimension of the vector space for this irrep, calculated as `2*l + 1`.

*   **Initialization Examples:**
    ```python
    # O(3) Irreps (with parity)
    irrep_0e = Irrep(0, 'e')  # Scalar, even parity (l=0, p=+1)
    irrep_1o = Irrep(1, 'o')  # Vector, odd parity (l=1, p=-1)
    irrep_2e = Irrep("2e")    # Rank-2 tensor, even parity (l=2, p=+1)
    irrep_3o = Irrep((3, -1)) # Rank-3 tensor, odd parity (l=3, p=-1)

    # SO(3) Irreps (without parity)
    irrep_0 = Irrep(0)       # Scalar (l=0, p=0)
    irrep_1 = Irrep("1")     # Vector (l=1, p=0)
    irrep_2 = Irrep((2, 0))  # Rank-2 tensor (l=2, p=0)
    irrep_2_none = Irrep((2, None)) # Equivalent to Irrep(2, 0)
    ```

*   **Representation:** An `Irrep` object prints as a concise string:
    ```python
    print(Irrep(2, 'e'))  # Output: 2e
    print(Irrep(1))       # Output: 1
    print(Irrep(3, 'o'))  # Output: 3o
    ```

*   **Tensor Product (`*`):** Calculates the decomposition of the tensor product of two irreps. The resulting irreps have `l` ranging from `|l1 - l2|` to `l1 + l2`, and parity `p = p1 * p2`.
    ```python
    # Vector (1o) x Vector (1o) -> Scalar (0e) + Pseudovector (1e) + Rank-2 Tensor (2e)
    print(Irrep("1o") * Irrep("1o")) # Output: Irreps(0e+1e+2e)

    # Vector (1) x Rank-2 Tensor (2) -> Vector (1) + Rank-2 (2) + Rank-3 (3)
    print(Irrep(1) * Irrep(2))      # Output: Irreps(1+2+3)
    ```

### `Irreps` Class

Represents a *collection* (direct sum) of `Irrep` objects, often with **multiplicities** indicating how many times each irrep type appears *structurally* within the representation definition. This defines the abstract geometric structure.

*   **Key Attributes:**
    *   `irrep_groups`: A list of `(Irrep, multiplicity)` tuples defining the collection. The `multiplicity` here refers to the structural count within the `Irreps` definition itself.
    *   `dim`: The total dimension of the tensor space represented by this collection (sum of `irrep.dim * multiplicity` for all groups). This is the size of the `irreps_dim` axis in a corresponding tensor *without* considering feature channels.

*   **Important Distinction: Multiplicity vs. Channels:**
    *   The **multiplicity** (e.g., the `3` in `3x0e`) defined within an `Irreps` string or object describes the *composition* of the representation type.
    *   The **channels** dimension (the last axis in the standard tensor shape `(N, irreps_dim, channels)`) represents independent feature vectors associated with the *entire* geometric structure defined by the `Irreps`. Network layers operate on and mix these channels. The number of channels in a tensor is often uniform across all irreps and determined by the network architecture (e.g., `channels_in`, `channels_out` in layers), not directly by the individual multiplicities in the `Irreps` definition.

*   **Initialization Examples:**
    ```python
    # From a formatted string (multiplicity x irrep, separated by +)
    irreps_a = Irreps("3x0e + 2x1o + 1x2e") # 3 scalars (0e), 2 vectors (1o), 1 tensor (2e)
    irreps_b = Irreps("0e+1o+2e")           # Equivalent to 1x0e + 1x1o + 1x2e

    # From a list of Irrep objects, strings, or (Irrep, multiplicity) tuples
    irreps_c = Irreps([Irrep(0), "1o", (Irrep("2e"), 5)]) # 1x0 + 1x1o + 5x2e

    # From a range tuple (min_l, max_l) for SO(3) irreps
    irreps_d = Irreps((0, 2)) # Creates Irreps(0+1+2)

    # From a single Irrep
    irreps_e = Irreps(Irrep("1o")) # Creates Irreps(1o)
    ```

*   **Representation:** Prints as a string showing the irreps and multiplicities.
    ```python
    print(Irreps("3x0e + 2x1o + 1x2e")) # Output: Irreps(3x0e+2x1o+1x2e)
    print(Irreps((0, 2)))              # Output: Irreps(0+1+2)
    ```

*   **Key Methods:**
    *   `simplified()`: Merges identical irreps by summing their multiplicities and sorts the result. Useful for canonical representation.
        ```python
        print(Irreps("1o+0e+1o+2x0e").simplified()) # Output: Irreps(3x0e+2x1o)
        ```
    *   `__iter__()`: Allows iteration over individual `Irrep` objects, respecting multiplicities.
        ```python
        for ir in Irreps("2x0e+1x1o"):
            print(ir)
        # Output:
        # 0e
        # 0e
        # 1o
        ```
    *   `__len__()`: Returns the total number of irreps including multiplicities.
        ```python
        print(len(Irreps("2x0e+1x1o"))) # Output: 3
        ```
    *   `__getitem__[index]`: Accesses the `Irrep` at a specific index in the *expanded* list.
        ```python
        irreps = Irreps("2x0e+1x1o")
        print(irreps[0]) # Output: 0e
        print(irreps[1]) # Output: 0e
        print(irreps[2]) # Output: 1o
        ```
    *   `dim`: Property giving the total dimension.
        ```python
        # Irreps("1x0e + 2x1o") -> dim = (1*1) + (2*3) = 7
        print(Irreps("1x0e + 2x1o").dim) # Output: 7
        ```

## 2. Tensor Shape Convention

The standard shape for equivariant tensors in `equitorch` is:

`(batch, irreps_dim, channels)`

*   `batch`: data batch dimensions.
*   `irreps_dim`: The core dimension representing the geometric features. Its size is equal to the `.dim` attribute of the corresponding `Irreps` object. This dimension transforms according to the defined irreps under O(3)/SO(3) rotations/reflections.
*   `channels`: The feature channel dimension (last axis). It represents multiple independent feature vectors, each transforming according to the *entire* geometric structure defined by the `Irreps` object. Operations like `IrrepsLinear` mix information across this channel dimension.

**Example:**

Consider a tensor representing features for `N` atoms. The *type* of features is defined by `Irreps("2x0e + 1x1o")`. Let's say the network layer using these features operates with `C=16` channels.

*   The `Irreps("2x0e + 1x1o")` definition tells us the geometric structure: it's composed of two scalar parts and one vector part.
*   Total `irreps_dim` = `(2 * Irrep('0e').dim) + (1 * Irrep('1o').dim)` = `(2 * 1) + (1 * 3) = 5`. This is the size of the second-to-last dimension.
*   The `channels` dimension size is `16`, as specified by the layer or the data pipeline.

The tensor shape would be `(N, 5, 16)`.

*   The first 2 elements along the `irreps_dim` axis (`tensor[:, 0:2, :]`) correspond to the basis vectors spanning the two `0e` irreps defined in the `Irreps` structure.
*   The next 3 elements along the `irreps_dim` axis (`tensor[:, 2:5, :]`) correspond to the basis vectors spanning the single `1o` irrep defined in the `Irreps` structure.
*   Each of the 16 slices along the `channels` axis (`tensor[:, :, c]` for `c` in 0..15) represents a complete, independent feature vector that transforms according to `Irreps("2x0e + 1x1o")`.

### Comparison with e3nn Tensor Convention

While both `equitorch` and the popular `e3nn` library handle O(3) equivariant tensors, they employ different conventions for representing channels and structuring the final tensor dimension. Understanding these differences is crucial when comparing or translating between the libraries.

*   **`equitorch`:**
    *   **Shape:** `(batch, irreps_dim, channels)`
    *   **Channels:** Uses an explicit, separate `channels` dimension as the *last* axis. This dimension represents multiple independent feature vectors, each transforming according to the geometric structure defined by the `Irreps`.
    *   **`Irreps` Definition:** The `Irreps` object (e.g., `"2x0e + 1x1o"`) defines the geometric components. The `.dim` attribute of this `Irreps` object (e.g., `(2*1 + 1*3) = 5`) gives the size of the `irreps_dim` axis.
    *   **SO(3) Support:** Explicitly supports both O(3) irreps (with parity `e`/`o`) and SO(3) irreps (with parity `0`).

*   **`e3nn`:**
    *   **Shape:** `(batch, irreps_dim)`
    *   **Channels:** Does *not* typically use a separate `channels` axis. Instead, the concept of channels is integrated directly into the `Irreps` definition via multiplicities. Each irrep's multiplicity effectively acts as its channel count.
    *   **`Irreps` Definition:** To represent the equivalent of an `equitorch` tensor with `C` channels and geometric structure `irreps_geom`, the `e3nn` `Irreps` definition would multiply the multiplicity of *each* irrep in `irreps_geom` by `C`. For example, `equitorch`'s `Irreps("2x0e + 1x1o")` with `channels=16` corresponds conceptually to `e3nn`'s `Irreps("32x0e + 16x1o")`.
    *   **`irreps_dim` Calculation:** The size of the final `irreps_dim` axis in `e3nn` is the sum of `multiplicity * irrep.dim` across this *channel-multiplied* `Irreps` definition. For the example `Irreps("32x0e + 16x1o")`, the dimension is `(32 * 1) + (16 * 3) = 80`. In this `e3nn` representation, the first 32 elements along the `irreps_dim` axis are individual `0e` features (scalars), and the subsequent 48 elements are grouped into 16 sets of 3, where each set of 3 forms a `1o` feature (vector). Therefore, although this `e3nn` feature `Irreps("32x0e + 16x1o")` is mathematically equivalent to an `equitorch` feature `Irreps("2x0e + 1x1o")` with `channels=16`, the `e3nn` representation does not allow for an explicit, regular partitioning of a separate channel dimension in the same way `equitorch` does. The "channels" are interleaved with the irrep components.
    *   **SO(3) Support:** Primarily designed for O(3) irreps and does not have the same explicit built-in support for parity=0 (SO(3)) irreps as `equitorch`.

**Key Differences Summary:**

1.  **Channel Handling:** `equitorch` separates geometric structure (`irreps_dim`) and feature channels (`channels`), while `e3nn` combines them into a single `irreps_dim` by adjusting multiplicities in the `Irreps` definition.
2.  **Tensor Shape:** This leads to different final tensor shapes and dimension interpretations.
3.  **SO(3) Irreps:** `equitorch` provides explicit support for SO(3) irreps (parity 0).

Be mindful of these distinctions when working with or comparing the two libraries.

## 3. Irrep Interactions and Equivariance (Schur's Lemma)

Equivariant neural networks rely on operations that respect the symmetry group (O(3) or SO(3)). How different irreps can interact is governed by fundamental principles, notably **Schur's Lemma**.

*   **Schur's Lemma Implications:** In essence, Schur's Lemma restricts the possible linear maps between irreducible representations that commute with the group action (i.e., equivariant maps).
    *   For a linear map `L` from `irrep_A` to `irrep_B`:
        *   If `irrep_A` and `irrep_B` are **not equivalent** (e.g., different `l` or incompatible parity), the *only* equivariant linear map is the **zero map** (`L = 0`).
        *   If `irrep_A` and `irrep_B` are **equivalent** (identical `l` and `p`), any equivariant linear map `L` must be proportional to the **identity matrix** (`L = c * I`, where `c` is a scalar).

*   **Tensor Products (`TensorProduct`, `SO3Linear`):**
    *   The interaction between two input irreps (`ir1`, `ir2`) to produce an output irrep (`ir_out`) is governed by the tensor product decomposition rules. An output `ir_out` is possible if and only if:
        1.  Its degree `l` is within the range `[|l1 - l2|, l1 + l2]`.
        2.  Its parity `p` matches the product of input parities (`p = p1 * p2`), **unless** the output irrep is SO(3) (`p=0`), in which case the parity constraint is ignored (any `p1`, `p2` can contribute).
    *   The specific way these irreps combine is determined by the **Clebsch-Gordan coefficients**. These coefficients are non-zero only for allowed couplings satisfying the rules above.
    *   The `equitorch.utils._structs.prepare_so3` function calculates these allowed paths and their corresponding coefficients. The `has_path(ir_out, ir1, ir2)` function encapsulates these rules, checking if a coupling `(ir1, ir2) -> ir_out` is allowed.
    *   Learnable weights in `TensorProduct` or `SO3Linear` modules scale these fundamental coupling coefficients for each allowed path and channel combination.

*   **Linear Layers (`IrrepsLinear`, `IrrepWiseLinear`):**
    *   These layers map input features (`irreps_in`) to output features (`irreps_out`) while potentially mixing channels. They represent equivariant linear maps.
    *   `IrrepsLinear`: This layer implements the constraints from Schur's Lemma more directly for linear maps (which can be seen as a special case of tensor product with a scalar `0e` irrep). A path from an input `ir_in` to an output `ir_out` is allowed if and only if:
        1.  Their degrees match: `l_out == l_in`.
        2.  Their parities match (`p_out == p_in`), **unless** the output irrep is SO(3) (`p_out=0`), in which case the parity constraint is ignored (any `p_in` can contribute).
    *   The `equitorch.utils._structs.prepare_irreps_linear` function generates paths only for these allowed pairs, consistent with `has_path(ir_out, ir_in)`. The learnable weights provide the scalar (`c`) for the identity map (`I`) for each channel associated with that specific allowed path.
    *   `IrrepWiseLinear`: This is a simpler case where each irrep block is mapped only to itself (`ir_out == ir_in`), trivially satisfying the conditions. The weights perform standard linear mixing on the *channel* dimension within each block.

Understanding these interaction rules is crucial for designing valid and effective equivariant network architectures. The `Irreps` definitions determine *what* geometric features exist, while the interaction rules dictate *how* they can be combined or transformed by network layers.

## 4. Module Input/Output Shape Examples

Let's illustrate the shapes with specific modules, assuming `N` batch items and a network operating with `channels=16` or `channels=32` as specified. Remember that `channels_in`/`channels_out` refer to the size of the last tensor dimension.

**a) `IrrepsLinear`**

*   Mixes channels while preserving the irrep structure. `irreps_in` and `irreps_out` must be isomorphic.
*   Example: `IrrepsLinear(irreps_in="2x0e+1x1o", irreps_out="1x1o+2x0e", channels_in=16, channels_out=32)`
    *   `irreps_in.dim` = `(2*1 + 1*3) = 5`
    *   `irreps_out.dim` = `(1*3 + 2*1) = 5`
    *   Input shape: `(N, 5, 16)`
    *   Output shape: `(N, 5, 32)`

**b) `IrrepWiseLinear`**

*   Mixes channels *within* each irrep type independently.
*   Example: `IrrepWiseLinear(irreps="2x0e+1x1o", channels_in=16, channels_out=32)`
    *   `irreps.dim` = 5
    *   Input shape: `(N, 5, 16)`
    *   Output shape: `(N, 5, 32)` (The irrep structure `2x0e+1x1o` is unchanged).

**c) `SO3Linear` (`feature_mode='uv'`)**

*   Fully connected linear, second input has no channel dim.
*   Example: `SO3Linear(irreps_in1="2x1e", irreps_in2="0e+1e", irreps_out="3x1o", channels_in=16, channels_out=32)`
    *   `irreps_in1.dim` = `2 * 3 = 6`
    *   `irreps_in2.dim` = `1 + 3 = 4`
    *   `irreps_out.dim` = `3 * 3 = 9`
    *   Input1 shape: `(N, 6, 16)`
    *   Input2 shape: `(N, 4)`
    *   Output shape: `(N, 9, 32)`

**d) `TensorProduct` (`feature_mode='uvw'`)**

*   Fully connected tensor product between two tensors with channels.
*   Example: `TensorProduct(irreps_in1="1x1o", irreps_in2="1x1o", irreps_out="1x0e+1x1e+1x2e", channels_in1=16, channels_in2=16, channels_out=32)`
    *   `irreps_in1.dim` = 3
    *   `irreps_in2.dim` = 3
    *   `irreps_out.dim` = `1 + 3 + 5 = 9`
    *   Input1 shape: `(N, 3, 16)`
    *   Input2 shape: `(N, 3, 16)`
    *   Output shape: `(N, 9, 32)`

**e) `TensorProduct` (`feature_mode='uuu'`)**

*   Depthwise tensor product (often self-interaction). Assumes input channels are the same.
*   Example: `TensorProduct(irreps_in1="1x1o", irreps_in2="1x1o", irreps_out="1x0e+1x1e+1x2e", channels_in1=16, channels_in2=16, channels_out=16)` (Note: `channels_in1`, `channels_in2`, `channels_out` are often equal, represented by a single `channels` parameter in practice).
    *   `irreps_in1.dim` = 3
    *   `irreps_in2.dim` = 3
    *   `irreps_out.dim` = 9
    *   Input1 shape: `(N, 3, 16)`
    *   Input2 shape: `(N, 3, 16)`
    *   Output shape: `(N, 9, 16)`

**f) `TensorDot` (`feature_mode='uv'`)**

*   Channel-wise dot product. Reduces `irreps_dim` to 1 (scalar per channel).
*   Example: `TensorDot(irreps="2x1e+1x2o", feature_mode='uv')`
    *   `irreps.dim` = `(2*3) + (1*5) = 11`
    *   Input1 shape: `(N, 11, 16)`
    *   Input2 shape: `(N, 11, 16)`
    *   Output shape: `(N, 16)`
