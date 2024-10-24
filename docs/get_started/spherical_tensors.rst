.. Equivariant Tensors
.. ===================

.. First of all, we will give a brief introduction on equivariances and spherical tensors, a basic data structure widely used in SO(3)-equivariant neural networks.


.. Symmetries and Equivariances
.. ----------------------------

.. To be simplest, symmetries is a concept regarding to how properties or laws remains the same under a set of transformations.

.. Here we give some example of symmetries in machine learning:

.. In image classification: if the input is a cat,


Introduction to Spherical Tensors
=================================

.. note::

    In this introduction, we aim to minimize complex mathematical details and instead focus on the key properties and rules necessary in building neural networks. While this approach might omit the origins of some concepts and make them seem less intuitive, we apologize for any inconvenience and will strive to clarify these ideas through visualizations and examples.

`Spherical tensors` (or irreps tensors/features), is a basic data structure of a widely used family of SO(3)-equivariant (3D rotational-equivariant) neural networks. 

Shapes and components
---------------------

A spherical tensor :math:`\mathbf{x}` is a tensor that can be indexed with "degree" :math:`l` and "order" :math:`m`, with :math:`l \ge 0` and :math:`m=-l,\dots,0,\dots,l`. The components of the tensor can be denoted as :math:`\mathbf{x}_{m}^{(l)}`, and we denote its degree-:math:`l` component as :math:`\mathbf{x}^{(l)}\in\mathbb{R}^{2l+1}`.

For example, a spherical tensor :math:`\mathbf{x}` with degrees ranging from :math:`0` to :math:`2` can be represented as

.. figure:: ./imgs/1_spherical.png
    :align: center
    :scale: 40%

or write as 

.. math::
    \begin{aligned}
    &\mathbf{x}= \begin{bmatrix}
    \mathbf{x}_{0}^{(0)} & \mathbf{x}_{-1}^{(1)} & \mathbf{x}_{0}^{(1)} & \mathbf{x}_{1}^{(1)} & \mathbf{x}_{-2}^{(2)} & \mathbf{x}_{-1}^{(2)} & \mathbf{x}_{0}^{(2)} & \mathbf{x}_{1}^{(2)} & \mathbf{x}_{2}^{(2)}
    \end{bmatrix}^\top\\
    \end{aligned}

in which the three single-degree components are

.. math::
    \begin{aligned}
    &\mathbf{x}^{(0)}=\begin{bmatrix}
    \mathbf{x}_{0}^{(0)}
    \end{bmatrix}^\top\\
    &\mathbf{x}^{(1)}=\begin{bmatrix}
    \mathbf{x}_{-1}^{(1)} & \mathbf{x}_{0}^{(1)} & \mathbf{x}_{1}^{(1)}
    \end{bmatrix}^\top\\
    &\mathbf{x}^{(2)}=\begin{bmatrix}
    \mathbf{x}_{-2}^{(2)} & \mathbf{x}_{-1}^{(2)} & \mathbf{x}_{0}^{(2)} & \mathbf{x}_{1}^{(2)} & \mathbf{x}_{2}^{(2)}
    \end{bmatrix}^\top\\
    \end{aligned}

The total number of orders of :math:`\mathbf{x}` above is :math:`9`.

In Equitorch, the spherical tensors are simply represented using :obj:`torch.Tensor` and constrained to have continuous degree range, that is, :math:`l_{\min}\le l\le l_{\max}`. We use the type :obj:`~equitorch.typing.DegreeRange` to represent the range of degrees of a spherical tensor. A :obj:`~equitorch.typing.DegreeRange` can be a tuple of two integers, denoting the minimum and maximum degree :math:`(l_{\min},l_{\max})`, or a single integer, denoting the maximum degree  :math:`l_{\max}` and the minimum degree is treated as :math:`0` by default. (You can use :obj:`~equitorch.utils.check_degree_range` to cast both cases to tuples.) In this documentation, we often use :math:`L` to represent a degree range and :math:`l\in L` for the degrees in the range.

When considering batched, multi-channel, spherical tensors, we follow the convention that it should be in shape :math:`(N, \text{num_orders}, C)`, where :math:`N` is the batch-size, :math:`C` is the channel numbers and :math:`\text{num_orders}` is the total number of orders in the spherical tensor.

.. figure:: ./imgs/1_multichannels.png
    :align: center
    :scale: 40%

    A batched multi-channel spherical tensor with shape :math:`(4,9,4)`.


For example, we can simply create a random spherical tensor by

.. code-block:: python

    >>> N, C = 5, 3
    >>> x = torch.randn(N, num_orders_in(3), C) # degree ranges form 0 to 3
    >>> print(x.shape)
    torch.Size([5, 16, 3])
    >>> y = torch.randn(N, num_orders_in((2,5)), C) # degree ranges form 2 to 5
    >>> print(y.shape) # could you interprete the results?
    torch.Size([5, 32, 3])


In :obj:`equitorch.utils`, we provide many related functions to manipulate and convert the spherical indicies of 0-based sequential indicies, degrees and orders. (Strating from :obj:`~equitorch.utils.check_degree_range` in the documentation.)  

Transformation
--------------

Besides being of the proper shape, what makes a tensor "spherical" is that it should work compatible with equivariant transformations. To be specific, this means for a equivariant transformation :math:`f: f(\mathbf{x})=\mathbf{y}` that takes a spherical tensor :math:`\mathbf{x}` and produces a spherical tensor :math:`\mathbf{y}`, we should have

.. math::

    f(\mathbf{D}_{\text{in}}(\mathbf{R})\mathbf{x})=\mathbf{D}_{\text{out}}(\mathbf{R})\mathbf{y},

for any rotation :math:`\mathbf{R}\in\text{SO(3)}`. Reversely, we can call the functions with such a property to be "equivariant". Here, :math:`\mathbf{D}_{\text{in}}(\mathbf{R})` and :math:`\mathbf{D}_{\text{out}}(\mathbf{R})` are the so-called `Wigner D matrices <https://en.wikipedia.org/wiki/Wigner_D-matrix>`_, with the degree-range of the input and output space, respectively. Given a degree-range :math:`L`, the corresponding Wigner D matrix is block diagonal :math:`\mathbf{D}(\mathbf{R})=\bigoplus_{l\in L}\mathbf{D}^{(l)}(\mathbf{R})` with each blocks :math:`\mathbf{D}^{(l)}(\mathbf{R})\in\mathbb{R}^{(2l+1)\times(2l+1)}` being othrogonal.

The equivariance property can also be represented via the following diagrams:


.. figure:: ./imgs/1_equiv.png
    :align: center
    :scale: 60%

    (In our diagram, the :math:`\mathbf{D}`'s before and after :math:`f` actually represent :math:`\mathbf{D}_{\text{in}}` and :math:`\mathbf{D}_{\text{out}}`, respectively, although this is not explicitly shown.)

The most important property of Wigner D matrices is that they will multiply as rotations, that is, :math:`\mathbf{D}(\mathbf{R}_1)\mathbf{D}(\mathbf{R}_2)=\mathbf{D}(\mathbf{R}_1\mathbf{R}_2)`, thus, the Wigner D :math:`\mathbf{D}(\mathbf{R})` can be considered as a "**representation**" of the 3D rotation :math:`\mathbf{R}` on the spaces of spherical tensors.

Specifically, when we take the output of the transformation to be degree-0, or :math:`\mathbb{R}`, the Wigner D matrices on the output space will be constant to :math:`[\ 1\ ]`, and the property of equivariant will write as

.. math::

    f(\mathbf{D}_{\text{in}}(\mathbf{R})\mathbf{x})=y.

In such cases, we will call the function "invariant". The degree-0 component of a larger spherical tensor is also called the invariant part.

By our diagram, the invariance condition can be represented as 

.. figure:: ./imgs/1_inv.png
    :align: center
    :scale: 60%
    
In Equitorch, the Wigner D matrices can be obtained using :obj:`~equitorch.math.wigner_D`, and it can be applied to batched, multi-channel spherical tensors by :obj:`~equitorch.utils.rot_on`.

We may notice from the block-diagonality of Wigner D's that the spherical tensors are actually transformed degree-wise:

.. figure:: ./imgs/1_wigner.png
    :align: center
    :scale: 50%

    The block diagonal action of the Wigner D.

In the following sections we will see that, when talking about equivariant operations and spherical tensors, **the components within a same degree are often treated as a whole**.

   .. This means to keep the equivariances, we may not be able to transform the elements arbitrarily. Some supported operations for the degree-:math:`l` component :math:`\mathbf{x}^{(l)}` include:
..
   .. - adding or subtracting with another spherical tensor component with the same degree,
   .. - scalar multiplying,
   .. - linear transform on other dimensions (like data and channel dimensions),
   .. - dot product with another spherical tensor component with the same degree,
   .. - tensor product with another spherical tensor component (see :ref:`tp`),
   .. - calculate the norm (which is the dot product with itself),
..
   .. while the unsupported operations include:
..
   .. - arbitrarily unary functions applied to each components other than scalar multiplying,
   .. - direct summation or product,
   .. - dot product with a non-zero constant tensor.
..
   .. Since we directly use :obj:`torch.Tensor` to represent the spherical tensors, thus it is not forbiddened to perform the unsupported operations. In such cases, the result will be of no clear meanings.

Spherical Harmonics 
-------------------

.. is a set of functions :math:`Y_m^{(l)}(\hat{\mathbf{r}})` defined on the unit sphere :math:`S^2`. By combining the scalar functions, we can get the spherical harmonics tensor :math:`\mathbf{Y}(\hat{\mathbf{r}})`.


`Spherical harmoics <https://en.wikipedia.org/wiki/Spherical_harmonics>`_, :math:`\mathbf{Y}(\hat{\mathbf{r}})`, is a special spherical tensor determined by a vector :math:`\hat{\mathbf{r}}` on the unit sphere :math:`S^2`. When rotating the vector :math:`\hat{\mathbf{r}}` to :math:`\mathbf{R}\hat{\mathbf{r}}`, the corresponding spherical harmonics will be rotated to 

.. math::
    \mathbf{Y}(\mathbf{R}\hat{\mathbf{r}})=\mathbf{D}(\mathbf{R})\mathbf{Y}(\hat{\mathbf{r}}),

which is also a case of equivariances.

.. figure:: ./imgs/1_eqY.png
    :align: center
    :scale: 60%

    The equivariance of spherical harmonics

The spherical harmonics are usually used to encode a direction vector :math:`\hat{\mathbf{r}}\in S^2` to the space of spherical tensors. Especially, the three components of degree-:math:`1` spherical harmonics is just :math:`\frac{1}{\sqrt{3}}` times the input vector :math:`\hat{\mathbf{r}}` in :math:`(y,z,x)` order.

In Equitorch, we can get the spherical harmonics by :obj:`~equitorch.math.spherical_harmonics`.

For example:

.. code-block:: python

    >>> N, L = 5, 2
    >>> r = rand_spherical_xyz(N) # Generate N random points on the unit sphere
    >>> print(r.shape)
    torch.Size([5, 3])
    >>> Y = spherical_harmonics(r, L) # Calculate the spherical harmonics
    >>> print(Y.shape)
    torch.Size([5, 9])

    >>> rotation_angles = rand_rotation_angles(N) # Generate N random rotations in Euler's angle
    >>> R = angles_to_matrix(*rotation_angles) # Get the matrix representation of the rotation
    >>> print(R.shape)
    torch.Size([5, 3, 3])
    >>> D = wigner_D(L, *rotation_angles) # Get the corresponding Wigner D
    >>> print(D.shape)
    torch.Size([5, 9, 9])
    
    >>> Y_rotated = spherical_harmonics(rot_on(R, r), L) # Calculate the spherical harmonics of rotated input
    
    >>> print(torch.allclose(Y_rotated, rot_on(D, Y), atol=1e-4)) # Check the equivariance
    True
    >>> print((Y_rotated-rot_on(D, Y)).abs().max())
    tensor(2.1011e-06)

.. .. note::

.. When checking the explicit form of spherical haromics and Wigner D matrices on Wikipedia or other materials, you may see these functions expressed in complex form containing factors like :math:`\mathrm{e}^{im\alpha}`, :math:`\mathrm{e}^{im\gamma}` or :math:`\mathrm{e}^{im\phi}`. To translate 