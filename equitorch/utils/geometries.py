import e3nn.math
import torch
from torch import Tensor
from typing import Optional
from torch_geometric.utils import segment
import e3nn
import e3nn.o3 as o3

from ..math.so3 import wigner_D, xyz_to_angles, angles_to_matrix


from .indices import check_degree_range, degrees_in_range

from ..typing import DegreeRange

def get_center_of_mass(pos: Tensor, ptr: Tensor, atom_mass: Optional[Tensor] = None):
    r"""Calculate the center of mass for each molecule in a batch.

    Parameters
    ----------
    pos : :obj:`~torch.Tensor`
        Tensor of shape :math:`(N, 3)` containing the positions of all atoms.
    ptr : :obj:`~torch.Tensor`
        Tensor containing the cumulative sum of atoms for each molecule in the batch.
    atom_mass : :obj:`~torch.Tensor`, optional
        Tensor of shape :math:`(N,)` containing the mass of each atom. 
        If None, all atoms are assumed to have equal mass.

    Returns
    -------
    :obj:`~torch.Tensor`
        Tensor of shape :math:`(B, 3)` containing the center of mass for each molecule in the batch,
        where :math:`B` is the number of molecules.
    """

    if atom_mass:
        return segment(pos * atom_mass, ptr) / segment(atom_mass, ptr)
    else:
        return segment(pos, ptr, reduce='mean') 
    
def zero_center_of_mass(pos: Tensor, batch: Tensor, center_of_mass: Tensor):
    r"""Shift the positions of atoms in each molecule so that their center of mass is at the origin.

    Parameters
    ----------
    pos : :obj:`~torch.Tensor`
        Tensor of shape :math:`(N, 3)` containing the positions of all atoms.
    batch : :obj:`~torch.Tensor`
        Tensor of shape :math:`(N,)` indicating which molecule each atom belongs to.
    center_of_mass : :obj:`~torch.Tensor`
        Tensor of shape :math:`(B, 3)` containing the center of mass for each molecule,
        where B is the number of molecules.

    Returns
    -------
    :obj:`~torch.Tensor`
        Tensor of shape :math:`(N, 3)` containing the shifted positions of all atoms.

    """
    return pos - center_of_mass.index_select(dim=0, index=batch)

def rand_spherical_angles(*shape, dtype=None, device=None):
    r"""Returns the uniformly sampled points on the unit sphere in spherical coordinates.  

    Parameters
    ----------
    *shape: int

    Returns
    -------
    theta : :obj:`Tensor` 
        Polar angle, tensor of shape :math:`(\text{shape})`
    phi : :obj:`Tensor`
        Azimuthal angle, tensor of shape :math:`(\text{shape})`
    """
    phi = 2 * torch.pi * torch.rand(*shape, dtype=dtype, device=device)
    theta = torch.rand(shape, dtype=dtype, device=device).mul(2).sub(1).acos()
    return theta, phi

def rand_spherical_xyz(*shape, dtype=None, device=None):
    r"""Returns the uniformly sampled points on the unit sphere in euclidean coordinates.  

    Parameters
    ----------
    *shape: int

    Returns
    -------
    :obj:`~torch.Tensor`
        The euclidean coordinates of sampled points, tensor of shape :math:`(\text{shape}, 3)`
    """
    xyz = torch.randn(*shape, 3, dtype=dtype, device=device)
    return xyz / xyz.norm(dim=-1, keepdim=True)

def rand_rotation_angles(*shape, dtype=None, device=None):
    r"""Returns the uniformly sampled rotations in Z-Y-Z Euler angles.  

    Parameters
    ----------
    *shape: int

    Returns
    -------
    alpha : :obj:`Tensor` 
        Third rotation angle around Z-axis, tensor of shape :math:`(\text{shape})`
    beta : :obj:`Tensor` 
        Second rotation angle around Y-axis, tensor of shape :math:`(\text{shape})`
    gamma : :obj:`Tensor` 
        First rotation angle around Z-axis, tensor of shape :math:`(\text{shape})`
    """
    return o3.rand_angles(*shape, dtype=dtype, device=device)

def rand_rotation_matrix(*shape, dtype=None, device=None):
    r"""Returns the uniformly sampled rotations in rotation matrices.

    Parameters
    ----------
    *shape: int

    Returns
    -------
    :obj:`~torch.Tensor`
        The rotation matrices of shape :math:`(\text{shape}, 3, 3)`
    """
    return angles_to_matrix(*rand_rotation_angles(shape, dtype=dtype, device=device))

def align_to_z_mat(r: Tensor, need_normalize:bool=True):
    r"""Calculate the rotation matrix :math:`\mathbf{R}` that 
    aligns the vector :math:`\mathbf{r}` with the z-axis.

    .. math::
        \mathbf{R}\mathbf{r}=\|\mathbf{r}\|\hat{\mathbf{z}}

    Parameters
    ----------
    r : :obj:`~torch.Tensor`
        Input vector to be aligned.
    need_normalize : bool, optional
        Whether to normalize the input vector. Default is :obj:`True`.

    Returns
    -------
    :obj:`~torch.Tensor`
        The rotation matrix.
    """

    theta, phi = xyz_to_angles(r, need_normalize)
    return o3.matrix_y(-theta) @ o3.matrix_z(-phi)

def edge_align_to_z_mat(pos: Tensor, edge_index: Tensor):
    r"""Calculate the rotation matrix :math:`\mathbf{R}`
    that aligns the edge vector with the z-axis.
    
    .. math::
        \mathbf{R}(\mathbf{x}_i-\mathbf{x}_j)=\|\mathbf{x}_i-\mathbf{x}_j\|\hat{\mathbf{z}}

    Parameters
    ----------
    pos : :obj:`~torch.Tensor`
        Tensor of node positions.
    edge_index : :obj:`~torch.Tensor`
        Tensor of edge indices.

    Returns
    -------
    :obj:`~torch.Tensor`
        The rotation matrix.
    """
    to_pos = pos.index_select(dim=-1, index=edge_index[1])
    from_pos = pos.index_select(dim=-1, index=edge_index[0])
    return align_to_z_mat(from_pos - to_pos, need_normalize=True)

def edge_spherical_angles(pos: Tensor, edge_index: Tensor):
    r"""Calculate the spherical angles :math:`(\phi, \theta)` 
     of the edge vectors.

    Parameters
    ----------
    pos : :obj:`~torch.Tensor`
        Tensor of node positions.
    edge_index : :obj:`~torch.Tensor`
        Tensor of edge indices.

    Returns
    -------
        A tuple :math:`(\phi, \theta)` where:

        - :math:`\hat{\mathbf{r}}` can be generated by rotating :math:`\hat{\mathbf{z}}` around y by :math:`\theta`, and then rotating around z by :math:`\phi`.

        - :math:`\hat{\mathbf{r}}` can be aligned to z by rotating around z by :math:`-\phi`, and then around y by :math:`-\theta`.
    """

    to_pos = pos.index_select(dim=-1, index=edge_index[1])
    from_pos = pos.index_select(dim=-1, index=edge_index[0])
    return xyz_to_angles(from_pos - to_pos, need_normalize=True)

def align_to_z_wigner(r: Tensor, L: DegreeRange, need_normalize:bool=True):
    r"""Returns the Wigner D matrix corresponding to the return of :obj:`align_to_z_mat`.

    Parameters
    ----------
    r : :obj:`~torch.Tensor`
        Input vector to be aligned.
    L : :obj:`~equitorch.typing.DegreeRange`
        The degree range for the Wigner D matrix.
    need_normalize : bool, optional
        Whether to normalize the input vector. Default is :obj:`True`.

    Returns
    -------
    :obj:`~torch.Tensor`
        Wigner D matrix corresponding to the rotation returned by :obj:`align_to_z_mat`.
    """

    L = check_degree_range(L)
    theta, phi = xyz_to_angles(r, need_normalize)
    return wigner_D(L, torch.zeros_like(phi), -theta, -phi).to(r.device)

def edge_align_to_z_wigner(pos: Tensor, edge_index: Tensor, L: DegreeRange):
    r"""Calculate the Wigner D matrix corresponding to the rotation that aligns 
    an edge vector with the z-axis as :obj:`edge_align_to_z_mat`.

    Parameters
    ----------
    pos : :obj:`~torch.Tensor`
        Tensor of node positions.
    edge_index : :obj:`~torch.Tensor`
        Tensor of edge indices.
    L : :obj:`~equitorch.typing.DegreeRange`
        The degree range of the Wigner D matrix.

    Returns
    -------
    :obj:`~torch.Tensor`
        Wigner D matrix corresponding to the rotation 
        that aligns the edge vector with the z-axis 
        as :obj:`edge_align_to_z_mat`.
    """
    L = check_degree_range(L)
    theta, phi = edge_spherical_angles(pos, edge_index)
    return wigner_D(L, torch.zeros_like(phi), -theta, -phi).to(pos.device)


def rot_on(mat: Tensor, x: Tensor):
    r"""Performs rotation of matrix :obj:`mat` on tensor :obj:`x`.

    This function applies the rotation defined by :obj:`mat` to the tensor :obj:`x`. The behavior
    depends on the dimensions of both :obj:`mat` and :obj:`x`.

    Parameters
    ----------
    mat : :obj:`~torch.Tensor`
        The rotation matrix or matrices. Can be 2D, 3D, or 4D.
        - If 2D: A single rotation matrix applied to all samples in x.
        - If 3D: Batched rotation matrices applied sample-wise to x.
        - If 4D: Batched rotation matrices applied both sample-wise and channel-wise to x.
    x : :obj:`~torch.Tensor`
        The tensor to be rotated. Can be 2D or 3D.
        - If 2D: Interpreted as a batch of vectors.
        - If 3D: Interpreted as a batch of matrices or multi-channel vectors.

    Returns
    -------
    :obj:`~torch.Tensor`
        The rotated tensor. The shape will match the input 'x'.

    """
    d = x.ndim
    if d == 2:
        x = x.unsqueeze(-1)
    if mat.ndim == 3:
        ret = mat @ x
    elif mat.ndim == 2:
        ret = mat.unsqueeze(0) @ x
    elif mat.ndim == 4:
        ret = torch.einsum('NPQC, NQC -> NPC', mat, x)
    else:
        assert "The dimension of mat should only be 2, 3 or 4."
        ret = x
    if d==2:
        ret = ret.squeeze(-1)

    return ret