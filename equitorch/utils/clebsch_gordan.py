from typing import Optional, Union, Tuple, Callable
from torch import Tensor
from torch_geometric.utils import scatter
import e3nn
import torch

from .indices import check_degree_range, degree_order_to_index, degrees_in_range

from ..typing import DegreeRange
from ..math.so3 import _so3_clebsch_gordan

def dense_CG(L: DegreeRange, L1: DegreeRange, L2: DegreeRange, condition:Optional[Callable]=None, dtype=None, device=None):
    r"""Generate dense Clebsch-Gordan (CG) matrices for given angular momentum ranges.

    This function computes the Clebsch-Gordan coefficients for coupling two angular momenta
    :math:`l_1` and :math:`l_2` to a total angular momentum :math:`l`. The coefficients are
    organized into a dense matrix format.

    Parameters
    ----------
    L : :obj:`~equitorch.typing.DegreeRange`
        Range of total angular momentum :math:`l`.
    L1 : :obj:`~equitorch.typing.DegreeRange`
        Range of first angular momentum :math:`l_1`.
    L2 : :obj:`~equitorch.typing.DegreeRange`
        Range of second angular momentum :math:`l_2`.
    condition : callable, optional
        A function that takes :math:`(l, l_1, l_2)` as arguments and returns a boolean or float.
        If provided, it is used to filter the CG coefficients.
    dtype : torch.dtype, optional
        The desired data type of the output tensor. Default is None for :obj:`torch.get_default_dtype`.
    device : torch.device, optional
        The desired device of the output tensor. Default is None.

    Returns
    -------
    :obj:`~torch.Tensor`
        A dense tensor of Clebsch-Gordan coefficients. The shape of the tensor is
        :math:`(l_{max}^2 - l_{min}^2,l_{1,max}^2 - l_{1,min}^2, l_{2,max}^2 - l_{2,min}^2)`.
    """
    l_min, l_max = check_degree_range(L)
    l1_min, l1_max = check_degree_range(L1)
    l2_min, l2_max = check_degree_range(L2)
    if condition is not None:
        CGs = [
            [
                [
                    _so3_clebsch_gordan(l,l1,l2)
                        .nan_to_num(0) * condition(l,l1,l2)
                for l2 in range(l2_min, l2_max+1)]
            for l1 in range(l1_min,l1_max+1)]
        for l in range(l_min,l_max+1)]
    else:
        CGs = [
            [
                [
                    _so3_clebsch_gordan(l,l1,l2)
                        .nan_to_num(0)
                for l2 in range(l2_min, l2_max+1)]
            for l1 in range(l1_min,l1_max+1)]
        for l in range(l_min,l_max+1)]
    CG = torch.cat(
        [torch.cat(
            [torch.cat(
                CGs_LL1, dim=-1) 
            for CGs_LL1 in CGs_L], dim=-2
        ) for CGs_L in CGs], dim=-3)
    if dtype is None:
        dtype = torch.get_default_dtype()
    return CG.to(device=device, dtype=dtype)

def blocked_CG(L: DegreeRange, L1: DegreeRange, L2:DegreeRange, condition:Optional[Callable]=None, device=None, dtype=None):
    r"""Generate blocked Clebsch-Gordan (CG) coefficients for given angular momentum ranges.

    This function computes the Clebsch-Gordan coefficients for coupling two angular momenta
    :math:`l_1` and :math:`l_2` to a total angular momentum :math:`l`. The coefficients are
    organized into a dictionary of tensors, where each tensor corresponds to a specific
    :math:`(l, l_1, l_2)` combination.

    Parameters
    ----------
    L : :obj:`~equitorch.typing.DegreeRange`
        Range of total angular momentum :math:`l`.
    L1 : :obj:`~equitorch.typing.DegreeRange`
        Range of first angular momentum :math:`l_1`.
    L2 : :obj:`~equitorch.typing.DegreeRange`
        Range of second angular momentum :math:`l_2`.
    condition : callable, optional
        A function that takes :math:`(l, l_1, l_2)` as arguments and returns a boolean.
        If provided, it is used to filter the CG coefficients.
    dtype : torch.dtype, optional
        The desired data type of the output tensors. Default is torch.float.

    Returns
    -------
    :obj:`Dict[Tuple[int, int, int], Tensor]`
        A dictionary where each key is a tuple :math:`(l, l_1, l_2)`, and each value
        is a tensor of Clebsch-Gordan coefficients for that combination.
    """
    L = check_degree_range(L)
    L1 = check_degree_range(L1)
    L2 = check_degree_range(L2)
    if condition is None:
        return {
            (l, l1, l2):_so3_clebsch_gordan(l,l1,l2).to(device=device, dtype=dtype)
                for l in degrees_in_range(L)
                for l1 in degrees_in_range(L1)
                for l2 in degrees_in_range(L2)
                if l >= abs(l1-l2) and l <= l1+l2
        }
    else:
        return {
            (l, l1, l2):_so3_clebsch_gordan(l,l1,l2).to(device=device, dtype=dtype)
                for l in degrees_in_range(L)
                for l1 in degrees_in_range(L1)
                for l2 in degrees_in_range(L2)
                if l >= abs(l1-l2) and l <= l1+l2 and condition(l,l1,l2)
        }
def coo_CG(L:DegreeRange, L1:DegreeRange, L2:DegreeRange, 
           condition:Optional[Callable]=None, dtype=None, device=None):
    r"""Generate sparse Clebsch-Gordan (CG) coefficients in coordinate format.

    This function computes the non-zero Clebsch-Gordan coefficients for coupling two angular momenta
    :math:`l_1` and :math:`l_2` to a total angular momentum :math:`l`, and returns them in a sparse format.
    The results are sorted in the order of (l, M, l1, l2, M1, M2).

    Note
    ----
    The output is strictly sorted in the order of (l, M, l1, l2, M1, M2) lexigraphically.

    Parameters
    ----------
    L : :obj:`~equitorch.typing.DegreeRange`
        Range of total angular momentum :math:`l`.
    L1 : :obj:`~equitorch.typing.DegreeRange`
        Range of first angular momentum :math:`l_1`.
    L2 : :obj:`~equitorch.typing.DegreeRange`
        Range of second angular momentum :math:`l_2`.
    condition : callable, optional
        A function that takes :math:`(l, l_1, l_2)` as arguments and returns a boolean.
        If provided, it is used to filter the CG coefficients.
    dtype : torch.dtype, optional
        The desired data type of the output tensors. Default is torch.float.
    device : torch.device, optional
        The desired device of the output tensors. Default is :obj:`None` (uses current device).


    Returns
    -------
    Cs : :obj:`~torch.Tensor` of shape (N,)
        Non-zero CG coefficient values.
    Ms : :obj:`~torch.Tensor` of shape (3, N)
        Indices (M, M1, M2) of non-zero CG values.
    ls_cg : :obj:`~torch.Tensor` of shape (3, N)
        Corresponding degrees (l, l1, l2) for each non-zero CG value.
    l_ind : :obj:`~torch.Tensor` of shape (N,)
        Indices of (l, l1, l2) combinations.

        Where N is the number of non-zero CG coefficients.

    """
    L = check_degree_range(L)
    L1 = check_degree_range(L1)
    L2 = check_degree_range(L2)
    CG_blocks = blocked_CG(L,L1,L2, condition=condition, dtype=torch.float)
    ret = sorted([
        ( l, degree_order_to_index(l,m,L[0]), l1, l2, 
        degree_order_to_index(l1,m1,L1[0]), 
        degree_order_to_index(l2,m2,L2[0]),
        l_ind,
        CG[m+l,m1+l1,m2+l2].item()) 
            for l_ind, ((l, l1, l2), CG) in enumerate(CG_blocks.items()) 
            for m in range(-l,l+1) 
            for m1 in range(-l1,l1+1) 
            for m2 in range(-l2,l2+1) 
            if CG_blocks[l,l1,l2][m+l,m1+l1,m2+l2] != 0
    ])
    # [ l, M, l1, l2, M1, M2, l_ind, CGs[l,l1,l2][m,m1,m2] ]
    ls_cg = torch.tensor([[t[0], t[2], t[3]] for t in ret]).T
    l_ind = torch.tensor([t[6] for t in ret])
    Ms = torch.tensor([[t[1], t[4], t[5]] for t in ret]).T
    Cs = torch.tensor([t[7] for t in ret])
    return Cs.to(device=device, dtype=dtype), Ms.to(device=device), ls_cg.to(device=device), l_ind.to(device=device)

