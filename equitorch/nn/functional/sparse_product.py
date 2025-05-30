from typing import Optional
import torch
from torch import Tensor
from torch.autograd import Function

from ...ops.batched_sparse_dense_op import (
    indexed_mul_scale_gather,
    indexed_outer_scale_gather,
    indexed_inner_scale_gather,
    indexed_vecmat_scale_gather,
    indexed_vecsca_scale_gather,
    indexed_mat_t_vec_scale_gather,
    indexed_scavec_scale_gather,
)

from ...structs import SparseProductInfo

from equitorch.irreps import check_irreps, Irreps

class SparseMul(Function):

    @staticmethod
    def forward(ctx, input1: Tensor, input2: Tensor, info_fwd: SparseProductInfo, 
                info_bwd1: Optional[SparseProductInfo] = None, info_bwd2: Optional[SparseProductInfo] = None, out_accumulated: bool = False) -> Tensor:
        ret = indexed_mul_scale_gather(
            input1, input2,
            info_fwd.scale,
            info_fwd.index1,
            info_fwd.index2,
            info_fwd.seg_out,
            info_fwd.gather_index,
            info_fwd.index_out,
            out_accumulated=out_accumulated,
            out_size=info_fwd.out_size,
            )
        ctx.save_for_backward(input1 if input2.requires_grad else None, input2 if input1.requires_grad else None)
        ctx.infos = (info_fwd, info_bwd1, info_bwd2)
        # Determine shared status based on input dimensions heuristic
        ctx.shared1 = input1.ndim < 3
        ctx.shared2 = input2.ndim < 3
        ctx.out_accumulated = out_accumulated # Save for backward logic
        return ret

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None, None, None, None, None
            
        grad = grad_output
        input1, input2 = ctx.saved_tensors
        info_fwd, info_bwd1, info_bwd2 = ctx.infos
        left_shared_fwd = ctx.shared1
        right_shared_fwd = ctx.shared2

        grad1, grad2 = None, None
        if ctx.needs_input_grad[0]:
            # gx = op_bx(y, gz) -> out_accumulated_bx = left_shared_fwd
            out_accumulated_bwd1 = left_shared_fwd
            grad1 = SparseMul.apply(input2, grad, info_bwd1, info_bwd2, info_fwd, out_accumulated_bwd1)

        if ctx.needs_input_grad[1]:
            # gy = op_by(gz, x) -> out_accumulated_by = right_shared_fwd
            out_accumulated_bwd2 = right_shared_fwd
            grad2 = SparseMul.apply(grad, input1, info_bwd2, info_fwd, info_bwd1, out_accumulated_bwd2)
        else:
            grad2 = None

        # Return grads corresponding to input1, input2, out_accumulated, info_fwd, info_bwd1, info_bwd2
        return grad1, grad2, None, None, None, None

def sparse_mul(input1: Tensor, input2: Tensor, info_fwd: SparseProductInfo, 
                info_bwd1: Optional[SparseProductInfo] = None, info_bwd2: Optional[SparseProductInfo] = None, out_accumulated: bool = False) -> Tensor:
    r"""
    Computes sparse element-wise multiplication using indexed operations.

    This function performs an element-wise product of two input tensors, ``input1`` and ``input2``,
    based on the indexing and scaling information provided in ``info_fwd``.
    The operation is "sparse" in the sense that it uses predefined indexing schemes
    to select elements for multiplication, rather than a dense matrix multiplication.

    The backward pass information can be optionally provided via ``info_bwd1`` and ``info_bwd2``
    for custom gradient calculations if needed.

    Args:
        input1 (torch.Tensor): The first input tensor.
        input2 (torch.Tensor): The second input tensor.
        info_fwd (SparseProductInfo): Contains scaling factors, indices for ``input1`` and ``input2``,
            output segmentation, gather indices, and output indices for the forward pass.
        info_bwd1 (SparseProductInfo, optional): Information for the backward pass with respect to ``input1``.
            Defaults to None.
        info_bwd2 (SparseProductInfo, optional): Information for the backward pass with respect to ``input2``.
            Defaults to None.
        out_accumulated (bool, optional): If ``True``, the output is accumulated into an existing tensor
            (not fully supported by all underlying ops, behavior might vary). Defaults to ``False``.

    Returns:
        torch.Tensor: The result of the sparse element-wise multiplication.
    """
    return SparseMul.apply(input1, input2, info_fwd, info_bwd1, info_bwd2, out_accumulated)

class SparseOuter(Function):

    @staticmethod
    def forward(ctx, input1: Tensor, input2: Tensor, info_fwd: SparseProductInfo, 
                info_bwd1: Optional[SparseProductInfo] = None, info_bwd2: Optional[SparseProductInfo] = None, out_accumulated: bool = False) -> Tensor:
        ret = indexed_outer_scale_gather(
            input1, input2,
            info_fwd.scale,
            info_fwd.index1,
            info_fwd.index2,
            info_fwd.seg_out,
            info_fwd.gather_index,
            info_fwd.index_out,
            out_accumulated=out_accumulated,
            out_size=info_fwd.out_size,
            )
        ctx.save_for_backward(input1 if input2.requires_grad else None, input2 if input1.requires_grad else None)
        ctx.infos = (info_fwd, info_bwd1, info_bwd2)
        ctx.shared1 = input1.ndim < 3 # is vec shared?
        ctx.shared2 = input2.ndim < 3 # is vec shared?
        ctx.out_accumulated = out_accumulated
        return ret

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        grad = grad_output
        input1, input2 = ctx.saved_tensors
        info_fwd, info_bwd1, info_bwd2 = ctx.infos
        left_shared_fwd = ctx.shared1
        right_shared_fwd = ctx.shared2

        grad1, grad2 = None, None
        if ctx.needs_input_grad[0]:
            # gx = vecmat(y, gz^T) -> out_accumulated_bx = left_shared_fwd
            out_accumulated_bwd1 = left_shared_fwd
            grad1 = SparseVecMat.apply(input2, grad.transpose(-1,-2).contiguous(), info_bwd1, info_bwd2, info_fwd, out_accumulated_bwd1)

        if ctx.needs_input_grad[1]:
            # gy = mat_t_vec(gz, x) -> out_accumulated_by = right_shared_fwd
            out_accumulated_bwd2 = right_shared_fwd
            grad2 = SparseMatTVec.apply(grad, input1, info_bwd2, info_fwd, info_bwd1, out_accumulated_bwd2)
        else:
            grad2 = None

        return grad1, grad2, None, None, None, None
def sparse_outer(input1: Tensor, input2: Tensor, info_fwd: SparseProductInfo, 
                info_bwd1: Optional[SparseProductInfo] = None, info_bwd2: Optional[SparseProductInfo] = None, out_accumulated: bool = False) -> Tensor:
    r"""
    Computes sparse outer product using indexed operations.

    This function performs an outer product of ``input1`` and ``input2`` based on the indexing
    and scaling information in ``info_fwd``. The result is typically a higher-rank tensor.

    Args:
        input1 (torch.Tensor): The first input tensor (often a vector).
        input2 (torch.Tensor): The second input tensor (often a vector).
        info_fwd (SparseProductInfo): Information for the forward pass.
        info_bwd1 (SparseProductInfo, optional): Information for the backward pass w.r.t. ``input1``.
        info_bwd2 (SparseProductInfo, optional): Information for the backward pass w.r.t. ``input2``.
        out_accumulated (bool, optional): If ``True``, accumulate to output. Defaults to ``False``.

    Returns:
        torch.Tensor: The result of the sparse outer product.
    """
    return SparseOuter.apply(input1, input2, info_fwd, info_bwd1, info_bwd2, out_accumulated)

class SparseInner(Function):

    @staticmethod
    def forward(ctx, input1: Tensor, input2: Tensor, info_fwd: SparseProductInfo, 
                info_bwd1: Optional[SparseProductInfo] = None, info_bwd2: Optional[SparseProductInfo] = None, out_accumulated: bool = False) -> Tensor:
        ret = indexed_inner_scale_gather(
            input1, input2,
            info_fwd.scale,
            info_fwd.index1,
            info_fwd.index2,
            info_fwd.seg_out,
            info_fwd.gather_index,
            info_fwd.index_out,
            out_accumulated=out_accumulated,
            out_size=info_fwd.out_size,
            )
        ctx.save_for_backward(input1 if input2.requires_grad else None, input2 if input1.requires_grad else None)
        ctx.infos = (info_fwd, info_bwd1, info_bwd2)
        ctx.shared1 = input1.ndim < 3 # vec
        ctx.shared2 = input2.ndim < 3 # vec
        ctx.out_accumulated = out_accumulated
        return ret

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        grad = grad_output
        input1, input2 = ctx.saved_tensors
        info_fwd, info_bwd1, info_bwd2 = ctx.infos
        left_shared_fwd = ctx.shared1
        right_shared_fwd = ctx.shared2

        grad1, grad2 = None, None
        if ctx.needs_input_grad[0]:
            # gx = vecsca(y, gz) -> out_accumulated_bx = left_shared_fwd
            out_accumulated_bwd1 = left_shared_fwd
            # Ensure grad has correct shape (scalar-like) for vecsca backward
            # Inner product output grad is (N, O) or (O), needs expansion for vecsca which expects (N, O, C) or (O, C)
            grad1 = SparseVecSca.apply(input2, grad, info_bwd1, info_bwd2, info_fwd, out_accumulated_bwd1)

        if ctx.needs_input_grad[1]:
            # gy = scavec(gz, x) -> out_accumulated_by = right_shared_fwd
            out_accumulated_bwd2 = right_shared_fwd
            grad2 = SparseScaVec.apply(grad, input1, info_bwd2, info_fwd, info_bwd1, out_accumulated_bwd2)
        else:
            grad2 = None

        return grad1, grad2, None, None, None, None
def sparse_inner(input1: Tensor, input2: Tensor, info_fwd: SparseProductInfo, 
                info_bwd1: Optional[SparseProductInfo] = None, info_bwd2: Optional[SparseProductInfo] = None, out_accumulated: bool = False) -> Tensor:
    r"""
    Computes sparse inner product using indexed operations.

    This function performs an inner product (dot product) between ``input1`` and ``input2``
    based on the indexing and scaling information in ``info_fwd``. The result is typically a scalar or a lower-rank tensor.

    Args:
        input1 (torch.Tensor): The first input tensor.
        input2 (torch.Tensor): The second input tensor.
        info_fwd (SparseProductInfo): Information for the forward pass.
        info_bwd1 (SparseProductInfo, optional): Information for the backward pass w.r.t. ``input1``.
        info_bwd2 (SparseProductInfo, optional): Information for the backward pass w.r.t. ``input2``.
        out_accumulated (bool, optional): If ``True``, accumulate to output. Defaults to ``False``.

    Returns:
        torch.Tensor: The result of the sparse inner product.
    """
    return SparseInner.apply(input1, input2, info_fwd, info_bwd1, info_bwd2, out_accumulated)

class SparseVecMat(Function):

    @staticmethod
    def forward(ctx, input1: Tensor, input2: Tensor, info_fwd: SparseProductInfo, 
                info_bwd1: Optional[SparseProductInfo] = None, info_bwd2: Optional[SparseProductInfo] = None, out_accumulated: bool = False) -> Tensor:
        ret = indexed_vecmat_scale_gather(
            input1, input2,
            info_fwd.scale,
            info_fwd.index1,
            info_fwd.index2,
            info_fwd.seg_out,
            info_fwd.gather_index,
            info_fwd.index_out,
            out_accumulated=out_accumulated,
            out_size=info_fwd.out_size,
            )
        ctx.save_for_backward(input1 if input2.requires_grad else None, input2 if input1.requires_grad else None)
        ctx.infos = (info_fwd, info_bwd1, info_bwd2)
        ctx.shared1 = input1.ndim < 3 # vec
        ctx.shared2 = input2.ndim < 4 # mat
        ctx.out_accumulated = out_accumulated
        return ret

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        grad = grad_output
        input1, input2 = ctx.saved_tensors
        info_fwd, info_bwd1, info_bwd2 = ctx.infos
        left_shared_fwd = ctx.shared1
        right_shared_fwd = ctx.shared2

        grad1, grad2 = None, None
        if ctx.needs_input_grad[0]:
            # gx = mat_t_vec(y^T, gz) -> out_accumulated_bx = left_shared_fwd
            out_accumulated_bwd1 = left_shared_fwd
            grad1 = SparseMatTVec.apply(input2.transpose(-1,-2).contiguous(), grad, info_bwd1, info_bwd2, info_fwd, out_accumulated_bwd1)

        if ctx.needs_input_grad[1]:
            # gy = outer(gz, x)^T -> out_accumulated_by = right_shared_fwd
            out_accumulated_bwd2 = right_shared_fwd
            grad2 = SparseOuter.apply(grad, input1, info_bwd2, info_fwd, info_bwd1, out_accumulated_bwd2).transpose(-1,-2).contiguous()
        else:
            grad2 = None

        return grad1, grad2, None, None, None, None
def sparse_vecmat(input1: Tensor, input2: Tensor, info_fwd: SparseProductInfo, 
                info_bwd1: Optional[SparseProductInfo] = None, info_bwd2: Optional[SparseProductInfo] = None, out_accumulated: bool = False) -> Tensor:
    r"""
    Computes sparse vector-matrix multiplication using indexed operations.

    Multiplies a vector (``input1``) by a matrix (``input2``) based on ``info_fwd``.

    Args:
        input1 (torch.Tensor): The input vector.
        input2 (torch.Tensor): The input matrix.
        info_fwd (SparseProductInfo): Information for the forward pass.
        info_bwd1 (SparseProductInfo, optional): Information for the backward pass w.r.t. ``input1``.
        info_bwd2 (SparseProductInfo, optional): Information for the backward pass w.r.t. ``input2``.
        out_accumulated (bool, optional): If ``True``, accumulate to output. Defaults to ``False``.

    Returns:
        torch.Tensor: The result of the sparse vector-matrix multiplication.
    """
    return SparseVecMat.apply(input1, input2, info_fwd, info_bwd1, info_bwd2, out_accumulated)

class SparseVecSca(Function):

    @staticmethod
    def forward(ctx, input1: Tensor, input2: Tensor, info_fwd: SparseProductInfo, 
                info_bwd1: Optional[SparseProductInfo] = None, info_bwd2: Optional[SparseProductInfo] = None, out_accumulated: bool = False) -> Tensor:
        ret = indexed_vecsca_scale_gather(
            input1, input2,
            info_fwd.scale,
            info_fwd.index1,
            info_fwd.index2,
            info_fwd.seg_out,
            info_fwd.gather_index,
            info_fwd.index_out,
            out_accumulated=out_accumulated,
            out_size=info_fwd.out_size,
            )
        ctx.save_for_backward(input1 if input2.requires_grad else None, input2 if input1.requires_grad else None)
        ctx.infos = (info_fwd, info_bwd1, info_bwd2)
        ctx.shared1 = input1.ndim < 3 # vec
        ctx.shared2 = input2.ndim < 2 # scalar
        ctx.out_accumulated = out_accumulated
        return ret

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        grad = grad_output
        input1, input2 = ctx.saved_tensors
        info_fwd, info_bwd1, info_bwd2 = ctx.infos
        left_shared_fwd = ctx.shared1
        right_shared_fwd = ctx.shared2

        grad1, grad2 = None, None
        if ctx.needs_input_grad[0]:
            # gx = scavec(y, gz) -> out_accumulated_bx = left_shared_fwd
            out_accumulated_bwd1 = left_shared_fwd
            grad1 = SparseScaVec.apply(input2, grad, info_bwd1, info_bwd2, info_fwd, out_accumulated_bwd1)
        else:
            grad1 = None
        if ctx.needs_input_grad[1]:
            # gy = inner(gz, x) -> out_accumulated_by = right_shared_fwd
            out_accumulated_bwd2 = right_shared_fwd
            grad2 = SparseInner.apply(grad, input1, info_bwd2, info_fwd, info_bwd1, out_accumulated_bwd2)
        else:
            grad2 = None

        # Return grads corresponding to input1, input2, out_accumulated, info_fwd, info_bwd1, info_bwd2
        return grad1, grad2, None, None, None, None
def sparse_vecsca(input1: Tensor, input2: Tensor, info_fwd: SparseProductInfo, 
                info_bwd1: Optional[SparseProductInfo] = None, info_bwd2: Optional[SparseProductInfo] = None, out_accumulated: bool = False) -> Tensor:
    r"""
    Computes sparse vector-scalar multiplication using indexed operations.

    Multiplies a vector (``input1``) by a scalar (``input2``) element-wise, guided by ``info_fwd``.

    Args:
        input1 (torch.Tensor): The input vector.
        input2 (torch.Tensor): The input scalar (or tensor broadcastable to scalar for each operation).
        info_fwd (SparseProductInfo): Information for the forward pass.
        info_bwd1 (SparseProductInfo, optional): Information for the backward pass w.r.t. ``input1``.
        info_bwd2 (SparseProductInfo, optional): Information for the backward pass w.r.t. ``input2``.
        out_accumulated (bool, optional): If ``True``, accumulate to output. Defaults to ``False``.

    Returns:
        torch.Tensor: The result of the sparse vector-scalar multiplication.
    """
    return SparseVecSca.apply(input1, input2, info_fwd, info_bwd1, info_bwd2, out_accumulated)

class SparseScaVec(Function):

    @staticmethod
    def forward(ctx, input1: Tensor, input2: Tensor, info_fwd: SparseProductInfo, 
                info_bwd1: Optional[SparseProductInfo] = None, info_bwd2: Optional[SparseProductInfo] = None, out_accumulated: bool = False) -> Tensor:
        ret = indexed_scavec_scale_gather(
            input1, input2,
            info_fwd.scale,
            info_fwd.index1,
            info_fwd.index2,
            info_fwd.seg_out,
            info_fwd.gather_index,
            info_fwd.index_out,
            out_accumulated=out_accumulated,
            out_size=info_fwd.out_size,
            )
        ctx.save_for_backward(input1 if input2.requires_grad else None, input2 if input1.requires_grad else None)
        ctx.infos = (info_fwd, info_bwd1, info_bwd2)
        ctx.shared1 = input1.ndim < 2 # scalar
        ctx.shared2 = input2.ndim < 3 # vec
        ctx.out_accumulated = out_accumulated
        return ret

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        grad = grad_output
        input1, input2 = ctx.saved_tensors
        info_fwd, info_bwd1, info_bwd2 = ctx.infos
        left_shared_fwd = ctx.shared1
        right_shared_fwd = ctx.shared2

        grad1, grad2 = None, None
        if ctx.needs_input_grad[0]:
            # gx = inner(y, gz) -> out_accumulated_bx = left_shared_fwd
            out_accumulated_bwd1 = left_shared_fwd
            grad1 = SparseInner.apply(input2, grad, info_bwd1, info_bwd2, info_fwd, out_accumulated_bwd1)
        else:
            grad1 = None
        if ctx.needs_input_grad[1]:
            # gy = vecsca(gz, x) -> out_accumulated_by = right_shared_fwd
            out_accumulated_bwd2 = right_shared_fwd
            grad2 = SparseVecSca.apply(grad, input1, info_bwd2, info_fwd, info_bwd1, out_accumulated_bwd2)
        else:
            grad2 = None

        # Return grads corresponding to input1, input2, out_accumulated, info_fwd, info_bwd1, info_bwd2
        return grad1, grad2, None, None, None, None
def sparse_scavec(input1: Tensor, input2: Tensor, info_fwd: SparseProductInfo, 
                info_bwd1: Optional[SparseProductInfo] = None, info_bwd2: Optional[SparseProductInfo] = None, out_accumulated: bool = False) -> Tensor:
    r"""
    Computes sparse scalar-vector multiplication using indexed operations.

    Multiplies a scalar (``input1``) by a vector (``input2``) element-wise, guided by ``info_fwd``.

    Args:
        input1 (torch.Tensor): The input scalar (or tensor broadcastable to scalar for each operation).
        input2 (torch.Tensor): The input vector.
        info_fwd (SparseProductInfo): Information for the forward pass.
        info_bwd1 (SparseProductInfo, optional): Information for the backward pass w.r.t. ``input1``.
        info_bwd2 (SparseProductInfo, optional): Information for the backward pass w.r.t. ``input2``.
        out_accumulated (bool, optional): If ``True``, accumulate to output. Defaults to ``False``.

    Returns:
        torch.Tensor: The result of the sparse scalar-vector multiplication.
    """
    return SparseScaVec.apply(input1, input2, info_fwd, info_bwd1, info_bwd2, out_accumulated)

class SparseMatTVec(Function):

    @staticmethod
    def forward(ctx, input1: Tensor, input2: Tensor, info_fwd: SparseProductInfo, 
                info_bwd1: Optional[SparseProductInfo] = None, info_bwd2: Optional[SparseProductInfo] = None, out_accumulated: bool = False) -> Tensor:
        ret = indexed_mat_t_vec_scale_gather(
            input1, input2,
            info_fwd.scale,
            info_fwd.index1,
            info_fwd.index2,
            info_fwd.seg_out,
            info_fwd.gather_index,
            info_fwd.index_out,
            out_accumulated=out_accumulated,
            out_size=info_fwd.out_size,
            )
        ctx.save_for_backward(input1 if input2.requires_grad else None, input2 if input1.requires_grad else None)
        ctx.infos = (info_fwd, info_bwd1, info_bwd2)
        ctx.shared1 = input1.ndim < 4 # mat
        ctx.shared2 = input2.ndim < 3 # vec
        ctx.out_accumulated = out_accumulated
        return ret

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        grad = grad_output
        input1, input2 = ctx.saved_tensors
        info_fwd, info_bwd1, info_bwd2 = ctx.infos
        left_shared_fwd = ctx.shared1
        right_shared_fwd = ctx.shared2

        grad1, grad2 = None, None
        if ctx.needs_input_grad[0]:
            # gx = outer(y, gz) -> out_accumulated_bx = left_shared_fwd
            out_accumulated_bwd1 = left_shared_fwd
            grad1 = SparseOuter.apply(input2, grad, info_bwd1, info_bwd2, info_fwd, out_accumulated_bwd1)
        else:
            grad1 = None
        if ctx.needs_input_grad[1]:
            # gy = vecmat(gz, x^T) -> out_accumulated_by = right_shared_fwd
            out_accumulated_bwd2 = right_shared_fwd
            grad2 = SparseVecMat.apply(grad, input1.transpose(-1,-2).contiguous(), info_bwd2, info_fwd, info_bwd1, out_accumulated_bwd2)
        else:
            grad2 = None

        # Return grads corresponding to input1, input2, out_accumulated, info_fwd, info_bwd1, info_bwd2
        return grad1, grad2, None, None, None, None
def sparse_mat_t_vec(input1: Tensor, input2: Tensor, info_fwd: SparseProductInfo, 
                info_bwd1: Optional[SparseProductInfo] = None, info_bwd2: Optional[SparseProductInfo] = None, out_accumulated: bool = False) -> Tensor:
    r"""
    Computes sparse matrix-transposed-vector multiplication using indexed operations.

    Multiplies a transposed matrix (``input1.T``) by a vector (``input2``) based on ``info_fwd``.
    Effectively ``input1.T @ input2`` but using sparse indexing.

    Args:
        input1 (torch.Tensor): The input matrix.
        input2 (torch.Tensor): The input vector (to be effectively transposed).
        info_fwd (SparseProductInfo): Information for the forward pass.
        info_bwd1 (SparseProductInfo, optional): Information for the backward pass w.r.t. ``input1``.
        info_bwd2 (SparseProductInfo, optional): Information for the backward pass w.r.t. ``input2``.
        out_accumulated (bool, optional): If ``True``, accumulate to output. Defaults to ``False``.

    Returns:
        torch.Tensor: The result of the sparse matrix-transposed-vector multiplication.
    """
    return SparseMatTVec.apply(input1, input2, info_fwd, info_bwd1, info_bwd2, out_accumulated)
