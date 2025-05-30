from typing import Optional
import torch
from torch import Tensor
from torch.autograd import Function

from ...ops.indexed_scale_segment_op import (
    indexed_scale_segment
)

from ...structs import SparseScaleInfo

from equitorch.irreps import check_irreps, Irreps

class SparseScale(Function):
    '''
    Currently only support Square-matrix Transformation
    '''
    @staticmethod
    def forward(ctx, input, info_fwd, info_bwd):

        ret = indexed_scale_segment(
            input,
            info_fwd.scale,
            info_fwd.index,
            info_fwd.seg_out,
            info_fwd.out_size,
        )

        ctx.save_for_backward(input)
        ctx.infos = (info_fwd, info_bwd)
        return ret
    
    @staticmethod
    def backward(ctx, grad_output):
        grad = grad_output
        (input,) = ctx.saved_tensors
        info_fwd, info_bwd = ctx.infos

        grad_in = None
        
        if ctx.needs_input_grad[0]:
            grad_in = SparseScale.apply(grad, info_bwd, info_fwd)

        return grad_in, None, None
    
def sparse_scale(input: Tensor, info_fwd: SparseScaleInfo, info_bwd: Optional[SparseScaleInfo] = None) -> Tensor:
    return SparseScale.apply(input, info_fwd, info_bwd)