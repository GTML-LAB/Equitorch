import torch
from torch import Tensor
from torch.autograd import Function

from .sparse_product import sparse_mat_t_vec, sparse_mul, sparse_outer, sparse_vecmat

from ...structs import IrrepsInfo, SparseProductInfo, TensorProductInfo

class TensorProductUUUDummy(Function):

    @staticmethod
    def forward(ctx, input1: Tensor, input2: Tensor, weight: Tensor,
                tp_info_forward: TensorProductInfo,
                tp_info_backward1: TensorProductInfo,
                tp_info_backward2: TensorProductInfo):
        
        inter = sparse_mul(input1, input2, tp_info_forward.info_Mij_fwd)
        ret = sparse_mul(inter, weight, tp_info_forward.info_M_fwd)

        grad_weight = weight.requires_grad
        grad_input1 = input1.requires_grad
        grad_input2 = input2.requires_grad
        
        ctx.weight_ndim = weight.ndim
        if not grad_weight:
            inter = None
        if (not grad_input1) and (not grad_input2):
            weight = None
        if not grad_input1:
            input2 = None
        if not grad_input2:
            input1 = None
        ctx.save_for_backward(input1, input2, weight, inter)
        ctx.tp_info = (tp_info_forward, tp_info_backward1, tp_info_backward2)
        return ret, inter

    @staticmethod
    def backward(ctx, grad_output, grad_inter):
        grad = grad_output
        input1, input2, weight, inter = ctx.saved_tensors
        tp_info_forward, tp_info_backward1, tp_info_backward2 = ctx.tp_info

        # Ensure grad_inter has correct shape for higher-order derivatives
        if grad_inter is not None:
            grad_inter = torch.broadcast_to(grad_inter, inter.shape)
        else:
            grad_inter = torch.zeros_like(inter)
        if ctx.needs_input_grad[0]:
            grad1 = tensor_product_uuu(
                input2, grad, weight, 
                tp_info_backward1, tp_info_backward2, tp_info_forward)
            # if grad1.requires_grad:
            grad1 = grad1 + sparse_mul(input2, grad_inter,
                                    tp_info_forward.info_Mij_bwd1,
                                    tp_info_forward.info_Mij_bwd2,
                                    tp_info_forward.info_Mij_fwd)
        else:
            grad1 = None
            
        if ctx.needs_input_grad[1]:
            grad2 = tensor_product_uuu(
                grad, input1, weight, 
                tp_info_backward2, tp_info_forward, tp_info_backward1)
            # if grad2.requires_grad:
            grad2 = grad2 + sparse_mul(grad_inter, input1,
                                    tp_info_forward.info_Mij_bwd2,
                                    tp_info_forward.info_Mij_fwd,
                                    tp_info_forward.info_Mij_bwd1)
        else:
            grad2 = None

        if ctx.needs_input_grad[2]:
            grad_W = sparse_mul(grad_output, inter,
                                tp_info_forward.info_M_bwd2,
                                tp_info_forward.info_M_fwd,
                                tp_info_forward.info_M_bwd1,
                                out_accumulated=ctx.weight_ndim == 2)
        else:
            grad_W = None
        return grad1, grad2, grad_W, None, None, None

def tensor_product_uuu(input1: Tensor, input2: Tensor, weight: Tensor,
                tp_info_forward: TensorProductInfo,
                tp_info_backward1: TensorProductInfo = None,
                tp_info_backward2: TensorProductInfo = None):
    ret, _ =  TensorProductUUUDummy.apply(input1, input2, weight,
                                       tp_info_forward,
                                       tp_info_backward1,
                                       tp_info_backward2)
    return ret

class TensorProductUVWDummy(Function):

    @staticmethod
    def forward(ctx, input1: Tensor, input2: Tensor, weight: Tensor,
                tp_info_forward: TensorProductInfo,
                tp_info_backward1: TensorProductInfo,
                tp_info_backward2: TensorProductInfo):
        
        inter = sparse_outer(input1, input2, tp_info_forward.info_Mij_fwd)
        ret = sparse_vecmat(inter.flatten(-2), weight.flatten(-3,-2), tp_info_forward.info_M_fwd)

        grad_weight = weight.requires_grad
        grad_input1 = input1.requires_grad
        grad_input2 = input2.requires_grad
        
        ctx.weight_ndim = weight.ndim
        if not grad_weight:
            inter = None
        if (not grad_input1) and (not grad_input2):
            weight = None
        if not grad_input1:
            input2 = None
        if not grad_input2:
            input1 = None
        ctx.save_for_backward(input1, input2, weight, inter)
        ctx.tp_info = (tp_info_forward, tp_info_backward1, tp_info_backward2)
        return ret, inter

    @staticmethod
    def backward(ctx, grad_output, grad_inter):
        grad = grad_output
        input1, input2, weight, inter = ctx.saved_tensors
        tp_info_forward, tp_info_backward1, tp_info_backward2 = ctx.tp_info

        grad_inter = torch.broadcast_to(grad_inter, inter.shape)
        if ctx.needs_input_grad[0]:
            grad1 = tensor_product_uvw(
                input2, grad, 
                weight.permute(*range(0,weight.ndim-3), -2, -1, -3).contiguous(),
                tp_info_backward1, tp_info_backward2, tp_info_forward)
            # if grad1.requires_grad:
            grad1 = grad1 + sparse_vecmat(input2, grad_inter.transpose(-1,-2).contiguous(),
                                    tp_info_forward.info_Mij_bwd1,
                                    tp_info_forward.info_Mij_bwd2,
                                    tp_info_forward.info_Mij_fwd)
        else:
            grad1 = None
            
        if ctx.needs_input_grad[1]:
            grad2 = tensor_product_uvw(
                grad, input1,
                weight.permute(*range(0,weight.ndim-3), -1, -3, -2).contiguous(),
                tp_info_backward2, tp_info_forward, tp_info_backward1)
            # if grad2.requires_grad:
            grad2 = grad2 + sparse_mat_t_vec(grad_inter, input1,
                                    tp_info_forward.info_Mij_bwd2,
                                    tp_info_forward.info_Mij_fwd,
                                    tp_info_forward.info_Mij_bwd1)
        else:
            grad2 = None

        if ctx.needs_input_grad[2]:
            grad_W = sparse_outer(grad_output, inter.flatten(-2),
                                tp_info_forward.info_M_bwd2,
                                tp_info_forward.info_M_fwd,
                                tp_info_forward.info_M_bwd1,
                                out_accumulated=ctx.weight_ndim == 4).transpose(-1,-2).contiguous()
            grad_W = grad_W.unflatten(-2, weight.shape[-3:-1])
        else:
            grad_W = None
        return grad1, grad2, grad_W, None, None, None

def tensor_product_uvw(input1: Tensor, input2: Tensor, weight: Tensor,
                tp_info_forward: TensorProductInfo,
                tp_info_backward1: TensorProductInfo = None,
                tp_info_backward2: TensorProductInfo = None):
    ret, _ = TensorProductUVWDummy.apply(input1, input2, weight,
                                       tp_info_forward,
                                       tp_info_backward1,
                                       tp_info_backward2)
    return ret
class TensorDotUU(Function):

    @staticmethod
    def forward(ctx, input1: Tensor, input2: Tensor, 
                irreps_info: IrrepsInfo, scaled: bool = True):

        grad_input1 = input1.requires_grad
        grad_input2 = input2.requires_grad

        info_fwd = SparseProductInfo(seg_out=irreps_info.irrep_seg)
        info_bwd1 = SparseProductInfo(index2=irreps_info.irrep_index)
        info_bwd2 = SparseProductInfo(index1=irreps_info.irrep_index)
        ctx.infos = (info_fwd, info_bwd1, info_bwd2)
        
        ret = sparse_mul(input1, input2, info_fwd, info_bwd1, info_bwd2)
        if scaled:
            ret = ret * irreps_info.rsqrt_dims.unsqueeze(-1)

        if not grad_input1:
            input2 = None
        if not grad_input2:
            input1 = None
        ctx.save_for_backward(input1, input2)
        ctx.irreps_info = irreps_info
        ctx.scaled = scaled
        return ret
    
    @staticmethod
    def backward(ctx, grad_output):
        grad = grad_output
        input1, input2 = ctx.saved_tensors
        scaled = ctx.scaled
        irreps_info = ctx.irreps_info
        info_fwd = SparseProductInfo(seg_out=irreps_info.irrep_seg)
        info_bwd1 = SparseProductInfo(index2=irreps_info.irrep_index)
        info_bwd2 = SparseProductInfo(index1=irreps_info.irrep_index)

        if scaled:
            grad = grad * irreps_info.rsqrt_dims.unsqueeze(-1)

        grad1 = None
        if ctx.needs_input_grad[0]:
            grad1 = sparse_mul(input2, grad, info_bwd1, info_bwd2, info_fwd)

        grad2 = None
        if ctx.needs_input_grad[1]:
            grad2 = sparse_mul(grad, input1, info_bwd2, info_fwd, info_bwd1)

        return grad1, grad2, None, None

tensor_dot_uu = TensorDotUU.apply

class TensorDotUV(Function):

    @staticmethod
    def forward(ctx, input1: Tensor, input2: Tensor, 
                irreps_info: IrrepsInfo, scaled: bool = True):

        raise NotImplementedError()    

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError()    

tensor_dot_uv = TensorDotUV.apply
