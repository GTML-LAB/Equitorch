import torch
from torch import Tensor
from torch.autograd import Function

from torch_geometric.utils import scatter

from .sparse_product import sparse_inner, sparse_mat_t_vec, sparse_mul, sparse_outer, sparse_scavec, sparse_vecmat, sparse_vecsca
from .sparse_scale import sparse_scale

from ...ops.indexed_product_scale_segment_op import indexed_vecmat_scale_segment
from ...ops.indexed_product_segment_op import (
    indexed_outer_segment, 
)
from ...structs import IrrepsInfo, IrrepsLinearInfo, SparseProductInfo, TensorProductInfo

class TensorProductU1UDummy(Function):

    @staticmethod
    def forward(ctx, input1: Tensor, input2: Tensor, weight: Tensor,
                tp_info_forward: TensorProductInfo,
                tp_info_backward1: TensorProductInfo,
                tp_info_backward2: TensorProductInfo):
        
        inter = sparse_vecsca(input1, input2, tp_info_forward.info_Mij_fwd)
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

        grad_inter = torch.broadcast_to(grad_inter, inter.shape)
        if ctx.needs_input_grad[0]:
            grad1 = tensor_product_1uu(
                input2, grad, weight, 
                tp_info_backward1, tp_info_backward2, tp_info_forward)
            # if grad1.requires_grad:
            grad1 = grad1 + sparse_scavec(input2, grad_inter,
                                    tp_info_forward.info_Mij_bwd1,
                                    tp_info_forward.info_Mij_bwd2,
                                    tp_info_forward.info_Mij_fwd)
        else:
            grad1 = None
            
        if ctx.needs_input_grad[1]:
            grad2 = tensor_product_uu1(
                grad, input1, weight, 
                tp_info_backward2, tp_info_forward, tp_info_backward1)
            # if grad2.requires_grad:
            grad2 = grad2 + sparse_inner(grad_inter, input1,
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

def tensor_product_u1u(input1: Tensor, input2: Tensor, weight: Tensor,
                tp_info_forward: TensorProductInfo,
                tp_info_backward1: TensorProductInfo=None,
                tp_info_backward2: TensorProductInfo=None):
    ret, _ = TensorProductU1UDummy.apply(input1, input2, weight,
                tp_info_forward,
                tp_info_backward1,
                tp_info_backward2)
    return ret
so3_linear_uu = tensor_product_u1u

class TensorProduct1UUDummy(Function):

    @staticmethod
    def forward(ctx, input1: Tensor, input2: Tensor, weight: Tensor,
                tp_info_forward: TensorProductInfo,
                tp_info_backward1: TensorProductInfo,
                tp_info_backward2: TensorProductInfo):
        
        inter = sparse_scavec(input1, input2, tp_info_forward.info_Mij_fwd).squeeze(-2)
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

        grad_inter = torch.broadcast_to(grad_inter, inter.shape)
        if ctx.needs_input_grad[0]:
            grad1 = tensor_product_uu1(
                input2, grad, weight, 
                tp_info_backward1, tp_info_backward2, tp_info_forward)
            # if grad1.requires_grad:
            grad1 = grad1 + sparse_inner(input2, grad_inter,
                                    tp_info_forward.info_Mij_bwd1,
                                    tp_info_forward.info_Mij_bwd2,
                                    tp_info_forward.info_Mij_fwd)
        else:
            grad1 = None
            
        if ctx.needs_input_grad[1]:
            grad2 = tensor_product_u1u(
                grad, input1, weight, 
                tp_info_backward2, tp_info_forward, tp_info_backward1)
            # if grad2.requires_grad:
            grad2 = grad2 + sparse_vecsca(grad_inter, input1,
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

def tensor_product_1uu(input1: Tensor, input2: Tensor, weight: Tensor,
                tp_info_forward: TensorProductInfo,
                tp_info_backward1: TensorProductInfo = None,
                tp_info_backward2: TensorProductInfo = None):
    ret, _ = TensorProduct1UUDummy.apply(input1, input2, weight,
                tp_info_forward,
                tp_info_backward1,
                tp_info_backward2)
    return ret

class TensorProductUU1Dummy(Function):

    @staticmethod
    def forward(ctx, input1: Tensor, input2: Tensor, weight: Tensor,
                tp_info_forward: TensorProductInfo,
                tp_info_backward1: TensorProductInfo,
                tp_info_backward2: TensorProductInfo):
        
        inter = sparse_mul(input1, input2, tp_info_forward.info_Mij_fwd)
        ret = sparse_inner(inter, weight, tp_info_forward.info_M_fwd)

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
            grad1 = tensor_product_u1u(
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
            grad2 = tensor_product_1uu(
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
            grad_W = sparse_scavec(grad_output, inter,
                                tp_info_forward.info_M_bwd2,
                                tp_info_forward.info_M_fwd,
                                tp_info_forward.info_M_bwd1,
                                out_accumulated=ctx.weight_ndim == 2)
        else:
            grad_W = None
        return grad1, grad2, grad_W, None, None, None

def tensor_product_uu1(input1: Tensor, input2: Tensor, weight: Tensor,
                tp_info_forward: TensorProductInfo,
                tp_info_backward1: TensorProductInfo = None,
                tp_info_backward2: TensorProductInfo = None):
    ret, _ =  TensorProductUU1Dummy.apply(input1, input2, weight,
                tp_info_forward,
                tp_info_backward1,
                tp_info_backward2)
    return ret

class TensorProductU1VDummy(Function):

    @staticmethod
    def forward(ctx, input1: Tensor, input2: Tensor, weight: Tensor,
                tp_info_forward: TensorProductInfo,
                tp_info_backward1: TensorProductInfo,
                tp_info_backward2: TensorProductInfo):
        
        inter = sparse_vecsca(input1, input2, tp_info_forward.info_Mij_fwd)
        ret = sparse_vecmat(inter, weight, tp_info_forward.info_M_fwd)

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
            grad1 = tensor_product_1vu(
                input2, grad, 
                weight.transpose(-1,-2).contiguous(),
                tp_info_backward1, tp_info_backward2, tp_info_forward)
            # if grad1.requires_grad:
            grad1 = grad1 + sparse_scavec(input2, grad_inter,
                                    tp_info_forward.info_Mij_bwd1,
                                    tp_info_forward.info_Mij_bwd2,
                                    tp_info_forward.info_Mij_fwd)
        else:
            grad1 = None
            
        if ctx.needs_input_grad[1]:
            grad2 = tensor_product_vu1(
                grad, input1, 
                weight.transpose(-1,-2).contiguous(),
                tp_info_backward2, tp_info_forward, tp_info_backward1)
            # if grad2.requires_grad:
            grad2 = grad2 + sparse_inner(grad_inter, input1,
                                    tp_info_forward.info_Mij_bwd2,
                                    tp_info_forward.info_Mij_fwd,
                                    tp_info_forward.info_Mij_bwd1)
        else:
            grad2 = None

        if ctx.needs_input_grad[2]:
            grad_W = sparse_outer(grad_output, inter,
                                tp_info_forward.info_M_bwd2,
                                tp_info_forward.info_M_fwd,
                                tp_info_forward.info_M_bwd1,
                                out_accumulated=ctx.weight_ndim == 3).transpose(-1,-2).contiguous()

        else:
            grad_W = None
        return grad1, grad2, grad_W, None, None, None
    
def tensor_product_u1v(input1: Tensor, input2: Tensor, weight: Tensor,
                tp_info_forward: TensorProductInfo,
                tp_info_backward1: TensorProductInfo = None,
                tp_info_backward2: TensorProductInfo = None):
    ret, _ =  TensorProductU1VDummy.apply(input1, input2, weight,
                tp_info_forward,
                tp_info_backward1,
                tp_info_backward2)
    return ret

so3_linear_uv = tensor_product_u1v

class TensorProduct1VUDummy(Function):

    @staticmethod
    def forward(ctx, input1: Tensor, input2: Tensor, weight: Tensor,
                tp_info_forward: TensorProductInfo,
                tp_info_backward1: TensorProductInfo,
                tp_info_backward2: TensorProductInfo):
        inter = sparse_scavec(input1, input2, tp_info_forward.info_Mij_fwd)
        ret = sparse_vecmat(inter, weight, tp_info_forward.info_M_fwd)

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
            grad1 = tensor_product_vu1(
                input2, grad, 
                weight,
                tp_info_backward1, tp_info_backward2, tp_info_forward)
            # if grad1.requires_grad:
            grad1 = grad1 + sparse_inner(input2, grad_inter,
                                    tp_info_forward.info_Mij_bwd1,
                                    tp_info_forward.info_Mij_bwd2,
                                    tp_info_forward.info_Mij_fwd)
        else:
            grad1 = None
        if ctx.needs_input_grad[1]:
            grad2 = tensor_product_u1v(
                grad, input1, 
                weight.transpose(-1,-2).contiguous(),
                tp_info_backward2, tp_info_forward, tp_info_backward1)
            # if grad2.requires_grad:
            grad2 = grad2 + sparse_vecsca(grad_inter, input1,
                                    tp_info_forward.info_Mij_bwd2,
                                    tp_info_forward.info_Mij_fwd,
                                    tp_info_forward.info_Mij_bwd1)
        else:
            grad2 = None
        if ctx.needs_input_grad[2]:
            grad_W = sparse_outer(grad_output, inter,
                                tp_info_forward.info_M_bwd2,
                                tp_info_forward.info_M_fwd,
                                tp_info_forward.info_M_bwd1,
                                out_accumulated=ctx.weight_ndim == 3).transpose(-1,-2).contiguous()
        else:
            grad_W = None
        return grad1, grad2, grad_W, None, None, None

def tensor_product_1vu(input1: Tensor, input2: Tensor, weight: Tensor,
                tp_info_forward: TensorProductInfo,
                tp_info_backward1: TensorProductInfo,
                tp_info_backward2: TensorProductInfo):
    ret, _ = TensorProduct1VUDummy.apply(input1, input2, weight,
                tp_info_forward, tp_info_backward1, tp_info_backward2)
    return ret

class TensorProductVU1Dummy(Function):

    @staticmethod
    def forward(ctx, input1: Tensor, input2: Tensor, weight: Tensor,
                tp_info_forward: TensorProductInfo,
                tp_info_backward1: TensorProductInfo,
                tp_info_backward2: TensorProductInfo):

        inter = sparse_vecmat(input1, weight, tp_info_forward.info_kM1j_fwd)
        inter = sparse_inner(inter, input2, tp_info_forward.info_kM1M2_fwd)
        ret = sparse_scale(inter.unsqueeze(-1), tp_info_forward.info_M_kM1M2_fwd).squeeze(-1)
        inter_save = None
        # else:
        #     inter = sparse_outer(input1, input2, tp_info_forward.info_Mij_fwd)
        #     ret = sparse_inner(inter, weight, tp_info_forward.info_M_fwd)
        #     inter_save = inter
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
        ctx.save_for_backward(input1, input2, weight, inter_save)
        ctx.tp_info = (tp_info_forward, tp_info_backward1, tp_info_backward2)
        return ret, None
    
    @staticmethod
    def backward(ctx, grad_output, _):
        grad = grad_output
        input1, input2, weight, inter = ctx.saved_tensors
        tp_info_forward, tp_info_backward1, tp_info_backward2 = ctx.tp_info

        if ctx.needs_input_grad[0]:
            grad1 = tensor_product_u1v(
                input2, grad, 
                weight.transpose(-1,-2).contiguous(),
                tp_info_backward1, tp_info_backward2, tp_info_forward)
        else:
            grad1 = None
        if ctx.needs_input_grad[1]:
            grad2 = tensor_product_1vu(
                grad, input1, 
                weight,
                tp_info_backward2, tp_info_forward, tp_info_backward1)
        else:
            grad2 = None
        if ctx.needs_input_grad[2]:
            grad_inter = sparse_scale(grad_output.unsqueeze(-1), 
                                      tp_info_forward.info_M_kM1M2_bwd,
                                      tp_info_forward.info_M_kM1M2_fwd).squeeze(-1)
            # inter = sparse_inner(inter, input2, tp_info_forward.info_kM1M2_fwd)
            grad_inter = sparse_vecsca(input2, grad_inter, 
                                       tp_info_forward.info_kM1M2_bwd1,
                                       tp_info_forward.info_kM1M2_bwd2,
                                       tp_info_forward.info_kM1M2_fwd)
            # inter = sparse_vecmat(input1, weight, tp_info_forward.info_kM1j_fwd)
            grad_W = sparse_outer(grad_inter, input1, 
                                       tp_info_forward.info_kM1j_bwd2,
                                       tp_info_forward.info_kM1j_fwd,
                                       tp_info_forward.info_kM1j_bwd1
                                       ).transpose(-1,-2).contiguous()
        else:
            grad_W = None
        return grad1, grad2, grad_W, None, None, None
    
def tensor_product_vu1(input1: Tensor, input2: Tensor, weight: Tensor,
                tp_info_forward: TensorProductInfo,
                tp_info_backward1: TensorProductInfo,
                tp_info_backward2: TensorProductInfo):
    ret, _ = TensorProductVU1Dummy.apply(input1, input2, weight,
                tp_info_forward, tp_info_backward1, tp_info_backward2)
    return ret

class IrrepWiseLinear(Function):

    @staticmethod
    def forward(ctx, input: Tensor, weight: Tensor,
                irreps_info: IrrepsInfo):

        info_fwd = SparseProductInfo(index2=irreps_info.irrep_index)
        info_bwd1 = SparseProductInfo(index1=irreps_info.irrep_index)
        info_bwd2 = SparseProductInfo(seg_out=irreps_info.irrep_seg)

        out = sparse_vecmat(input, weight, info_fwd)
        
        ctx.save_for_backward(input, weight)
        ctx.infos = (info_fwd, info_bwd1, info_bwd2)
        ctx.weight_dim = weight.ndim
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        
        input, weight = ctx.saved_tensors
        info_fwd, info_bwd1, info_bwd2 = ctx.infos

        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = sparse_mat_t_vec(weight.transpose(-1,-2), grad_output, 
                                          info_bwd1, info_bwd2, info_fwd)

        grad_weight = None
        if ctx.needs_input_grad[1]:
            grad_weight = sparse_outer(grad_output, input, 
                                       info_bwd2, info_fwd, info_bwd1,
                                       ctx.weight_dim==3).transpose(-1,-2).contiguous()

        return grad_input, grad_weight, None
    
irrep_wise_linear = IrrepWiseLinear.apply

class IrrepsLinear(Function):

    
    @staticmethod
    def forward(ctx, input: Tensor, weight: Tensor,
                irreps_linear_info_forward: IrrepsLinearInfo,
                irreps_linear_info_backward: IrrepsLinearInfo):
        out = indexed_vecmat_scale_segment(
            input, weight, 
            index1=irreps_linear_info_forward.M0_MM0,
            index2=irreps_linear_info_forward.ii0_MM0,
            seg=irreps_linear_info_forward.M_seg_MM0,
            scale=irreps_linear_info_forward.scale_MM0)
        out = scatter(out, irreps_linear_info_forward.M_out, 
                      dim=-2, dim_size=irreps_linear_info_forward.out_size)
        ctx.save_for_backward(input, weight)
        ctx.irreps_linear_info = irreps_linear_info_forward, irreps_linear_info_backward
        ctx.weight_dim = weight.ndim
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        
        input, weight = ctx.saved_tensors
        irreps_info_forward, irreps_info_backward = ctx.irreps_linear_info

        if ctx.needs_input_grad[0]:
            grad_input = IrrepsLinear.apply(grad_output, weight.transpose(-1,-2), irreps_info_backward, irreps_info_forward)
        else:
            grad_input = None

        if ctx.needs_input_grad[1]:
            grad_weight = indexed_outer_segment(input, grad_output,
                                        index1=irreps_info_forward.M0_ii0MM0,
                                        index2=irreps_info_forward.M_ii0MM0, 
                                        seg=irreps_info_forward.ii0_seg_ii0MM0,
                                        accumulated=ctx.weight_dim==3) * irreps_info_forward.scales_ii0.unsqueeze(-1).unsqueeze(-1)
        else:
            grad_weight = None

        return grad_input, grad_weight, None, None
    
irreps_linear = IrrepsLinear.apply

so2_linear_uu = sparse_mul
so2_linear_uv = sparse_vecmat