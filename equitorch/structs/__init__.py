"""Defines specialized data structures (NamedTuples) for managing sparse operations and tensor product information."""
import collections
from typing import NamedTuple, Callable, Any, Optional
from torch import Tensor
import torch

def add_operation_methods(cls):
    # """类装饰器：为 NamedTuple 动态添加 to/cuda/cpu 等方法"""
    def _apply(self, func: Callable[[Any], Any]):
        processed = []
        for field in self._fields:
            value = getattr(self, field)
            # 递归处理 Tensor 或同类型实例
            if isinstance(value, Tensor):
                processed.append(func(value))
            elif isinstance(value, self.__class__):
                processed.append(func(value))  # 递归处理同类型字段
            elif hasattr(value, '_apply'):
                processed.append(value._apply(func))
            else:
                processed.append(value)
        return self.__class__(*processed)
    
    def to(self, *args, **kwargs):
        # 解析参数
        device, dtype, non_blocking, _ = torch._C._nn._parse_to(*args, **kwargs)
        
        # 定义转换函数
        def convert(t):
            if isinstance(t, Tensor):
                # 只对浮点张量应用dtype转换
                target_dtype = dtype if (dtype is not None and t.is_floating_point()) else None
                return t.to(device=device, dtype=target_dtype, non_blocking=non_blocking)
            return t
        
        return self._apply(convert)

    def cuda(self, *args, **kwargs):
        return self._apply(lambda x: x.cuda(*args, **kwargs))

    def cpu(self, *args, **kwargs):
        return self._apply(lambda x: x.cpu(*args, **kwargs))

    cls._apply = _apply
    cls.to = to
    cls.cuda = cuda
    cls.cpu = cpu
    return cls

@add_operation_methods
class SparseScaleInfo(NamedTuple):
    '''
        z_M = sum_{t in Ind*[M]} s_t * x_Ind'[t]

        or

        z_M = sum_{M'} s_{MM'} x_M'
    '''
    scale: Optional[Tensor] = None # (num_t,), floating
    index: Optional[Tensor] = None # (num_t,), int in [0, num_M')
    seg_out: Optional[Tensor] = None # (num_M_nonzero+1,), increasing int in [0, num_t]
    index_out: Optional[Tensor] = None # (num_M_nonzero,), int in [0, num_M)
    out_size: Optional[int] = None # num_M


@add_operation_methods
class SparseProductInfo(NamedTuple):
    '''
        z_M = sum_{t in Ind*[M]} s_t * x_Ind1[t] * y_Ind2[t]

        or

        z_M = sum_{M1M2} s_{MM1M2} x_M1 * y_M2
    '''
    scale: Optional[Tensor] = None # (num_t,), floating
    index1: Optional[Tensor] = None # (num_t,), int in [0, num_M1)
    index2: Optional[Tensor] = None # (num_t,), int in [0, num_M2)
    seg_out: Optional[Tensor] = None # (num_M_nonzero+1,), increasing int in [0, num_t]
    gather_index: Optional[Tensor] = None # (num_M_nonzero,) int in [0, num_t)
    index_out: Optional[Tensor] = None # (num_M_nonzero,), int in [0, num_M)
    out_size: Optional[int] = None # num_M


@add_operation_methods
class TensorProductInfo(NamedTuple):
    info_Mij_fwd: SparseProductInfo
    info_Mij_bwd1: SparseProductInfo
    info_Mij_bwd2: SparseProductInfo
    
    info_M_fwd: SparseProductInfo
    info_M_bwd1: SparseProductInfo
    info_M_bwd2: SparseProductInfo
    
    info_kM1j_fwd: SparseProductInfo
    info_kM1j_bwd1: SparseProductInfo
    info_kM1j_bwd2: SparseProductInfo
    info_kM1M2_fwd: SparseProductInfo
    info_kM1M2_bwd1: SparseProductInfo
    info_kM1M2_bwd2: SparseProductInfo
    info_M_kM1M2_fwd: SparseScaleInfo
    info_M_kM1M2_bwd: SparseScaleInfo
    out_size: int


@add_operation_methods
class IrrepsInfo(NamedTuple):
    rsqrt_dims: Tensor 
    rdims: Tensor 
    irrep_index: Tensor
    irrep_seg: Tensor
    num_irreps: int


@add_operation_methods
class IrrepsLinearInfo(NamedTuple):

    scale_MM0: Tensor
    M_seg_MM0: Tensor
    ii0_MM0: Tensor
    M0_MM0: Tensor
    M_MM0: Tensor
    M_out: Tensor

    # for grad on weight
    ii0_seg_ii0MM0: Tensor
    M_ii0MM0: Tensor
    M0_ii0MM0: Tensor
    scales_ii0: Tensor
    
    out_size: int

@add_operation_methods
class WignerRotationInfo(NamedTuple):
    j_matrix_info: SparseScaleInfo
    rotate_z_info_fwd: SparseProductInfo
    rotate_z_info_bwd_input: Optional[SparseProductInfo] = None
    rotate_z_info_bwd_cs: Optional[SparseProductInfo] = None
    sign: Optional[Tensor] = None # for O(3) representation. But leave it None currently.
    max_m: Optional[int] = None


