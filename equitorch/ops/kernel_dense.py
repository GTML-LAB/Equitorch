import triton
import triton.language as tl

from .kernel_utils import (
    load_block_scalar,
    load_block_vec,
    load_block_mat
)


@triton.jit
def kernel_mul(input1_bases, input2_bases,
               n_mask,
               c_offsets, c_mask,
               stride_c1, stride_c2,
               SHARED_INPUT1: tl.constexpr = False,  # 新增SHARED_INPUT1参数
               SHARED_INPUT2: tl.constexpr = False,
               OUT_ACCUMULATED: tl.constexpr = False):
    # 使用load_block_vec加载输入数据
    input1 = load_block_vec(
        base_offsets=input1_bases,
        base_mask=n_mask,
        c_offsets=c_offsets,
        c_mask=c_mask,
        stride=stride_c1,
        SHARED=SHARED_INPUT1
    )

    input2 = load_block_vec(
        base_offsets=input2_bases,
        base_mask=n_mask,
        c_offsets=c_offsets,
        c_mask=c_mask,
        stride=stride_c2,
        SHARED=SHARED_INPUT2
    )

    if OUT_ACCUMULATED:
        return tl.sum(input1 * input2, axis=0)
    else:
        return input1 * input2

@triton.jit
def kernel_outer(input1_bases, input2_bases,
                 n_mask,
                 c1_offsets, c2_offsets,
                 c1_mask, c2_mask,
                 stride_c1, stride_c2,
                 SHARED_INPUT1: tl.constexpr = False,  # 新增参数
                 SHARED_INPUT2: tl.constexpr = False,
                 OUT_ACCUMULATED: tl.constexpr = False):
    # 加载输入向量并扩展维度
    input1 = load_block_vec(
        base_offsets=input1_bases,
        base_mask=n_mask,
        c_offsets=c1_offsets,
        c_mask=c1_mask,
        stride=stride_c1,
        SHARED=SHARED_INPUT1
    ).expand_dims(-1)

    input2 = load_block_vec(
        base_offsets=input2_bases,
        base_mask=n_mask,
        c_offsets=c2_offsets,
        c_mask=c2_mask,
        stride=stride_c2,
        SHARED=SHARED_INPUT2
    ).expand_dims(-2)

    if OUT_ACCUMULATED:
        return tl.sum(input1 * input2, axis=0)
    else:
        return input1 * input2

@triton.jit
def kernel_inner(input1_bases, input2_bases,
                 n_mask,
                 c_in,
                 stride_c1, stride_c2,
                 BLOCK_SIZE_N: tl.constexpr,
                 BLOCK_SIZE_C:  tl.constexpr,
                 SHARED_INPUT1: tl.constexpr = False,
                 SHARED_INPUT2: tl.constexpr = False,
                 OUT_ACCUMULATED: tl.constexpr = False,
                 NUM_STAGES: tl.constexpr = None,
                 LOOP_UNROLL_FACTOR: tl.constexpr = None,
                 OUT_DTYPE: tl.dtype = None):
    # 初始化累加器
    if OUT_ACCUMULATED:
        accumulator = tl.zeros((), dtype=OUT_DTYPE)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_N, ), dtype=OUT_DTYPE)
    for block_start_c in tl.range(0, c_in, BLOCK_SIZE_C, NUM_STAGES, LOOP_UNROLL_FACTOR):

        c_offsets = block_start_c + tl.arange(0, BLOCK_SIZE_C)
        c_mask = c_offsets < c_in
        
        # 向量加载
        input1 = load_block_vec(
            base_offsets=input1_bases,
            base_mask=n_mask,
            c_offsets=c_offsets,
            c_mask=c_mask,
            stride=stride_c1,
            SHARED=SHARED_INPUT1
        )
        input2 = load_block_vec(
            base_offsets=input2_bases,
            base_mask=n_mask,
            c_offsets=c_offsets,
            c_mask=c_mask,
            stride=stride_c2,
            SHARED=SHARED_INPUT2
        )
        
        # 累加逻辑
        if OUT_ACCUMULATED:
            accumulator += tl.sum(input1 * input2)
        else:
            accumulator += tl.sum(input1 * input2, axis=-1)

    return accumulator

@triton.jit
def kernel_vecmat(input1_bases: tl.tensor, input2_bases,
                  n_mask,
                  c_out_offsets, c_out_mask,
                  c_in,
                  stride_c_in1, stride_c_in2, stride_c_out2,
                  BLOCK_SIZE_N: tl.constexpr,
                  BLOCK_SIZE_C_OUT: tl.constexpr,
                  BLOCK_SIZE_C_IN: tl.constexpr,
                  SHARED_INPUT1: tl.constexpr = False,
                  SHARED_INPUT2: tl.constexpr = False,
                  OUT_ACCUMULATED: tl.constexpr = False,
                  NUM_STAGES: tl.constexpr = None,
                  LOOP_UNROLL_FACTOR: tl.constexpr = None,
                  OUT_DTYPE: tl.constexpr = None):
    # 初始化累加器
    if OUT_ACCUMULATED:
        accumulator = tl.zeros((BLOCK_SIZE_C_OUT,), dtype=OUT_DTYPE)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_C_OUT), dtype=OUT_DTYPE)

    for block_start_c_in in tl.range(0, c_in, BLOCK_SIZE_C_IN, NUM_STAGES, LOOP_UNROLL_FACTOR):
        c_in_offsets = block_start_c_in + tl.arange(0, BLOCK_SIZE_C_IN)
        c_in_mask = c_in_offsets < c_in

        # 加载输入向量
        input1 = load_block_vec(
            base_offsets=input1_bases,
            base_mask=n_mask,
            c_offsets=c_in_offsets,
            c_mask=c_in_mask,
            stride=stride_c_in1,
            SHARED=SHARED_INPUT1
        )
        
        # 矩阵加载
        input2 = load_block_mat(
            base_offsets=input2_bases,
            base_mask=n_mask,
            c_in_offsets=c_in_offsets,
            c_in_mask=c_in_mask,
            c_out_offsets=c_out_offsets,
            c_out_mask=c_out_mask,
            stride_c_in=stride_c_in2,
            stride_c_out=stride_c_out2,
            SHARED=SHARED_INPUT2
        )
        
        # 乘积累加
        if OUT_ACCUMULATED:
            accumulator += tl.sum(tl.sum(input2 * input1.expand_dims(-1),axis=-2),axis=0)
        else:
            accumulator += tl.sum(input2 * input1.expand_dims(-1), axis=-2)

    return accumulator

@triton.jit
def kernel_vecsca(input1_bases, input2_bases,
                  n_mask,
                  c_offsets, c_mask,
                  stride_c1,
                  SHARED_INPUT1: tl.constexpr = False,
                  SHARED_INPUT2: tl.constexpr = False,
                  OUT_ACCUMULATED: tl.constexpr = False):
    # 加载向量输入
    input1 = load_block_vec(
        base_offsets=input1_bases,
        base_mask=n_mask,
        c_offsets=c_offsets,
        c_mask=c_mask,
        stride=stride_c1,
        SHARED=SHARED_INPUT1
    )
    
    # 加载标量输入
    input2 = load_block_scalar(
        base_offsets=input2_bases,
        base_mask=n_mask,
        SHARED=SHARED_INPUT2
    ).expand_dims(-1)

    if OUT_ACCUMULATED:
        return tl.sum(input1 * input2, axis=0)
    else:
        return input1 * input2
    
