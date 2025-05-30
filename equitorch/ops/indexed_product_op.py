from .batched_sparse_dense_op import (
    indexed_mul_scale_gather,
    indexed_inner_scale_gather,
    indexed_outer_scale_gather,
    indexed_vecmat_scale_gather,
    indexed_vecsca_scale_gather
)



def indexed_mul(
    input1, input2, index1=None, index2=None, out = None,
    block_size_n = 64,
    block_size_c = 64,
):
    assert index1 is not None or index2 is not None

    return indexed_mul_scale_gather(
        input1, input2,
        index1=index1, index2=index2,
        out=out,
        block_size_n=block_size_n,
        block_size_c=block_size_c,
        num_stages=0
    )

def indexed_outer(
    input1, 
    input2, 
    index1=None, 
    index2=None, 
    out=None,
    block_size_n=16,  # 外积需要更高并行度
    block_size_c1=64,
    block_size_c2=64,
):
    r"""带索引的批量外积运算"""
    assert index1 is not None or index2 is not None
    
    return indexed_outer_scale_gather(
        input1, input2,
        index1=index1,
        index2=index2,
        out=out,
        block_size_n=block_size_n,
        block_size_c1=block_size_c1,
        block_size_c2=block_size_c2,
        num_stages=1  # 外积内存压力大，减少流水线阶段
    )

def indexed_inner(
    input1,
    input2,
    index1=None,
    index2=None,
    out=None,
    block_size_n=32,  # 内积计算密集，增大块大小
    block_size_c=32,
    loop_unroll_factor=4,  # 显式循环展开
):
    r"""带索引的批量内积运算"""
    assert index1 is not None or index2 is not None
    
    return indexed_inner_scale_gather(
        input1, input2,
        index1=index1,
        index2=index2,
        out=out,
        block_size_n=block_size_n,
        block_size_c=block_size_c,  # 固定通道分块大小
        num_stages=2,
        loop_unroll_factor=loop_unroll_factor
    )

def indexed_vecmat(
    input1,  # [N, M1, C_in]
    input2,  # [N, M2, C_in, C_out]
    index1=None,
    index2=None,
    out=None,
    block_size_n=16,  # 矩阵乘法需要更高并行
    block_size_c_out=32,  # 输出通道分块
    block_size_c_in=32,  # 输入通道分块
):
    r"""带索引的向量-矩阵乘法"""
    assert index1 is not None or index2 is not None
    
    return indexed_vecmat_scale_gather(
        input1, input2,
        index1=index1,
        index2=index2,
        out=out,
        block_size_n=block_size_n,
        block_size_c_out=block_size_c_out,
        block_size_c_in=block_size_c_in,
        num_stages=1,  # 减少寄存器压力
        loop_unroll_factor=4
    )

def indexed_vecsca(
    input1,  # [N, M1, C]
    input2,  # [N, M2] 或 [M2] 或 [N, M2, 1]
    index1=None,
    index2=None,
    out=None,
    block_size_n=32,
    block_size_c=32,
):
    r"""带索引的向量缩放运算"""
    assert index1 is not None or index2 is not None
    
    return indexed_vecsca_scale_gather(
        input1, input2,
        index1=index1,
        index2=index2,
        out=out,
        block_size_n=block_size_n,
        block_size_c=block_size_c,
        num_stages=2  # 缩放计算简单，可用更多流水线
    )