from .batched_sparse_dense_op import (
    indexed_mul_scale_gather,
    indexed_inner_scale_gather,
    indexed_outer_scale_gather,
    indexed_vecmat_scale_gather,
    indexed_vecsca_scale_gather
)



def mul_segment(
    input1, 
    input2, 
    seg, 
    out=None,
    block_size_n=64, 
    block_size_c=64, 
    num_stages=2,
    accumulated=False
):
    r"""分段聚集的逐元素乘法（必须提供seg且至少一个索引）"""
    
    return indexed_mul_scale_gather(
        input1, input2,
        seg=seg,
        out=out,
        block_size_n=block_size_n,
        block_size_c=block_size_c,
        num_stages=num_stages,
        out_accumulated=accumulated
    )

def outer_segment(
    input1, 
    input2, 
    seg, 
    out=None,
    block_size_n=8, 
    block_size_c1=32, 
    block_size_c2=32, 
    num_stages=1,
    accumulated=False
):
    r"""分段聚集的逐元素外积（必须提供seg且至少一个索引）"""
    
    return indexed_outer_scale_gather(
        input1, input2,
        seg=seg,
        out=out,
        block_size_n=block_size_n,
        block_size_c1=block_size_c1,
        block_size_c2=block_size_c2,
        num_stages=num_stages,
        out_accumulated=accumulated
    )

def inner_segment(
    input1, 
    input2, 
    seg, 
    out=None,
    block_size_n=32, 
    block_size_c=32, 
    num_stages=2,
    accumulated=False
):
    r"""分段聚集的逐元素内积（必须提供seg且至少一个索引）"""
    
    return indexed_inner_scale_gather(
        input1, input2,
        seg=seg,
        out=out,
        block_size_n=block_size_n,
        block_size_c=block_size_c,
        num_stages=num_stages,
        accumulated=accumulated
    )

def vecmat_segment(
    input1, 
    input2, 
    seg, 
    out=None,
    block_size_n=8, 
    block_size_c_in=32, 
    block_size_c_out=32, 
    num_stages=1,
    accumulated=False
):
    r"""分段聚集的逐元素向量-矩阵乘（必须提供seg且至少一个索引）"""
    
    return indexed_vecmat_scale_gather(
        input1, input2,
        seg=seg,
        out=out,
        block_size_n=block_size_n,
        block_size_c_in=block_size_c_in,
        block_size_c_out=block_size_c_out,
        num_stages=num_stages,
        accumulated=accumulated
    )

def vecsca_segment(
    input1, 
    input2, 
    seg, 
    out=None,
    block_size_n=64, 
    block_size_c=64, 
    num_stages=2,
    accumulated=False
):
    r"""分段聚集的逐元素向量缩放（必须提供seg且至少一个索引）"""
    
    return indexed_vecsca_scale_gather(
        input1, input2,
        seg=seg,
        out=out,
        block_size_n=block_size_n,
        block_size_c=block_size_c,
        num_stages=num_stages,
        accumulated=accumulated
    )

