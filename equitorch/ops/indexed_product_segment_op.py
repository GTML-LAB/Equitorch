from .batched_sparse_dense_op import (
    indexed_mul_scale_gather,
    indexed_inner_scale_gather,
    indexed_outer_scale_gather,
    indexed_vecmat_scale_gather,
    indexed_vecsca_scale_gather
)



def indexed_mul_segment(
    input1, 
    input2, 
    index1=None, 
    index2=None, 
    seg=None, 
    out=None,
    block_size_n=64, 
    block_size_c=64, 
    num_stages=2
):
    r"""分段聚集的索引乘法（必须提供seg且至少一个索引）"""
    assert seg is not None, "seg cannot be None for segment operations"
    assert index1 is not None or index2 is not None, "At least one of index1 or index2 must be provided"
    
    return indexed_mul_scale_gather(
        input1, input2,
        index1=index1,
        index2=index2,
        seg=seg,
        out=out,
        block_size_n=block_size_n,
        block_size_c=block_size_c,
        num_stages=num_stages
    )

def indexed_outer_segment(
    input1, 
    input2, 
    index1=None, 
    index2=None, 
    seg=None, 
    out=None,
    accumulated=False,
    block_size_n=8, 
    block_size_c1=32, 
    block_size_c2=32, 
    num_stages=1, 
):
    r"""分段聚集的索引外积（必须提供seg且至少一个索引）"""
    assert seg is not None, "seg cannot be None for segment operations"
    assert index1 is not None or index2 is not None, "At least one of index1 or index2 must be provided"
    
    return indexed_outer_scale_gather(
        input1, input2,
        index1=index1,
        index2=index2,
        seg=seg,
        out=out,
        block_size_n=block_size_n,
        block_size_c1=block_size_c1,
        block_size_c2=block_size_c2,
        num_stages=num_stages,
        out_accumulated=accumulated
    )

def indexed_inner_segment(
    input1, 
    input2, 
    index1=None, 
    index2=None, 
    seg=None, 
    out=None,
    block_size_n=32, 
    block_size_c=32, 
    num_stages=2
):
    r"""分段聚集的索引内积（必须提供seg且至少一个索引）"""
    assert seg is not None, "seg cannot be None for segment operations"
    assert index1 is not None or index2 is not None, "At least one of index1 or index2 must be provided"
    
    return indexed_inner_scale_gather(
        input1, input2,
        index1=index1,
        index2=index2,
        seg=seg,
        out=out,
        block_size_n=block_size_n,
        block_size_c=block_size_c,
        num_stages=num_stages
    )

def indexed_vecmat_segment(
    input1, 
    input2, 
    index1=None, 
    index2=None, 
    seg=None, 
    out=None,
    block_size_n=8, 
    block_size_c_in=32, 
    block_size_c_out=32, 
    num_stages=1
):
    r"""分段聚集的索引向量-矩阵乘（必须提供seg且至少一个索引）"""
    assert seg is not None, "seg cannot be None for segment operations"
    assert index1 is not None or index2 is not None, "At least one of index1 or index2 must be provided"
    
    return indexed_vecmat_scale_gather(
        input1, input2,
        index1=index1,
        index2=index2,
        seg=seg,
        out=out,
        block_size_n=block_size_n,
        block_size_c_in=block_size_c_in,
        block_size_c_out=block_size_c_out,
        num_stages=num_stages
    )

def indexed_vecsca_segment(
    input1, 
    input2, 
    index1=None, 
    index2=None, 
    seg=None, 
    out=None,
    block_size_n=64, 
    block_size_c=64, 
    num_stages=2
):
    r"""分段聚集的索引向量缩放（必须提供seg且至少一个索引）"""
    assert seg is not None, "seg cannot be None for segment operations"
    assert index1 is not None or index2 is not None, "At least one of index1 or index2 must be provided"
    
    return indexed_vecsca_scale_gather(
        input1, input2,
        index1=index1,
        index2=index2,
        seg=seg,
        out=out,
        block_size_n=block_size_n,
        block_size_c=block_size_c,
        num_stages=num_stages
    )

