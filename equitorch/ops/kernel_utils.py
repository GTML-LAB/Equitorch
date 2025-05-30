import triton
import triton.language as tl

@triton.jit
def indexed_ptr(
    base_ptr,
    index_ptr,
    index_offset,
    n_offsets,
    stride_n,
    stride_idx,
    SHARED: tl.constexpr,
):
    if index_ptr is not None: 
        index = tl.load(index_ptr + index_offset)
    else:
        index = index_offset
    if not SHARED:
        rows_base = base_ptr + index * stride_idx + n_offsets * stride_n
    else:
        rows_base = base_ptr + index * stride_idx
    return rows_base


@triton.jit
def load_block_scalar(
    base_offsets,
    base_mask,
    SHARED: tl.constexpr,
):
    # 直接使用基础偏移作为指针地址
    ptr = base_offsets
    # 根据 SHARED 标志确定掩码
    if SHARED:
        mask = True  # 共享内存数据无需额外掩码
    else:
        mask = base_mask
    # 加载标量数据并返回
    return tl.load(ptr, mask=mask, other=0.0)

@triton.jit
def load_block_vec(
    base_offsets,
    base_mask,
    c_offsets,
    c_mask,
    stride,
    SHARED: tl.constexpr,
):
    # 计算指针地址
    ptr = base_offsets.expand_dims(-1) + c_offsets * stride
    # 根据SHARED标志确定掩码
    if SHARED:
        mask = c_mask
    else:
        mask = c_mask & base_mask.expand_dims(-1)
    # 加载数据并返回
    return tl.load(ptr, mask=mask, other=0.0)

@triton.jit
def load_block_mat(
    base_offsets,
    base_mask,
    c_in_offsets,
    c_in_mask,
    c_out_offsets,
    c_out_mask,
    stride_c_in,
    stride_c_out,
    SHARED: tl.constexpr,
):
    # 计算二维偏移量：输入通道和输出通道的组合
    offsets = c_in_offsets[:, None] * stride_c_in + c_out_offsets[None, :] * stride_c_out
    ptrs = base_offsets.expand_dims((-1,-2)) + offsets
    # 组合输入和输出通道的掩码
    mask = c_in_mask[:, None] & c_out_mask[None, :]
    # 根据SHARED标志添加基掩码
    if not SHARED:
        mask = mask & base_mask.expand_dims((-1,-2))
    # 加载矩阵数据并返回
    return tl.load(ptrs, mask=mask, other=0.0)



@triton.jit
def store_block_scalar(
    value,
    base_offsets,
    base_mask,
    ACCUMULATED: tl.constexpr,
):
    # 直接使用基础偏移作为指针地址
    ptr = base_offsets
    # 根据 SHARED 标志确定掩码
    if ACCUMULATED:
        tl.atomic_add(ptr, value)
    else:
        mask = base_mask
        tl.store(ptr, value, mask)



@triton.jit
def store_block_indexed_vec(
    value,
    base_ptr,
    index_ptr,
    index_offset,
    n_offsets,
    stride_n,
    stride_idx,
    c_offsets,
    base_mask,
    c_mask,
    stride_c,
    ACCUMULATED: tl.constexpr,
):
    if index_ptr is not None: 
        index = tl.load(index_ptr + index_offset)
    else:
        index = index_offset
    
    if not ACCUMULATED:
        rows_base = base_ptr + n_offsets * stride_n + index * stride_idx
    else:
        rows_base = base_ptr + index * stride_idx
    
    # 计算存储指针并执行存储操作（原store_block_vec的功能）
    ptr = rows_base[:,None] + c_offsets[None,:] * stride_c
    
    if ACCUMULATED:
        mask = c_mask
        tl.atomic_add(ptr, value, mask)
    else:
        mask = c_mask & base_mask.expand_dims(-1)
        tl.store(ptr, value, mask)


@triton.jit
def store_block_vec(
    value,
    base_offsets,
    base_mask,
    c_offsets,
    c_mask,
    stride,
    ACCUMULATED: tl.constexpr,
):
    ptr = base_offsets.expand_dims(-1) + c_offsets * stride
    if ACCUMULATED:
        mask = c_mask
        tl.atomic_add(ptr, value, mask)
    else:
        mask = c_mask & base_mask.expand_dims(-1)
        tl.store(ptr, value, mask)

@triton.jit
def store_block_mat(
    value,
    base_offsets,
    base_mask,
    c_in_offsets,
    c_in_mask,
    c_out_offsets,
    c_out_mask,
    stride_c_in,
    stride_c_out,
    ACCUMULATED: tl.constexpr,
):
    # 计算二维偏移量：输入通道和输出通道的组合
    offsets = c_in_offsets[:, None] * stride_c_in + c_out_offsets[None, :] * stride_c_out
    ptrs = base_offsets.expand_dims((-1,-2)) + offsets
    # 组合输入和输出通道的掩码
    mask = c_in_mask[:, None] & c_out_mask[None, :]
    if ACCUMULATED:
        tl.atomic_add(ptrs, value, mask)
    else:
        mask = mask & base_mask.expand_dims((-1,-2))
        tl.store(ptrs, value, mask)




@triton.jit
def block2_ptr(
    base_ptr,
    n_offsets, stride_n,
    idx, stride_idx,
    c_offsets, stride_c,
):
    return base_ptr + idx * stride_idx + n_offsets[:,None] * stride_n + c_offsets[None,:] * stride_c

@triton.jit
def block2_outer_ptr(
    base_ptr,
    n_offsets, stride_n,
    idx, stride_idx,
    c1_offsets, stride_c1,
    c2_offsets, stride_c2,
):
    return (base_ptr + idx * stride_idx 
            + n_offsets[:,None,None] * stride_n 
            + c1_offsets[None,:,None] * stride_c1
            + c2_offsets[None,None,:] * stride_c2
    )
