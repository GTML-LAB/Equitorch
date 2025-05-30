import torch

import triton
import triton.language as tl

from .kernel_dense import (
    kernel_mul,
    kernel_outer,
    kernel_inner,
    kernel_vecmat,
    kernel_vecsca
)

from .kernel_utils import (
    indexed_ptr,
    store_block_vec,
    store_block_scalar,
    store_block_mat,
)

from torch_geometric.utils import segment, scatter

@triton.jit
def batched_single_mul(
        input1_ptr, index1_ptr,
        input2_ptr, index2_ptr,        
        scale_ptr,
        sparse_idx,
        n_offsets, n_mask,
        c_offsets, c_mask,
        stride_n1, stride_idx1, stride_c1,
        stride_n2, stride_idx2, stride_c2,
        LEFT_SHARED: tl.constexpr,
        RIGHT_SHARED: tl.constexpr,
        OUT_ACCUMULATED: tl.constexpr,
        ):

        input1_bases = indexed_ptr(
            input1_ptr, index1_ptr, 
            sparse_idx, n_offsets, stride_n1 if not LEFT_SHARED else 0,
            stride_idx1, LEFT_SHARED) 

        input2_bases = indexed_ptr(
            input2_ptr, index2_ptr, 
            sparse_idx, n_offsets, stride_n2 if not RIGHT_SHARED else 0,
            stride_idx2, RIGHT_SHARED) 
    
        product = kernel_mul(
            input1_bases, input2_bases,
            n_mask,
            c_offsets, c_mask, 
            stride_c1, stride_c2,
            LEFT_SHARED, RIGHT_SHARED,
            OUT_ACCUMULATED)

        if scale_ptr is not None: # Check None directly
            scale = tl.load(scale_ptr + sparse_idx)
            product = product * scale

        return product


@triton.jit
def indexed_mul_scale_gather_kernel(
    input1_ptr, input2_ptr, scale_ptr,
    index1_ptr, index2_ptr,
    seg_ptr, gather_index_ptr, index_out_ptr,
    output_ptr,
    size_n, size_c,
    stride_n1, stride_idx1, stride_c1,
    stride_n2, stride_idx2, stride_c2,
    stride_n_out, stride_idx_out, stride_c_out,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    LEFT_SHARED: tl.constexpr,
    RIGHT_SHARED: tl.constexpr,
    OUT_ACCUMULATED: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    
    pid_n = tl.program_id(0)
    pid_seg = tl.program_id(1)
    pid_c = tl.program_id(2)
    
    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    n_mask = n_offsets < size_n

    c_offsets = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    c_mask = c_offsets < size_c
    
    if seg_ptr is not None: # Check None for OUT_GATHERED

        if not OUT_ACCUMULATED:
            accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_C) , dtype=output_ptr.dtype.element_ty)
        else:
            accumulator = tl.zeros((BLOCK_SIZE_C,), dtype=output_ptr.dtype.element_ty)

        
        loop_start = tl.load(seg_ptr + pid_seg)
        loop_end = tl.load(seg_ptr + pid_seg + 1)
        for loop_idx in tl.range(loop_start, loop_end, num_stages=NUM_STAGES):

            if gather_index_ptr is None: # Check None for OUT_SEGMENTED
                sparse_idx = loop_idx  
            else:
                sparse_idx = tl.load(gather_index_ptr+loop_idx)
                
            product = batched_single_mul(
                input1_ptr, index1_ptr, 
                input2_ptr, index2_ptr, 
                scale_ptr, 
                sparse_idx,
                n_offsets, n_mask,
                c_offsets, c_mask,
                stride_n1 if not LEFT_SHARED else 0, stride_idx1, stride_c1,
                stride_n2 if not RIGHT_SHARED else 0, stride_idx2, stride_c2,
                LEFT_SHARED, RIGHT_SHARED, 
                OUT_ACCUMULATED) 
            
            accumulator += product 
    else: # Not OUT_GATHERED (seg_ptr is None)
        sparse_idx = pid_seg
    
        accumulator = batched_single_mul(
                input1_ptr, index1_ptr, 
                input2_ptr, index2_ptr, 
                scale_ptr, 
                sparse_idx,
                n_offsets, n_mask,
                c_offsets, c_mask,
                stride_n1 if not LEFT_SHARED else 0, stride_idx1, stride_c1,
                stride_n2 if not RIGHT_SHARED else 0, stride_idx2, stride_c2,
                LEFT_SHARED, RIGHT_SHARED, 
                OUT_ACCUMULATED) 

    output_bases = indexed_ptr(output_ptr, 
                               index_out_ptr, 
                               pid_seg, 
                               n_offsets,
                               stride_n_out if not OUT_ACCUMULATED else 0,
                               stride_idx_out,
                               OUT_ACCUMULATED) 
    
    store_block_vec(
        accumulator,
        output_bases,
        n_mask,
        c_offsets,
        c_mask,
        stride_c_out,
        OUT_ACCUMULATED
    )



def indexed_mul_scale_gather_gpu(
        input1, input2, 
        scale=None, index1=None, index2=None,
        seg=None, gather_index=None, 
        index_out=None, out=None,
        out_accumulated=False,
        out_size=None,
        block_size_n=64, block_size_c=64,
        num_stages=2):
    scaled = scale is not None
    left_indexed = index1 is not None
    right_indexed = index2 is not None

    if input1.ndim==2:
        left_shared = True
        input1 = input1.unsqueeze(0)
    else:
        left_shared = False
    if input2.ndim==2:
        right_shared = True
        input2 = input2.unsqueeze(0)
    else:
        right_shared = False

    out_gathered = seg is not None
    out_segmented = gather_index is None
    out_indexed = index_out is not None

    sparse_grid_size = (seg.shape[0] - 1) if out_gathered else index1.shape[0] if left_indexed else index2.shape[0] if right_indexed else input1.shape[-2]

    if out_size is None:
        out_size = sparse_grid_size

    size_n = max(input1.shape[0], input2.shape[0])

    if out is None:
        if out_accumulated:
            out = torch.zeros(out_size, input1.shape[-1], dtype=torch.result_type(input1, input2), device=input1.device)
        else:
            if out_indexed:
                out = torch.zeros(size_n, out_size, input1.shape[-1], dtype=torch.result_type(input1, input2), device=input1.device)
            else:
                out = torch.empty(size_n, out_size, input1.shape[-1], dtype=torch.result_type(input1, input2), device=input1.device)
    else:
        out = out.contiguous()

    if out_accumulated:
        out = out.unsqueeze(0)

    # Ensure inputs are contiguous
    input1 = input1.contiguous()
    input2 = input2.contiguous()


    grid = lambda meta: (
        triton.cdiv(size_n, meta['BLOCK_SIZE_N']),
        sparse_grid_size,
        triton.cdiv(input1.shape[2], meta['BLOCK_SIZE_C']),
    )

    indexed_mul_scale_gather_kernel[grid](
        input1, input2, scale,
        index1, index2,
        seg, gather_index, index_out, out, 
        size_n, input1.shape[-1],
        *input1.stride(), *input2.stride(),
        *out.stride(),
        block_size_n, block_size_c,
        left_shared, right_shared,
        out_accumulated,
        num_stages
    )

    if out_accumulated:
        out = out.squeeze(0)

    return out
    
def indexed_mul_scale_gather_cpu(
        input1, input2, 
        scale=None, index1=None, index2=None,
        seg=None, gather_index=None, 
        index_out=None, out=None,
        out_accumulated=False,
        out_size=None):
    r"""
    CPU implementation of indexed_mul_scale_gather.
    """
    # Handle input indexing
    if index1 is not None:
        input1 = input1.index_select(-2, index1)
    if index2 is not None:
        input2 = input2.index_select(-2, index2)
    
    # Core multiplication
    inter = input1 * input2
    
    # Apply scaling if provided
    if scale is not None:
        inter = inter * scale.unsqueeze(-1)
    
    # Handle segmentation/gathering
    if seg is not None:
        if gather_index is not None:
            # Gather then segment
            gathered = inter.index_select(-2, gather_index)
            inter = segment(gathered, seg.unsqueeze(0))
        else:
            # Direct segment
            inter = segment(inter, seg.unsqueeze(0))
    
    # Handle output indexing
    if index_out is not None:
        inter = scatter(inter, index_out, dim=-2, dim_size=out_size)
    
    # Handle accumulation
    if out_accumulated:
        inter = inter.sum(dim=0)
    
    return inter

def indexed_mul_scale_gather(
        input1, input2, 
        scale=None, index1=None, index2=None,
        seg=None, gather_index=None, 
        index_out=None, out=None,
        out_accumulated=False,
        out_size=None,
        block_size_n=64, block_size_c=64,
        num_stages=2):
    r"""
    Dispatches to GPU or CPU implementation based on input device.
    """
    if input1.device.type == 'cuda':
        return indexed_mul_scale_gather_gpu(
            input1, input2, scale, index1, index2,
            seg, gather_index, index_out, out,
            out_accumulated, out_size,
            block_size_n, block_size_c, num_stages)
    else:
        return indexed_mul_scale_gather_cpu(
            input1, input2, scale, index1, index2,
            seg, gather_index, index_out, out,
            out_accumulated, out_size)


@triton.jit
def batched_single_outer(
        input1_ptr, index1_ptr,
        input2_ptr, index2_ptr,        
        scale_ptr,
        sparse_idx,
        n_offsets, n_mask,
        c1_offsets, c1_mask,
        c2_offsets, c2_mask,
        stride_n1, stride_idx1, stride_c1,
        stride_n2, stride_idx2, stride_c2,
        LEFT_SHARED: tl.constexpr,
        RIGHT_SHARED: tl.constexpr,
        OUT_ACCUMULATED: tl.constexpr,
        # SCALED: tl.constexpr # Removed
        ):

        input1_bases = indexed_ptr(
            input1_ptr, index1_ptr, 
            sparse_idx, n_offsets, stride_n1 if not LEFT_SHARED else 0,
            stride_idx1, LEFT_SHARED) 

        input2_bases = indexed_ptr(
            input2_ptr, index2_ptr, 
            sparse_idx, n_offsets, stride_n2 if not RIGHT_SHARED else 0,
            stride_idx2, RIGHT_SHARED) 
    
        product = kernel_outer(
            input1_bases, input2_bases,
            n_mask,
            c1_offsets, c2_offsets, c1_mask, c2_mask,
            stride_c1, stride_c2,
            LEFT_SHARED, RIGHT_SHARED,
            OUT_ACCUMULATED)

        if scale_ptr is not None: # Check None directly
            scale = tl.load(scale_ptr + sparse_idx)
            product = product * scale

        return product

@triton.jit
def indexed_outer_scale_gather_kernel(
    input1_ptr, input2_ptr, scale_ptr,
    index1_ptr, index2_ptr,
    seg_ptr, gather_index_ptr, index_out_ptr,
    output_ptr,
    size_n, size_c1, size_c2,
    stride_n1, stride_idx1, stride_c1,
    stride_n2, stride_idx2, stride_c2,
    stride_n_out, stride_idx_out, 
    stride_c1_out, stride_c2_out,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C1: tl.constexpr,
    BLOCK_SIZE_C2: tl.constexpr,
    LEFT_SHARED: tl.constexpr,
    RIGHT_SHARED: tl.constexpr,
    OUT_ACCUMULATED: tl.constexpr,
    NUM_STAGES: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_seg = tl.program_id(1)
    pid_c = tl.program_id(2)
    
    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    n_mask = n_offsets < size_n

    c1_offsets = (pid_c // tl.cdiv(size_c2, BLOCK_SIZE_C2)) * BLOCK_SIZE_C1 + tl.arange(0, BLOCK_SIZE_C1)
    c1_mask = c1_offsets < size_c1
    c2_offsets = (pid_c % tl.cdiv(size_c2, BLOCK_SIZE_C2)) * BLOCK_SIZE_C2 + tl.arange(0, BLOCK_SIZE_C2)
    c2_mask = c2_offsets < size_c2
    
    if seg_ptr is not None: # Check None for OUT_GATHERED
        if not OUT_ACCUMULATED:
            accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_C1, BLOCK_SIZE_C2), dtype=output_ptr.dtype.element_ty)
        else:
            accumulator = tl.zeros((BLOCK_SIZE_C1, BLOCK_SIZE_C2), dtype=output_ptr.dtype.element_ty)

        loop_start = tl.load(seg_ptr + pid_seg)
        loop_end = tl.load(seg_ptr + pid_seg + 1)
        for loop_idx in tl.range(loop_start, loop_end, num_stages=NUM_STAGES):
            if gather_index_ptr is None: # Check None for OUT_SEGMENTED
                sparse_idx = loop_idx  
            else:
                sparse_idx = tl.load(gather_index_ptr+loop_idx)
                
            product = batched_single_outer(
                input1_ptr, index1_ptr, 
                input2_ptr, index2_ptr, 
                scale_ptr, 
                sparse_idx,
                n_offsets, n_mask,
                c1_offsets, c1_mask,
                c2_offsets, c2_mask,
                stride_n1 if not LEFT_SHARED else 0, stride_idx1, stride_c1,
                stride_n2 if not RIGHT_SHARED else 0, stride_idx2, stride_c2,
                LEFT_SHARED, RIGHT_SHARED, 
                OUT_ACCUMULATED) 
            
            accumulator += product 
    else: # Not OUT_GATHERED (seg_ptr is None)
        sparse_idx = pid_seg
    
        accumulator = batched_single_outer(
                input1_ptr, index1_ptr, 
                input2_ptr, index2_ptr, 
                scale_ptr, 
                sparse_idx,
                n_offsets, n_mask,
                c1_offsets, c1_mask,
                c2_offsets, c2_mask,
                stride_n1 if not LEFT_SHARED else 0, stride_idx1, stride_c1,
                stride_n2 if not RIGHT_SHARED else 0, stride_idx2, stride_c2,
                LEFT_SHARED, RIGHT_SHARED, 
                OUT_ACCUMULATED) 

    output_bases = indexed_ptr(output_ptr, 
                               index_out_ptr, 
                               pid_seg, 
                               n_offsets,
                               stride_n_out if not OUT_ACCUMULATED else 0,
                               stride_idx_out,
                               OUT_ACCUMULATED) 
    
    store_block_mat(
        accumulator,
        output_bases,
        n_mask,
        c1_offsets, c1_mask,
        c2_offsets, c2_mask,
        stride_c1_out, stride_c2_out,
        OUT_ACCUMULATED
    )

def indexed_outer_scale_gather_gpu(
        input1, input2, 
        scale=None, index1=None, index2=None,
        seg=None, gather_index=None, 
        index_out=None, out=None,
        out_accumulated=False,
        out_size=None,
        block_size_n=8, block_size_c1=32, block_size_c2=32,
        num_stages=2):
    scaled = scale is not None
    left_indexed = index1 is not None
    right_indexed = index2 is not None

    if input1.ndim==2:
        left_shared = True
        input1 = input1.unsqueeze(0)
    else:
        left_shared = False
    if input2.ndim==2:
        right_shared = True
        input2 = input2.unsqueeze(0)
    else:
        right_shared = False

    out_gathered = seg is not None
    out_segmented = gather_index is None
    out_indexed = index_out is not None

    sparse_grid_size = (seg.shape[0] - 1) if out_gathered else index1.shape[0] if left_indexed else index2.shape[0] if right_indexed else input1.shape[-2]

    if out_size is None:
        out_size = sparse_grid_size

    size_n = max(input1.shape[0], input2.shape[0])

    if out is None:
        if out_accumulated:
            out = torch.zeros(out_size, input1.shape[-1], input2.shape[-1], dtype=torch.result_type(input1, input2), device=input1.device)
        else:
            if out_indexed:
                out = torch.zeros(size_n, out_size, input1.shape[-1], input2.shape[-1], dtype=torch.result_type(input1, input2), device=input1.device)
            else:
                out = torch.empty(size_n, out_size, input1.shape[-1], input2.shape[-1], dtype=torch.result_type(input1, input2), device=input1.device)
    else:
        out = out.contiguous()

    if out_accumulated:
        out = out.unsqueeze(0)

    # Ensure inputs are contiguous
    input1 = input1.contiguous()
    input2 = input2.contiguous()


    grid = lambda meta: (
        triton.cdiv(size_n, meta['BLOCK_SIZE_N']),
        sparse_grid_size,
        (triton.cdiv(input1.shape[2], meta['BLOCK_SIZE_C1']) 
         * triton.cdiv(input2.shape[2], meta['BLOCK_SIZE_C2'])),
    )

    indexed_outer_scale_gather_kernel[grid](
        input1, input2, scale,
        index1, index2,
        seg, gather_index, index_out, out, 
        size_n, input1.shape[-1], input2.shape[-1],
        *input1.stride(), *input2.stride(),
        *out.stride(),
        block_size_n, block_size_c1, block_size_c2,
        left_shared, right_shared,
        out_accumulated,
        num_stages
    )
    if out_accumulated:
        out = out.squeeze(0)

    return out
    

def indexed_outer_scale_gather_cpu(
        input1, input2, 
        scale=None, index1=None, index2=None,
        seg=None, gather_index=None, 
        index_out=None, out=None,
        out_accumulated=False,
        out_size=None):
    r"""
    CPU implementation of indexed_outer_scale_gather.
    """
    # Handle input indexing
    if index1 is not None:
        input1 = input1.index_select(-2, index1)
    if index2 is not None:
        input2 = input2.index_select(-2, index2)
    
    # Core outer product
    inter = input1.unsqueeze(-1) * input2.unsqueeze(-2)
    
    # Apply scaling if provided
    if scale is not None:
        inter = inter * scale.view(1, -1, 1, 1)
    
    # Handle segmentation/gathering
    if seg is not None:
        if gather_index is not None:
            # Gather then segment
            gathered = inter.index_select(-3, gather_index)
            inter = segment(gathered, seg.unsqueeze(0))
        else:
            # Direct segment
            inter = segment(inter, seg.unsqueeze(0))
    
    # Handle output indexing
    if index_out is not None:
        inter = scatter(inter, index_out, dim=-3, dim_size=out_size)
    
    # Handle accumulation
    if out_accumulated:
        inter = inter.sum(dim=0)
    
    return inter
def indexed_outer_scale_gather(
        input1, input2, 
        scale=None, index1=None, index2=None,
        seg=None, gather_index=None, 
        index_out=None, out=None,
        out_accumulated=False,
        out_size=None,
        block_size_n=8, block_size_c1=32, block_size_c2=32,
        num_stages=2):
    r"""
    Dispatches to GPU or CPU implementation based on input device.
    """
    if input1.device.type == 'cuda':
        return indexed_outer_scale_gather_gpu(
            input1, input2, scale, index1, index2,
            seg, gather_index, index_out, out,
            out_accumulated, out_size,
            block_size_n, block_size_c1, block_size_c2, num_stages)
    else:
        return indexed_outer_scale_gather_cpu(
            input1, input2, scale, index1, index2,
            seg, gather_index, index_out, out,
            out_accumulated, out_size)


@triton.jit
def batched_single_inner(
        input1_ptr, index1_ptr,
        input2_ptr, index2_ptr,        
        scale_ptr,
        sparse_idx,
        n_offsets, n_mask,
        c_in,
        stride_n1, stride_idx1, stride_c1,
        stride_n2, stride_idx2, stride_c2,
        LEFT_SHARED: tl.constexpr,
        RIGHT_SHARED: tl.constexpr,
        OUT_ACCUMULATED: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_C: tl.constexpr,
        NUM_STAGES: tl.constexpr,
        LOOP_UNROLL_FACTOR: tl.constexpr,
        OUT_DTYPE: tl.constexpr
        ):

        input1_bases = indexed_ptr(
            input1_ptr, index1_ptr, 
            sparse_idx, n_offsets, stride_n1 if not LEFT_SHARED else 0,
            stride_idx1, LEFT_SHARED) 

        input2_bases = indexed_ptr(
            input2_ptr, index2_ptr, 
            sparse_idx, n_offsets, stride_n2 if not RIGHT_SHARED else 0,
            stride_idx2, RIGHT_SHARED) 
    
        product = kernel_inner(
            input1_bases, input2_bases,
            n_mask,
            c_in,
            stride_c1, stride_c2,
            BLOCK_SIZE_N, BLOCK_SIZE_C,
            LEFT_SHARED, RIGHT_SHARED,
            OUT_ACCUMULATED,
            NUM_STAGES, LOOP_UNROLL_FACTOR, OUT_DTYPE)

        if scale_ptr is not None: # Check None directly
            scale = tl.load(scale_ptr + sparse_idx)
            product = product * scale

        return product

@triton.jit
def indexed_inner_scale_gather_kernel(
    input1_ptr, input2_ptr, scale_ptr,
    index1_ptr, index2_ptr,
    seg_ptr, gather_index_ptr, index_out_ptr,
    output_ptr,
    size_n, size_c,
    stride_n1, stride_idx1, stride_c1,
    stride_n2, stride_idx2, stride_c2,
    stride_n_out, stride_idx_out,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    LEFT_SHARED: tl.constexpr,
    RIGHT_SHARED: tl.constexpr,
    OUT_ACCUMULATED: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    LOOP_UNROLL_FACTOR: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_seg = tl.program_id(1)
    
    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    n_mask = n_offsets < size_n
    
    if seg_ptr is not None: # Check None for OUT_GATHERED
        if not OUT_ACCUMULATED:
            accumulator = tl.zeros((BLOCK_SIZE_N,), dtype=output_ptr.dtype.element_ty)
        else:
            accumulator = tl.zeros((), dtype=output_ptr.dtype.element_ty)

        loop_start = tl.load(seg_ptr + pid_seg)
        loop_end = tl.load(seg_ptr + pid_seg + 1)
        for loop_idx in tl.range(loop_start, loop_end, num_stages=NUM_STAGES):
            if gather_index_ptr is None: # Check None for OUT_SEGMENTED
                sparse_idx = loop_idx  
            else:
                sparse_idx = tl.load(gather_index_ptr+loop_idx)
                
            product = batched_single_inner(
                input1_ptr, index1_ptr, 
                input2_ptr, index2_ptr, 
                scale_ptr, 
                sparse_idx,
                n_offsets, n_mask,
                size_c,
                stride_n1 if not LEFT_SHARED else 0, stride_idx1, stride_c1,
                stride_n2 if not RIGHT_SHARED else 0, stride_idx2, stride_c2,
                LEFT_SHARED, RIGHT_SHARED, 
                OUT_ACCUMULATED, 
                BLOCK_SIZE_N, BLOCK_SIZE_C,
                NUM_STAGES, LOOP_UNROLL_FACTOR, output_ptr.dtype.element_ty)
            
            accumulator += product 
    else: # Not OUT_GATHERED (seg_ptr is None)
        sparse_idx = pid_seg
    
        accumulator = batched_single_inner(
                input1_ptr, index1_ptr, 
                input2_ptr, index2_ptr, 
                scale_ptr, 
                sparse_idx,
                n_offsets, n_mask,
                size_c,
                stride_n1 if not LEFT_SHARED else 0, stride_idx1, stride_c1,
                stride_n2 if not RIGHT_SHARED else 0, stride_idx2, stride_c2,
                LEFT_SHARED, RIGHT_SHARED, 
                OUT_ACCUMULATED, 
                BLOCK_SIZE_N, BLOCK_SIZE_C,
                NUM_STAGES, LOOP_UNROLL_FACTOR, output_ptr.dtype.element_ty)

    output_bases = indexed_ptr(output_ptr, 
                               index_out_ptr, 
                               pid_seg, 
                               n_offsets,
                               stride_n_out if not OUT_ACCUMULATED else 0,
                               stride_idx_out,
                               OUT_ACCUMULATED) 
    
    store_block_scalar(
        accumulator,
        output_bases,
        n_mask,
        OUT_ACCUMULATED
    )

def indexed_inner_scale_gather_gpu(
        input1, input2, 
        scale=None, index1=None, index2=None,
        seg=None, gather_index=None, 
        index_out=None, out=None,
        out_accumulated=False,
        out_size=None,
        block_size_n=32, block_size_c=32,
        num_stages=2, loop_unroll_factor=4):
    scaled = scale is not None
    left_indexed = index1 is not None
    right_indexed = index2 is not None

    if input1.ndim==2:
        left_shared = True
        input1 = input1.unsqueeze(0)
    else:
        left_shared = False
    if input2.ndim==2:
        right_shared = True
        input2 = input2.unsqueeze(0)
    else:
        right_shared = False

    out_gathered = seg is not None
    out_segmented = gather_index is None
    out_indexed = index_out is not None

    sparse_grid_size = (seg.shape[0] - 1) if out_gathered else index1.shape[0] if left_indexed else index2.shape[0] if right_indexed else input1.shape[-2]

    if out_size is None:
        out_size = sparse_grid_size

    size_n = max(input1.shape[0], input2.shape[0])

    if out is None:
        if out_accumulated:
            out = torch.zeros(out_size, dtype=torch.result_type(input1, input2), device=input1.device)
        else:
            if out_indexed:
                out = torch.zeros(size_n, out_size, dtype=torch.result_type(input1, input2), device=input1.device)
            else:
                out = torch.empty(size_n, out_size, dtype=torch.result_type(input1, input2), device=input1.device)
    else:
        out = out.contiguous()

    if out_accumulated:
        out = out.unsqueeze(0)

    # Ensure inputs are contiguous
    input1 = input1.contiguous()
    input2 = input2.contiguous()


    grid = lambda meta: (
        triton.cdiv(size_n, meta['BLOCK_SIZE_N']),
        sparse_grid_size,
    )

    indexed_inner_scale_gather_kernel[grid](
        input1, input2, scale,
        index1, index2,
        seg, gather_index, index_out, out, 
        size_n, input1.shape[-1],
        *input1.stride(), *input2.stride(),
        *out.stride(),
        block_size_n, block_size_c,
        left_shared, right_shared,
        out_accumulated,
        num_stages,
        loop_unroll_factor
    )
    if out_accumulated:
        out = out.squeeze(0)

    return out
    
def indexed_inner_scale_gather_cpu(
        input1, input2, 
        scale=None, index1=None, index2=None,
        seg=None, gather_index=None, 
        index_out=None, out=None,
        out_accumulated=False,
        out_size=None):
    r"""
    CPU implementation of indexed_inner_scale_gather.
    """
    # Handle input indexing
    if index1 is not None:
        input1 = input1.index_select(-2, index1)
    if index2 is not None:
        input2 = input2.index_select(-2, index2)
    
    # Core inner product (element-wise mul + sum over C)
    inter = (input1 * input2).sum(dim=-1)
    
    # Apply scaling if provided
    if scale is not None:
        inter = inter * scale # scale is (T,)
    
    # Handle segmentation/gathering
    if seg is not None:
        if gather_index is not None:
            # Gather then segment
            gathered = inter.index_select(-1, gather_index) # inter is (N, T) or (T,)
            inter = segment(gathered, seg.unsqueeze(0))
        else:
            # Direct segment
            inter = segment(inter, seg.unsqueeze(0))
    
    # Handle output indexing
    if index_out is not None:
        inter = scatter(inter, index_out, dim=-1, dim_size=out_size)
    
    # Handle accumulation
    if out_accumulated:
        inter = inter.sum(dim=0)
    
    return inter

def indexed_inner_scale_gather(
        input1, input2, 
        scale=None, index1=None, index2=None,
        seg=None, gather_index=None, 
        index_out=None, out=None,
        out_accumulated=False,
        out_size=None,
        block_size_n=32, block_size_c=32,
        num_stages=2, loop_unroll_factor=4):
    r"""
    Dispatches to GPU or CPU implementation based on input device.
    """
    if input1.device.type == 'cuda':
        return indexed_inner_scale_gather_gpu(
            input1, input2, scale, index1, index2,
            seg, gather_index, index_out, out,
            out_accumulated, out_size,
            block_size_n, block_size_c, num_stages, loop_unroll_factor)
    else:
        return indexed_inner_scale_gather_cpu(
            input1, input2, scale, index1, index2,
            seg, gather_index, index_out, out,
            out_accumulated, out_size)



# ---- scavec (scalar * vector) ----

@triton.jit
def batched_single_scavec(
        input1_ptr, index1_ptr, # scalar
        input2_ptr, index2_ptr, # vector
        scale_ptr,
        sparse_idx,
        n_offsets, n_mask,
        c_offsets, c_mask, # vector channel offsets/mask
        stride_n1, stride_idx1, # scalar strides (stride_c1 is implicitly 0 or 1)
        stride_n2, stride_idx2, stride_c2, # vector strides
        LEFT_SHARED: tl.constexpr, # scalar shared
        RIGHT_SHARED: tl.constexpr, # vector shared
        OUT_ACCUMULATED: tl.constexpr,
        ):

        # Note: kernel_vecsca expects input1=vector, input2=scalar
        # So we swap the roles here

        # Load vector bases (becomes input1 for kernel_vecsca)
        vec_bases = indexed_ptr(
            input2_ptr, index2_ptr,
            sparse_idx, n_offsets, stride_n2 if not RIGHT_SHARED else 0,
            stride_idx2, RIGHT_SHARED)

        # Load scalar bases (becomes input2 for kernel_vecsca)
        scalar_bases = indexed_ptr(
            input1_ptr, index1_ptr,
            sparse_idx, n_offsets, stride_n1 if not LEFT_SHARED else 0,
            stride_idx1, LEFT_SHARED)

        product = kernel_vecsca(
            vec_bases, scalar_bases, # Call kernel_vecsca(vector, scalar)
            n_mask,
            c_offsets, c_mask,
            stride_c2, # Use vector's channel stride
            RIGHT_SHARED, # vector shared flag becomes SHARED_INPUT1
            LEFT_SHARED,  # scalar shared flag becomes SHARED_INPUT2
            OUT_ACCUMULATED)

        if scale_ptr is not None: # Apply optional external scale
            scale = tl.load(scale_ptr + sparse_idx)
            product = product * scale

        return product

@triton.jit
def indexed_scavec_scale_gather_kernel(
    input1_ptr, input2_ptr, scale_ptr, # input1=scalar, input2=vector
    index1_ptr, index2_ptr,
    seg_ptr, gather_index_ptr, index_out_ptr,
    output_ptr,
    size_n, size_c, # size_c is vector channel size
    stride_n1, stride_idx1, # scalar strides (stride_c1 assumed 0 or 1)
    stride_n2, stride_idx2, stride_c2, # vector strides
    stride_n_out, stride_idx_out, stride_c_out, # output strides (matches vector)
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr, # Block size for vector channels
    LEFT_SHARED: tl.constexpr, # scalar shared
    RIGHT_SHARED: tl.constexpr, # vector shared
    OUT_ACCUMULATED: tl.constexpr,
    NUM_STAGES: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_seg = tl.program_id(1)
    pid_c = tl.program_id(2)

    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    n_mask = n_offsets < size_n

    c_offsets = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    c_mask = c_offsets < size_c # Mask for vector channels

    if seg_ptr is not None: # Check None for OUT_GATHERED
        if not OUT_ACCUMULATED:
            # Output shape matches vector shape
            accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_C), dtype=output_ptr.dtype.element_ty)
        else:
            accumulator = tl.zeros((BLOCK_SIZE_C,), dtype=output_ptr.dtype.element_ty)

        loop_start = tl.load(seg_ptr + pid_seg)
        loop_end = tl.load(seg_ptr + pid_seg + 1)
        for loop_idx in tl.range(loop_start, loop_end, num_stages=NUM_STAGES):
            if gather_index_ptr is None: # Check None for OUT_SEGMENTED
                sparse_idx = loop_idx
            else:
                sparse_idx = tl.load(gather_index_ptr+loop_idx)

            product = batched_single_scavec(
                input1_ptr, index1_ptr, # scalar
                input2_ptr, index2_ptr, # vector
                scale_ptr,
                sparse_idx,
                n_offsets, n_mask,
                c_offsets, c_mask, # vector channel offsets/mask
                stride_n1, stride_idx1,
                stride_n2, stride_idx2, stride_c2, # vector strides
                LEFT_SHARED, RIGHT_SHARED,
                OUT_ACCUMULATED)

            accumulator += product
    else: # Not OUT_GATHERED (seg_ptr is None)
        sparse_idx = pid_seg

        accumulator = batched_single_scavec(
                input1_ptr, index1_ptr, # scalar
                input2_ptr, index2_ptr, # vector
                scale_ptr,
                sparse_idx,
                n_offsets, n_mask,
                c_offsets, c_mask, # vector channel offsets/mask
                stride_n1, stride_idx1,
                stride_n2, stride_idx2, stride_c2, # vector strides
                LEFT_SHARED, RIGHT_SHARED,
                OUT_ACCUMULATED)

    output_bases = indexed_ptr(output_ptr,
                               index_out_ptr,
                               pid_seg,
                               n_offsets,
                               stride_n_out if not OUT_ACCUMULATED else 0,
                               stride_idx_out,
                               OUT_ACCUMULATED)

    # Store as vector
    store_block_vec(
        accumulator,
        output_bases,
        n_mask,
        c_offsets, c_mask,
        stride_c_out,
        OUT_ACCUMULATED
    )

def indexed_scavec_scale_gather(
        input1, input2, # input1=scalar, input2=vector
        scale=None, index1=None, index2=None,
        seg=None, gather_index=None,
        index_out=None, out=None,
        out_accumulated=False,
        out_size=None,
        block_size_n=64, block_size_c=64,
        num_stages=2):
    r"""
    Computes indexed/gathered scalar * vector.
    input1 (scalar): Tensor of shape (N1, M1) or (M1,) or (N1, M1, 1)
    input2 (vector): Tensor of shape (N2, M2, C) or (M2, C)
    Computes indexed/gathered scalar * vector.
    input1 (scalar): Tensor of shape (N1, M1) or (M1,) or (N1, M1, 1)
    input2 (vector): Tensor of shape (N2, M2, C) or (M2, C)
    Output shape matches vector input shape (adjusted for N, M_out).

    Note: This function internally calls indexed_vecsca_scale_gather by swapping
          input1 (scalar) and input2 (vector) because the underlying kernel
          expects (vector, scalar) and the direct implementation might be buggy.
    """
    # Call the direct implementation (indexed_vecsca_scale_gather)
    # with swapped inputs and indices.
    # indexed_vecsca_scale_gather expects (vector, scalar)
    return indexed_vecsca_scale_gather(
        input1=input2,  # Pass original vector as input1
        input2=input1,  # Pass original scalar as input2
        scale=scale,
        index1=index2,  # Pass vector's index as index1
        index2=index1,  # Pass scalar's index as index2
        seg=seg,
        gather_index=gather_index,
        index_out=index_out,
        out=out,
        out_accumulated=out_accumulated,
        out_size=out_size,
        # Pass block sizes according to indexed_vecsca_scale_gather's signature
        block_size_n=block_size_n,
        block_size_c=block_size_c, # Corresponds to C of the vector (input2 here)
        num_stages=num_stages
    )


# ---- vecmat (vector @ matrix) using mat_t_vec naming ----
# Note: Despite the name containing "mat_t_vec", this function computes vec @ mat,
#       where input1 is the matrix (..., C_in, C_out) and input2 is the vector (..., C_in).
#       It reuses kernel_vecmat internally.

@triton.jit
def batched_single_mat_t_vec( # Computes vec @ mat
        input1_ptr, index1_ptr, # matrix
        input2_ptr, index2_ptr, # vector
        scale_ptr,
        sparse_idx,
        n_offsets, n_mask,
        c_out_offsets, c_out_mask, # Output channel offsets/mask (matrix C_out dim)
        c_in, # Input channel size (vector size C_in, matrix C_in dim)
        stride_n1, stride_idx1, stride_c_out1, stride_c_in1, # matrix strides
        stride_n2, stride_idx2, stride_c2, # vector strides (stride_c2 is C_in stride)
        LEFT_SHARED: tl.constexpr, # matrix shared
        RIGHT_SHARED: tl.constexpr, # vector shared
        OUT_ACCUMULATED: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_C_OUT: tl.constexpr, # Corresponds to matrix C_out dim
        BLOCK_SIZE_C_IN: tl.constexpr,  # Corresponds to vector C_in dim
        NUM_STAGES: tl.constexpr,
        LOOP_UNROLL_FACTOR: tl.constexpr,
        OUT_DTYPE: tl.constexpr
        ):

        # Note: kernel_vecmat computes vec @ mat.
        # Here, input1=matrix (..., C_in, C_out), input2=vector (..., C_in).
        # We load input2 (vector) as vec_bases (input1 for kernel_vecmat)
        # and input1 (matrix) as mat_bases (input2 for kernel_vecmat).

        # Load vector bases (input2 -> becomes input1 for kernel_vecmat)
        vec_bases = indexed_ptr(
            input2_ptr, index2_ptr, # input2 is the vector
            sparse_idx, n_offsets, stride_n2 if not RIGHT_SHARED else 0,
            stride_idx2, RIGHT_SHARED)

        # Load matrix bases (input1 -> becomes input2 for kernel_vecmat)
        mat_bases = indexed_ptr(
            input1_ptr, index1_ptr, # input1 is the matrix
            sparse_idx, n_offsets, stride_n1 if not LEFT_SHARED else 0,
            stride_idx1, LEFT_SHARED)

        # kernel_vecmat(vector, matrix)
        product = kernel_vecmat(
            vec_bases, mat_bases,
            n_mask,
            c_out_offsets, c_out_mask, # Output dimension (matrix C_out)
            c_in, # Reduction dimension (vector C_in, matrix C_in)
            stride_c2, # vector's C_in stride (reduction dim stride for input1)
            stride_c_in1, stride_c_out1, # matrix's C_in, C_out strides (strides for input2)
            BLOCK_SIZE_N, BLOCK_SIZE_C_OUT, BLOCK_SIZE_C_IN, # N, Output_Dim, Reduction_Dim blocks
            RIGHT_SHARED, # vector shared flag (input1 shared)
            LEFT_SHARED,  # matrix shared flag (input2 shared)
            OUT_ACCUMULATED,
            NUM_STAGES, LOOP_UNROLL_FACTOR, OUT_DTYPE)

        if scale_ptr is not None: # Apply optional external scale
            scale = tl.load(scale_ptr + sparse_idx)
            product = product * scale

        return product

@triton.jit
def indexed_mat_t_vec_scale_gather_kernel(
    input1_ptr, input2_ptr, scale_ptr, # input1=matrix, input2=vector
    index1_ptr, index2_ptr,
    seg_ptr, gather_index_ptr, index_out_ptr,
    output_ptr,
    size_n, size_c_in, size_c_out, # c_in=vector size, c_out=matrix first dim size
    stride_n1, stride_idx1, stride_c_out1, stride_c_in1, # matrix strides
    stride_n2, stride_idx2, stride_c2, # vector strides
    stride_n_out, stride_idx_out, stride_c_out, # output strides (vector of size c_out)
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C_OUT: tl.constexpr, # Block size for matrix C_out dim
    BLOCK_SIZE_C_IN: tl.constexpr,  # Block size for vector C_in dim
    LEFT_SHARED: tl.constexpr, # matrix shared
    RIGHT_SHARED: tl.constexpr, # vector shared
    OUT_ACCUMULATED: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    LOOP_UNROLL_FACTOR: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_seg = tl.program_id(1)
    pid_c_out = tl.program_id(2) # Grid dim over output channels (matrix C_out)

    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    n_mask = n_offsets < size_n

    # Offsets/mask for the output dimension (matrix C_out)
    c_out_offsets = pid_c_out * BLOCK_SIZE_C_OUT + tl.arange(0, BLOCK_SIZE_C_OUT)
    c_out_mask = c_out_offsets < size_c_out

    if seg_ptr is not None: # Check None for OUT_GATHERED
        # Accumulator shape matches output vector shape
        if not OUT_ACCUMULATED:
            accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_C_OUT), dtype=output_ptr.dtype.element_ty)
        else:
            accumulator = tl.zeros((BLOCK_SIZE_C_OUT,), dtype=output_ptr.dtype.element_ty)

        loop_start = tl.load(seg_ptr + pid_seg)
        loop_end = tl.load(seg_ptr + pid_seg + 1)
        for loop_idx in tl.range(loop_start, loop_end, num_stages=NUM_STAGES):
            if gather_index_ptr is None: # Check None for OUT_SEGMENTED
                sparse_idx = loop_idx
            else:
                sparse_idx = tl.load(gather_index_ptr+loop_idx)

            product = batched_single_mat_t_vec(
                input1_ptr, index1_ptr, # matrix
                input2_ptr, index2_ptr, # vector
                scale_ptr,
                sparse_idx,
                n_offsets, n_mask,
                c_out_offsets, c_out_mask, # Output channel offsets/mask
                size_c_in, # Inner dimension size (vector size)
                stride_n1, stride_idx1, stride_c_out1, stride_c_in1, # matrix strides
                stride_n2, stride_idx2, stride_c2, # vector strides
                LEFT_SHARED, RIGHT_SHARED,
                OUT_ACCUMULATED,
                BLOCK_SIZE_N, BLOCK_SIZE_C_OUT, BLOCK_SIZE_C_IN,
                NUM_STAGES, LOOP_UNROLL_FACTOR, output_ptr.dtype.element_ty)

            accumulator += product
    else: # Not OUT_GATHERED (seg_ptr is None)
        sparse_idx = pid_seg

        accumulator = batched_single_mat_t_vec(
                input1_ptr, index1_ptr, # matrix
                input2_ptr, index2_ptr, # vector
                scale_ptr,
                sparse_idx,
                n_offsets, n_mask,
                c_out_offsets, c_out_mask, # Output channel offsets/mask
                size_c_in, # Inner dimension size (vector size)
                stride_n1, stride_idx1, stride_c_out1, stride_c_in1, # matrix strides
                stride_n2, stride_idx2, stride_c2, # vector strides
                LEFT_SHARED, RIGHT_SHARED,
                OUT_ACCUMULATED,
                BLOCK_SIZE_N, BLOCK_SIZE_C_OUT, BLOCK_SIZE_C_IN,
                NUM_STAGES, LOOP_UNROLL_FACTOR, output_ptr.dtype.element_ty)

    output_bases = indexed_ptr(output_ptr,
                               index_out_ptr,
                               pid_seg,
                               n_offsets,
                               stride_n_out if not OUT_ACCUMULATED else 0,
                               stride_idx_out,
                               OUT_ACCUMULATED)

    # Store the result vector
    store_block_vec(
        accumulator,
        output_bases,
        n_mask,
        c_out_offsets, c_out_mask, # Use output offsets/mask
        stride_c_out, # Use output stride
        OUT_ACCUMULATED
    )


def indexed_mat_t_vec_scale_gather(
        input1, input2, # input1=matrix, input2=vector
        scale=None, index1=None, index2=None,
        seg=None, gather_index=None,
        index_out=None, out=None,
        out_accumulated=False,
        out_size=None,
        block_size_n=8, block_size_c_in=32, block_size_c_out=32, # c_in=reduction, c_out=output
        num_stages=1, loop_unroll_factor=4):
    r"""
    Computes indexed/gathered vec @ mat, using index1 for matrix and index2 for vector.
    input1 (matrix): Tensor of shape (N1, M1, C_in, C_out) or (M1, C_in, C_out)
    input2 (vector): Tensor of shape (N2, M2, C_in) or (M2, C_in)
    Output (vector): Tensor of shape (N_out, M_out, C_out) or (M_out, C_out)

    Note: This function internally calls indexed_vecmat_scale_gather by swapping
          input1 (matrix) and input2 (vector) because the underlying kernel
          expects (vector, matrix) and the direct implementation here was buggy.
    """
    # Call the correct implementation with swapped inputs and indices
    return indexed_vecmat_scale_gather(
        input1=input2,  # Pass original vector as input1
        input2=input1,  # Pass original matrix as input2
        scale=scale,
        index1=index2,  # Pass vector's index as index1
        index2=index1,  # Pass matrix's index as index2
        seg=seg,
        gather_index=gather_index,
        index_out=index_out,
        out=out,
        out_accumulated=out_accumulated,
        out_size=out_size,
        # Pass block sizes according to indexed_vecmat_scale_gather's signature
        block_size_n=block_size_n,
        block_size_c_out=block_size_c_out, # Corresponds to C_out of the matrix (input1 here)
        block_size_c_in=block_size_c_in,   # Corresponds to C_in of the vector/matrix
        num_stages=num_stages,
        loop_unroll_factor=loop_unroll_factor
    )

@triton.jit
def batched_single_vecmat(
        input1_ptr, index1_ptr,
        input2_ptr, index2_ptr,        
        scale_ptr,
        sparse_idx,
        n_offsets, n_mask,
        c_out_offsets, c_out_mask,
        c_in,
        stride_n1, stride_idx1, stride_c1,
        stride_n2, stride_idx2, stride_c_in2, stride_c_out2,
        LEFT_SHARED: tl.constexpr,
        RIGHT_SHARED: tl.constexpr,
        OUT_ACCUMULATED: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_C_OUT: tl.constexpr,
        BLOCK_SIZE_C_IN: tl.constexpr,
        NUM_STAGES: tl.constexpr,
        LOOP_UNROLL_FACTOR: tl.constexpr,
        OUT_DTYPE: tl.constexpr
        ):

        input1_bases = indexed_ptr(
            input1_ptr, index1_ptr, 
            sparse_idx, n_offsets, stride_n1 if not LEFT_SHARED else 0,
            stride_idx1, LEFT_SHARED) 

        input2_bases = indexed_ptr(
            input2_ptr, index2_ptr, 
            sparse_idx, n_offsets, stride_n2 if not RIGHT_SHARED else 0,
            stride_idx2, RIGHT_SHARED) 
    
        product = kernel_vecmat(
            input1_bases, input2_bases,
            n_mask,
            c_out_offsets, c_out_mask,
            c_in,
            stride_c1, stride_c_in2, stride_c_out2,
            BLOCK_SIZE_N, BLOCK_SIZE_C_OUT, BLOCK_SIZE_C_IN,
            LEFT_SHARED, RIGHT_SHARED,
            OUT_ACCUMULATED,
            NUM_STAGES, LOOP_UNROLL_FACTOR, OUT_DTYPE)

        if scale_ptr is not None: # Check None directly
            scale = tl.load(scale_ptr + sparse_idx)
            product = product * scale

        return product

@triton.jit
def indexed_vecmat_scale_gather_kernel(
    input1_ptr, input2_ptr, scale_ptr,
    index1_ptr, index2_ptr,
    seg_ptr, gather_index_ptr, index_out_ptr,
    output_ptr,
    size_n, size_c_in, size_c_out,
    stride_n1, stride_idx1, stride_c1,
    stride_n2, stride_idx2, stride_c_in2, stride_c_out2,
    stride_n_out, stride_idx_out, stride_c_out,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C_OUT: tl.constexpr,
    BLOCK_SIZE_C_IN: tl.constexpr,
    LEFT_SHARED: tl.constexpr,
    RIGHT_SHARED: tl.constexpr,
    OUT_ACCUMULATED: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    LOOP_UNROLL_FACTOR: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_seg = tl.program_id(1)
    pid_c_out = tl.program_id(2)
    
    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    n_mask = n_offsets < size_n

    c_out_offsets = pid_c_out * BLOCK_SIZE_C_OUT + tl.arange(0, BLOCK_SIZE_C_OUT)
    c_out_mask = c_out_offsets < size_c_out
    
    if seg_ptr is not None: # Check None for OUT_GATHERED
        if not OUT_ACCUMULATED:
            accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_C_OUT), dtype=output_ptr.dtype.element_ty)
        else:
            accumulator = tl.zeros((BLOCK_SIZE_C_OUT,), dtype=output_ptr.dtype.element_ty)

        loop_start = tl.load(seg_ptr + pid_seg)
        loop_end = tl.load(seg_ptr + pid_seg + 1)
        for loop_idx in tl.range(loop_start, loop_end, num_stages=NUM_STAGES):
            if gather_index_ptr is None: # Check None for OUT_SEGMENTED
                sparse_idx = loop_idx  
            else:
                sparse_idx = tl.load(gather_index_ptr+loop_idx)
                
            product = batched_single_vecmat(
                input1_ptr, index1_ptr, 
                input2_ptr, index2_ptr, 
                scale_ptr, 
                sparse_idx,
                n_offsets, n_mask,
                c_out_offsets, c_out_mask,
                size_c_in,
                stride_n1 if not LEFT_SHARED else 0, stride_idx1, stride_c1,
                stride_n2 if not RIGHT_SHARED else 0, stride_idx2, stride_c_in2, stride_c_out2,
                LEFT_SHARED, RIGHT_SHARED, 
                OUT_ACCUMULATED, 
                BLOCK_SIZE_N, BLOCK_SIZE_C_OUT, BLOCK_SIZE_C_IN,
                NUM_STAGES, LOOP_UNROLL_FACTOR, output_ptr.dtype.element_ty)
            
            accumulator += product 
    else: # Not OUT_GATHERED (seg_ptr is None)
        sparse_idx = pid_seg
    
        accumulator = batched_single_vecmat(
                input1_ptr, index1_ptr, 
                input2_ptr, index2_ptr, 
                scale_ptr, 
                sparse_idx,
                n_offsets, n_mask,
                c_out_offsets, c_out_mask,
                size_c_in,
                stride_n1 if not LEFT_SHARED else 0, stride_idx1, stride_c1,
                stride_n2 if not RIGHT_SHARED else 0, stride_idx2, stride_c_in2, stride_c_out2,
                LEFT_SHARED, RIGHT_SHARED, 
                OUT_ACCUMULATED, 
                BLOCK_SIZE_N, BLOCK_SIZE_C_OUT, BLOCK_SIZE_C_IN,
                NUM_STAGES, LOOP_UNROLL_FACTOR, output_ptr.dtype.element_ty)

    output_bases = indexed_ptr(output_ptr, 
                               index_out_ptr, 
                               pid_seg, 
                               n_offsets,
                               stride_n_out if not OUT_ACCUMULATED else 0,
                               stride_idx_out,
                               OUT_ACCUMULATED) 
    
    store_block_vec(
        accumulator,
        output_bases,
        n_mask,
        c_out_offsets, c_out_mask,
        stride_c_out,
        OUT_ACCUMULATED
    )

def indexed_vecmat_scale_gather_gpu(
        input1, input2, 
        scale=None, index1=None, index2=None,
        seg=None, gather_index=None, 
        index_out=None, out=None,
        out_accumulated=False,
        out_size=None,
        block_size_n=8, block_size_c_out=32, block_size_c_in=32,
        num_stages=1, loop_unroll_factor=4):
    scaled = scale is not None
    left_indexed = index1 is not None
    right_indexed = index2 is not None

    if input1.ndim==2:
        left_shared = True
        input1 = input1.unsqueeze(0)
    else:
        left_shared = False
    if input2.ndim==3:
        right_shared = True
        input2 = input2.unsqueeze(0)
    else:
        right_shared = False

    out_gathered = seg is not None
    out_segmented = gather_index is None
    out_indexed = index_out is not None

    sparse_grid_size = (seg.shape[0] - 1) if out_gathered else index1.shape[0] if left_indexed else index2.shape[0] if right_indexed else input1.shape[-2]

    if out_size is None:
        out_size = sparse_grid_size

    size_n = max(input1.shape[0], input2.shape[0])

    if out is None:
        if out_accumulated:
            out = torch.zeros(out_size, input2.shape[-1], dtype=torch.result_type(input1, input2), device=input1.device)
        else:
            if out_indexed:
                out = torch.zeros(size_n, out_size, input2.shape[-1], dtype=torch.result_type(input1, input2), device=input1.device)
            else:
                out = torch.empty(size_n, out_size, input2.shape[-1], dtype=torch.result_type(input1, input2), device=input1.device)
    else:
        out = out.contiguous()

    if out_accumulated:
        out = out.unsqueeze(0)

    # Ensure inputs are contiguous
    input1 = input1.contiguous()
    input2 = input2.contiguous()


    grid = lambda meta: (
        triton.cdiv(size_n, meta['BLOCK_SIZE_N']),
        sparse_grid_size,
        triton.cdiv(input2.shape[-1], meta['BLOCK_SIZE_C_OUT']),
    )

    indexed_vecmat_scale_gather_kernel[grid](
        input1, input2, scale,
        index1, index2,
        seg, gather_index, index_out, out, 
        size_n, input1.shape[-1], input2.shape[-1],
        *input1.stride(), *input2.stride(),
        *out.stride(),
        block_size_n, block_size_c_out, block_size_c_in,
        left_shared, right_shared,
        out_accumulated,
        num_stages,
        loop_unroll_factor
    )
    if out_accumulated:
        out = out.squeeze(0)

    return out
    
def indexed_vecmat_scale_gather_cpu(
        input1, input2, 
        scale=None, index1=None, index2=None,
        seg=None, gather_index=None, 
        index_out=None, out=None,
        out_accumulated=False,
        out_size=None):
    r"""
    CPU implementation of indexed_vecmat_scale_gather.
    input1: vector (..., C_in), input2: matrix (..., C_in, C_out)
    Output: vector (..., C_out)
    """
    # Handle input indexing
    if index1 is not None:
        input1 = input1.index_select(-2, index1) # (N, T, Cin) or (T, Cin)
    if index2 is not None:
        input2 = input2.index_select(-3, index2) # (N, T, Cin, Cout) or (T, Cin, Cout)
    
    # Core vec-mat product using einsum
    # input1: (..., T, Cin), input2: (..., T, Cin, Cout) -> (..., T, Cout)
    inter = torch.einsum('...ti,...tij->...tj', input1, input2)
    
    # Apply scaling if provided
    if scale is not None:
        inter = inter * scale.unsqueeze(-1) # scale is (T,) -> (1, T, 1) or (T,1) for broadcasting
    
    # Handle segmentation/gathering
    if seg is not None:
        if gather_index is not None:
            # Gather then segment
            # inter shape: (N, T, Cout) or (T, Cout)
            # gather_index refers to T dimension, which is -2 for 3D, -2 for 2D
            dim_to_gather = -2 if inter.ndim > 1 else 0 # Should be -2 if batched, 0 if not
            gathered = inter.index_select(dim_to_gather, gather_index)
            inter = segment(gathered, seg.unsqueeze(0))
        else:
            # Direct segment
            inter = segment(inter, seg.unsqueeze(0))
    
    # Handle output indexing
    if index_out is not None:
        # scatter along the sparse dimension (now -2 after segment)
        inter = scatter(inter, index_out, dim=-2, dim_size=out_size)
    
    # Handle accumulation
    if out_accumulated:
        inter = inter.sum(dim=0)
    
    return inter

def indexed_vecmat_scale_gather(
        input1, input2, 
        scale=None, index1=None, index2=None,
        seg=None, gather_index=None, 
        index_out=None, out=None,
        out_accumulated=False,
        out_size=None,
        block_size_n=8, block_size_c_out=32, block_size_c_in=32,
        num_stages=1, loop_unroll_factor=4):
    r"""
    Dispatches to GPU or CPU implementation based on input device.
    """
    if input1.device.type == 'cuda':
        return indexed_vecmat_scale_gather_gpu(
            input1, input2, scale, index1, index2,
            seg, gather_index, index_out, out,
            out_accumulated, out_size,
            block_size_n, block_size_c_out, block_size_c_in,
            num_stages, loop_unroll_factor)
    else:
        return indexed_vecmat_scale_gather_cpu(
            input1, input2, scale, index1, index2,
            seg, gather_index, index_out, out,
            out_accumulated, out_size)



@triton.jit
def batched_single_vecsca(
        input1_ptr, index1_ptr,
        input2_ptr, index2_ptr,        
        scale_ptr,
        sparse_idx,
        n_offsets, n_mask,
        c_offsets, c_mask,
        stride_n1, stride_idx1, stride_c1,
        stride_n2, stride_idx2,
        LEFT_SHARED: tl.constexpr,
        RIGHT_SHARED: tl.constexpr,
        OUT_ACCUMULATED: tl.constexpr,
        # SCALED: tl.constexpr # Removed
        ):

        input1_bases = indexed_ptr(
            input1_ptr, index1_ptr, 
            sparse_idx, n_offsets, stride_n1 if not LEFT_SHARED else 0,
            stride_idx1, LEFT_SHARED) 

        input2_bases = indexed_ptr(
            input2_ptr, index2_ptr, 
            sparse_idx, n_offsets, stride_n2 if not RIGHT_SHARED else 0,
            stride_idx2, RIGHT_SHARED) 
    
        product = kernel_vecsca(
            input1_bases, input2_bases,
            n_mask,
            c_offsets, c_mask,
            stride_c1,
            LEFT_SHARED, RIGHT_SHARED,
            OUT_ACCUMULATED)

        if scale_ptr is not None: # Check None directly
            scale = tl.load(scale_ptr + sparse_idx)
            product = product * scale

        return product

@triton.jit
def indexed_vecsca_scale_gather_kernel(
    input1_ptr, input2_ptr, scale_ptr,
    index1_ptr, index2_ptr,
    seg_ptr, gather_index_ptr, index_out_ptr,
    output_ptr,
    size_n, size_c,
    stride_n1, stride_idx1, stride_c1,
    stride_n2, stride_idx2,
    stride_n_out, stride_idx_out, stride_c_out,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    LEFT_SHARED: tl.constexpr,
    RIGHT_SHARED: tl.constexpr,
    OUT_ACCUMULATED: tl.constexpr,
    NUM_STAGES: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_seg = tl.program_id(1)
    pid_c = tl.program_id(2)
    
    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    n_mask = n_offsets < size_n

    c_offsets = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    c_mask = c_offsets < size_c
    
    if seg_ptr is not None: # Check None for OUT_GATHERED
        if not OUT_ACCUMULATED:
            accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_C), dtype=output_ptr.dtype.element_ty)
        else:
            accumulator = tl.zeros((BLOCK_SIZE_C,), dtype=output_ptr.dtype.element_ty)

        loop_start = tl.load(seg_ptr + pid_seg)
        loop_end = tl.load(seg_ptr + pid_seg + 1)
        for loop_idx in tl.range(loop_start, loop_end, num_stages=NUM_STAGES):
            if gather_index_ptr is None: # Check None for OUT_SEGMENTED
                sparse_idx = loop_idx  
            else:
                sparse_idx = tl.load(gather_index_ptr+loop_idx)
                
            product = batched_single_vecsca(
                input1_ptr, index1_ptr, 
                input2_ptr, index2_ptr, 
                scale_ptr, 
                sparse_idx,
                n_offsets, n_mask,
                c_offsets, c_mask,
                stride_n1 if not LEFT_SHARED else 0, stride_idx1, stride_c1,
                stride_n2 if not RIGHT_SHARED else 0, stride_idx2,
                LEFT_SHARED, RIGHT_SHARED, 
                OUT_ACCUMULATED) 
            
            accumulator += product 
    else: # Not OUT_GATHERED (seg_ptr is None)
        sparse_idx = pid_seg
    
        accumulator = batched_single_vecsca(
                input1_ptr, index1_ptr, 
                input2_ptr, index2_ptr, 
                scale_ptr, 
                sparse_idx,
                n_offsets, n_mask,
                c_offsets, c_mask,
                stride_n1 if not LEFT_SHARED else 0, stride_idx1, stride_c1,
                stride_n2 if not RIGHT_SHARED else 0, stride_idx2,
                LEFT_SHARED, RIGHT_SHARED, 
                OUT_ACCUMULATED) 

    output_bases = indexed_ptr(output_ptr, 
                               index_out_ptr, 
                               pid_seg, 
                               n_offsets,
                               stride_n_out if not OUT_ACCUMULATED else 0,
                               stride_idx_out,
                               OUT_ACCUMULATED) 
    
    store_block_vec(
        accumulator,
        output_bases,
        n_mask,
        c_offsets, c_mask,
        stride_c_out,
        OUT_ACCUMULATED
    )


def indexed_vecsca_scale_gather_gpu(
        input1, input2, 
        scale=None, index1=None, index2=None,
        seg=None, gather_index=None, 
        index_out=None, out=None,
        out_accumulated=False,
        out_size=None,
        block_size_n=64, block_size_c=64,
        num_stages=2):
    scaled = scale is not None
    left_indexed = index1 is not None
    right_indexed = index2 is not None

    # 处理输入共享维度
    if input1.ndim==2:
        left_shared = True
        input1 = input1.unsqueeze(0)
    else:
        left_shared = False
        
    if input2.ndim==1:
        right_shared = True
        input2 = input2.unsqueeze(0)
    else:
        right_shared = False

    # 确定输出模式
    out_gathered = seg is not None
    out_segmented = gather_index is None
    out_indexed = index_out is not None

    # 计算稀疏网格大小
    sparse_grid_size = (seg.shape[0] - 1) if out_gathered else index1.shape[0] if left_indexed else index2.shape[0] if right_indexed else input1.shape[-2]

    # 确定输出大小
    if out_size is None:
        out_size = sparse_grid_size

    # 确定batch维度大小
    size_n = max(input1.shape[0], input2.shape[0])

    # 初始化输出张量
    if out is None:
        if out_accumulated:
            out = torch.zeros(out_size, input1.shape[-1], dtype=torch.result_type(input1, input2), device=input1.device)
        else:
            if out_indexed:
                out = torch.zeros(size_n, out_size, input1.shape[-1], dtype=torch.result_type(input1, input2), device=input1.device)
            else:
                out = torch.empty(size_n, out_size, input1.shape[-1], dtype=torch.result_type(input1, input2), device=input1.device)
    else:
        out = out.contiguous()

    # 处理输出维度
    if out_accumulated:
        out = out.unsqueeze(0)

    # Ensure inputs are contiguous
    input1 = input1.contiguous()
    input2 = input2.contiguous()


    # 定义计算网格
    grid = lambda meta: (
        triton.cdiv(size_n, meta['BLOCK_SIZE_N']),
        sparse_grid_size,
        triton.cdiv(input1.shape[-1], meta['BLOCK_SIZE_C']),
    )

    # 调用内核函数
    indexed_vecsca_scale_gather_kernel[grid](
        input1, input2, scale,
        index1, index2,
        seg, gather_index, index_out, out, 
        size_n, input1.shape[-1],
        *input1.stride(), *input2.stride(),
        *out.stride(),
        block_size_n, block_size_c,
        left_shared, right_shared,
        out_accumulated,
        num_stages
    )
    if out_accumulated:
        out = out.squeeze(0)

    return out
    
def indexed_vecsca_scale_gather_cpu(
        input1, input2, 
        scale=None, index1=None, index2=None,
        seg=None, gather_index=None, 
        index_out=None, out=None,
        out_accumulated=False,
        out_size=None):
    r"""
    CPU implementation of indexed_vecsca_scale_gather.
    input1: vector, input2: scalar
    """
    # Handle input indexing
    if index1 is not None:
        input1 = input1.index_select(-2, index1)
    if index2 is not None:
        input2 = input2.index_select(-1, index2) # Scalar indexed along its sparse dim
    
    # Core vec-sca product
    inter = input1 * input2.unsqueeze(-1) # (N, T, C) * (N, T, 1) or (T, C) * (T,1)
    
    # Apply scaling if provided
    if scale is not None:
        inter = inter * scale.unsqueeze(-1)
    
    # Handle segmentation/gathering
    if seg is not None:
        if gather_index is not None:
            # Gather then segment
            gathered = inter.index_select(-2, gather_index)
            inter = segment(gathered, seg.unsqueeze(0))
        else:
            # Direct segment
            inter = segment(inter, seg.unsqueeze(0))
    
    # Handle output indexing
    if index_out is not None:
        inter = scatter(inter, index_out, dim=-2, dim_size=out_size)
    
    # Handle accumulation
    if out_accumulated:
        inter = inter.sum(dim=0)
    
    return inter

def indexed_vecsca_scale_gather(
        input1, input2, 
        scale=None, index1=None, index2=None,
        seg=None, gather_index=None, 
        index_out=None, out=None,
        out_accumulated=False,
        out_size=None,
        block_size_n=64, block_size_c=64,
        num_stages=2):
    r"""
    Dispatches to GPU or CPU implementation based on input device.
    """
    if input1.device.type == 'cuda':
        return indexed_vecsca_scale_gather_gpu(
            input1, input2, scale, index1, index2,
            seg, gather_index, index_out, out,
            out_accumulated, out_size,
            block_size_n, block_size_c, num_stages)
    else:
        return indexed_vecsca_scale_gather_cpu(
            input1, input2, scale, index1, index2,
            seg, gather_index, index_out, out,
            out_accumulated, out_size)
