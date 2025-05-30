import torch
import triton
import triton.language as tl

from torch_geometric.utils import segment

@triton.jit
def indexed_scale_segment_kernel(
    # Pointers
    input_ptr,
    scale_ptr,
    index_ptr,
    seg_ptr,
    output_ptr,
    # Tensor dimensions
    N, C,
    # Memory strides
    stride_input_n, stride_input_idx, stride_input_c,
    stride_output_n, stride_output_idx, stride_output_c,
    # Block sizes
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    # Optimization
    NUM_STAGES: tl.constexpr,
):
    # ---- Grid Dimensions ----
    pid_n = tl.program_id(0)    # Batch dimension block
    pid_seg = tl.program_id(1)  # Segment dimension
    pid_c = tl.program_id(2)    # Channel dimension block
    
    # ---- Offsets and Masks ----
    # Batch offsets
    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    n_mask = n_offsets < N
    
    # Channel offsets
    c_offsets = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    c_mask = c_offsets < C
    
    # ---- Segment Range ----
    if seg_ptr is not None:
        loop_start = tl.load(seg_ptr + pid_seg)
        loop_end = tl.load(seg_ptr + pid_seg + 1)
        
        # ---- Accumulation ----
        accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_C), 
                            dtype=output_ptr.dtype.element_ty)
        
        # ---- Main Computation ----
        for loop_idx in tl.range(loop_start, loop_end, num_stages=NUM_STAGES):
            # 1. Load index and scale
            idx = tl.load(index_ptr + loop_idx)
            if scale_ptr is not None:
                scale_val = tl.load(scale_ptr + loop_idx)
            
            # 2. Calculate input pointers
            input_ptrs = (
                input_ptr + 
                n_offsets[:, None] * stride_input_n + 
                idx * stride_input_idx + 
                c_offsets[None, :] * stride_input_c
            )
            
            # 3. Load and scale input values
            input_vals = tl.load(input_ptrs, 
                                mask=n_mask[:, None] & c_mask[None, :], 
                                other=0.0)
            if scale_ptr is not None:
                scaled_vals = input_vals * scale_val
            else: 
                scaled_vals = input_vals        
            # 4. Accumulate
            accumulator += scaled_vals
    else:
        
        # 1. Load index and scale
        idx = tl.load(index_ptr + pid_seg)
        if scale_ptr is not None:
            scale_val = tl.load(scale_ptr + pid_seg)
        
        # 2. Calculate input pointers
        input_ptrs = (
            input_ptr + 
            n_offsets[:, None] * stride_input_n + 
            idx * stride_input_idx + 
            c_offsets[None, :] * stride_input_c
        )
        
        # 3. Load and scale input values
        input_vals = tl.load(input_ptrs, 
                            mask=n_mask[:, None] & c_mask[None, :], 
                            other=0.0)
        if scale_ptr is not None:
            scaled_vals = input_vals * scale_val
        else: 
            scaled_vals = input_vals        
        # 4. Accumulate
        accumulator = scaled_vals
            
    # ---- Write Back Result ----
    output_ptrs = (
        output_ptr + 
        n_offsets[:, None] * stride_output_n + 
        pid_seg * stride_output_idx + 
        c_offsets[None, :] * stride_output_c
    )
    tl.store(output_ptrs, accumulator, 
            mask=n_mask[:, None] & c_mask[None, :])

def indexed_scale_segment_gpu(input, scale, index, seg, out_size=None, out=None,
                            block_size_n=64, block_size_c=64):
    r"""
    GPU (Triton) implementation of indexed_scale_segment.
    """
    N, _, C = input.shape
    if seg is not None:
        num_segments = seg.shape[0] - 1
    else:
        num_segments = out_size
    out_size = out_size or num_segments
    
    if out is None:
        out = torch.empty((N, out_size, C), 
                         device=input.device, dtype=input.dtype)
    
    grid = lambda meta: (
        triton.cdiv(N, meta['BLOCK_SIZE_N']),
        num_segments,
        triton.cdiv(C, meta['BLOCK_SIZE_C'])
    )
    
    indexed_scale_segment_kernel[grid](
        input, scale, index, seg, out,
        N, C,
        *input.stride(), *out.stride(),
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_C=block_size_c,
        NUM_STAGES=2
    )
    
    return out

def indexed_scale_segment_cpu(input, scale, index, seg, out_size=None, out=None):
    r"""
    CPU implementation of indexed_scale_segment.
    """
    N, _, C = input.shape
    num_segments = seg.shape[0] - 1
    out_size = out_size or num_segments
    
    if out is None:
        out = torch.empty((N, out_size, C), 
                         device=input.device, dtype=input.dtype)
    
    # Apply indexing and scaling
    if index is not None:
        indexed = input.index_select(-2, index)
    else: 
        indexed = input
    if scale is not None:
        scaled = indexed * scale.unsqueeze(-1)
    else:
        scaled = indexed
    
    # Segment the result
    if seg is not None:
        segmented = segment(scaled, seg.unsqueeze(0))
    else:
        segmented = scaled
    return segmented

def indexed_scale_segment(input, scale, index, seg, out_size=None, out=None,
                         block_size_n=64, block_size_c=64):
    r"""
    Dispatches to GPU or CPU implementation based on input device.
    """
    if input.device.type == 'cuda':
        return indexed_scale_segment_gpu(input, scale, index, seg, out_size, out,
                                       block_size_n, block_size_c)
    else:
        return indexed_scale_segment_cpu(input, scale, index, seg, out_size, out)
