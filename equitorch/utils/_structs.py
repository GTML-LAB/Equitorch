from typing import List, Tuple, Dict
from torch import tensor

from ..irreps import Irreps, check_irreps, has_path, irrep_segments

from ..constants import so3_clebsch_gordan, j_matrix

from ..structs import IrrepsInfo, IrrepsLinearInfo, SparseProductInfo, SparseScaleInfo, TensorProductInfo, WignerRotationInfo
from ._indices import extract_batch_segments, extract_scatter_indices, sort_by_column_key

from typing import List, Tuple, Dict
from torch import tensor

from equitorch.irreps import Irreps, check_irreps, has_path, irrep_segments

from equitorch.constants import so3_clebsch_gordan, j_matrix

from equitorch.structs import IrrepsInfo, IrrepsLinearInfo, SparseProductInfo, SparseScaleInfo, TensorProductInfo, WignerRotationInfo
from equitorch.utils._indices import extract_batch_segments, extract_scatter_indices, sort_by_column_key


def sparse_scale_info(index=None, index_out=None, scale=None, out_size=None):
    assert index is not None or index_out is not None, "At least one index should be not None"
    inter_size = len(index or index_out)
    index = index or list(range(inter_size))
    index_out = index_out or list(range(inter_size))
    if scale is not None:
        index_out_MM1, index_MM1, scale_MM1 = sort_by_column_key(
            [index_out, index, scale]
        )
        batch_M_MM1, seg_M_MM1, (index_out_M,) = extract_batch_segments(
            [index_out_MM1]
        )
    else:
        index_out_MM1, index_MM1 = sort_by_column_key(
            [index_out, index]
        )
        batch_M_MM1, seg_M_MM1, (index_out_M,) = extract_batch_segments(
            [index_out_MM1]
        )
        scale_MM1 = None

    
    if out_size is None:
        out_size = len(index_out_M)

    seg_new = [0]
    out_count = 0
    for out_M in range(out_size):
        if out_count < len(index_out_M) and index_out_M[out_count] == out_M:
            seg_new.append(seg_M_MM1[out_count+1])
            out_count+=1
        else:
            seg_new.append(seg_new[-1])

    index_out_M = None
    if index_MM1 == list(range(inter_size)):
        index_MM1 = None

    if seg_new == list(range(out_size+1)):#len(batch_M_MM1)+1:
        seg_new = None
    # if index_out_M == list(range(out_size)):
    #     index_out_M = None

    return SparseScaleInfo(
        tensor(scale_MM1) if scale_MM1 is not None else None, 
        tensor(index_MM1) if index_MM1 is not None else None, 
        # tensor(seg_M_MM1) if seg_M_MM1 is not None else None, 
        tensor(seg_new) if seg_new is not None else None, 
        tensor(index_out_M) if index_out_M is not None else None,
        out_size)

def sparse_scale_infos(index=None, index_out=None, scale=None, out_size=None, in_size=None):
    return (
        sparse_scale_info(index, index_out, scale, out_size),
        sparse_scale_info(index_out, index, scale, in_size)
    )

def sparse_product_info(index1=None, index2=None, index=None, scale=None, out_size=None):

    assert index1 is not None or index2 is not None or index is not None, "At least one of the indices should be not None"
    # assert index is None or index_out is None, "index and index_out cannot be both not None " 

    inter_size = len(index1 or index2 or index)
    index1 = index1 or list(range(inter_size))
    index2 = index2 or list(range(inter_size))
    index = index or list(range(inter_size))
    if scale is not None:
        index_MM1M2, index1_MM1M2, index2_MM1M2, scale_MM1M2 = sort_by_column_key(
            [index, index1, index2, scale])
        batch_M_MM1M2, seg_M_MM1M2, (index_M,) = extract_batch_segments(
            [index_MM1M2]
        )
    else:
        index_MM1M2, index1_MM1M2, index2_MM1M2 = sort_by_column_key(
            [index, index1, index2])
        batch_M_MM1M2, seg_M_MM1M2, (index_M,) = extract_batch_segments(
            [index_MM1M2]
        )
        scale_MM1M2 = None

    if out_size is None:
        out_size = len(index_M)

    if index_M[0] != 0:
        seg_M_MM1M2 = [0] * index_M[0] + seg_M_MM1M2
    if index_M[-1] != out_size:
        seg_M_MM1M2 = seg_M_MM1M2 + [inter_size] * (out_size-index_M[-1]-1)


    if index1_MM1M2 == list(range(inter_size)):
        index1_MM1M2 = None
    if index2_MM1M2 == list(range(inter_size)):
        index2_MM1M2 = None

    seg_new = [0]
    out_count = 0
    for out_M in range(out_size):
        if out_count < len(index_M) and index_M[out_count] == out_M:
            seg_new.append(seg_M_MM1M2[out_count+1])
            out_count+=1
        else:
            seg_new.append(seg_new[-1])

    if seg_new == list(range(out_size+1)):
        seg_new = None
    # if len(seg_M_MM1M2) == len(batch_M_MM1M2)+1:
    #     seg_M_MM1M2 = None
    # if index_M == list(range(out_size)):
    index_M = None
            
    return SparseProductInfo(
        scale=tensor(scale_MM1M2) if scale_MM1M2 is not None else None,
        index1=tensor(index1_MM1M2) if index1_MM1M2 is not None else None,
        index2=tensor(index2_MM1M2) if index2_MM1M2 is not None else None,
        seg_out=tensor(seg_new) if seg_new is not None else None,
        # seg_out=tensor(seg_M_MM1M2) if seg_M_MM1M2 is not None else None,
        index_out=tensor(index_M) if index_M is not None else None,
        out_size=out_size
    )

def sparse_product_infos(index1=None, index2=None, index=None, scale=None, out_size=None, in1_size=None, in2_size=None):
    return (
        sparse_product_info(index1,index2,index,scale,out_size),
        sparse_product_info(index2,index,index1,scale,in1_size),
        sparse_product_info(index,index1,index2,scale,in2_size),
    )

def generate_fully_connected_tp_paths(irreps_out: Irreps, 
                           irreps1: Irreps, irreps2: Irreps):

    paths = []
    for (k,ir_out) in enumerate(irreps_out):
        for (i,ir1) in enumerate(irreps1):
            for (j, ir2) in enumerate(irreps2):
                if has_path(ir_out, ir1, ir2): 
                    paths.append((k,i,j))

    return paths


def prepare_so3(
        irreps_out: Irreps,
        irreps1: Irreps, irreps2: Irreps,
        path: List[Tuple[int, int, int]]=None,
        path_norm: bool = True,
        channel_norm: bool = False,
        channel_scale: float = 1.0):
    
    if not path:
        path = generate_fully_connected_tp_paths(irreps_out, irreps1, irreps2)
    
    seg_out = irrep_segments(irreps_out)
    seg1 = irrep_segments(irreps1)
    seg2 = irrep_segments(irreps2)

    current_k = None
    path_count = {}
    for w_idx, (k, i, j) in enumerate(path):
        if k == current_k:
            path_count[k] += 1
        else:
            path_count[k] = 1
            current_k = k

    cg_vals = []
    Ms = []
    M1s = []
    M2s = []
    k_s = []
    i_s = []
    j_s = []
    w_idcs = []

    for w_idx, (k,i,j) in enumerate(path):
        l, l1, l2 = (
            irreps_out[k].l, irreps1[i].l, irreps2[j].l
        )
        cg = so3_clebsch_gordan(l, l1, l2)
        for km in range(2 * l + 1):
            # Apply path normalization and channel normalization
            norm_scale = (path_count[k] ** (-0.5) if path_norm else 1.0) * \
                         (channel_scale if channel_norm else 1.0)
            for im in range(2 * l1 + 1):
                for jm in range(2 * l2 + 1):
                    cg_val = float(cg[km, im, jm]) * norm_scale
                    if cg_val != 0.0:
                        cg_vals.append(cg_val)
                        Ms.append(seg_out[k] + km)
                        M1s.append(seg1[i]+im)
                        M2s.append(seg2[j]+jm)
                        k_s.append(k)
                        i_s.append(i)
                        j_s.append(j)
                        w_idcs.append(w_idx)
    return cg_vals, Ms, M1s, M2s, k_s, i_s, j_s, w_idcs, len(path)


def create_tp_info(cg_vals, M, M1, M2, k, i, j, w_idx, out_size, in1_size, in2_size):
    k_MM1M2, M_MM1M2, i_MM1M2, j_MM1M2, M1_MM1M2, M2_MM1M2, w_idx_MM1M2, cg_vals = sort_by_column_key(
        [k, M, i, j, M1, M2, w_idx, cg_vals])
    
    # first xy 
    Mij_MM1M2, Mij_seg_MM1M2, (k_Mij, M_Mij, i_Mij, j_Mij, w_idx_Mij)  = extract_batch_segments(
        [k_MM1M2, M_MM1M2, i_MM1M2, j_MM1M2, w_idx_MM1M2]
    )
    infos_inter = sparse_product_infos(M1_MM1M2, M2_MM1M2, Mij_MM1M2, cg_vals, in1_size=in1_size, in2_size=in2_size)
    M_batch_Mij, M_seg_Mij, (M_out,) = extract_batch_segments(
        [M_Mij]
    )
    infos_M = sparse_product_infos(index2=w_idx_Mij, index=M_batch_Mij,out_size=out_size)


    # first W
    # k_MM1M2, M_MM1M2, i_MM1M2, M1_MM1M2, j_MM1M2, M2_MM1M2, w_idx_MM1M2, cg_vals = sort_by_column_key(
    #     [k, M, i, M1, j, M2, w_idx, cg_vals])

    # kM1M2_MM1M2, kM1M2_seg_MM1M2, (k_kM1M2, i_kM1M2,  M1_kM1M2, j_kM1M2, M2_kM1M2, w_idx_kM1M2) = extract_batch_segments(
    #     [k_MM1M2, i_MM1M2,  M1_MM1M2, j_MM1M2, M2_MM1M2, w_idx_MM1M2]
    # )

    # infos_M_kM1M2 = sparse_scale_infos(index = kM1M2_MM1M2, index_out=M_MM1M2, scale=cg_vals, out_size=out_size)

    # kM1j_batch_kM1M2, kM1j_seg_kM1M2, (k_kM1j, i_kM1j,  M1_kM1j, j_kM1j, w_idx_kM1j) = extract_batch_segments(
    #     [k_kM1M2, i_kM1M2,  M1_kM1M2, j_kM1M2, w_idx_kM1M2]
    # )

    # infos_kM1M2_kM1j = sparse_product_infos(index1=kM1j_batch_kM1M2, index2=M2_kM1M2, in2_size=in2_size)
    # infos_kM1j_M1 = sparse_product_infos(index1=M1_kM1j, index2=w_idx_kM1j, in1_size=in1_size)
    # k_kM1M2M, i_kM1M2M, M1_kM1M2M, j_kM1M2M, M2_kM1M2M, w_idx_kM1M2M, cg_vals, M_kM1M2M = sort_by_column_key(
    #     [k, i, M1, j, M2, w_idx, cg_vals, M])

    # kM1M2_batch_kM1M2M, kM1M2_seg_kM1M2M, (k_kM1M2, i_kM1M2,  M1_kM1M2, j_kM1M2, M2_kM1M2, w_idx_kM1M2) = extract_batch_segments(
    #     [k_kM1M2M, i_kM1M2M,  M1_kM1M2M, j_kM1M2M, M2_kM1M2M, w_idx_kM1M2M]
    # )

    # infos_M_kM1M2 = sparse_scale_infos(index = kM1M2_batch_kM1M2M, index_out=M_kM1M2M, scale=cg_vals, out_size=out_size)

    # kM1j_batch_kM1M2, kM1j_seg_kM1M2, (k_kM1j, i_kM1j,  M1_kM1j, j_kM1j, w_idx_kM1j) = extract_batch_segments(
    #     [k_kM1M2, i_kM1M2,  M1_kM1M2, j_kM1M2, w_idx_kM1M2]
    # )

    # infos_kM1M2_kM1j = sparse_product_infos(index1=kM1j_batch_kM1M2, index2=M2_kM1M2, in2_size=in2_size)
    # infos_kM1j_M1 = sparse_product_infos(index1=M1_kM1j, index2=w_idx_kM1j, in1_size=in1_size)
    # first W
    kijM1M2_batch_MijM1M2, (k_kijM1M2, i_kijM1M2, j_kijM1M2, M1_kijM1M2, M2_kijM1M2, w_idx_kijM1M2) = extract_scatter_indices(
        [k_MM1M2, i_MM1M2, j_MM1M2, M1_MM1M2, M2_MM1M2, w_idx_MM1M2]
    ) 
    kijM1_batch_kM1M2, (k_kijM1, i_kijM1, M1_kijM1, j_kijM1, w_idx_kijM1) = extract_scatter_indices(
        [k_kijM1M2, i_kijM1M2, M1_kijM1M2, j_kijM1M2, w_idx_kijM1M2]
    )

    M_batch_MM1M2, M_seg_MM1M2, (M_out,) = extract_batch_segments(
        [M_MM1M2]
    )


    infos_kM1j_M1 = sparse_product_infos(index1=M1_kijM1, index2=w_idx_kijM1, in1_size=in1_size)
    infos_kM1M2_kM1j = sparse_product_infos(index1=kijM1_batch_kM1M2, index2=M2_kijM1M2, in2_size=in2_size)
    infos_M_kM1M2 = sparse_scale_infos(index = kijM1M2_batch_MijM1M2, index_out=[M_out[M] for M in M_batch_MM1M2], 
                                       scale=cg_vals, out_size=out_size)

    tp_info = TensorProductInfo(
        *infos_inter,
        *infos_M,
        *infos_kM1j_M1,
        *infos_kM1M2_kM1j,
        *infos_M_kM1M2,
        out_size=out_size
    )
    return tp_info


    tp_info = TensorProductInfo(
        *infos_inter,
        *infos_M,
        *infos_kM1j_M1,
        *infos_kM1M2_kM1j,
        *infos_M_kM1M2,
        out_size=out_size
    )
    return tp_info

def tp_info(
        irreps_out: Irreps,
        irreps1: Irreps, irreps2: Irreps,
        path: List[Tuple[int, int, int]]=None,
        path_norm: bool = True,
        channel_norm: bool = False,
        channel_scale: float = 1.0):
    cg_vals, M, M1, M2, k, i, j, w_idx, num_paths = prepare_so3(
        irreps_out, irreps1, irreps2, path, path_norm, channel_norm, channel_scale)

    tp_info = create_tp_info(cg_vals, M, M1, M2, k, i, j, w_idx, irreps_out.dim)
    return tp_info, num_paths


def tp_infos(
        irreps_out: Irreps,
        irreps1: Irreps, irreps2: Irreps,
        path: List[Tuple[int, int, int]]=None,
        path_norm: bool = True,
        channel_norm: bool = False,
        channel_scale: float = 1.0):
    # Note: channel_norm and channel_scale are only applied to the forward pass CG coefficients
    # Backward passes use the original (forward) coefficients
    cg_vals, M, M1, M2, k, i, j, w_idx, num_paths = prepare_so3(
        irreps_out, irreps1, irreps2, path, path_norm, channel_norm, channel_scale)

    tp_forward = create_tp_info(cg_vals, M, M1, M2, k, i, j, w_idx, irreps_out.dim, irreps1.dim, irreps2.dim)
    tp_backward1 = create_tp_info(cg_vals, M1, M2, M, i, j, k, w_idx, irreps1.dim, irreps2.dim, irreps_out.dim)
    tp_backward2 = create_tp_info(cg_vals, M2, M, M1, j, k, i, w_idx, irreps2.dim, irreps_out.dim, irreps1.dim)
    return tp_forward, tp_backward1, tp_backward2, num_paths


def generate_fully_connected_irreps_linear_paths(
        irreps_out: Irreps, irreps_in: Irreps):
    paths = []
    for i, ir_out in enumerate(irreps_out):
        for i0, ir_in in enumerate(irreps_in):
            if has_path(ir_out, ir_in):
                paths.append((i,i0))
    return paths


def prepare_irreps_linear(irreps_out, irreps_in, path, path_norm, channel_norm, channel_scale: float = 1.0):

    if not path:
        path = generate_fully_connected_irreps_linear_paths(irreps_out, irreps_in)

    seg_out = irrep_segments(irreps_out)
    seg_in = irrep_segments(irreps_in)
    
    current_i = None
    path_count = {}
    for (i,i0) in path:
        if i == current_i:
            path_count[i] += 1
        else: 
            path_count[i] = 1
            current_i = i

    scales = []
    M_s = []
    M0_s = []
    i_s = []
    i0_s = []
    ii0_s = []

    for w_idx, (i,i0) in enumerate(path):
        l = irreps_out[i].l
        norm_scale = path_count[i] ** (-0.5) if path_norm else 1
        if channel_norm:
            norm_scale *= channel_scale
        for m in range(0, 2*l+1):
            scales.append(norm_scale)
            M_s.append(seg_out[i]+m)
            M0_s.append(seg_in[i0]+m)
            i_s.append(i)
            i0_s.append(i0)
            ii0_s.append(w_idx)

    return scales, M_s, M0_s, i_s, i0_s, ii0_s, len(path)

def create_irreps_linear_info(scales, M_s, M0_s, ii0_s, out_size):
    
    M_MM0, M0_MM0, ii0_MM0, scales_MM0 = sort_by_column_key(
        [M_s, M0_s, ii0_s, scales]
    )

    M_batch_MM0, M_seg_MM0, (M,) = extract_batch_segments(
        [M_MM0]
    )

    # for grad on weight
    ii0_ii0MM0, M_ii0MM0, M0_ii0MM0, scales_ii0MM0 = sort_by_column_key(
        [ii0_s, M_s, M0_s, scales]
    )

    ii0_batch_ii0MM0, ii0_seg_ii0MM0, [ii0_ii0, scales_ii0] = extract_batch_segments(
        [ii0_ii0MM0, scales_ii0MM0]
    )

    return IrrepsLinearInfo(
        tensor(scales_MM0), 
        tensor(M_seg_MM0), 
        tensor(ii0_MM0), 
        tensor(M0_MM0), tensor(M_MM0), 
        tensor(M),
        tensor(ii0_seg_ii0MM0), 
        tensor(M_ii0MM0), tensor(M0_ii0MM0), tensor(scales_ii0),
        out_size
    )


# def irreps_linear_info(
#         irreps_out: Irreps, irreps_in: Irreps, 
#         path: List[Tuple[int, int]]=None,
#         path_norm: bool = True):
#     scales, M_s, M0_s, i_s, i0_s, ii0_s, num_paths = prepare_irreps_linear(
#         irreps_out, irreps_in, path, path_norm)

#     irreps_linear_info = create_irreps_linear_info(
#         scales, M_s, M0_s, ii0_s, irreps_out.dim
#     )

#     return irreps_linear_info, num_paths


def irreps_linear_infos(
        irreps_out: Irreps, irreps_in: Irreps, 
        path: List[Tuple[int, int]]=None,
        path_norm: bool = True, channel_norm:bool=False, channel_scale: float = 1.0):
    
    scales, M_s, M0_s, i_s, i0_s, ii0_s, num_paths = prepare_irreps_linear(
        irreps_out, irreps_in, path, path_norm, channel_norm, channel_scale)

    irreps_linear_info_forward = create_irreps_linear_info(scales, M_s, M0_s, ii0_s, irreps_out.dim)
    irreps_linear_info_backward = create_irreps_linear_info(scales, M0_s, M_s, ii0_s, irreps_in.dim)

    return irreps_linear_info_forward, irreps_linear_info_backward, num_paths


def irreps_info(irreps:Irreps):
    seg = []
    idx = []
    rsqrt_dim = []
    rdims = []
    acc = 0
    for i,irrep in enumerate(irreps):
        seg.append(acc)
        idx.extend(irrep.dim * [i])
        acc += irrep.dim
        rsqrt_dim.append(irrep.dim ** (-0.5))
        rdims.append(1/irrep.dim)
    seg.append(acc)
    return IrrepsInfo(tensor(rsqrt_dim), tensor(rdims), tensor(idx), tensor(seg), len(irreps))


def prepare_z_rotation(irreps):
    indices1: List[int] = [] # Index into x (input1)
    indices2: List[int] = [] # Index into cs (input2)
    indices_out: List[int] = [] # Output index M
    scales: List[float] = [] # Scale factor sigma_t

    # 1. Create mapping from M -> (irrep_idx, l, m) and (irrep_idx, l, m) -> M
    m_to_ilm: Dict[int, Tuple[int, int, int]] = {}
    lm_to_M: Dict[Tuple[int, int, int], int] = {}
    current_m_index = 0
    max_l = 0
    for irrep_idx, irrep in enumerate(irreps):
        l = irrep.l
        max_l = max(max_l, l)
        for m in range(-l, l + 1):
            m_to_ilm[current_m_index] = (irrep_idx, l, m)
            # Use irrep_idx from the original list as part of the key
            lm_to_M[(irrep_idx, l, m)] = current_m_index
            current_m_index += 1

    if irreps.dim != current_m_index:
        # This indicates an issue with Irreps definition or parsing
        raise RuntimeError(f"Calculated dimension ({current_m_index}) does not match irreps.dim ({irreps.dim})")

    cs_dim = 1 + 2 * max_l # Size of the dimension in cs tensor

    # 2. Iterate through output indices M and generate sparse interactions
    for M in range(irreps.dim):
        irrep_idx, l, m = m_to_ilm[M]

        if m == 0:
            # x'_{M} = 1.0 * x_{M}
            # Interaction t: M_out=M, scale=1.0, cs_idx=0 (for 1.0), x_idx=M
            indices_out.append(M)
            scales.append(1.0)
            indices1.append(M)
            indices2.append(0) # Index 0 in cs holds 1.0
        else:
            # x'_{M} = cos(m*phi) * x_{M} + sin(m*phi) * x_{M(-m)}
            abs_m = abs(m)
            sign_m = float(m > 0) - float(m < 0)

            # Find M_neg_m index for the same irrep_idx and l, but opposite m
            M_neg_m = lm_to_M.get((irrep_idx, l, -m))
            if M_neg_m is None:
                 # This should not happen for valid irreps
                 raise ValueError(f"Could not find index for irrep_idx={irrep_idx}, l={l}, m={-m} corresponding to M={M}")

            # Interaction t1 (cos term): M_out=M, scale=1.0, cs_idx=2*|m|, x_idx=M
            indices_out.append(M)
            scales.append(1.0)
            indices1.append(M)
            indices2.append(2 * abs_m) # Index for cos(m*phi)

            # Interaction t2 (sin term): M_out=M, scale=-sign(m), cs_idx=2*|m|-1, x_idx=M_neg_m
            indices_out.append(M)
            scales.append(-sign_m)
            indices1.append(M_neg_m) # Index into x for the sin term
            indices2.append(2 * abs_m - 1) # Index for sin(|m|*phi)

    return indices1, indices2, indices_out, scales, cs_dim


def z_rotation_infos(irreps: Irreps) -> Tuple[SparseProductInfo, SparseProductInfo, SparseProductInfo]:

    indices1, indices2, indices_out, scales, cs_dim = prepare_z_rotation(irreps)

    info = sparse_product_info(
        index1=indices1,    # Indices into x tensor (Input 1)
        index2=indices2,    # Indices into cs tensor (Input 2)
        index=indices_out,   # Defines output index M for segmentation
        scale=scales,        # Scaling factor sigma_t
        out_size=irreps.dim, # Size of the output sparse dimension (x')
    )
    return info

def z_rotation_infos(irreps: Irreps) -> Tuple[SparseProductInfo, SparseProductInfo, SparseProductInfo]:
    """
    Generates SparseProductInfo for performing z-axis rotation on a tensor
    with the given Irreps structure using the sparse_scavec operation.

    Rotation formula:
        m=0: x'_{nimc} = x_{ni0c}
        m!=0: x'_{nimc} = cos(m*phi) * x_{nimc} + sin(m*phi) * x_{ni(-m)c}

    Assumes sparse_scavec computes:
        output[n, M] = sum_t scale[t] * input[n, index2[t]] * cs[n, index1[t]]
    where the sum is segmented by the output index M (implicit via index_out).

    Assumed `cs` Tensor Structure (Input 2): shape (N, cs_dim)
        cs_dim = 1 + 2 * max_l
        cs[:, 0] = 1.0
        cs[:, 2*m - 1] = sin(m*phi) for m = 1..max_l
        cs[:, 2*m]     = cos(m*phi) for m = 1..max_l

    Args:
        irreps: The Irreps object describing the geometric structure of the
                input tensor (Input 1, shape (N, irreps.dim, C)) and
                output tensor (shape (N, irreps.dim, C)).

    Returns:
        A tuple (info_fwd, info_bwd1, info_bwd2) containing SparseProductInfo
        objects for the forward pass and gradients w.r.t. x (input1) and
        cs (input2).
    """

    indices1, indices2, indices_out, scales, cs_dim = prepare_z_rotation(irreps)

    # 3. Create SparseProductInfo using the helper that returns all three infos
    # sparse_product_infos handles sorting by index_out ('index' arg) and creates segments.
    # It requires sizes for input1 (x), input2 (cs), and output (x').
    # Input1 (x) size = irreps.dim
    # Input2 (cs) size = cs_dim
    # Output (x') size = irreps.dim
    infos = sparse_product_infos(
        index1=indices1,    # Indices into x tensor (Input 1)
        index2=indices2,    # Indices into cs tensor (Input 2)
        index=indices_out,   # Defines output index M for segmentation
        scale=scales,        # Scaling factor sigma_t
        out_size=irreps.dim, # Size of the output sparse dimension (x')
        in1_size=irreps.dim,  # Size of the input1 sparse dimension (x)
        in2_size=cs_dim,     # Size of the input2 sparse dimension (cs)
    )
    # info_fwd, info_bwd1 (grad_cs), info_bwd2 (grad_x)
    return infos


def j_matrix_info(irreps: Irreps) -> SparseScaleInfo:
    """
    Generates SparseScaleInfo for multiplying by the J matrix by extracting
    non-zero elements from the dense blocks computed by the user's j_matrix(l) function.

    Operation: y_M' = sum_M J_{M', M} x_M

    Args:
        irreps: The Irreps object describing the geometric structure.

    Returns:
        A SparseScaleInfo object for the forward pass.
    """
    indices_in: List[int] = []  # Input index M (column index of J)
    indices_out: List[int] = [] # Output index M' (row index of J)
    scales: List[float] = []    # Non-zero value J_{M', M}

    current_offset = 0
    # Correct iteration: Use irreps directly which respects multiplicity
    for irrep in irreps: # This iterates through each Irrep instance, including multiplicities
        l = irrep.l
        dim = irrep.dim
        # Get the dense J matrix block for this l using the provided function
        # Convert from sympy Matrix to numpy array if necessary
        j_dense_block = j_matrix(l)

        # Iterate through the dense block elements for this irrep instance
        for row_idx in range(dim): # Corresponds to m'
            for col_idx in range(dim): # Corresponds to m
                scale_val = j_dense_block[row_idx, col_idx]
                if abs(scale_val) > 1e-9: # Check for non-zero with tolerance
                    # Map block indices to flattened indices using current_offset
                    M_out = current_offset + row_idx
                    M_in = current_offset + col_idx

                    indices_out.append(M_out)
                    indices_in.append(M_in)
                    scales.append(float(scale_val))
        # Move offset to the start of the next irrep instance's block
        current_offset += dim

    # Create SparseScaleInfo using the singular helper from _structs
    # sparse_scale_info handles sorting by index_out and creating segments.
    info = sparse_scale_info(
        index=indices_in,     # Indices into input tensor's M dimension (col index)
        index_out=indices_out,# Defines output index M (row index)
        scale=scales,         # Scaling factor (non-zero matrix element)
        out_size=irreps.dim,  # Size of the output sparse dimension
    )
    return info

def wigner_d_info(irreps: Irreps) -> WignerRotationInfo:
    """Prepares all necessary info objects for arbitrary Wigner D rotation."""
    irreps = check_irreps(irreps)
    # Get z-rotation infos (fwd, bwd_x, bwd_cs)
    dz_info_fwd, dz_info_bwd_x, dz_info_bwd_cs = z_rotation_infos(irreps)
    # Get J matrix info (forward pass only needed)
    j_info = j_matrix_info(irreps)
    sign = [ir.p ** ir.l for ir in irreps for _ in ir]

    return WignerRotationInfo(
        j_matrix_info=j_info,
        rotate_z_info_fwd=dz_info_fwd,
        rotate_z_info_bwd_input=dz_info_bwd_x, # Assuming this is grad_x info
        rotate_z_info_bwd_cs=dz_info_bwd_cs,
        max_m = max(ir.l for ir in irreps),
        sign = tensor(sign)
    )

def irreps_blocks_infos(irreps: Irreps) -> SparseProductInfo:

    irreps = check_irreps(irreps)

    acc = 0
    index_in = []
    index_out = []
    index_mat = []
    for ir in irreps:
        for M_out in range(acc, acc+ir.dim):
            for M_in in range(acc, acc+ir.dim):
                index_in.append(M_in)
                index_out.append(M_out)
                index_mat.append(len(index_mat))
        acc += ir.dim
    return sparse_product_infos(index_mat, index_in, index_out)


def prepare_so2_linear(irreps_out, irreps_in, path=None, path_norm=True, channel_norm=False, channel_scale=1.0):
    if path is None:
        path = [(k, i) for k in range(len(irreps_out)) for i in range(len(irreps_in))]
    # 1. Create mapping from M -> (irrep_idx, l, m) and (irrep_idx, l, m) -> M
    m_to_ilm_in: Dict[int, Tuple[int, int, int]] = {}
    ilm_to_M_in: Dict[Tuple[int, int, int], int] = {}
    current_m_index = 0
    max_l = 0
    for irrep_idx, irrep in enumerate(irreps_in):
        l = irrep.l
        max_l = max(max_l, l)
        for m in range(-l, l + 1):
            m_to_ilm_in[current_m_index] = (irrep_idx, l, m)
            # Use irrep_idx from the original list as part of the key
            ilm_to_M_in[(irrep_idx, l, m)] = current_m_index
            current_m_index += 1

    m_to_klm_out: Dict[int, Tuple[int, int, int]] = {}
    klm_to_M_out: Dict[Tuple[int, int, int], int] = {}
    current_m_index = 0
    max_l = 0
    for irrep_idx, irrep in enumerate(irreps_out):
        l = irrep.l
        max_l = max(max_l, l)
        for m in range(-l, l + 1):
            m_to_klm_out[current_m_index] = (irrep_idx, l, m)
            # Use irrep_idx from the original list as part of the key
            klm_to_M_out[(irrep_idx, l, m)] = current_m_index
            current_m_index += 1

    indices1: List[int] = [] # Index into x (input1)
    indices2: List[int] = [] # Index into weights (input2)
    indices_out: List[int] = [] # Output index M
    scales: List[float] = [] # Scale factor: 

    # 2. Iterate through output indices M and generate sparse interactions
    kim_to_w_idx: Dict[Tuple[int, int, int], int] = {}
    w_idx_to_kim: Dict[int, Tuple[int, int, int]] = {}
    path_count_km_out: Dict[Tuple[int, int], int] = {}
    # for ir_idx_out, ir_out in enumerate(irreps_out):
    #     for ir_idx_in, ir_in in enumerate(irreps_in):
    for ir_idx_out, ir_idx_in in path:
        ir_out = irreps_out[ir_idx_out]
        ir_in = irreps_in[ir_idx_in]
        for m in range(0, min(ir_out.l, ir_in.l)+1):
            if m == 0:
                w_idx_to_kim[len(w_idx_to_kim)] = (ir_idx_out, ir_idx_in, m)
                kim_to_w_idx[(ir_idx_out, ir_idx_in, m)] = len(w_idx_to_kim) - 1
                path_count_km_out[(ir_idx_out, m)] = path_count_km_out.get((ir_idx_out, m),0)+1
            else:
                w_idx_to_kim[len(w_idx_to_kim)] = (ir_idx_out, ir_idx_in, -m)
                kim_to_w_idx[(ir_idx_out, ir_idx_in, -m)] = len(w_idx_to_kim) - 1
                path_count_km_out[(ir_idx_out, m)] = path_count_km_out.get((ir_idx_out, -m),0)+1
                w_idx_to_kim[len(w_idx_to_kim)] = (ir_idx_out, ir_idx_in, m)
                kim_to_w_idx[(ir_idx_out, ir_idx_in, m)] = len(w_idx_to_kim) - 1
                path_count_km_out[(ir_idx_out, m)] = path_count_km_out.get((ir_idx_out, -m),0)+1
    
    for ir_idx_out, ir_idx_in in path:
        ir_out = irreps_out[ir_idx_out]
        ir_in = irreps_in[ir_idx_in]
        for m in range(0, min(ir_out.l, ir_in.l)+1):
            if m == 0:
                if path_norm:
                    scales.append(path_count_km_out[(ir_idx_out, m)] ** (-0.5))
                else:
                    scales.append(1.0)  
                indices_out.append(klm_to_M_out[(ir_idx_out, ir_out.l, m)])
                indices1.append(ilm_to_M_in[(ir_idx_in, ir_in.l, m)])
                indices2.append(kim_to_w_idx[(ir_idx_out, ir_idx_in, m)])
            else:
                if path_norm:
                    scales.append((path_count_km_out[(ir_idx_out, m)]*2) ** (-0.5))
                else:
                    scales.append(2**(-0.5))
                indices_out.append(klm_to_M_out[(ir_idx_out, ir_out.l, m)])
                indices1.append(ilm_to_M_in[(ir_idx_in, ir_in.l, m)])         
                indices2.append(kim_to_w_idx[(ir_idx_out, ir_idx_in, m)])         

                if path_norm:
                    scales.append((path_count_km_out[(ir_idx_out, m)]*2) ** (-0.5))
                else:
                    scales.append(2**(-0.5))
                indices_out.append(klm_to_M_out[(ir_idx_out, ir_out.l, -m)])
                indices1.append(ilm_to_M_in[(ir_idx_in, ir_in.l, -m)])         
                indices2.append(kim_to_w_idx[(ir_idx_out, ir_idx_in, m)])         

                if path_norm:
                    scales.append(-(path_count_km_out[(ir_idx_out, m)]*2) ** (-0.5))
                else:
                    scales.append(-2**(-0.5))    
                indices_out.append(klm_to_M_out[(ir_idx_out, ir_out.l, m)])
                indices1.append(ilm_to_M_in[(ir_idx_in, ir_in.l, -m)])         
                indices2.append(kim_to_w_idx[(ir_idx_out, ir_idx_in, -m)])         

                if path_norm:
                    scales.append((path_count_km_out[(ir_idx_out, m)]*2) ** (-0.5))
                else:
                    scales.append(2**(-0.5))
                indices_out.append(klm_to_M_out[(ir_idx_out, ir_out.l, -m)])
                indices1.append(ilm_to_M_in[(ir_idx_in, ir_in.l, m)])         
                indices2.append(kim_to_w_idx[(ir_idx_out, ir_idx_in, -m)])         
    if channel_norm:
        scales = [s * channel_scale for s in scales]
    num_weights = len(w_idx_to_kim)
    return indices1, indices2, indices_out, scales, num_weights

def so2_linear_info(irreps_out, irreps_in, path=None, path_norm=True, channel_norm=False, channel_scale=1.0):
    indices1, indices2, indices_out, scales, num_weights = prepare_so2_linear(irreps_out, irreps_in, path=path, path_norm=path_norm, channel_norm=channel_norm, channel_scale=channel_scale)
    return sparse_product_info(indices1, indices2, indices_out, scales, irreps_out.dim), num_weights

def so2_linear_infos(irreps_out, irreps_in, path=None, path_norm=True, channel_norm=False, channel_scale=1.0):
    indices1, indices2, indices_out, scales, num_weights = prepare_so2_linear(irreps_out, irreps_in, path=path, path_norm=path_norm, channel_norm=channel_norm, channel_scale=channel_scale)
    return *sparse_product_infos(indices1, indices2, indices_out, scales, irreps_out.dim, irreps_in.dim, num_weights), num_weights
