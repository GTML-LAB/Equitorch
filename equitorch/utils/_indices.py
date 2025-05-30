import bisect
from typing import List, Tuple, Any

import torch

def expand_left(source: torch.Tensor, target:torch.Tensor, dim:int):
    if dim < 0:
        dim = source.ndim + dim
    target = target.view([1]*dim+[-1])
    return target

def extract_batch_segments(keys: List[List[int]]):
    r"""
    Process sorted integer key lists to generate batch indices, boundary pointers, and key values.

    Parameters
    ----------
    keys : List[List[int]]
        A list of sorted integer key lists. All lists must have the same length.

    Returns
    -------
    batch : List[int]
        A list where each element indicates the batch index it belongs to.
    seg : List[int]
        A list of boundary pointers indicating the start and end of each batch.
    val : List[List[int]]
        A list of lists containing the key values at the boundary points for each key list.

    Notes
    -----
    - The input key lists must be sorted in ascending order.
    - If the input is empty, the function returns empty lists for `batch`, `seg`, and `val`.

    Examples
    --------
    >>> keys = [
    ...     [1, 1, 2, 2],
    ...     [1, 1, 2, 2]
    ... ]
    >>> extract_batch_seg_native(keys)
    ([0, 0, 1, 1], [0, 2, 4], [[1, 2], [1, 2]])

    >>> keys = [
    ...     [5, 5, 5],
    ...     [5, 5, 5]
    ... ]
    >>> extract_batch_seg_native(keys)
    ([0, 0, 0], [0, 3], [[5], [5]])

    >>> keys = [
    ...     [1, 1, 2, 3, 3],
    ...     [1, 2, 2, 3, 3]
    ... ]
    >>> extract_batch_seg_native(keys)
    ([0, 1, 2, 3, 3], [0, 1, 2, 3, 5], [[1, 1, 2, 3], [1, 2, 2, 3]])
    """
    if not keys or not keys[0]:
        return [], [], []
    
    length = len(keys[0])
    seg = [0]  # 初始化分界指针
    
    # 生成分界指针
    for i in range(length):
        last_idx = seg[-1]
        # 检查所有键在当前索引i处是否与上一个分界点的值不同
        if any(key[i] != key[last_idx] for key in keys):
            seg.append(i)
    
    seg.append(length)  # 添加最终边界
    
    # 生成批次索引
    batch = [0] * length
    for batch_idx in range(1, len(seg)):
        start = seg[batch_idx-1]
        end = seg[batch_idx]
        for i in range(start, end):
            batch[i] = batch_idx - 1
    
    # 提取分界点键值
    val = [
        [key[boundary] for boundary in seg[:-1]]  # 排除最后一个边界
        for key in keys
    ]
    
    return batch, seg, val

def sort_by_column_key(to_sort: List[List[Any]], key: List[List[Any]] = None) -> List[List[Any]]:
    """
    Sort the columns of the first 2D list based on the column-wise lexicographical order of the key 2D list.

    Parameters
    ----------
    to_sort : List[List[Any]]
        The first 2D list whose columns are to be sorted.
    key : List[List[Any]]
        The key 2D list used to determine the sorting order of columns.

    Returns
    -------
    List[List[Any]]
        The first 2D list with columns sorted according to the column-wise lexicographical order of the key.

    Raises
    ------
    ValueError
        If either `to_sort` or `key` is empty, or if their lengths do not match.

    Examples
    --------
    >>> to_sort = [[1, 2, 3], 
    ...            [4, 5, 6]]
    >>> key = [[2, 1, 3], 
    ...        [1, 3, 2]]
    >>> sort_by_column_key(to_sort, key)
    [[2, 1, 3], 
     [5, 4, 6]]
    """

    if key is None:
        key = to_sort

    # 将两个列表转置为列优先的形式
    to_sort_transposed = list(zip(*to_sort))
    key_transposed = list(zip(*key))
    # 将转置后的列表组合成 [(to_sort_col, key_col)] 的形式
    combined = list(zip(to_sort_transposed, key_transposed))
    # 根据 key 的列字典序进行排序
    sorted_combined = sorted(combined, key=lambda x: x[1])
    # 提取排序后的 to_sort 的列
    sorted_to_sort_transposed = [item[0] for item in sorted_combined]
    # 将转置后的结果还原为原始形式
    sorted_to_sort = list(zip(*sorted_to_sort_transposed))
    # 将元组转换为列表
    sorted_to_sort = [list(row) for row in sorted_to_sort]

    return sorted_to_sort

def extract_scatter_indices(keys: List[List[int]]) -> Tuple[List[int], List[List[int]]]:
    """
    Process integer key lists to generate scatter indices and sorted unique keys.

    Parameters
    ----------
    keys : List[List[int]]
        A list of integer key lists. All lists must have the same length.

    Returns
    -------
    indices : List[int]
        A list where each element is the index of the corresponding key tuple in the sorted unique list.
    scatter_keys : List[List[int]]
        A list of lists containing the sorted unique key values for each original key list.

    Notes
    -----
    - If the input is empty, the function returns empty lists for `indices` and `scatter_keys`.

    Examples
    --------
    >>> keys = [
    ...     [1, 1, 2, 2],
    ...     [1, 1, 2, 2]
    ... ]
    >>> extract_scatter_indices(keys)
    ([0, 0, 1, 1], [[1, 2], [1, 2]])

    >>> keys = [
    ...     [5, 5, 5],
    ...     [5, 5, 5]
    ... ]
    >>> extract_scatter_indices(keys)
    ([0, 0, 0], [[5], [5]])

    >>> keys = [
    ...     [1, 1, 2, 3, 3],
    ...     [1, 2, 2, 3, 3]
    ... ]
    >>> extract_scatter_indices(keys)
    ([0, 1, 2, 3, 3], [[1, 1, 2, 3], [1, 2, 2, 3]])
    """
    if not keys or not keys[0]:
        return [], []
    
    # 将每个位置的键打包成元组
    tuple_list = list(zip(*keys))
    
    # 排序并去重得到唯一的元组列表
    scatter = sorted(set(tuple_list))
    
    # 生成每个原始元组的索引
    indices = []
    for t in tuple_list:
        idx = bisect.bisect_left(scatter, t)
        indices.append(idx)
    
    # 解压唯一元组列表为各键列表
    if not scatter:
        scatter_keys = []
    else:
        scatter_keys = [list(sk) for sk in zip(*scatter)]
    
    return indices, scatter_keys
