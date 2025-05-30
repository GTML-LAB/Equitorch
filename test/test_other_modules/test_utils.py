import traceback
import e3nn.math
import torch
from torch.profiler import profile, record_function, ProfilerActivity

import equitorch
from equitorch.irreps import check_irreps, Irreps
import equitorch.nn
import equitorch.nn.functional
import equitorch.nn.functional.rotations
import equitorch.nn.functional.wigner_d
import equitorch.utils
import equitorch.utils._random
import equitorch.utils._structs


def rand_irreps_feature(irreps: Irreps, N:int, C:int=None, dtype=None, device=None):
    irreps = check_irreps(irreps)
    if C is not None:
        return torch.randn(N,irreps.dim,C,dtype=dtype, device=device)
    else:
        return torch.randn(N,irreps.dim,dtype=dtype, device=device)

def rand_rotation_dict(irreps: Irreps, N:int, dtype=None, device=None):
    
    irreps = check_irreps(irreps)

    a, b, c = equitorch.utils._random.rand_rotation_angles(N, dtype=dtype, device=device)

    rotation_matrix = equitorch.nn.functional.rotations.angles_to_matrix(a,b,c)
    wigner_d_info = equitorch.utils._structs.wigner_d_info(irreps).to(dtype=dtype, device=device)
    wigner_D = equitorch.nn.functional.wigner_d.wigner_d_matrix(torch.eye(irreps.dim, device=device, dtype=dtype), a,b,c, info=wigner_d_info)

    return {
        "irreps": irreps,
        "angles": (a,b,c),
        "matrix": rotation_matrix,
        "wigner_D": wigner_D
    }

def profile_func(func, func_name, trace_name, repeat, *args, **kwargs):
    activities = [
        ProfilerActivity.CUDA,
        ProfilerActivity.CPU,
        # ProfilerActivity.XPU
    ]

    # for _ in range(warmup):
    #     func(*args, **kwargs)
    with profile(activities=activities, record_shapes=True) as prof:
        with record_function(func_name):
            for _ in range(repeat):
                func(*args, **kwargs)
    if trace_name is not None:
        prof.export_chrome_trace(trace_name)
    print(prof.key_averages().table())

def max_abs_diff(x,y, transform_x=None, transform_y=None):
    if transform_x is not None:
        x = transform_x(x)
    if transform_y is not None:
        y = transform_y(y)
    return (x - y).abs().max().item() 


def max_abs_diff_list(list1, list2):
    return [max_abs_diff(a,b) for a,b in zip(list1, list2)]
def rate_mean2_std(x, y):
    rate = (x/y).nan_to_num(nan=1)
    return (rate.mean().item()**2, rate.std().item())

def rate_mean2_std_list(list1, list2):
    return [rate_mean2_std(a,b) for a,b in zip(list1, list2)]



def compare_funcs(
        funcs,
        compare_func,
        *args,
        **kwargs
):
    res = []
    for func in funcs:
        try:
            res.append(func(*args, **kwargs))
        except torch.OutOfMemoryError as e:
            traceback.print_exc()
            res.append(torch.tensor(float('nan')))
    compared = [[compare_func(ri,rj) for ri in res] for rj in res]
    for i, ci in enumerate(compared):
        for j, cij in enumerate(ci):
            print(f"compare {i} and {j}: \t {cij}")
            
def profile_funcs(
        funcs,
        names,
        trace_name_func,
        repeat,
        *args,
        **kwargs
):
    if trace_name_func is None:
        trace_name_func = lambda *args, **kwargs: None
    for func, name in zip(funcs, names):
        try:
            profile_func(func, name, trace_name_func(name, repeat), repeat, 
                        *args, **kwargs)
        except torch.OutOfMemoryError as e:
            traceback.print_exc()


import time
import torch
from torch.profiler import profile, ProfilerActivity, record_function
import traceback
from typing import Callable, Dict, Tuple, List, Any, Optional
import itertools

class FunctionTester:
    def __init__(self, funcs_to_test: Dict[str, Tuple[Callable, list, dict]], test_data: Any = None):
        """
        初始化函数测试器
        
        参数:
            funcs_to_test: 字典，格式为 {'函数名': (函数, 位置参数列表, 关键字参数字典)}
            test_data: 可选的测试数据，可以在测试中使用
        """
        self.funcs_to_test = funcs_to_test
        self.test_data = test_data
    
    def compare(self, post_transforms: Optional[List[Callable]] = None, 
                         compare_func: Callable = max_abs_diff) -> Dict[Tuple[str, str], bool]:
        """
        执行一致性测试，比较所有函数的结果是否一致
        
        参数:
            post_transforms: 可选的转换函数列表，应用于每个函数的结果
            compare_func: 用于比较两个结果的函数，默认使用 == 操作符
            
        返回:
            字典，键是函数名对，值是比较结果
        """
        results = {}
        transformed_results = []
        func_names = []
        
        # 收集所有函数的结果
        for func_name, (func, args, kwargs) in self.funcs_to_test.items():
            res = func(*args, **kwargs)
            if post_transforms is not None:
                if isinstance(post_transforms, list):
                    # 为应用对应的转换
                    idx = list(self.funcs_to_test.keys()).index(func_name)
                    if idx < len(post_transforms):
                        res = post_transforms[idx](res)
                else:
                    # 应用相同的转换
                    res = post_transforms(res)
            transformed_results.append(res)
            func_names.append(func_name)
        
        # 执行两两比较
        comparisons = {}
        for (i, name1), (j, name2) in itertools.combinations(enumerate(func_names), 2):
            comp_result = compare_func(transformed_results[i], transformed_results[j])
            comparisons[(name1, name2)] = comp_result
        
        return comparisons
    
    def profile(self, repeat: int = 1, trace_name_func: Optional[Callable] = None):
        """
        对所有函数进行性能分析
        
        参数:
            repeat: 每个函数执行的次数
            trace_name_func: 生成跟踪文件名的函数，接受函数名和repeat作为参数
        """
        if trace_name_func is None:
            trace_name_func = lambda name, r: None
            
        for func_name, (func, args, kwargs) in self.funcs_to_test.items():
            try:
                self._profile_func(
                    func, 
                    func_name, 
                    trace_name_func(func_name, repeat), 
                    repeat, 
                    *args, 
                    **kwargs
                )
            except torch.OutOfMemoryError as e:
                traceback.print_exc()
    
    def _profile_func(self, func: Callable, func_name: str, trace_name: Optional[str], 
                     repeat: int, *args, **kwargs):
        """
        内部方法: 分析单个函数的性能
        """
        activities = [
            ProfilerActivity.CUDA,
            ProfilerActivity.CPU,
        ]
        
        with profile(activities=activities, record_shapes=True) as prof:
            with record_function(func_name):
                for _ in range(repeat):
                    func(*args, **kwargs)
        
        if trace_name is not None:
            prof.export_chrome_trace(trace_name)
        print(prof.key_averages().table())
    
    def timeit(self, repeat: int = 1, warmup: int = 0) -> Dict[str, float]:
        """
        简单计时测试
        
        参数:
            repeat: 每个函数执行的次数
            warmup: 预热次数
            
        返回:
            字典，包含每个函数的平均执行时间(秒)
        """
        timings = {}
        
        for func_name, (func, args, kwargs) in self.funcs_to_test.items():
            # 预热
            for _ in range(warmup):
                func(*args, **kwargs)
            
            # 计时
            start = time.time()
            for _ in range(repeat):
                func(*args, **kwargs)
            end = time.time()
            
            avg_time = (end - start) / repeat
            timings[func_name] = avg_time
        
        return timings