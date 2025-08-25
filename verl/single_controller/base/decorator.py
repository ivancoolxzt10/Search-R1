# 版权声明，表明代码归属 Bytedance Ltd. 及其关联公司所有
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# 许可证声明，采用 Apache License 2.0，允许合规使用和分发
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# you may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非法律要求或书面同意，否则按“原样”分发，不提供任何明示或暗示的担保
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入 inspect 模块，用于检查对象类型和参数
import inspect
# 导入 partial 和 wraps，用于函数式编程和装饰器实现
from functools import partial, wraps
# 导入 FunctionType，用于类型判断
from types import FunctionType

# 导入分布式协议相关的 DataProtoFuture 和 _padding_size_key
from verl.protocol import DataProtoFuture, _padding_size_key
# 导入动态枚举工具类 DynamicEnum
from verl.utils.py_functional import DynamicEnum

# 定义魔法属性名，避免用户自定义函数冲突
MAGIC_ATTR = "attrs_3141562937"

# 定义分发模式枚举类，用于分布式计算的数据分发策略
class Dispatch(DynamicEnum):
    """分布式计算的分发模式枚举类。
    每种模式代表一种数据在分布式系统中分发的策略，
    用于控制数据在不同 worker group 间的分区和处理方式。
    """

    # 注册表，用于存储所有分发模式
    _registry = {}
    # 下一个分发模式的枚举值
    _next_value = 0


# 初始化预定义分发模式，将常用模式注册到 Dispatch 枚举类
# 包括 RANK_ZERO、ONE_TO_ALL、ALL_TO_ALL 等常见分布式策略
# DIRECT_ROLLOUT_METHOD 为 vllm 特殊分发模式
def init_predefined_dispatch_mode():
    Dispatch.register("RANK_ZERO")
    Dispatch.register("ONE_TO_ALL")
    Dispatch.register("ALL_TO_ALL")
    Dispatch.register("DP_COMPUTE")
    Dispatch.register("DP_COMPUTE_PROTO")
    Dispatch.register("DP_COMPUTE_PROTO_WITH_FUNC")
    Dispatch.register("DP_COMPUTE_METRIC")
    # vllm 专用分发模式
    Dispatch.register("DIRECT_ROLLOUT_METHOD")


# 定义 Execute 枚举类，用于后续分布式执行策略
class Execute(DynamicEnum):
    """Enum class defining different execution modes for distributed computation.

    These modes control how a function should be executed across different ranks
    in a distributed system.
    """

    _registry = {}
    _next_value = 0


# 初始化预定义执行模式，将常用执行策略注册到 Execute 枚举类
# 包括 ALL 和 RANK_ZERO 两种模式
def init_predefined_execute_mode():
    Execute.register("ALL")
    Execute.register("RANK_ZERO")


# 初始化分发和执行的动态枚举类
init_predefined_dispatch_mode()
init_predefined_execute_mode()


# 将输入的 args 和 kwargs 按照指定的 chunks 拆分为多个部分
# 每个部分的大小由 chunks 决定，适用于 DataProto 和 DataProtoFuture 类型
def _split_args_kwargs_data_proto(chunks, *args, **kwargs):
    from verl.protocol import DataProto, DataProtoFuture

    splitted_args = []
    for arg in args:
        assert isinstance(arg, DataProto | DataProtoFuture)
        splitted_args.append(arg.chunk(chunks=chunks))

    splitted_kwargs = {}
    for key, val in kwargs.items():
        assert isinstance(val, DataProto | DataProtoFuture)
        splitted_kwargs[key] = val.chunk(chunks=chunks)

    return splitted_args, splitted_kwargs


# 拆分参数的同时自动进行填充，确保每个 DataProto 对象的长度一致
# 填充大小由 padding_size 决定，并作为关键字参数传递
def _split_args_kwargs_data_proto_with_auto_padding(chunks, *args, **kwargs):
    from verl.protocol import DataProto, DataProtoFuture

    data_proto_len = None
    padding_size = None

    def _padding_and_split_data(obj, chunks):
        nonlocal data_proto_len, padding_size
        assert isinstance(obj, DataProto | DataProtoFuture)
        if isinstance(obj, DataProto) and obj.is_padding_enabled():
            # 对于需要填充的 DataProto，确保所有对象具有相同的长度
            if data_proto_len is None:
                data_proto_len = len(obj)
                padding_size = (chunks - (data_proto_len % chunks)) if (data_proto_len % chunks > 0) else 0
            else:
                assert data_proto_len == len(obj), (
                    f"expecting all arg share same length of {data_proto_len}, but got {len(obj)}"
                )
            obj.padding(padding_size=padding_size)
        return obj.chunk(chunks=chunks)

    splitted_args = [_padding_and_split_data(arg, chunks) for arg in args]
    splitted_kwargs = {key: _padding_and_split_data(val, chunks) for key, val in kwargs.items()}
    if padding_size is not None:
        splitted_kwargs[_padding_size_key] = padding_size

    return splitted_args, splitted_kwargs


# 一对多的分发方式，将数据从一个源头分发到所有工作节点
def dispatch_one_to_all(worker_group, *args, **kwargs):
    args = tuple([arg] * worker_group.world_size for arg in args)
    kwargs = {k: [v] * worker_group.world_size for k, v in kwargs.items()}
    return args, kwargs


# 禁止的直接回滚调用，抛出未实现异常
def dummy_direct_rollout_call(worker_group, *args, **kwargs):
    raise NotImplementedError("Direct rollout call is forbidden.")


# 全到全的分发方式，数据在所有工作节点之间均匀分配
def dispatch_all_to_all(worker_group, *args, **kwargs):
    return args, kwargs


# 收集全到全的结果，直接返回输出
def collect_all_to_all(worker_group, output):
    return output


# 连接多个 DataProto 或 DataProtoFuture 对象为一个
# 确保所有元素类型一致，支持 ray.ObjectRef 类型
def _concat_data_proto_or_future(output: list):
    import ray

    from verl.protocol import DataProto, DataProtoFuture

    # 确保输出中所有元素具有相同的类型
    for o in output:
        assert type(o) is type(output[0])

    o = output[0]

    if isinstance(o, DataProto):
        return DataProto.concat(output)
    elif isinstance(o, ray.ObjectRef):
        return DataProtoFuture.concat(output)
    else:
        raise NotImplementedError


# 分发数据并进行计算，适用于数据并行计算场景
# 确保所有输入参数的长度与工作组大小一致
def dispatch_dp_compute(worker_group, *args, **kwargs):
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)
    for arg in args:
        assert isinstance(arg, tuple | list) and len(arg) == worker_group.world_size
    for k, v in kwargs.items():
        assert isinstance(v, tuple | list) and len(v) == worker_group.world_size
    return args, kwargs


# 收集数据并行计算的结果，确保输出结果的顺序与工作组一致
def collect_dp_compute(worker_group, output):
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)
    assert len(output) == worker_group.world_size
    return output


# 分发数据并进行计算，支持 DataProto 类型
# 在数据并行计算中，自动进行填充以适应不同大小的数据
def dispatch_dp_compute_data_proto(worker_group, *args, **kwargs):
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)
    # 注意：为 dp compute 的 DataProto 启用自动填充
    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto_with_auto_padding(
        worker_group.world_size,
        *args,
        **kwargs,
    )
    return splitted_args, splitted_kwargs


# 分发数据并进行计算，支持函数作为参数的情况
# 将函数复制到每个工作节点，并分发其余参数
def dispatch_dp_compute_data_proto_with_func(worker_group, *args, **kwargs):
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)
    assert isinstance(args[0], FunctionType)  # NOTE: The first one args is a function!

    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto(worker_group.world_size, *args[1:], **kwargs)
    splitted_args_with_func = [[args[0]] * worker_group.world_size] + splitted_args
    return splitted_args_with_func, splitted_kwargs


# 收集数据并行计算的结果，支持 DataProto 类型
# 在数据并行计算中，自动进行结果的合并与还原
def collect_dp_compute_data_proto(worker_group, output):
    import ray

    from verl.protocol import DataProto

    for o in output:
        assert isinstance(o, DataProto | ray.ObjectRef), f"expecting {o} to be DataProto, but got {type(o)}"

    output = collect_dp_compute(worker_group, output)
    return _concat_data_proto_or_future(output)


# 分发计算任务到指定的工作节点，支持灵活的节点映射
# 根据 dp_rank_mapping 将数据分发到不同的工作节点
def dispatch_nd_compute(dp_rank_mapping: list[int], dp_size, worker_group, *args, **kwargs):
    import ray

    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)

    args = [[ray.put(dp_arg) for dp_arg in arg] for arg in args]
    kwargs = {k: [ray.put(dp_v) for dp_v in v] for k, v in kwargs.items()}

    all_args = []
    for arg in args:
        assert isinstance(arg, tuple | list) and len(arg) == dp_size
        transformed_args = []
        for i in range(worker_group.world_size):
            local_dp_rank = dp_rank_mapping[i]
            transformed_args.append(arg[local_dp_rank])
        all_args.append(transformed_args)
    all_args = tuple(all_args)

    all_kwargs = {}
    for k, v in kwargs.items():
        assert isinstance(v, tuple | list) and len(v) == dp_size
        transformed_v = []
        for i in range(worker_group.world_size):
            local_dp_rank = dp_rank_mapping[i]
            transformed_v.append(v[local_dp_rank])
        all_kwargs[k] = transformed_v
    return all_args, all_kwargs


# 收集指定工作节点的计算结果
# 根据 collect_mask 决定是否收集某个节点的结果
def collect_nd_compute(collect_mask: list[bool], worker_group, output):
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)
    assert len(output) == worker_group.world_size

    output_in_dp = []
    for global_rank in range(worker_group.world_size):
        collect_dp_rank = collect_mask[global_rank]
        if collect_dp_rank:
            output_in_dp.append(output[global_rank])
    return output_in_dp


# 分发数据并进行计算，支持 DataProto 类型
# 根据数据并行大小和工作节点映射，将数据分发到指定节点
def dispatch_nd_compute_dataproto(dp_rank_mapping: list[int], dp_size, worker_group, *args, **kwargs):
    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto(dp_size, *args, **kwargs)
    return dispatch_nd_compute(dp_rank_mapping, dp_size, worker_group, *splitted_args, **splitted_kwargs)


# 收集数据并行计算的结果，支持 DataProto 类型
# 根据节点的收集掩码，决定是否收集某个节点的结果
def collect_nd_compute_dataproto(collect_mask: list[bool], worker_group, output):
    output = collect_nd_compute(collect_mask, worker_group, output)
    import ray

    from verl.protocol import DataProto

    for o in output:
        assert isinstance(o, DataProto | ray.ObjectRef), f"expecting {o} to be DataProto, but got {type(o)}"
    return _concat_data_proto_or_future(output)


# 延迟计算的分发，支持动态的 mesh_name
# 根据 mesh_name 查询调度信息，并将计算任务分发到对应的工作节点
def dispatch_lazy_compute_data_proto(mesh_name, worker_group, *args, **kwargs):
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)

    # 查询工作组的调度信息
    if mesh_name not in worker_group._dispatch_info:
        worker_group._dispatch_info[mesh_name] = worker_group._query_dispatch_info(mesh_name)
        assert len(worker_group._dispatch_info[mesh_name]) == worker_group.world_size

    dp_rank_mapping = worker_group._dispatch_info[mesh_name]
    # 执行分发
    dp_size = max(dp_rank_mapping) + 1
    return dispatch_nd_compute_dataproto(dp_rank_mapping, dp_size, worker_group, *args, **kwargs)


# 收集延迟计算的结果，支持动态的 mesh_name
# 根据 mesh_name 查询收集信息，并决定是否收集某个节点的结果
def collect_lazy_compute_data_proto(mesh_name, worker_group, *args, **kwargs):
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)

    # 调度信息存储在工作组中
    assert mesh_name in worker_group._dispatch_info

    if mesh_name not in worker_group._collect_info:
        worker_group._collect_info[mesh_name] = worker_group._query_collect_info(mesh_name)
        assert len(worker_group._collect_info[mesh_name]) == worker_group.world_size

    # 收集掩码，指示哪些 dp_rank 需要被收集
    collect_mask = worker_group._collect_info[mesh_name]
    # 执行收集
    return collect_nd_compute_dataproto(collect_mask, worker_group, *args, **kwargs)


# 创建针对特定 mesh_name 的分发和收集函数
# 返回一个字典，包含 dispatch_fn 和 collect_fn
def make_nd_compute_dataproto_dispatch_fn(mesh_name):
    return {
        "dispatch_fn": partial(dispatch_lazy_compute_data_proto, mesh_name),
        "collect_fn": partial(collect_lazy_compute_data_proto, mesh_name),
    }


# 全局注册表，存储预定义的分发模式及其对应的函数
DISPATCH_MODE_FN_REGISTRY = {
    Dispatch.ONE_TO_ALL: {
        "dispatch_fn": dispatch_one_to_all,
        "collect_fn": collect_all_to_all,
    },
    Dispatch.ALL_TO_ALL: {
        "dispatch_fn": dispatch_all_to_all,
        "collect_fn": collect_all_to_all,
    },
    Dispatch.DP_COMPUTE: {"dispatch_fn": dispatch_dp_compute, "collect_fn": collect_dp_compute},
    Dispatch.DP_COMPUTE_PROTO: {
        "dispatch_fn": dispatch_dp_compute_data_proto,
        "collect_fn": collect_dp_compute_data_proto,
    },
    Dispatch.DP_COMPUTE_PROTO_WITH_FUNC: {
        "dispatch_fn": dispatch_dp_compute_data_proto_with_func,
        "collect_fn": collect_dp_compute_data_proto,
    },
    Dispatch.DP_COMPUTE_METRIC: {"dispatch_fn": dispatch_dp_compute_data_proto, "collect_fn": collect_dp_compute},
    Dispatch.DIRECT_ROLLOUT_METHOD: {
        "dispatch_fn": dummy_direct_rollout_call,
        "collect_fn": dummy_direct_rollout_call,
    },
}


# 根据分发模式获取对应的分发和收集函数
def get_predefined_dispatch_fn(dispatch_mode):
    return DISPATCH_MODE_FN_REGISTRY[dispatch_mode]


# 注册新的分发模式
# 将用户自定义的分发模式添加到全局注册表中
def register_dispatch_mode(dispatch_mode_name, dispatch_fn, collect_fn):
    """
    Register a new dispatch mode.
    """
    dispatch_mode = Dispatch.register(dispatch_mode_name)
    _check_dispatch_mode(dispatch_mode)
    assert dispatch_mode not in DISPATCH_MODE_FN_REGISTRY, f"dispatch_mode_name {dispatch_mode_name} already exists"
    DISPATCH_MODE_FN_REGISTRY[dispatch_mode] = {"dispatch_fn": dispatch_fn, "collect_fn": collect_fn}


# 更新已注册的分发模式
# 修改分发模式对应的函数
def update_dispatch_mode(dispatch_mode, dispatch_fn, collect_fn):
    """
    Update the dispatch mode.
    """
    _check_dispatch_mode(dispatch_mode)
    assert dispatch_mode in DISPATCH_MODE_FN_REGISTRY, f"dispatch_mode {dispatch_mode} not found"
    DISPATCH_MODE_FN_REGISTRY[dispatch_mode] = {"dispatch_fn": dispatch_fn, "collect_fn": collect_fn}


# 获取预定义的执行函数
# 返回一个字典，包含执行函数的名称
def get_predefined_execute_fn(execute_mode):
    """
    Note that here we only asks execute_all and execute_rank_zero to be implemented
    Leave the choice of how these two functions handle argument 'blocking' to users
    """
    predefined_execute_mode_fn = {
        Execute.ALL: {"execute_fn_name": "execute_all"},
        Execute.RANK_ZERO: {"execute_fn_name": "execute_rank_zero"},
    }
    return predefined_execute_mode_fn[execute_mode]


# 检查分发模式的有效性
# 确保分发模式是 Dispatch 类型或包含必要键的字典
def _check_dispatch_mode(dispatch_mode):
    assert isinstance(dispatch_mode, Dispatch | dict), (
        f"dispatch_mode must be a Dispatch or a Dict. Got {dispatch_mode}"
    )
    if isinstance(dispatch_mode, dict):
        necessary_keys = ["dispatch_fn", "collect_fn"]
        for key in necessary_keys:
            assert key in dispatch_mode, f"key {key} should be in dispatch_mode if it is a dictionary"


# 检查执行模式的有效性
# 确保执行模式是 Execute 类型
def _check_execute_mode(execute_mode):
    assert isinstance(execute_mode, Execute), f"execute_mode must be a Execute. Got {execute_mode}"


# 材料化未来对象
# 将 DataProtoFuture 类型的参数转换为实际值
def _materialize_futures(*args, **kwargs):
    new_args = []
    for arg in args:
        if isinstance(arg, DataProtoFuture):
            arg = arg.get()
        # add more type to materialize
        new_args.append(arg)
    for k, v in kwargs.items():
        if isinstance(v, DataProtoFuture):
            kwargs[k] = v.get()

    new_args = tuple(new_args)
    return new_args, kwargs


# 注册装饰器，配置分发和执行的相关参数
# 支持参数包括 dispatch_mode、execute_mode、blocking 和 materialize_futures
def register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.ALL, blocking=True, materialize_futures=True):
    """Register a function with distributed execution configuration.

    This decorator registers a function with specific dispatch and execution modes
    for distributed computation. It handles both synchronous and asynchronous
    functions, and optionally materializes futures before execution.

    Args:
        dispatch_mode:
            Dispatch mode for computation distribution. Default: Dispatch.ALL_TO_ALL.
        execute_mode:
            Execute mode for computation distribution. Default: Execute.ALL.
        blocking:
            Whether the execution should be blocking. Defaults to True.
        materialize_futures:
            Whether to materialize the data before dispatching. Defaults to True.

    Returns:
        A decorator that wraps the original function with distributed execution
        configuration.
    """
    _check_dispatch_mode(dispatch_mode=dispatch_mode)
    _check_execute_mode(execute_mode=execute_mode)

    def decorator(func):
        @wraps(func)
        def inner(*args, **kwargs):
            if materialize_futures:
                args, kwargs = _materialize_futures(*args, **kwargs)
            return func(*args, **kwargs)

        @wraps(func)
        async def async_inner(*args, **kwargs):
            if materialize_futures:
                args, kwargs = _materialize_futures(*args, **kwargs)
            return await func(*args, **kwargs)

        wrapper = async_inner if inspect.iscoroutinefunction(func) else inner
        attrs = {"dispatch_mode": dispatch_mode, "execute_mode": execute_mode, "blocking": blocking}
        setattr(wrapper, MAGIC_ATTR, attrs)
        return wrapper

    return decorator
