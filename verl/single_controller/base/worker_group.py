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
"""
WorkerGroup 类定义文件，负责分布式工作组的抽象与资源管理
"""

# 导入日志、信号、线程、时间等标准库模块
import logging
import signal
import threading
import time
from typing import Any, Callable

# 导入分发相关常量和函数
from .decorator import MAGIC_ATTR, Dispatch, get_predefined_dispatch_fn, get_predefined_execute_fn

# 资源池类，管理多节点的进程数和 GPU 分配
class ResourcePool:
    """
    管理多节点的资源池，跟踪进程数和 GPU 分配。
    提供计算 world_size、本地 world_size 和本地 rank 的方法。
    """

    def __init__(self, process_on_nodes=None, max_colocate_count: int = 10, n_gpus_per_node=8) -> None:
        """初始化资源池，设置节点进程数和 GPU 配置。
        Args:
            process_on_nodes (List[int], optional): 每个节点的进程数列表。默认为空列表。
            max_colocate_count (int, optional): 单节点最大可并置进程数。默认为 10。
            n_gpus_per_node (int, optional): 每节点 GPU 数量。默认为 8。
        """
        if process_on_nodes is None:
            process_on_nodes = []
        self._store = process_on_nodes
        self.max_colocate_count = max_colocate_count
        self.n_gpus_per_node = n_gpus_per_node  # 预留给未来如华为 16 GPU 节点

    def add_node(self, process_count):
        # 向资源池添加一个节点及其进程数
        self._store.append(process_count)

    @property
    def world_size(self):
        """资源池中所有节点的总进程数。"""
        return sum(self._store)

    def __call__(self) -> Any:
        # 返回资源池的进程数列表
        return self._store

    @property
    def store(self):
        # 获取资源池的底层存储列表
        return self._store

    def local_world_size_list(self) -> list[int]:
        """返回每个进程对应的本地 world size 列表。"""
        nested_local_world_size_list = [
            [local_world_size for _ in range(local_world_size)] for local_world_size in self._store
        ]
        return [item for row in nested_local_world_size_list for item in row]

    def local_rank_list(self) -> list[int]:
        """返回所有进程的本地 rank 列表。"""
        nested_local_rank_list = [[i for i in range(local_world_size)] for local_world_size in self._store]
        return [item for row in nested_local_rank_list for item in row]

# 延迟实例化类，存储构造参数，便于远程或延迟创建对象
class ClassWithInitArgs:
    """
    包装类，存储构造参数以便延迟实例化。
    适用于远程类实例化场景，实际构造在不同时间或位置发生。
    """

    def __init__(self, cls, *args, **kwargs) -> None:
        """初始化包装类，存储类和构造参数。
        Args:
            cls: 需要延迟实例化的类
            *args: 类构造的参数
            **kwargs: 类构造的关键字参数
        """
        self.cls = cls
        self.args = args
        self.kwargs = kwargs

        self.fused_worker_used = False

    def __call__(self) -> Any:
        """使用存储的参数实例化类对象。"""
        return self.cls(*self.args, **self.kwargs)


def check_workers_alive(workers: list, is_alive: Callable, gap_time: float = 1) -> None:
    """持续监控工作进程的存活状态，如有进程死亡则向主线程发送 SIGABRT 信号。

    Args:
        workers (List):
            要监控的工作进程列表
        is_alive (Callable):
            检查工作进程是否存活的函数
        gap_time (float):
            检查间隔时间
    """
    import time

    while True:
        for worker in workers:
            if not is_alive(worker):
                logging.warning(f"worker {worker} is not alive sending signal to main thread")
                signal.raise_signal(signal.SIGABRT)
        time.sleep(gap_time)


class WorkerGroup:
    """
    管理分布式系统中工作进程的基类。
    提供工作进程管理、存活检查和方法绑定等功能。
    """

    fused_worker_execute_fn_name = "_fuw_execute"

    def __init__(self, resource_pool: ResourcePool, **kwargs) -> None:
        # 初始化时检查资源池是否为 None，以区分是新建工作组还是附加到现有工作组
        self._is_init_with_detached_workers = resource_pool is None

        self.fused_worker_used = False

        if resource_pool is not None:
            # 处理 WorkerGroup 附加到现有资源池的情况
            self._procecss_dispatch_config = resource_pool()
        else:
            self._procecss_dispatch_config = None

        # 初始化工作进程和工作进程名称的列表
        self._workers = []
        self._worker_names = []

        # 初始化分发信息和收集信息的字典
        self._dispatch_info = {}
        self._collect_info = {}

        # 主节点地址和端口的初始化
        self._master_addr = None
        self._master_port = None

        # 检查线程的初始化
        self._checker_thread: threading.Thread = None

    def _is_worker_alive(self, worker):
        """检查工作进程是否存活。必须在派生类中实现。"""
        raise NotImplementedError("WorkerGroup._is_worker_alive called, should be implemented in derived class.")

    def _block_until_all_workers_alive(self) -> None:
        """阻塞直到工作组中的所有工作进程都存活。"""
        while True:
            all_state = [self._is_worker_alive(worker) for worker in self._workers]
            if False in all_state:
                time.sleep(1)
            else:
                break

    def start_worker_aliveness_check(self, every_n_seconds=1) -> None:
        """启动后台线程监控工作进程的存活状态。

        Args:
            every_n_seconds (int): 存活检查的时间间隔
        """
        # 在开始检查工作进程存活状态之前，确保所有工作进程都已启动
        self._block_until_all_workers_alive()

        self._checker_thread = threading.Thread(
            target=check_workers_alive, args=(self._workers, self._is_worker_alive, every_n_seconds)
        )
        self._checker_thread.start()

    @property
    def world_size(self):
        """工作组中工作进程的数量。"""
        return len(self._workers)

    def _bind_worker_method(self, user_defined_cls, func_generator):
        """根据注册的属性将工作进程方法绑定到 WorkerGroup。

        Args:
            user_defined_cls (type): 包含要绑定方法的类
            func_generator (Callable): 生成绑定方法的函数

        Returns:
            List[str]: 成功绑定的方法名称列表
        """
        method_names = []
        for method_name in dir(user_defined_cls):
            try:
                method = getattr(user_defined_cls, method_name)
                assert callable(method), f"{method_name} in {user_defined_cls} is not callable"
            except Exception:
                # 如果是属性，将因 Class 没有实例属性而失败，故在此处捕获异常
                continue

            if hasattr(method, MAGIC_ATTR):
                # 方法被 register 装饰器装饰
                attribute = getattr(method, MAGIC_ATTR)
                assert isinstance(attribute, dict), f"attribute must be a dictionary. Got {type(attribute)}"
                assert "dispatch_mode" in attribute, "attribute must contain dispatch_mode in its key"

                dispatch_mode = attribute["dispatch_mode"]
                execute_mode = attribute["execute_mode"]
                blocking = attribute["blocking"]

                # 获取分发函数
                if isinstance(dispatch_mode, Dispatch):
                    # 获取默认的分发函数
                    fn = get_predefined_dispatch_fn(dispatch_mode=dispatch_mode)
                    dispatch_fn = fn["dispatch_fn"]
                    collect_fn = fn["collect_fn"]
                else:
                    assert isinstance(dispatch_mode, dict)
                    assert "dispatch_fn" in dispatch_mode
                    assert "collect_fn" in dispatch_mode
                    dispatch_fn = dispatch_mode["dispatch_fn"]
                    collect_fn = dispatch_mode["collect_fn"]

                # 获取执行模式
                execute_mode = get_predefined_execute_fn(execute_mode=execute_mode)
                wg_execute_fn_name = execute_mode["execute_fn_name"]

                # 从字符串中获取执行函数
                try:
                    execute_fn = getattr(self, wg_execute_fn_name)
                    assert callable(execute_fn), "execute_fn must be callable"
                except Exception:
                    print(f"execute_fn {wg_execute_fn_name} is invalid")
                    raise

                # 将新方法绑定到 RayWorkerGroup
                func = func_generator(
                    self,
                    method_name,
                    dispatch_fn=dispatch_fn,
                    collect_fn=collect_fn,
                    execute_fn=execute_fn,
                    blocking=blocking,
                )

                try:
                    setattr(self, method_name, func)
                    method_names.append(method_name)
                except Exception as e:
                    raise ValueError(f"Fail to set method_name {method_name}") from e

        return method_names
