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
Worker 类定义文件，负责分布式工作节点的抽象与管理
"""

# 导入 os 模块，用于系统环境变量和路径操作
import os
# 导入 socket 模块，用于网络通信和 IP 获取
import socket
# 导入 warnings 模块，用于警告提示
import warnings
# 导入 dataclass 装饰器，用于简化数据结构定义
from dataclasses import dataclass

# 导入 ray 分布式计算框架
import ray

# 导入设备相关工具函数
from verl.utils.device import get_torch_device, get_visible_devices_keyword

# 导入分发与执行模式枚举及注册装饰器
from .decorator import Dispatch, Execute, register

# 分布式 rank 信息的数据结构，包含张量并行、数据并行等 rank
@dataclass
class DistRankInfo:
    tp_rank: int  # 张量并行 rank
    dp_rank: int  # 数据并行 rank
    pp_rank: int  # 流水线并行 rank
    cp_rank: int  # 通信并行 rank

# 分布式全局信息的数据结构，包含各并行维度的大小
@dataclass
class DistGlobalInfo:
    tp_size: int  # 张量并行大小
    dp_size: int  # 数据并行大小
    pp_size: int  # 流水线并行大小
    cp_size: int  # 通信并行大小

# Worker 辅助工具类，提供静态方法
class WorkerHelper:
    @staticmethod
    def _get_node_ip():
        # 如果后端为 ray，则通过 ray 获取节点 IP
        if os.getenv("WG_BACKEND", None) == "ray":
            return ray.util.get_node_ip_address()
        else:
            raise NotImplementedError("WG_BACKEND now just support ray mode.")

    @staticmethod
    def _get_free_port():
        # 创建一个 socket 对象，绑定到任意可用端口
        with socket.socket() as sock:
            sock.bind(("", 0))
            # 返回分配的端口号
            return sock.getsockname()[1]

    def get_availale_master_addr_port(self):
        warnings.warn(
            "This function is deprecated due to typo in name; Please use `get_available_master_addr_port` instead",
            stacklevel=2,
        )
        return self.get_available_master_addr_port()

    def get_available_master_addr_port(self):
        # 获取节点 IP 和可用端口，组成 master 地址和端口
        return self._get_node_ip().strip("[]"), str(self._get_free_port())


# 我们假设在每个 WorkerGroup 中，至少有一个 Master Worker
class Worker(WorkerHelper):
    """一个分布式工作节点，负责分布式训练的初始化和配置管理。

    该类管理工作节点的初始化、配置，并提供执行分布式操作的方法。它处理通信设置、设备配置和工作节点
    元数据管理。
    """

    fused_worker_attr_name = "fused_worker_dict"

    def _register_dispatch_collect_info(self, mesh_name: str, dp_rank: int, is_collect: bool):
        """注册给定网格名称的 dp_rank。此函数旨在由工作节点调用

        Args:
            mesh_name (str):
                要注册 dp_rank 的网格名称。
            dp_rank (int):
                要注册的 dp_rank。
            is_collect (bool):
                dp_rank 是否用于收集。
        """
        if mesh_name in self.__dispatch_dp_rank or mesh_name in self.__collect_dp_rank:
            raise ValueError(f"mesh_name {mesh_name} has been registered")
        self.__dispatch_dp_rank[mesh_name] = dp_rank
        self.__collect_dp_rank[mesh_name] = is_collect

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def _query_dispatch_info(self, mesh_name: str):
        """查询给定网格名称的调度信息。

        Args:
            mesh_name (str):
                要查询调度信息的网格名称。

        Returns:
            int:
                给定网格名称的 dp_rank。
        """
        assert mesh_name in self.__dispatch_dp_rank, f"{mesh_name} is not registered in {self.__class__.__name__}"
        # 注意每个 rank 存储自己的 dp_rank
        return self.__dispatch_dp_rank[mesh_name]

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def _query_collect_info(self, mesh_name: str):
        """查询给定网格名称的收集信息。

        Args:
            mesh_name (str):
                要查询收集信息的网格名称。

        Returns:
            bool:
                dp_rank 是否用于收集。
        """
        assert mesh_name in self.__collect_dp_rank, f"{mesh_name} is not registered in {self.__class__.__name__}"
        return self.__collect_dp_rank[mesh_name]

    @classmethod
    def env_keys(cls):
        """配置 Worker 所需的环境变量的键."""
        return [
            "WORLD_SIZE",
            "RANK",
            "LOCAL_WORLD_SIZE",
            "LOCAL_RANK",
            "MASTER_ADDR",
            "MASTER_PORT",
            get_visible_devices_keyword().upper(),
        ]

    def __init__(self, cuda_visible_devices=None) -> None:
        """根据环境设置和设备配置初始化工作节点。

        Args:
            cuda_visible_devices (str, optional):
                CUDA 可见设备配置。默认为 None。
        """
        # 从环境变量构建元信息。注意导入必须在类内部，因为
        # 它是远程执行的
        import os

        self._setup_env_cuda_visible_devices()

        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        self._rank = rank
        self._world_size = world_size

        master_addr = os.environ["MASTER_ADDR"]
        master_port = os.environ["MASTER_PORT"]

        local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", "1"))
        local_rank = int(os.getenv("LOCAL_RANK", "0"))

        store = {
            "_world_size": world_size,
            "_rank": rank,
            "_local_world_size": local_world_size,
            "_local_rank": local_rank,
            "_master_addr": master_addr,
            "_master_port": master_port,
        }
        if cuda_visible_devices is not None:
            store[f"_{get_visible_devices_keyword()}".lower()] = cuda_visible_devices

        self._configure_with_store(store=store)

        self.fused_worker_dict = {}
        self.__dispatch_dp_rank = {}
        self.__collect_dp_rank = {}

    def get_fused_worker_by_name(self, worker_name: str):
        """通过名称获取融合工作节点。

        Args:
            worker_name (str):
                要检索的工作节点名称
        """
        return self.fused_worker_dict.get(worker_name, None)

    def _setup_env_cuda_visible_devices(self):
        from verl.utils.ray_utils import ray_noset_visible_devices

        is_ray_noset_visible_devices = ray_noset_visible_devices()

        # 防止冲突的 `{CUDA/HIP/ROCR}_VISIBLE_DEVICES``
        rocr_val = os.environ.get("ROCR_VISIBLE_DEVICES", None)
        hip_val = os.environ.get("HIP_VISIBLE_DEVICES", None)
        cuda_val = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if hip_val:
            # 将 HIP_VISIBLE_DEVICES 切换为 CUDA_VISIBLE_DEVICES 以保持一致性。
            # 确保此时 HIP_VISIBLE_DEVICES 的值与 CUDA_VISIBLE_DEVICES 相同
            val = os.environ.pop("HIP_VISIBLE_DEVICES")
            hip_val = None
            if cuda_val:
                assert val == cuda_val, (
                    f"Please use the same HIP_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES, inconsistant values "
                    f"found: {val} and {cuda_val}."
                )
            else:
                cuda_val = val
                os.environ["CUDA_VISIBLE_DEVICES"] = val
                # os.environ["HIP_VISIBLE_DEVICES"] = val

        if rocr_val:
            # 如果同时设置了 HIP/CUDA 和 ROCR 环境变量，必须小心，因为它们的
            # 含义不同。两个环境变量都接受整数列表或 UUID 列表。首先处理 ROCR 环境变量，
            # 然后减少 HIP 可选择的 GPU 数量。
            # https://github.com/pytorch/pytorch/pull/144026
            # 为了避免复杂性，如果同时设置了这两个变量，则给出错误。
            # （还要与 ray 在 2.45.0 版本的做法保持一致）。
            # 否则，我们将 ROCR_VISIBLE_DEVICES 设置为 CUDA_VISIBLE_DEVICES
            # 并删除 ROCR_VISIBLE_DEVICES。
            if cuda_val:
                raise ValueError("Please don't set ROCR_VISIBLE_DEVICES when HIP/CUDA_VISIBLE_DEVICES is set.")

            cuda_val = os.environ.pop("ROCR_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_val
            rocr_val = None

        if is_ray_noset_visible_devices:
            # 注意：Ray 会自动为每个 actor 设置 *_VISIBLE_DEVICES
            # 环境变量，除非设置了
            # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES 标志,
            # 所以我们需要在标志设置时设置本地 rank。
            local_rank = os.environ.get("RAY_LOCAL_RANK")
            os.environ["LOCAL_RANK"] = local_rank
            get_torch_device().set_device(int(local_rank))

    def _configure_with_store(self, store: dict):
        """
        此函数应仅在 WorkerGroup 内部调用
        """
        store_env_dict = {f"_{key.lower()}": store.get(f"_{key.lower()}", None) for key in type(self).env_keys()}
        self.__dict__.update(store_env_dict)  # 这很 hacky
        # print(f"__dict__: {self.__dict__}")
        for key in type(self).env_keys():
            val = self.__dict__.get(f"_{key.lower()}", None)
            if val is not None:
                # print(f"set {key} to {val}")
                os.environ[key] = str(val)
        os.environ["REDIS_STORE_SERVER_HOST"] = (
            str(self._master_addr).replace("[", "").replace("]", "") if self._master_addr else ""
        )

    def get_master_addr_port(self):
        """获取用于分布式通信的主地址和端口."""
        return self._master_addr, self._master_port

    def get_cuda_visible_devices(self):
        """获取 CUDA 可见设备配置."""
        import os

        visible_devices = os.environ.get(get_visible_devices_keyword().upper(), "not set")
        return visible_devices

    @property
    def world_size(self):
        """获取分布式设置中工作节点的总数."""
        return self._world_size

    @property
    def rank(self):
        """获取该工作节点在分布式设置中的排名."""
        return self._rank

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO_WITH_FUNC)
    def execute_with_func_generator(self, func, *args, **kwargs):
        """使用函数生成器调度模式执行函数。

        Args:
            func:
                要执行的函数
            *args:
                函数的位置信息参数
            **kwargs:
                函数的关键字参数
        """
        ret_proto = func(self, *args, **kwargs)
        return ret_proto

    @register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.RANK_ZERO)
    def execute_func_rank_zero(self, func, *args, **kwargs):
        """在零排名执行模式下执行函数。

        Args:
            func:
                要执行的函数
            *args:
                函数的位置信息参数
            **kwargs:
                函数的关键字参数
        """
        result = func(*args, **kwargs)
        return result
