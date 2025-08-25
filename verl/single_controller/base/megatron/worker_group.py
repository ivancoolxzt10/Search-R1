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

# 导入类型注解 Dict
from typing import Dict

# 导入 Megatron 分布式 rank/global 信息结构体
from .worker import DistRankInfo, DistGlobalInfo
# 导入基础资源池和工作组类
from verl.single_controller.base import ResourcePool, WorkerGroup


# MegatronWorkerGroup 类，继承自 WorkerGroup，专用于 Megatron 并行训练场景
class MegatronWorkerGroup(WorkerGroup):

    def __init__(self, resource_pool: ResourcePool, **kwargs):
        # 初始化，调用父类 WorkerGroup 的构造方法
        super().__init__(resource_pool=resource_pool, **kwargs)
        self._megatron_rank_info = None  # Megatron 各 rank 信息
        self._megatron_global_info: DistGlobalInfo = None  # Megatron 全局并行信息

    def init_megatron(self, default_megatron_kwargs: Dict = None):
        # 初始化 Megatron 并行环境，需在子类中实现
        raise NotImplementedError(f"MegatronWorkerGroup.init_megatron should be overwritten")

    def get_megatron_rank_info(self, rank: int) -> DistRankInfo:
        # 获取指定 rank 的 Megatron 并行 rank 信息
        assert 0 <= rank < self.world_size, f'rank must be from [0, world_size), Got {rank}'
        return self._megatron_rank_info[rank]

    @property
    def tp_size(self):
        # 获取张量并行大小
        assert self._megatron_global_info is not None, "MegatronWorkerGroup._megatron_global_info must be initialized"
        return self._megatron_global_info.tp_size

    @property
    def dp_size(self):
        # 获取数据并行大小
        assert self._megatron_global_info is not None, "MegatronWorkerGroup._megatron_global_info must be initialized"
        return self._megatron_global_info.dp_size

    @property
    def pp_size(self):
        # 获取流水线并行大小
        assert self._megatron_global_info is not None, "MegatronWorkerGroup._megatron_global_info must be initialized"
        return self._megatron_global_info.pp_size

    def get_megatron_global_info(self):
        # 获取 Megatron 全局并行信息（需在子类中实现）
        raise NotImplementedError("get_megatron_global_info should be implemented in subclass")
