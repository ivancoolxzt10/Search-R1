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

# 导入类型注解 Dict、Optional
from typing import Dict, Optional

# 导入 ray 分布式计算框架
import ray

# 导入 Ray 分布式工作组相关类
from .base import RayWorkerGroup, RayResourcePool, RayClassWithInitArgs
# 导入 Megatron 分布式 rank/global 信息结构体
from verl.single_controller.base.megatron.worker import DistRankInfo, DistGlobalInfo
from verl.single_controller.base.megatron.worker_group import MegatronWorkerGroup


# NVMegatronRayWorkerGroup 类，继承自 RayWorkerGroup 和 MegatronWorkerGroup，适配 Megatron-core
class NVMegatronRayWorkerGroup(RayWorkerGroup, MegatronWorkerGroup):
    """
    MegatronWorkerGroup 会查询每个 worker 的 megatron rank 信息并存储在 WorkerGroup 内，
    以便 dispatcher 用于数据分发。
    """

    def __init__(self, resource_pool: RayResourcePool, ray_cls_with_init: RayClassWithInitArgs, **kwargs):
        # 初始化，调用父类构造方法
        super().__init__(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init, **kwargs)
        # 查询所有 worker 的 megatron rank 信息
        self._megatron_rank_info: DistRankInfo = self.execute_all_sync(method_name='get_megatron_rank_info')
        # 查询 rank_zero 的 megatron global 信息
        self._megatron_global_info: DistGlobalInfo = ray.get(
            self.execute_rank_zero_async(method_name='get_megatron_global_info'))


# MegatronRayWorkerGroup 类，继承自 RayWorkerGroup 和 MegatronWorkerGroup，支持自定义初始化
class MegatronRayWorkerGroup(RayWorkerGroup, MegatronWorkerGroup):
    """
    MegatronWorkerGroup 会查询每个 worker 的 megatron rank 信息并存储在 WorkerGroup 内，
    以便 dispatcher 用于数据分发。
    """

    def __init__(self,
                 resource_pool: RayResourcePool,
                 ray_cls_with_init: RayClassWithInitArgs,
                 default_megatron_kwargs: Dict = None,
                 **kwargs):
        # 初始化，调用父类构造方法
        super().__init__(resource_pool=resource_pool,
                         ray_cls_with_init=ray_cls_with_init,
                         default_megatron_kwargs=default_megatron_kwargs,
                         **kwargs)
        # 初始化 Megatron 环境
        self.init_megatron(default_megatron_kwargs=default_megatron_kwargs)
        # 查询所有 worker 的 megatron rank 信息
        self._megatron_rank_info: DistRankInfo = self.execute_all_sync(method_name='get_megatron_rank_info')
        # 查询 rank_zero 的 megatron global 信息
        self._megatron_global_info: DistGlobalInfo = ray.get(
            self.execute_rank_zero_async(method_name='get_megatron_global_info'))

    def init_megatron(self, default_megatron_kwargs: Optional[Dict] = None):
        # 在所有 worker 上调用 init_megatron 方法进行初始化
        if not self._is_init_with_detached_workers:
            # 仅在新建 WorkerGroup 时初始化 Megatron
            self.execute_all_sync(method_name='init_megatron', default_megatron_kwargs=default_megatron_kwargs)
