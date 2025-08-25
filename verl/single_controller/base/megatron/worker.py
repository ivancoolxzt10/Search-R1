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

# 导入 os 模块，用于环境变量操作
import os
# 导入 dataclass，用于简化数据结构定义
from dataclasses import dataclass
# 导入 Worker 及分布式 rank/global 信息结构体
from verl.single_controller.base.worker import Worker, DistRankInfo, DistGlobalInfo


# MegatronWorker 类，继承自 Worker，专用于 Megatron 并行训练场景
class MegatronWorker(Worker):

    def __init__(self, cuda_visible_devices=None) -> None:
        # 初始化，调用父类 Worker 的构造方法
        super().__init__(cuda_visible_devices)

    def get_megatron_global_info(self):
        # 获取 Megatron 并行训练的全局信息（张量并行、数据并行、流水线并行大小）
        from megatron.core import parallel_state as mpu
        tp_size = mpu.get_tensor_model_parallel_world_size()
        dp_size = mpu.get_data_parallel_world_size()
        pp_size = mpu.get_pipeline_model_parallel_world_size()
        info = DistGlobalInfo(tp_size=tp_size, dp_size=dp_size, pp_size=pp_size)
        return info

    def get_megatron_rank_info(self):
        # 获取 Megatron 并行训练的 rank 信息（张量并行、数据并行、流水线并行 rank）
        from megatron.core import parallel_state as mpu
        tp_rank = mpu.get_tensor_model_parallel_rank()
        dp_rank = mpu.get_data_parallel_rank()
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        info = DistRankInfo(tp_rank=tp_rank, dp_rank=dp_rank, pp_rank=pp_rank)
        return info