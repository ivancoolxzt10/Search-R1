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

# 导入 ray 分布式计算框架
import ray

# 定义远程类 WorkerGroupRegisterCenter，用于分布式 worker group 信息注册与查询
@ray.remote
class WorkerGroupRegisterCenter:

    def __init__(self, rank_zero_info):
        # 初始化，存储 rank_zero_info 信息
        self.rank_zero_info = rank_zero_info

    def get_rank_zero_info(self):
        # 获取 rank_zero_info 信息
        return self.rank_zero_info

# 工厂函数，创建远程 WorkerGroupRegisterCenter 实例
# name: 注册中心名称，info: rank_zero 信息
def create_worker_group_register_center(name, info):
    return WorkerGroupRegisterCenter.options(name=name).remote(info)
