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

# 从 base.py 导入 Ray 分布式相关核心类和函数
from .base import (
    RayClassWithInitArgs,      # Ray 延迟实例化类
    RayResourcePool,           # Ray 资源池
    RayWorkerGroup,            # Ray 工作组
    create_colocated_worker_cls,        # 创建并置 worker 类
    create_colocated_worker_cls_fused,  # 创建融合并置 worker 类
)

# 设置 __all__，用于控制模块对外暴露的接口
__all__ = [
    "RayClassWithInitArgs",
    "RayResourcePool",
    "RayWorkerGroup",
    "create_colocated_worker_cls",
    "create_colocated_worker_cls_fused",
]
