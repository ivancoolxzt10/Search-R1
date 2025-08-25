# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# mcore 包初始化文件，导入核心函数并定义 __all__，用于模块接口暴露。
# 说明：本文件用于声明 mcore 包的合法性和初始化，实际功能代码在子模块中实现。
from .registry import (
    get_mcore_forward_fn,        # 获取 mcore 前向推理函数
    get_mcore_forward_fused_fn,  # 获取 mcore 融合前向推理函数
    get_mcore_weight_converter,  # 获取权重转换工具
    hf_to_mcore_config,          # HuggingFace 配置转 mcore 配置
    init_mcore_model,            # 初始化 mcore 模型
)

__all__ = [
    "hf_to_mcore_config",           # 暴露 HuggingFace 配置转换接口
    "init_mcore_model",             # 暴露模型初始化接口
    "get_mcore_forward_fn",         # 暴露前向推理接口
    "get_mcore_weight_converter",   # 暴露权重转换接口
    "get_mcore_forward_fused_fn",   # 暴露融合前向推理接口
]
