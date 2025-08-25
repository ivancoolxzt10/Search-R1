# Copyright 2024 Bytedance Ltd. and/or its affiliates
# 版权声明，表明该文件归 Bytedance 所有
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 按照 Apache 2.0 协议授权
# you may not use this file except in compliance with the License.
# 只有遵循协议才能使用本文件，除非符合协议规定
# You may obtain a copy of the License at
# 可在以下网址获取协议全文
#
#     http://www.apache.org/licenses/LICENSE-2.0
# 协议网址
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 本文件按“原样”分发
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 不提供任何明示或暗示的担保
# See the License for the specific language governing permissions and
# 具体权限请查阅协议
# limitations under the License.
# 协议中的限制条款

import importlib  # 动态导入模块
from typing import Optional  # 类型注解

import torch.nn as nn  # 神经网络模块

# 支持的 Megatron-LM 模型架构，映射到模块和类名
_MODELS = {
    "LlamaForCausalLM": (
        "llama",
        ("ParallelLlamaForCausalLMRmPadPP", "ParallelLlamaForValueRmPadPP", "ParallelLlamaForCausalLMRmPad"),
    ),
    "Qwen2ForCausalLM": (
        "qwen2",
        ("ParallelQwen2ForCausalLMRmPadPP", "ParallelQwen2ForValueRmPadPP", "ParallelQwen2ForCausalLMRmPad"),
    ),
    "MistralForCausalLM": (
        "mistral",
        ("ParallelMistralForCausalLMRmPadPP", "ParallelMistralForValueRmPadPP", "ParallelMistralForCausalLMRmPad"),
    ),
}


class ModelRegistry:
    """
    模型注册表，支持按架构动态加载 Megatron-LM 模型类。
    """

    @staticmethod
    def load_model_cls(model_arch: str, value=False) -> Optional[type[nn.Module]]:
        """
        根据模型架构名称和 value 标志动态加载模型类。
        参数:
            model_arch: 架构名称，如 'LlamaForCausalLM'
            value: 是否为价值模型（critic/rm），否则为 actor/ref
        返回:
            nn.Module 子类或 None
        """
        if model_arch not in _MODELS:
            return None

        megatron = "megatron"

        module_name, model_cls_name = _MODELS[model_arch]
        if not value:  # actor/ref
            model_cls_name = model_cls_name[0]
        elif value:  # critic/rm
            model_cls_name = model_cls_name[1]

        module = importlib.import_module(f"verl.models.{module_name}.{megatron}.modeling_{module_name}_megatron")
        return getattr(module, model_cls_name, None)

    @staticmethod
    def get_supported_archs() -> list[str]:
        """
        获取支持的模型架构列表。
        返回:
            架构名称列表
        """
        return list(_MODELS.keys())
