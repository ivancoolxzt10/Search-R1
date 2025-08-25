# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

# 该文件用于注册和获取模型权重的加载器和保存器
# 包含两个主要函数：get_weight_loader 和 get_weight_saver
# 分别用于根据模型架构获取对应的权重加载器和保存器


def get_weight_loader(arch: str):
    # 从 verl.models.mcore.loader 导入加载权重的函数
    from verl.models.mcore.loader import load_state_dict_to_megatron_gptmodel

    # 定义一个字典，将模型架构名称映射到对应的权重加载函数
    _MODEL_WEIGHT_MEGATRON_LOADER_REGISTRY = {
        "LlamaForCausalLM": load_state_dict_to_megatron_gptmodel,
        "Qwen2ForCausalLM": load_state_dict_to_megatron_gptmodel,
    }

    # 如果传入的架构名称在注册表中，返回对应的加载器
    if arch in _MODEL_WEIGHT_MEGATRON_LOADER_REGISTRY:
        return _MODEL_WEIGHT_MEGATRON_LOADER_REGISTRY[arch]
    # 否则，抛出一个错误，提示不支持的架构
    raise ValueError(
        f"Model architectures {arch} loader are not supported for now. Supported architectures: "
        f"{_MODEL_WEIGHT_MEGATRON_LOADER_REGISTRY.keys()}"
    )


def get_weight_saver(arch: str):
    # 从 verl.models.mcore.saver 导入合并权重的函数
    from verl.models.mcore.saver import (
        merge_megatron_ckpt_gptmodel,
        merge_megatron_ckpt_gptmodel_dpskv3,
        merge_megatron_ckpt_gptmodel_mixtral,
        merge_megatron_ckpt_gptmodel_qwen2_5_vl,
        merge_megatron_ckpt_gptmodel_qwen_moe,
    )

    # 定义一个字典，将模型架构名称映射到对应的权重保存函数
    _MODEL_WEIGHT_MEGATRON_SAVER_REGISTRY = {
        "LlamaForCausalLM": merge_megatron_ckpt_gptmodel,
        "Qwen2ForCausalLM": merge_megatron_ckpt_gptmodel,
        "MixtralForCausalLM": merge_megatron_ckpt_gptmodel_mixtral,
        "Qwen2MoeForCausalLM": merge_megatron_ckpt_gptmodel_qwen_moe,
        "Qwen2_5_VLForConditionalGeneration": merge_megatron_ckpt_gptmodel_qwen2_5_vl,
        "DeepseekV3ForCausalLM": merge_megatron_ckpt_gptmodel_dpskv3,
        "Qwen3ForCausalLM": merge_megatron_ckpt_gptmodel,
        "Qwen3MoeForCausalLM": merge_megatron_ckpt_gptmodel_qwen_moe,
    }
    # 如果传入的架构名称在注册表中，返回对应的保存器
    if arch in _MODEL_WEIGHT_MEGATRON_SAVER_REGISTRY:
        return _MODEL_WEIGHT_MEGATRON_SAVER_REGISTRY[arch]
    # 否则，抛出一个错误，提示不支持的架构
    raise ValueError(
        f"Model architectures {arch} saver are not supported for now. Supported architectures: "
        f"{_MODEL_WEIGHT_MEGATRON_SAVER_REGISTRY.keys()}"
    )
