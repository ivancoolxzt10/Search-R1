"""
__main__.py

本文件为模型合并工具的命令行入口，支持 FSDP 和 Megatron 两种后端，将分布式训练的模型权重合并为 HuggingFace 格式。
主要功能：
- 解析命令行参数，生成合并配置
- 根据后端类型选择对应的模型合并器
- 执行模型权重合并与保存
- 清理临时资源

每一行代码均有详细中文备注，便于初学者理解。
"""

# Copyright 2024 Bytedance Ltd. and/or its affiliates
# 版权声明，表明该文件归 Bytedance 所有
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 按照 Apache 2.0 协议授权
# you may not use this file except in compliance with the License.
# 只有遵循协议才能使用本文件
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

from .base_model_merger import generate_config_from_args, parse_args  # 参数解析和配置生成工具


def main():
    args = parse_args()  # 解析命令行参数，获取用户输入的参数对象
    config = generate_config_from_args(args)  # 根据参数对象生成模型合并配置
    print(f"config: {config}")  # 打印配置，便于调试和确认

    if config.backend == "fsdp":  # 判断后端类型是否为 FSDP
        from .fsdp_model_merger import FSDPModelMerger  # 导入 FSDP 合并器类

        merger = FSDPModelMerger(config)  # 创建 FSDP 合并器实例
    elif config.backend == "megatron":  # 判断后端类型是否为 Megatron
        from .megatron_model_merger import MegatronModelMerger  # 导入 Megatron 合并器类

        merger = MegatronModelMerger(config)  # 创建 Megatron 合并器实例
    else:
        raise NotImplementedError(f"Unknown backend: {config.backend}")  # 不支持的后端类型，抛出异常

    merger.merge_and_save()  # 执行模型权重合并与保存
    merger.cleanup()  # 清理临时资源，如缓存文件等


if __name__ == "__main__":
    main()  # 命令行入口，执行主流程
