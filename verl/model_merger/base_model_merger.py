"""
base_model_merger.py

本文件为分布式模型权重合并工具的基础模块，支持 FSDP、Megatron 等后端，将分布式训练的 checkpoint 合并为 HuggingFace 格式。
主要功能：
- 命令行参数解析，生成合并/测试配置
- 定义模型合并配置数据结构
- 提供模型合并、保存、上传、清理等抽象接口
- 适合初学者理解分布式模型权重合并的整体流程

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
import argparse  # 命令行参数解析库
import os        # 操作系统相关模块
from abc import ABC, abstractmethod  # 抽象基类和抽象方法
from dataclasses import dataclass, field  # 数据类和字段工具
from typing import Optional  # 类型注解

import torch  # 深度学习库
from accelerate import init_empty_weights  # HuggingFace 加速库
from transformers import (
    AutoConfig,  # 自动加载模型配置
    AutoModelForCausalLM,  # 自动加载因果语言模型
    AutoModelForTokenClassification,  # 自动加载 token 分类模型
    AutoModelForVision2Seq,  # 自动加载视觉到序列模型
    GenerationConfig,  # 文本生成配置
)  # transformers 相关模型和配置

from verl.utils import hf_processor, hf_tokenizer  # huggingface 工具


def parse_args():
    """命令行参数解析函数，支持 merge/test 两种操作。"""
    parser = argparse.ArgumentParser(description="verl model merger")  # 创建参数解析器
    subparsers = parser.add_subparsers(dest="operation", required=True, help="Specify 'merge' or 'test' operation.")  # 添加子命令解析器

    base_op_parser = argparse.ArgumentParser(add_help=False)  # 基础参数解析器
    base_op_parser.add_argument(
        "--backend", type=str, required=True, choices=["fsdp", "megatron"], help="The backend of the model"
    )  # 指定后端类型
    base_op_parser.add_argument("--local_dir", type=str, default=None, help="Path to the saved model checkpoints.")  # checkpoint 路径
    base_op_parser.add_argument(
        "--tie-word-embedding",
        action="store_true",
        help="Whether to tie word embedding weights (currently only Megatron supported)",
    )  # 是否绑定词嵌入权重
    base_op_parser.add_argument("--trust-remote-code", action="store_true", help="Whether to trust remote code")  # 是否信任远程代码
    base_op_parser.add_argument(
        "--is-value-model",
        action="store_true",
        help="Whether the model is a value model (currently only Megatron supported)",
    )  # 是否为 value model
    base_op_parser.add_argument(
        "--use_cpu_initialization",
        action="store_true",
        help="是否使用 CPU 初始化模型，适用于超大模型初始化。",
    )  # 是否使用 CPU 初始化

    merge_parser = subparsers.add_parser("merge", parents=[base_op_parser], help="Merge model checkpoints and save.")  # 合并命令参数
    merge_parser.add_argument(
        "--target_dir", default="tmp", type=str, help="Directory to save the merged huggingface model"
    )  # 合并后模型保存目录
    merge_parser.add_argument(
        "--hf_upload_path", default=None, type=str, help="Hugging Face repository ID to upload the model"
    )  # HuggingFace 上传路径
    merge_parser.add_argument(
        "--private", action="store_true", help="Whether to upload the model to a private Hugging Face repository"
    )  # 是否上传到私有库

    test_parser = subparsers.add_parser(
        "test", parents=[base_op_parser], help="Test merged model against a reference Hugging Face model"
    )  # 测试命令参数
    test_parser.add_argument(
        "--test_hf_dir", type=str, required=True, help="Path to the reference Hugging Face model directory for testing"
    )  # 测试参考模型路径

    args = parser.parse_args()  # 解析命令行参数
    return args  # 返回参数对象


@dataclass
class ModelMergerConfig:
    """模型合并操作的配置结构体。"""
    operation: str  # 操作类型：'merge' 或 'test'
    backend: str    # 后端类型：'fsdp' 或 'megatron'
    target_dir: Optional[str] = "tmp"  # 合并后模型保存目录
    hf_upload_path: Optional[str] = None  # HuggingFace 上传路径
    private: bool = False  # 是否上传到私有库
    test_hf_dir: Optional[str] = None  # 测试参考模型路径
    tie_word_embedding: bool = False  # 是否绑定词嵌入权重
    trust_remote_code: bool = False  # 是否信任远程代码
    is_value_model: bool = False  # 是否为 value model
    local_dir: Optional[str] = None  # checkpoint 路径
    hf_model_config_path: Optional[str] = None  # HuggingFace 配置路径
    hf_upload: bool = field(init=False)  # 是否上传到 HuggingFace
    use_cpu_initialization: bool = False  # 是否使用 CPU 初始化

    def __post_init__(self):
        self.hf_upload = self.operation == "merge" and bool(self.hf_upload_path)  # 判断是否需要上传
        if self.operation == "test":
            self.target_dir = None  # 测试时不保存模型
            self.hf_upload_path = None  # 测试时不上传
            self.private = False  # 测试时不设为私有


def generate_config_from_args(args: argparse.Namespace) -> ModelMergerConfig:
    """根据命令行参数生成模型合并配置。"""
    common_config_args = {
        "operation": args.operation,  # 操作类型
        "backend": args.backend,  # 后端类型
        "tie_word_embedding": args.tie_word_embedding,  # 是否绑定词嵌入权重
        "trust_remote_code": args.trust_remote_code,  # 是否信任远程代码
        "is_value_model": args.is_value_model,  # 是否为 value model
        "local_dir": args.local_dir,  # checkpoint 路径
        "hf_model_config_path": os.path.join(args.local_dir, "huggingface"),  # HuggingFace 配置路径
        "use_cpu_initialization": args.use_cpu_initialization,  # 是否使用 CPU 初始化
    }

    if args.operation == "merge":
        config = ModelMergerConfig(
            **common_config_args,
            target_dir=args.target_dir,  # 合并后模型保存目录
            hf_upload_path=args.hf_upload_path,  # HuggingFace 上传路径
            private=args.private,  # 是否上传到私有库
            test_hf_dir=None,  # 测试参考模型路径
        )
        os.makedirs(config.target_dir, exist_ok=True)  # 创建保存目录
    elif args.operation == "test":
        config = ModelMergerConfig(
            **common_config_args,
            test_hf_dir=args.test_hf_dir,  # 测试参考模型路径
            target_dir=None,  # 测试时不保存模型
            hf_upload_path=None,  # 测试时不上传
            private=False,  # 测试时不设为私有
        )
    else:
        raise NotImplementedError(f"Unknown operation: {args.operation}")  # 不支持的操作类型
    return config  # 返回配置对象


class BaseModelMerger(ABC):
    """
    分布式模型 checkpoint 合并为 HuggingFace 格式的抽象基类。
    支持 FSDP、Megatron 等后端，将分布式训练的 checkpoint 转为标准 HuggingFace 格式。
    支持 merge（合并保存）和 test（校验）两种操作。
    """

    def __init__(self, config: ModelMergerConfig):
        self.config = config  # 保存配置
        self.hf_model_config_path = config.hf_model_config_path  # HuggingFace 配置路径
        self.model_config = AutoConfig.from_pretrained(
            self.hf_model_config_path, trust_remote_code=self.config.trust_remote_code
        )  # 加载 HuggingFace 配置

    def get_transformers_auto_model_class(self):
        """根据模型配置自动获取 transformers 的模型类。"""
        has_remote_code = hasattr(self.model_config, "auto_map") and any(
            self.model_config.architectures[0] in val for val in self.model_config.auto_map.values()
        )
        if has_remote_code:
            auto_class = next(
                k for k, v in self.model_config.auto_map.items() if self.model_config.architectures[0] in v
            )
            match auto_class:
                case "AutoModelForCausalLM":
                    return AutoModelForCausalLM
                case "AutoModelForTokenClassification":
                    return AutoModelForTokenClassification
                case "AutoModelForVision2Seq":
                    return AutoModelForVision2Seq
                case _:
                    raise NotImplementedError(f"Unknown auto class {auto_class}")
        else:
            if "ForTokenClassification" in self.model_config.architectures[0]:
                return AutoModelForTokenClassification
            elif "ForCausalLM" in self.model_config.architectures[0]:
                return AutoModelForCausalLM
            elif "ForConditionalGeneration" in self.model_config.architectures[0]:
                return AutoModelForVision2Seq

            raise NotImplementedError(f"Unknown architecture {self.model_config.architectures}")

    def patch_model_generation_config(self, model):
        """
        The generation_config created from model config may be different to the pretrained model,
        this may lead to error when generating: https://github.com/volcengine/verl/issues/1246

        This function patch the generation_config created from model config to the pretrained model.
        """
        if model.can_generate():
            try:
                model.generation_config = GenerationConfig.from_pretrained(self.hf_model_config_path)
            except OSError:
                print(
                    f"Warning: Generation config file not found in {self.hf_model_config_path}, using a "
                    f"generation config created from the model config."
                )
        return model

    def save_lora_adapter(self, state_dict: dict[str, torch.Tensor]):
        """
        Save lora adapter to safetensors.

        Returns:
            lora_path: str, the path to the lora adapter. None if no lora adapter found.

        Note:
            This function change the 'state_dict' in place.
        """
        lora_params_names = [name for name in state_dict.keys() if "lora_" in name]

        if len(lora_params_names) == 0:
            return None

        import json
        from typing import OrderedDict

        import peft
        from safetensors.torch import save_file

        lora_params = OrderedDict()
        target_modules = set()
        lora_key = None

        for name in lora_params_names:
            lora_key = name.replace(".default.weight", ".weight")
            target_modules.add(lora_key.split(".")[-3])
            lora_params[lora_key] = state_dict.pop(name)

        lora_rank = min(lora_params[lora_key].shape[0], lora_params[lora_key].shape[1])
        peft_dict = {
            "r": lora_rank,
            "lora_alpha": 0,  # lora_alpha is not set. An error should be raised to inform the user to set it manually.
            "target_modules": list(target_modules),
        }
        peft_config = peft.LoraConfig(**peft_dict).to_dict()
        peft_config["task_type"] = peft_config["task_type"].value if peft_config["task_type"] else None
        peft_config["peft_type"] = peft_config["peft_type"].value if peft_config["peft_type"] else None
        peft_config["target_modules"] = list(peft_config["target_modules"])

        lora_path = os.path.join(self.config.target_dir, "lora_adapter")
        os.makedirs(lora_path, exist_ok=True)
        with open(os.path.join(lora_path, "adapter_config.json"), "w", encoding="utf-8") as f:
            json.dump(peft_config, f, ensure_ascii=False, indent=4)
        save_file(lora_params, os.path.join(lora_path, "adapter_model.safetensors"))

        for name in list(state_dict.keys()):
            key = (
                name.replace("base_model.model.", "")
                .replace(".base_layer.weight", ".weight")
                .replace(".base_layer.bias", ".bias")
            )
            state_dict[key] = state_dict.pop(name)

        return lora_path

    def save_hf_model_and_tokenizer(self, state_dict: dict[str, torch.Tensor]):
        auto_model_class = self.get_transformers_auto_model_class()
        with init_empty_weights():
            model = auto_model_class.from_config(
                self.model_config, torch_dtype=torch.bfloat16, trust_remote_code=self.config.trust_remote_code
            )
        model.to_empty(device="cpu")
        model = self.patch_model_generation_config(model)

        lora_path = self.save_lora_adapter(state_dict)
        if lora_path:
            print(f"Saving lora adapter to {lora_path}")

        print(f"Saving model to {self.config.target_dir}")
        model.save_pretrained(self.config.target_dir, state_dict=state_dict)
        del state_dict
        del model

        processor = hf_processor(self.hf_model_config_path, trust_remote_code=self.config.trust_remote_code)
        tokenizer = hf_tokenizer(self.hf_model_config_path, trust_remote_code=self.config.trust_remote_code)
        if processor is not None:
            print(f"Saving processor to {self.config.target_dir}")
            processor.save_pretrained(self.config.target_dir)
        if tokenizer is not None:
            print(f"Saving tokenizer to {self.config.target_dir}")
            tokenizer.save_pretrained(self.config.target_dir)

    def upload_to_huggingface(self):
        import requests
        from huggingface_hub import HfApi
        from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

        api = HfApi()
        try:
            # Attempt to create repository
            api.create_repo(repo_id=self.config.hf_upload_path, private=self.config.private, exist_ok=True)
        except HfHubHTTPError as e:
            # Handle authentication/API errors
            if e.response.status_code == 401:
                raise PermissionError(
                    "Hugging Face authentication failed. Verify your token is valid and has write permissions."
                ) from e
            elif e.response.status_code == 404:
                raise RepositoryNotFoundError(f"Repository path not found: {self.config.hf_upload_path}") from e
            else:
                raise ConnectionError(f"Failed to create repository ({e.response.status_code}): {e}") from e
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError("Network connection failed. Check your internet connection.") from e

        try:
            # 尝试上传文件夹到 HuggingFace Hub
            api.upload_folder(folder_path=self.config.target_dir, repo_id=self.config.hf_upload_path, repo_type="model")
        except HfHubHTTPError as e:
            # 捕获 HuggingFace Hub 的 HTTP 错误
            if e.response.status_code == 401:
                # 401 认证失败，可能是 token 过期或权限不足
                raise PermissionError("Authentication failed during upload. Token may have expired.") from e
            else:
                # 其他 HTTP 错误，抛出运行时异常并显示状态码
                raise RuntimeError(f"Upload failed ({e.response.status_code}): {e}") from e
        except requests.exceptions.ConnectionError as e:
            # 网络连接中断，提示用户检查网络
            raise ConnectionError("Network interruption during upload. Try again with stable connection.") from e
        except OSError as e:
            # 本地文件夹错误，可能路径不存在或权限问题
            raise FileNotFoundError(f"Local folder error: {self.config.target_dir} - {str(e)}") from e
        except Exception as e:
            # 其他未知错误，统一抛出运行时异常
            raise RuntimeError(f"Unexpected error during upload: {str(e)}") from e

    @abstractmethod
    def merge_and_save(self):
        # 抽象方法，子类需实现模型权重合并与保存逻辑
        raise NotImplementedError("Subclasses should implement this method")

    @abstractmethod
    def cleanup(self):
        # 抽象方法，子类需实现资源清理逻辑（如删除临时文件等）
        raise NotImplementedError("Subclasses should implement this method to clean up resources if needed")
