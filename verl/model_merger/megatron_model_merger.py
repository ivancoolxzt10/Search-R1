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

import json  # JSON 处理模块
import os    # 操作系统相关模块
import warnings  # 警告处理模块
from contextlib import contextmanager  # 上下文管理器
from pathlib import Path  # 路径处理工具
from typing import Any, Callable, ContextManager  # 类型注解

import numpy as np  # 数组库
import torch  # 深度学习库
import torch.distributed as dist  # 分布式训练工具
from accelerate import init_empty_weights  # HuggingFace 加速库
from megatron.core import mpu  # Megatron 并行工具
from megatron.core.models.gpt.gpt_model import ModelType  # Megatron GPT 模型类型
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed  # 并行随机种子
from safetensors.torch import load_file  # safetensors 权重加载
from transformers import (
    AutoConfig,
    PretrainedConfig,
)  # transformers 配置

from verl.models.mcore import hf_to_mcore_config  # HuggingFace 到 mcore 配置转换
from verl.utils.device import get_device_name, get_nccl_backend, get_torch_device  # 设备工具
from verl.utils.megatron.dist_checkpointing import load_dist_checkpointing  # Megatron 分布式 checkpoint 加载
from verl.utils.megatron_utils import get_model  # Megatron 模型获取工具
from verl.utils.tokenizer import hf_processor, hf_tokenizer  # huggingface 工具

from .base_model_merger import BaseModelMerger, ModelMergerConfig  # 合并基类和配置


@contextmanager
def noop_context() -> Any:
    """空上下文管理器，用于兼容部分流程。"""
    yield


def get_dynamic_pipeline_shards(layer_num: int, pp_size: int) -> list[int]:
    """计算 Megatron-LM 的 pipeline 分片配置。

    Args:
        layer_num: 总层数
        pp_size: pipeline 并行数

    Returns:
        每个 pp rank 的层数配置，尽量均匀
    """
    if layer_num < pp_size:
        raise ValueError(f"layer_num {layer_num} must be greater than pp_size {pp_size}.")

    if pp_size < 1:
        raise ValueError(f"pp_size must be at least 1, got {pp_size}.")
    if pp_size == 1:
        return [layer_num]

    if pp_size == 2:
        return [
            layer_num // 2,
            layer_num - layer_num // 2,
        ]

    middle_size = pp_size - 2
    shards_strategy = []
    for middle_layer_num in range(layer_num):
        first_last_layer_num = layer_num - middle_layer_num * middle_size
        first_layer_num = first_last_layer_num // 2
        last_layer_num = first_last_layer_num - first_last_layer_num // 2
        if 0 < first_layer_num <= middle_layer_num and 0 < last_layer_num <= middle_layer_num:
            shards_strategy.append(
                (
                    [first_layer_num] + [middle_layer_num] * middle_size + [last_layer_num],
                    abs(first_layer_num - middle_layer_num),
                )
            )

    # 按分片层数差异排序，尽量均匀分片
    res = sorted(shards_strategy, key=lambda x: x[1])[0][0]
    assert sum(res) == layer_num, f"sum(res)={sum(res)} != layer_num={layer_num}, pp_size={pp_size}"
    return res


class MegatronModelMerger(BaseModelMerger):
    """
    Megatron-LM 分布式 checkpoint 合并器。

    该类负责将 Megatron-LM 分布式 checkpoint 转换为 HuggingFace 格式。
    Megatron-LM 使用张量并行、流水线并行和数据并行将大语言模型分布在多个 GPU 上。
    此合并器通过加载分布式检查点并应用必要的转换来重建完整模型。

    主要功能：
    - 支持张量并行、流水线并行和数据并行配置
    - 自动参数名称映射从 Megatron 到 HuggingFace 约定
    - 处理 QKV 和 gate-up 张量的分割/合并
    - 支持绑定的词嵌入和价值模型
    - 与 Megatron 的分布式检查点系统集成

    合并器处理各种模型架构和配置：
    - 标准变压器模型（GPT 风格）
    - 具有绑定词嵌入的模型
    - 强化学习的价值模型
    - 多层注意力（MLA）架构
    - 专家混合（MoE）模型

    Args:
        config (ModelMergerConfig): 配置对象，包含 Megatron 特定设置
            包括 tie_word_embedding和 is_value_model 标志。

    示例:
        要合并 Megatron 检查点：
        ```python
        config = ModelMergerConfig(
            operation="merge",
            backend="megatron",
            local_dir="path/to/megatron/checkpoints",
            target_dir="path/to/output",
            tie_word_embedding=True
        )
        merger = MegatronModelMerger(config)
        merger.merge_and_save()
        ```
    """

    def __init__(self, config: ModelMergerConfig):
        super().__init__(config)
        # 仅用 1 个 rank 合并分布式 checkpoint，后续可扩展为多进程
        if "WORLD_SIZE" not in os.environ:
            os.environ["RANK"] = "0"
            os.environ["LOCAL_RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "12355"

        torch.distributed.init_process_group(get_nccl_backend())

        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()
        local_rank = os.environ.get("LOCAL_RANK", 0)
        get_torch_device().set_device(f"{get_device_name()}:{local_rank}")

        mpu.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=self.world_size,
            virtual_pipeline_model_parallel_size=None,
            context_parallel_size=1,
            expert_model_parallel_size=1,
        )
        model_parallel_cuda_manual_seed(0)
        self.hf_config = AutoConfig.from_pretrained(
            self.config.hf_model_config_path, trust_remote_code=self.config.trust_remote_code
        )
        print(self.hf_config, flush=True)

        self.params_mapping = {
            # Megatron GPT 模型名与 HuggingFace 模型名映射
            # 注意：当两个键具有相同前缀时，必须先处理包含关系较长的键。
            "embedding.word_embeddings": "model.embed_tokens",
            # input layer norm for dpskv3
            "input_layernorm.weight": "input_layernorm.weight",
            "input_layernorm.bias": "input_layernorm.bias",
            # attn
            "self_attention.linear_qkv.layer_norm_weight": "input_layernorm.weight",
            "self_attention.linear_qkv.layer_norm_bias": "input_layernorm.bias",
            "self_attention.linear_qkv": "self_attn.qkv_proj",
            "self_attention.q_layernorm": "self_attn.q_norm",
            "self_attention.k_layernorm": "self_attn.k_norm",
            "self_attention.linear_proj": "self_attn.o_proj",
            # mla
            "self_attention.linear_q_proj": "self_attn.q_proj",
            "self_attention.linear_q_down_proj": "self_attn.q_a_proj",
            "self_attention.linear_q_up_proj.layer_norm_weight": "self_attn.q_a_layernorm.weight",
            "self_attention.linear_q_up_proj": "self_attn.q_b_proj",
            "self_attention.linear_kv_down_proj": "self_attn.kv_a_proj_with_mqa",
            "self_attention.linear_kv_up_proj.layer_norm_weight": "self_attn.kv_a_layernorm.weight",
            "self_attention.linear_kv_up_proj": "self_attn.kv_b_proj",
            # mlp
            "pre_mlp_layernorm": "post_attention_layernorm",
            "mlp.linear_fc1.layer_norm_weight": "post_attention_layernorm.weight",
            "mlp.linear_fc1.layer_norm_bias": "post_attention_layernorm.bias",
            "mlp.linear_fc1": "mlp.gate_up_proj",
            "mlp.linear_fc2": "mlp.down_proj",
            # moe
            "mlp.router.expert_bias": "mlp.gate.e_score_correction_bias",
            "mlp.router": "mlp.gate",
            "mlp.shared_experts.linear_fc1": "mlp.shared_experts.gate_up_proj",
            "mlp.shared_experts.linear_fc2": "mlp.shared_experts.down_proj",
            "linear_fc1": "gate_up_proj",
            "linear_fc2": "down_proj",
            # output
            "final_layernorm": "norm",
            "output_layer": "lm_head",
        }

        if "Qwen2MoeForCausalLM" in self.hf_config.architectures:
            self.params_mapping["mlp.shared_experts.linear_fc1"] = "mlp.shared_expert.gate_up_proj"
            self.params_mapping["mlp.shared_experts.linear_fc2"] = "mlp.shared_expert.down_proj"
            self.params_mapping["mlp.shared_experts.gate_weight"] = "mlp.shared_expert_gate.weight"

    def _load_state_dicts(self, model_ckpt_path: str) -> dict[str, Any]:
        """_summary_
        使用 Megatron dist_checkpointing 从检查点目录加载模型状态字典。

        Args:
            model_ckpt_path (str): 模型检查点目录路径。

        Returns:
            包含模型参数的状态字典。
        """

        # 初始化 hf config
        self.pipeline_shards = get_dynamic_pipeline_shards(self.hf_config.num_hidden_layers, self.world_size)
        print(f"Pipeline shards: {self.pipeline_shards}, total layers: {sum(self.pipeline_shards)}")

        tf_config = hf_to_mcore_config(
            self.hf_config,
            torch.bfloat16,
            num_layers_in_first_pipeline_stage=self.pipeline_shards[0] if len(self.pipeline_shards) > 1 else None,
            num_layers_in_last_pipeline_stage=self.pipeline_shards[-1] if len(self.pipeline_shards) > 2 else None,
        )
        tf_config.use_cpu_initialization = self.config.use_cpu_initialization
        tie_word_embeddings = getattr(self.hf_config, "tie_word_embeddings", False)

        # 初始化 megatron 模型
        def megatron_model_provider(pre_process, post_process):
            from verl.models.mcore import init_mcore_model

            parallel_model = init_mcore_model(
                tf_config,
                self.hf_config,
                pre_process,
                post_process,
                share_embeddings_and_output_weights=tie_word_embeddings,
                value=False,
            )
            return parallel_model

        context: Callable[..., ContextManager] = (
            init_empty_weights if self.config.use_cpu_initialization else noop_context
        )
        with context():
            whole_model = get_model(
                model_provider_func=megatron_model_provider,
                model_type=ModelType.encoder_or_decoder,
                wrap_with_ddp=False,
                transformer_config=tf_config,
            )

        if self.config.use_cpu_initialization:
            # 将 meta 设备转换为空张量，以便使用 `copy_` 函数
            whole_model[0].module = whole_model[0].module.to_empty(device="cpu")

        # 加载状态字典
        sharded_state_dict = {}
        for vpp_rank, model in enumerate(whole_model):
            key = f"model{vpp_rank}" if len(whole_model) > 1 else "model"
            mpu.set_virtual_pipeline_model_parallel_rank(vpp_rank)
            sharded_state_dict[key] = model.sharded_state_dict()
        model_state_dict = load_dist_checkpointing(sharded_state_dict, model_ckpt_path)
        model_state_dict_list = []
        for vpp_rank, model in enumerate(whole_model):
            key = f"model{vpp_rank}" if len(whole_model) > 1 else "model"
            mpu.set_virtual_pipeline_model_parallel_rank(vpp_rank)
            model_state_dict_list.append(model_state_dict[key])

        return model_state_dict_list

    def _check_megatron_state_key(self, key: str) -> bool:
        """
        检查给定的 key 是否为有效的 Megatron 状态 key。

        目前合并器仅支持以 "decoder/embedding/output_layer" 开头的 TransformerLayer 中的键。
        不应使用以 "model." 开头的键。
        """
        if key.startswith("model."):
            raise ValueError(
                f"Invalid key {key} in Megatron state_dict. Expected keys to start with "
                f"'decoder/embedding/output_layer' in TransformerLayer."
            )

        skip_checking_keys = ["embedding.word_embeddings", "output_layer"]
        for skip_key in skip_checking_keys:
            if skip_key in key:
                print(f"skip checking key {key}")
                return

        # 排除额外的状态键
        if not key.startswith("decoder"):
            raise ValueError(
                f"Invalid key {key} in Megatron state_dict. Expected keys to start with 'decoder' in TransformerLayer."
            )

    def _split_tensors(
        self, key: str, tensor: torch.Tensor, config: PretrainedConfig, is_value_model: bool = False
    ) -> list[torch.Tensor]:
        """
        根据名称将张量拆分为多个张量。
        用于处理 qkv 和 gate_up 张量。
        """
        if "linear_fc1.weight" in key:
            # 如果张量是 gate 和 proj
            gate_lst = []
            up_lst = []
            gate, up = tensor.chunk(2)
            gate_lst.append(gate)
            up_lst.append(up)
            gate = torch.cat(gate_lst, dim=0)
            up = torch.cat(up_lst, dim=0)
            return [gate, up]
        elif "self_attention.linear_qkv." in key and "layer_norm" not in key:
            # 如果张量是 qkv，对于每个 param 在 tp 上，拆分为 q、k、v
            # 分别连接 q、k、v。
            q_lst, k_lst, v_lst = [], [], []
            assert config.num_attention_heads % config.num_key_value_heads == 0
            num_q_per_kv = config.num_attention_heads // config.num_key_value_heads
            assert tensor.shape[0] % (num_q_per_kv + 2) == 0, (
                f"Tensor shape {tensor.shape} is not divisible by {num_q_per_kv + 2}"
            )
            kv_size = tensor.shape[0] // (num_q_per_kv + 2)
            split_size = [kv_size * num_q_per_kv, kv_size, kv_size]

            num_query_groups_per_partition = config.num_key_value_heads
            for chunk in tensor.chunk(num_query_groups_per_partition):
                split_size = [
                    kv_size * num_q_per_kv // num_query_groups_per_partition,
                    kv_size // num_query_groups_per_partition,
                    kv_size // num_query_groups_per_partition,
                ]
                q, k, v = chunk.split(split_size)
                q_lst.append(q)
                k_lst.append(k)
                v_lst.append(v)

            return [torch.cat(q_lst, dim=0), torch.cat(k_lst, dim=0), torch.cat(v_lst, dim=0)]
        else:
            return [tensor]

    def _merge_state_dicts(self, model_state_dict_list: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """
        合并多个模型状态字典为一个字典。
        对于每个模型状态字典，检查并转换键名，拆分张量，处理专家模型的特殊情况。
        """
        state_dict = {}
        layers_cum = 0
        if self.world_size > 1:
            pipeline_cumsum = np.cumsum(self.pipeline_shards)
            layers_cum = 0 if self.rank == 0 else pipeline_cumsum[self.rank - 1]

        print(f"{layers_cum=}")
        for model_state_dict in model_state_dict_list:
            layers_handled = 0
            keys = model_state_dict.keys()
            for key in keys:
                if "extra_state" in key:
                    continue
                if self.config.tie_word_embedding and ("output_layer" in key):
                    print("skip lm_head and reward_head loading because of tie_word_embeddings")
                    continue

                self._check_megatron_state_key(key)
                hf_name = self._replace_name(key, self.params_mapping)
                assert hf_name is not None, f"Failed to convert layer name [{key}] from megatron to huggingface."
                if "model.layers." in hf_name:
                    local_layer_no = int(hf_name.split(".")[2])
                    layers_handled = max(local_layer_no, layers_handled)
                    global_layer_no = local_layer_no + layers_cum
                    new_key_list = hf_name.split(".")
                    new_key_list[2] = str(global_layer_no)
                    hf_name = ".".join(new_key_list)
                else:
                    warnings.warn(f"hf_name {hf_name} will not be fixed with layer number", stacklevel=2)

                if "mlp.experts." in hf_name and ".weight" in hf_name:
                    name_prefix, expert_id = hf_name.split(".weight")
                    for proj in ["gate_up", "down"]:
                        if f"{proj}_proj" in hf_name:
                            hf_name = hf_name.replace(
                                f"mlp.experts.{proj}_proj.weight{expert_id}",
                                f"mlp.experts.{expert_id}.{proj}_proj.weight",
                            )

                tensor = model_state_dict[key]
                split_tensor = self._split_tensors(
                    key, tensor, self.hf_config, is_value_model=self.config.is_value_model
                )

                if len(split_tensor) == 1:
                    state_dict[hf_name] = split_tensor[0]
                elif len(split_tensor) == 3:
                    # 拆分 qkv
                    for n, d in zip(["q", "k", "v"], split_tensor, strict=True):
                        state_dict[hf_name.replace("qkv", n)] = d
                elif len(split_tensor) == 2:
                    # 拆分 gate up
                    state_dict[hf_name.replace("gate_up", "gate")] = split_tensor[0]
                    state_dict[hf_name.replace("gate_up", "up")] = split_tensor[1]
                shape_info = (
                    split_tensor.shape if isinstance(split_tensor, torch.Tensor) else [t.shape for t in split_tensor]
                )
                print(f"converted {key} to {hf_name} with shape {shape_info}")

            layers_cum += layers_handled + 1  # 从零开始计数

        return state_dict

    def save_hf_model_and_tokenizer(self, merged_state_dict):
        """
        保存合并后的 HuggingFace 模型和分词器。
        如果是单卡，则直接调用父类方法；
        如果是多卡，则手动分片保存模型权重。
        """
        if self.world_size == 1:
            return super().save_hf_model_and_tokenizer(merged_state_dict)

        from safetensors.torch import save_file

        layer_num = self.hf_config.num_hidden_layers

        # FIXME: 可配置
        saves_per_layer = 1 if layer_num < 30 else 2
        saves_total = saves_per_layer * layer_num
        saves_indexes = {}

        # 计算每层起始索引和键的分片
        layer_this_rank = self.pipeline_shards[self.rank]
        pipeline_cumsum = np.cumsum(self.pipeline_shards)
        layer_start = 0 if self.rank == 0 else pipeline_cumsum[self.rank - 1]
        keys = list(merged_state_dict.keys())
        keys_chunk = np.array_split(np.array(keys), layer_this_rank * saves_per_layer)
        numel = 0

        assert len(keys_chunk) == layer_this_rank * saves_per_layer, (
            f"Expected {len(keys_chunk)} chunks, but got {layer_this_rank * saves_per_layer} for rank {self.rank}."
        )

        # 手动保存模型分片
        target_dir = Path(self.config.target_dir)
        for i, keys in enumerate(keys_chunk):
            sd_to_save = {k: merged_state_dict[k] for k in keys}
            numel += sum([sd_to_save[i].numel() for i in sd_to_save])
            save_idx = layer_start * saves_per_layer + i
            save_path = target_dir / f"model-{save_idx + 1:05d}-of-{saves_total:05d}.safetensors"

            save_file(sd_to_save, save_path)
            for k in keys:
                saves_indexes[k] = str(save_path.name)

        tensor = torch.tensor([numel]).to(get_device_name())
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        numel = tensor.cpu().item()

        all_save_indexes = [{} for _ in range(self.world_size)]
        dist.all_gather_object(all_save_indexes, saves_indexes)
        saves_indexes = {k: v for i in all_save_indexes for k, v in i.items()}
        if self.rank == 0:
            with open(target_dir / "model.safetensors.index.json", "w") as f:
                json.dump(
                    {
                        "metadata": {
                            "total_size": numel,
                        },
                        "weight_map": saves_indexes,
                    },
                    f,
                    indent=4,
                )
            print(f"model saved to {target_dir} with {numel=}")

            self.model_config.save_pretrained(self.config.target_dir)

            processor = hf_processor(self.hf_model_config_path, trust_remote_code=self.config.trust_remote_code)
            tokenizer = hf_tokenizer(self.hf_model_config_path, trust_remote_code=self.config.trust_remote_code)
            if processor is not None:
                print(f"Saving processor to {self.config.target_dir}")
                processor.save_pretrained(self.config.target_dir)
            if tokenizer is not None:
                print(f"Saving tokenizer to {self.config.target_dir}")
                tokenizer.save_pretrained(self.config.target_dir)

    def merge_and_save(self):
        """
        主合并与保存流程：
        1. 获取分布式检查点路径
        2. 加载状态字典
        3. 合并状态字典
        4. 保存 HuggingFace 模型和分词器
        5. 可选：上传到 HuggingFace Hub
        """
        from verl.utils.megatron_utils import get_dist_checkpoint_path

        model_ckpt_path = get_dist_checkpoint_path(self.config.local_dir)

        model_state_dict = self._load_state_dicts(model_ckpt_path)
        merged_state_dict = self._merge_state_dicts(model_state_dict)
        del model_state_dict

        if self.config.operation == "test":
            if not self.config.test_hf_dir:
                raise ValueError("test_hf_dir must be provided for test operation")
            self._validate_state_dict(merged_state_dict)
        elif self.config.operation == "merge":
            self.save_hf_model_and_tokenizer(merged_state_dict)
            if self.config.hf_upload:
                self.upload_to_huggingface()
        else:
            raise ValueError(f"Unknown operation: {self.config.operation}")

    def _validate_state_dict(self, state_dict: dict[str, torch.Tensor]):
        """
        将合并后的 Megatron 状态字典与参考 safetensors 模型进行比较。
        应用必要的名称映射从 Megatron 到 Hugging Face 的约定使用 _replace_name。
        """
        ref_state_dict = load_file(Path(self.config.test_hf_dir) / "model.safetensors")

        for name, loaded_weight in state_dict.items():
            # name = self._replace_name(original_name, self.params_mapping)
            if not name or name.endswith(".bias") and name not in ref_state_dict:
                continue
            if "rotary_emb.inv_freq" in name:
                continue
            if "lm_head.weight" in name:
                if self.config.is_value_model or self.config.tie_word_embedding:
                    continue
            if name not in ref_state_dict:
                raise RuntimeError(f"key: {name} not exist in state_dict")
            param = ref_state_dict[name]
            assert loaded_weight.dtype == param.dtype
            torch.testing.assert_close(loaded_weight.to("cpu"), param, atol=1e-2, rtol=5e-2)

    def _replace_name(self, megatron_name: str, name_mapping: dict[str, str]) -> str:
        """
        根据名称映射字典转换 Megatron 模型的参数名称为 HuggingFace 约定。
        """
        for m_name, v_name in name_mapping.items():
            if m_name not in megatron_name:
                continue

            megatron_name = megatron_name.replace("decoder", "model")
            param_name = megatron_name.replace(m_name, v_name)

            return param_name

        return None  # 如果没有找到映射，则返回 None

    def cleanup(self):
        """
        清理过程，销毁进程组。
        """
        torch.distributed.destroy_process_group()
