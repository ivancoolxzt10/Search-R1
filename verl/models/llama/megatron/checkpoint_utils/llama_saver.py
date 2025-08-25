# 初学者备注：
# 本文件用于保存 Megatron 分布式模型的权重，包含全局 rank 计算和 layer 映射等辅助函数。
# 适用于多卡训练和模型保存场景，主要面向分布式训练初学者。
# 典型用法：调用 _megatron_calc_global_rank(tp_rank, dp_rank, pp_rank) 获取全局 rank。
#
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

import time

import torch
import torch.distributed as dist
from megatron.core import mpu
from megatron.core.distributed import DistributedDataParallel as LocalDDP
from megatron.core.transformer.module import Float16Module
from torch.nn.parallel import DistributedDataParallel as torchDDP

from verl.utils.device import get_device_id, get_torch_device
from verl.utils.logger import print_rank_0
from verl.utils.megatron_utils import unwrap_model


def _megatron_calc_global_rank(tp_rank: int = 0, dp_rank: int = 0, pp_rank: int = 0):
    """
    初学者备注：
    该函数用于根据张量并行（TP）、数据并行（DP）、流水线并行（PP）的 rank 计算全局 rank。
    在分布式训练/保存时，确定每个进程的唯一编号。
    """
    tp_size = mpu.get_tensor_model_parallel_world_size()  # 获取张量并行总数
    dp_size = mpu.get_data_parallel_world_size()  # 获取数据并行总数
    pp_size = mpu.get_pipeline_model_parallel_world_size()  # 获取流水线并行总数
    assert tp_size * dp_size * pp_size == torch.distributed.get_world_size(), (
        f"{tp_size} x {dp_size} x {pp_size} != {torch.distributed.get_world_size()}"
    )  # 检查总进程数是否匹配
    # 只支持 TP-DP-PP 分组，保证参数重分布时正确
    return (pp_rank * dp_size + dp_rank) * tp_size + tp_rank  # 计算全局 rank，优先级：PP > DP > TP


def _megatron_calc_layer_map(config):
    """
    初学者备注：
    该函数用于计算全局层索引到本地层索引的映射，适用于 Megatron 分布式模型保存。
    输入参数 config 通常为模型配置对象，包含层数等信息。
    返回 layer_map 字典，key 为全局层索引，value 为 (pp_rank, virtual_pp_rank, 层索引)。
    """
    from megatron.core import mpu  # 导入 Megatron 并行工具模块

    pp_size = mpu.get_pipeline_model_parallel_world_size()  # 获取流水线并行总数
    virtual_pp_size = mpu.get_virtual_pipeline_model_parallel_world_size() or 1  # 获取虚拟流水线并行数，默认为1

    layer_map = dict()  # 初始化层映射字典
    num_layers_per_model = config.num_hidden_layers // pp_size // virtual_pp_size  # 计算每个模型分到的层数
    assert num_layers_per_model * pp_size * virtual_pp_size == config.num_hidden_layers  # 检查层数分配是否正确

    for pp_rank_idx in range(pp_size):  # 遍历所有流水线 rank
        for virtual_pp_rank_idx in range(virtual_pp_size):  # 遍历所有虚拟流水线 rank
            layer_offset = (
                virtual_pp_rank_idx * (config.num_hidden_layers // virtual_pp_size) + pp_rank_idx * num_layers_per_model
            )  # 计算当前 rank 的层偏移量
            for layer_idx in range(num_layers_per_model):  # 遍历本地模型的所有层
                layer_map[layer_offset + layer_idx] = (
                    pp_rank_idx,  # 流水线 rank
                    virtual_pp_rank_idx,  # 虚拟流水线 rank
                    layer_idx,  # 本地层索引
                )  # 将映射关系加入字典
    return layer_map  # 返回层映射字典


def merge_megatron_ckpt_llama(wrapped_models, config, dtype, is_value_model=False, tie_word_embeddings=False):
    """
    Megatron 分布式权重合并函数。
    该函数将分布在各 rank 的分片参数合并为完整的 state_dict，仅在 rank 0 返回合并结果，其它 rank 返回空字典。
    Args:
        wrapped_models: Megatron 分布式 DDP 包装模型列表
        config: HuggingFace 模型配置
        dtype: 参数数据类型
        is_value_model: 是否为 value model（兼容接口）
        tie_word_embeddings: 是否绑定词嵌入（llama 不用，仅为接口兼容）
    Returns:
        state_dict: rank 0 返回合并后的权重字典，其它 rank 返回空字典
    """
    start_time = time.time()  # 记录开始时间

    def _get_gpt_model(model):
        # 兼容不同包装，实际返回底层模型
        return model

    dp_rank = mpu.get_data_parallel_rank()  # 数据并行 rank
    pp_size = mpu.get_pipeline_model_parallel_world_size()  # 流水线并行分段数
    pp_rank = mpu.get_pipeline_model_parallel_rank()  # 当前流水线分段 rank
    virtual_pp_size = mpu.get_virtual_pipeline_model_parallel_world_size() or 1  # 虚拟流水线分段数
    mp_group = mpu.get_model_parallel_group()  # 模型并行通信组

    # rank0 校验分布式并行配置
    if dist.get_rank() == 0:
        assert mp_group.rank() == 0, f"mp_rank:[{mp_group.rank}] != 0 on rank #0"
        assert pp_rank == 0, f"pp_rank:[{pp_rank}] != 0 on rank #0"
        assert dp_rank == 0, f"dp_rank:[{dp_rank}] != 0 on rank #0"

    # 保证 wrapped_models 为列表
    if not isinstance(wrapped_models, list | tuple):
        wrapped_models = list(wrapped_models)

    assert len(wrapped_models) == virtual_pp_size  # 每个虚拟分段一个模型
    num_layers_per_model = config.num_hidden_layers // pp_size // virtual_pp_size  # 每个模型负责的层数
    assert num_layers_per_model * pp_size * virtual_pp_size == config.num_hidden_layers

    models = [None] * len(wrapped_models)

    # 解包 DDP，获得底层模型
    for i, wrapped_model in enumerate(wrapped_models):
        models[i] = unwrap_model(wrapped_model, (torchDDP, LocalDDP, Float16Module))
        assert len(models[i].model.layers) == num_layers_per_model, (
            "len model layers {} not equal to num_layers_per_model {}".format(
                len(models[i].model.layers), num_layers_per_model
            )
        )

    state_dict = dict()  # 用于收集合并后的参数

    def _get_cpu_tensor(tensor: torch.Tensor):
        # 将张量转为 CPU，便于保存和后续处理
        if tensor is None:
            return None
        if tensor.device == torch.device("cpu"):
            return tensor.detach().clone()
        return tensor.detach().cpu()

    def _broadcast_tensor(tensor, name, src_pp_rank) -> torch.Tensor:
        """
        在 mp_group 内广播张量，收集分布式参数到 rank 0。
        """
        nonlocal state_dict
        nonlocal mp_group
        src_rank = _megatron_calc_global_rank(tp_rank=0, dp_rank=0, pp_rank=src_pp_rank)  # 计算源 rank

        # 判断当前进程是否为源 rank
        if torch.distributed.get_rank() == src_rank:
            # 如果张量为 None，则权重和形状都为 None
            if tensor is None:
                weight = None
                tensor_shape = None
            else:
                # 否则获取张量和其形状
                weight = tensor
                tensor_shape = weight.shape
        else:
            # 非源 rank，权重和形状都为 None
            weight = None
            tensor_shape = None

        # 用于广播张量形状
        obj_list = [tensor_shape]
        dist.broadcast_object_list(obj_list, src=src_rank, group=mp_group)
        tensor_shape = obj_list[0]

        # 如果形状为 None，说明张量不存在，跳过
        if tensor_shape is None:
            # all or none ranks in the mp_group should reach here
            print_rank_0(f"tensor:[{name}] not exist, skip collect")
            return

        # 如果权重为 None，则新建一个空张量用于接收广播
        if weight is None:
            weight = torch.empty(
                tensor_shape,
                dtype=dtype,
                device=get_device_id(),
                requires_grad=False,
            )

        # 广播权重张量到所有进程，确保每个 rank 都拿到正确参数
        dist.broadcast(weight, src=src_rank, group=mp_group)

        if torch.distributed.get_rank() == 0:
            state_dict[name] = _get_cpu_tensor(weight)  # rank 0 收集参数到最终 state_dict

    def _broadcast_tp_shard_tensor(tensor, name, src_pp_rank, concat_dim=0, mutate_func=None) -> torch.Tensor:
        """
        在 mp_group 内广播分块张量，收集所有分片参数并在 rank 0 合并。
        """
        nonlocal state_dict  # 使用外部的 state_dict 字典
        nonlocal mp_group  # 使用外部的模型并行通信组
        tp_size = mpu.get_tensor_model_parallel_world_size()  # 获取张量并行总数
        src_rank = _megatron_calc_global_rank(tp_rank=0, dp_rank=0, pp_rank=src_pp_rank)  # 计算源 rank

        chunk_shape = tensor.shape if torch.distributed.get_rank() == src_rank else None  # 源 rank 获取张量形状

        obj_list = [chunk_shape]
        dist.broadcast_object_list(obj_list, src=src_rank, group=mp_group)  # 广播张量形状
        chunk_shape = obj_list[0]
        if chunk_shape is None:
            print_rank_0(f"tp_shard tensor:[{name}] not exist, skip collecting")  # 未找到参数则跳过
            return

        buffer_tensor = torch.empty(
            chunk_shape,
            dtype=dtype,
            device=get_device_id(),
            requires_grad=False,
        )  # 新建空张量用于接收分块参数

        chunk_tensors = [None] * tp_size  # 初始化分块张量列表

        for i in range(tp_size):  # 遍历所有张量并行 rank
            cur_src_rank = _megatron_calc_global_rank(tp_rank=i, dp_rank=0, pp_rank=src_pp_rank)  # 计算当前 rank
            sync_tensor = tensor if torch.distributed.get_rank() == cur_src_rank else buffer_tensor  # 源 rank 用原始张量，其它 rank 用 buffer
            dist.broadcast(sync_tensor, src=cur_src_rank, group=mp_group)  # 广播分块参数

            if torch.distributed.get_rank() == 0:
                chunk_tensors[i] = _get_cpu_tensor(sync_tensor)  # rank 0 收集所有分块参数

        if torch.distributed.get_rank() == 0:
            full_tensor = torch.concat(chunk_tensors, dim=concat_dim)  # 合并所有分块参数
            if mutate_func is not None:
                full_tensor = mutate_func(full_tensor)  # 可选变换参数
            state_dict[name] = full_tensor  # 保存到最终 state_dict

    def _broadcast_tp_shard_tensor_gate_up(tensor, gate_name, up_name, src_pp_rank) -> torch.Tensor:
        """
        在 mp_group 内广播 gate/up 分块张量，收集所有分片参数并在 rank 0 合并。
        """
        nonlocal state_dict
        nonlocal mp_group
        tp_size = mpu.get_tensor_model_parallel_world_size()
        src_rank = _megatron_calc_global_rank(tp_rank=0, dp_rank=0, pp_rank=src_pp_rank)

        chunk_shape = tensor.shape if torch.distributed.get_rank() == src_rank else None

        obj_list = [chunk_shape]
        dist.broadcast_object_list(obj_list, src=src_rank, group=mp_group)
        chunk_shape = obj_list[0]
        if chunk_shape is None:
            # all or none ranks in the mp_group should reach here
            print_rank_0(f"tp_shard tensor:[{gate_name, up_name}] not exist, skip collecting")
            return

        buffer_tensor = torch.empty(
            chunk_shape,
            dtype=dtype,
            device=get_device_id(),
            requires_grad=False,
        )

        chunk_tensors = [None] * tp_size

        for i in range(tp_size):
            cur_src_rank = _megatron_calc_global_rank(tp_rank=i, dp_rank=0, pp_rank=src_pp_rank)
            sync_tensor = tensor if torch.distributed.get_rank() == cur_src_rank else buffer_tensor
            dist.broadcast(sync_tensor, src=cur_src_rank, group=mp_group)

            if torch.distributed.get_rank() == 0:
                chunk_tensors[i] = _get_cpu_tensor(sync_tensor)

        if torch.distributed.get_rank() == 0:
            full_tensor = torch.concat(chunk_tensors, dim=0)
            intermediate_size_tp = config.intermediate_size // tp_size
            gate_weight_list = []
            up_weight_list = []
            for i in range(tp_size):
                gate_up_weight_tp = full_tensor[intermediate_size_tp * 2 * i : intermediate_size_tp * 2 * (i + 1)]
                gate_weight_tp = gate_up_weight_tp[:intermediate_size_tp]
                up_weight_tp = gate_up_weight_tp[intermediate_size_tp:]
                gate_weight_list.append(gate_weight_tp)
                up_weight_list.append(up_weight_tp)

            state_dict[gate_name] = torch.cat(gate_weight_list, dim=0)
            state_dict[up_name] = torch.cat(up_weight_list, dim=0)

    def _broadcast_tp_shard_tensor_qkv(tensor, q_name, k_name, v_name, src_pp_rank):
        """broadcast tensor in tp shards across mp_group"""
        nonlocal state_dict
        nonlocal mp_group
        tp_size = mpu.get_tensor_model_parallel_world_size()
        src_rank = _megatron_calc_global_rank(tp_rank=0, dp_rank=0, pp_rank=src_pp_rank)

        chunk_shape = tensor.shape if torch.distributed.get_rank() == src_rank else None

        obj_list = [chunk_shape]
        dist.broadcast_object_list(obj_list, src=src_rank, group=mp_group)
        chunk_shape = obj_list[0]
        if chunk_shape is None:
            # all or none ranks in the mp_group should reach here
            print_rank_0(f"tp_shard tensor:[{q_name}] not exist, skip collecting")
            return

        buffer_tensor = torch.empty(
            chunk_shape,
            dtype=dtype,
            device=get_device_id(),
            requires_grad=False,
        )

        chunk_tensors = [None] * tp_size

        for i in range(tp_size):
            cur_src_rank = _megatron_calc_global_rank(tp_rank=i, dp_rank=0, pp_rank=src_pp_rank)
            sync_tensor = tensor if torch.distributed.get_rank() == cur_src_rank else buffer_tensor
            dist.broadcast(sync_tensor, src=cur_src_rank, group=mp_group)

            if torch.distributed.get_rank() == 0:
                chunk_tensors[i] = _get_cpu_tensor(sync_tensor)

        if torch.distributed.get_rank() == 0:
            full_tensor = torch.concat(chunk_tensors, dim=0)
            q_weight_list = []
            k_weight_list = []
            v_weight_list = []
            hidden_size_per_head = config.hidden_size // config.num_attention_heads

            if config.num_key_value_heads >= tp_size:
                q_size_tp = config.hidden_size // tp_size
                kv_size_tp = hidden_size_per_head * config.num_key_value_heads // tp_size
                total_size = q_size_tp + 2 * kv_size_tp
                for i in range(tp_size):
                    qkv_part = full_tensor[i * total_size : (i + 1) * total_size]
                    q_part = qkv_part[:q_size_tp]
                    k_part = qkv_part[q_size_tp : q_size_tp + kv_size_tp]
                    v_part = qkv_part[q_size_tp + kv_size_tp : total_size]
                    q_weight_list.append(q_part)
                    k_weight_list.append(k_part)
                    v_weight_list.append(v_part)
            else:
                q_size_tp = config.hidden_size // tp_size
                kv_size_tp = hidden_size_per_head
                total_size = q_size_tp + 2 * kv_size_tp
                for i in range(tp_size):
                    qkv_part = full_tensor[i * total_size : (i + 1) * total_size]
                    q_part = qkv_part[:q_size_tp]
                    k_part = qkv_part[q_size_tp : q_size_tp + kv_size_tp]
                    v_part = qkv_part[q_size_tp + kv_size_tp : total_size]
                    q_weight_list.append(q_part)
                    if i * config.num_key_value_heads % tp_size == 0:
                        k_weight_list.append(k_part)
                        v_weight_list.append(v_part)

            state_dict[q_name] = torch.cat(q_weight_list, dim=0)
            state_dict[k_name] = torch.cat(k_weight_list, dim=0)
            state_dict[v_name] = torch.cat(v_weight_list, dim=0)

    # empty cache before collecting weights
    get_torch_device().empty_cache()
    # Embeddings
    # -------------------
    if dp_rank == 0:
        # Embeddings
        # -------------------
        print_rank_0("collecting embeddings...")
        gpt_model_module = _get_gpt_model(models[0])
        _broadcast_tp_shard_tensor(
            gpt_model_module.model.embed_tokens.weight if pp_rank == 0 else None,
            "model.embed_tokens.weight",
            src_pp_rank=0,
        )

        # Transformer layers
        # -------------------
        layer_map = _megatron_calc_layer_map(config)
        for layer in range(config.num_hidden_layers):
            print_rank_0(f"collecting layer #{layer}...")
            layer_name = f"model.layers.{layer}"
            src_pp_rank, src_virtual_pp_rank, src_layer_idx = layer_map[layer]

            gpt_model_module = _get_gpt_model(models[src_virtual_pp_rank])
            sync_layer = gpt_model_module.model.layers[src_layer_idx]

            _broadcast_tensor(
                sync_layer.input_layernorm.weight,
                f"{layer_name}.input_layernorm.weight",
                src_pp_rank=src_pp_rank,
            )

            _broadcast_tp_shard_tensor_qkv(
                sync_layer.self_attn.qkv_proj.weight,
                f"{layer_name}.self_attn.q_proj.weight",
                f"{layer_name}.self_attn.k_proj.weight",
                f"{layer_name}.self_attn.v_proj.weight",
                src_pp_rank=src_pp_rank,
            )

            _broadcast_tp_shard_tensor(
                sync_layer.self_attn.o_proj.weight,
                f"{layer_name}.self_attn.o_proj.weight",
                concat_dim=1,
                src_pp_rank=src_pp_rank,
            )

            _broadcast_tensor(
                sync_layer.post_attention_layernorm.weight,
                f"{layer_name}.post_attention_layernorm.weight",
                src_pp_rank=src_pp_rank,
            )

            _broadcast_tp_shard_tensor_gate_up(
                sync_layer.mlp.gate_up_proj.weight,
                f"{layer_name}.mlp.gate_proj.weight",
                f"{layer_name}.mlp.up_proj.weight",
                src_pp_rank=src_pp_rank,
            )

            _broadcast_tp_shard_tensor(
                sync_layer.mlp.down_proj.weight,
                f"{layer_name}.mlp.down_proj.weight",
                concat_dim=1,
                src_pp_rank=src_pp_rank,
            )

        # Final Layernorm
        # -------------------
        print_rank_0("collecting final layernorm...")
        gpt_model_module = _get_gpt_model(models[-1])
        _broadcast_tensor(
            getattr(gpt_model_module.model.norm, "weight", None),
            "model.norm.weight",
            src_pp_rank=pp_size - 1,
        )

        print_rank_0("collecting lm_head...")

        if is_value_model:
            if pp_rank == pp_size - 1:
                print(f"gpt_model_module.lm_head.weight: {gpt_model_module.lm_head.weight.shape}")
            _broadcast_tensor(
                gpt_model_module.lm_head.weight if pp_rank == pp_size - 1 else None,
                "lm_head.weight",
                src_pp_rank=pp_size - 1,
            )
            _broadcast_tensor(
                gpt_model_module.reward_head.weight
                if pp_rank == pp_size - 1 and getattr(gpt_model_module, "reward_weight", None) is not None
                else None,
                "reward_head.weight",
                src_pp_rank=pp_size - 1,
            )

        else:
            _broadcast_tp_shard_tensor(
                getattr(gpt_model_module.lm_head, "weight", None) if pp_rank == pp_size - 1 else None,
                "lm_head.weight",
                src_pp_rank=pp_size - 1,
            )

    dist.barrier()

    get_torch_device().empty_cache()
    if torch.distributed.get_rank() == 0:
        if dtype not in [torch.float16, torch.bfloat16, torch.float32]:
            print(f'Unknown/unsupported dtype to save: {dtype}"')
            exit(1)
        for k, v in state_dict.items():
            if dtype != v.dtype:
                state_dict[k] = v.to(dtype)

    print_rank_0(f"merge megatron ckpt done, time elapsed {time.time() - start_time}s")
    return state_dict
