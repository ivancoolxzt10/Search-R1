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

from verl.utils.device import get_device_id, get_torch_device

# ===================== Megatron 层映射计算 =====================
def _megatron_calc_layer_map(config):
    """
    计算全局层索引到本地层索引的映射关系。
    Megatron 分布式训练中，模型被分为多个流水线分段（pp）、每个分段又可有多个虚拟流水线（virtual pp），每个分段/虚拟分段只负责部分层。
    返回 layer_map: {global_layer_idx: (pp_rank, virtual_pp_rank, local_layer_idx)}
    """
    from megatron.core import mpu

    print(f"get megatron data parallel size: {mpu.get_data_parallel_world_size()}")

    pp_size = mpu.get_pipeline_model_parallel_world_size()  # 流水线并行分段数
    virtual_pp_size = mpu.get_virtual_pipeline_model_parallel_world_size() or 1  # 虚拟流水线分段数
    num_layers_per_model = config.num_hidden_layers // pp_size // virtual_pp_size  # 每个分段/虚拟分段负责的层数
    assert num_layers_per_model * pp_size * virtual_pp_size == config.num_hidden_layers  # 校验总层数

    layer_map = dict()
    for pp_rank_idx in range(pp_size):  # 遍历所有流水线分段
        for virtual_pp_rank_idx in range(virtual_pp_size):  # 遍历所有虚拟流水线分段
            # 计算当前分段的全局层偏移
            layer_offset = (
                virtual_pp_rank_idx * (config.num_hidden_layers // virtual_pp_size) + pp_rank_idx * num_layers_per_model
            )
            for layer_idx in range(num_layers_per_model):  # 遍历本地模型的所有层
                layer_map[layer_offset + layer_idx] = (
                    pp_rank_idx,  # 流水线分段编号
                    virtual_pp_rank_idx,  # 虚拟流水线分段编号
                    layer_idx,  # 本地层索引
                )  # 记录映射关系
    return layer_map  # 返回层映射字典

# ===================== Megatron 权重加载 =====================
def load_state_dict_to_megatron_llama(
    state_dict, wrapped_models, config, params_dtype, is_value_model=False, tie_word_embeddings=False
):
    """
    将合并后的 state_dict 加载到分布式 Megatron 模型中。
    Args:
        state_dict: 合并后的权重字典（通常由 rank0 收集）
        wrapped_models: Megatron 分布式模型的 DDP 包装列表
        config: 模型配置
        params_dtype: 参数数据类型
        is_value_model, tie_word_embeddings: 兼容其它模型接口，llama 不使用
    """
    from megatron.core import DistributedDataParallel as LocalDDP
    from megatron.core import mpu
    from megatron.core.transformer.module import Float16Module
    from torch.nn.parallel import DistributedDataParallel as torchDDP

    from verl.utils.logger import print_rank_0
    from verl.utils.megatron_utils import unwrap_model

    start_time = time.time()

    def _get_gpt_model(model):
        # 兼容不同包装，实际返回底层模型
        return model

    def broadcast_params(module):
        # 广播所有参数，确保各 rank 参数一致
        for param in module.parameters():
            torch.distributed.broadcast(
                param.data, src=mpu.get_data_parallel_src_rank(), group=mpu.get_data_parallel_group()
            )

    dp_rank = mpu.get_data_parallel_rank()  # 数据并行 rank
    pp_rank = mpu.get_pipeline_model_parallel_rank()  # 流水线并行 rank
    pp_size = mpu.get_pipeline_model_parallel_world_size()  # 流水线分段数
    virtual_pp_size = mpu.get_virtual_pipeline_model_parallel_world_size() or 1  # 虚拟流水线分段数
    mp_group = mpu.get_model_parallel_group()  # 模型并行通信组

    # rank0 校验分布式并行配置
    if torch.distributed.get_rank() == 0:
        assert mp_group.rank() == 0, f"mp_rank:[{mp_group.rank}] != 0 on rank #0"
        assert pp_rank == 0, f"pp_rank:[{pp_rank}] != 0 on rank #0"
        assert dp_rank == 0, f"dp_rank:[{dp_rank}] != 0 on rank #0"

    # 保证 wrapped_models 为列表
    if not isinstance(wrapped_models, list | tuple):
        wrapped_models = list(wrapped_models)

    assert len(wrapped_models) == virtual_pp_size  # 每个虚拟分段一个模型
    num_layers_per_model = config.num_hidden_layers // pp_size // virtual_pp_size  # 每个模型负责的层数
    assert num_layers_per_model * pp_size * virtual_pp_size == config.num_hidden_layers, (
        f"num_layers_per_model: {num_layers_per_model} * pp_size: {pp_size} * virtual_pp_size "
        f"{virtual_pp_size} != config.num_hidden_layers: {config.num_hidden_layers}"
    )

    models = [None] * len(wrapped_models)

    # 解包 DDP，获得底层模型
    for i, wrapped_model in enumerate(wrapped_models):
        models[i] = unwrap_model(wrapped_model, (torchDDP, LocalDDP, Float16Module))
        gpt_model_module = _get_gpt_model(models[i])
        assert len(gpt_model_module.model.layers) == num_layers_per_model  # 校验层数

    # ===================== 分布式张量广播辅助函数 =====================
    def _broadcast_tensor(tensor, name) -> torch.Tensor:
        """
        从 rank0 广播张量到所有 mp_group 成员。
        用于分布式权重加载。
        """
        nonlocal state_dict
        nonlocal mp_group
        if torch.distributed.get_rank() == 0:
            if name in state_dict:
                weight = state_dict[name]  # rank0 拿到权重
                tensor_shape = weight.shape
            else:
                tensor_shape = None  # 权重不存在
        else:
            weight = None
            tensor_shape = None

        obj_list = [tensor_shape]
        dist.broadcast_object_list(obj_list, src=0, group=mp_group)  # 广播张量形状
        tensor_shape = obj_list[0]

        if tensor_shape is None:
            # 所有 mp_group 成员都跳过
            print_rank_0(f"tensor:[{name}] not in state_dict, skip load")
            return

        if tensor is None:
            # 非 rank0 创建空张量用于接收广播
            tensor = torch.empty(
                tensor_shape,
                dtype=params_dtype,
                device=get_device_id(),
                requires_grad=False,
            )
        if torch.distributed.get_rank() == 0:
            tensor.data.copy_(weight)  # rank0 拷贝权重
        dist.broadcast(tensor, src=0, group=mp_group)  # 广播权重到所有 mp_group 成员

    def _broadcast_tp_shard_tensor_vocab(tensor, name, chunk_dim=0, mutate_func=None) -> torch.Tensor:
        """broadcast tensor in tp shards across mp_group"""
        nonlocal state_dict
        nonlocal mp_group
        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()

        if torch.distributed.get_rank() == 0:
            if name in state_dict:
                full_weight = state_dict[name]

                if mutate_func is not None:
                    full_weight = mutate_func(full_weight)
                tensor_chunk = torch.chunk(full_weight, tp_size, dim=chunk_dim)
                chunk_shape = tensor_chunk[0].shape
            else:
                chunk_shape = None
        else:
            chunk_shape = None

        obj_list = [chunk_shape]
        dist.broadcast_object_list(obj_list, src=0, group=mp_group)
        chunk_shape = obj_list[0]
        if chunk_shape is None:
            # all or none ranks in the mp_group should reach here
            print_rank_0(f"tp_shard tensor:[{name}] not in state_dict, skip loading")
            return

        if tensor is None:
            sync_tensor = torch.empty(
                chunk_shape,
                dtype=params_dtype,
                device=get_device_id(),
                requires_grad=False,
            )
        else:
            assert tensor.shape == chunk_shape, (
                f"rank #{torch.distributed.get_rank()} tensor {name} shape {tensor.shape} != {chunk_shape}"
            )
            sync_tensor = torch.empty_like(tensor, device=get_device_id(), requires_grad=False)

        for i in range(tp_size):
            if torch.distributed.get_rank() == 0:
                sync_tensor.data.copy_(tensor_chunk[i])
            dist.broadcast(sync_tensor, src=0, group=mp_group)
            if (i == tp_rank) and (tensor is not None):
                tensor.data.copy_(sync_tensor)

    def _broadcast_tp_shard_tensor(tensor, name, chunk_dim=0, mutate_func=None) -> torch.Tensor:
        """broadcast tensor in tp shards across mp_group"""
        nonlocal state_dict
        nonlocal mp_group
        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()

        if torch.distributed.get_rank() == 0:
            if name in state_dict:
                full_weight = state_dict[name]
                if mutate_func is not None:
                    full_weight = mutate_func(full_weight)
                tensor_chunk = torch.chunk(full_weight, tp_size, dim=chunk_dim)
                chunk_shape = tensor_chunk[0].shape
            else:
                chunk_shape = None
        else:
            chunk_shape = None

        obj_list = [chunk_shape]
        dist.broadcast_object_list(obj_list, src=0, group=mp_group)
        chunk_shape = obj_list[0]
        if chunk_shape is None:
            # all or none ranks in the mp_group should reach here
            print_rank_0(f"tp_shard tensor:[{name}] not in state_dict, skip loading")
            return

        if tensor is None:
            sync_tensor = torch.empty(
                chunk_shape,
                dtype=params_dtype,
                device=get_device_id(),
                requires_grad=False,
            )
        else:
            assert tensor.shape == chunk_shape, (
                f"rank #{torch.distributed.get_rank()} tensor {name} shape {tensor.shape} != {chunk_shape}"
            )
            sync_tensor = torch.empty_like(tensor, device=get_device_id(), requires_grad=False)

        for i in range(tp_size):
            if torch.distributed.get_rank() == 0:
                sync_tensor.data.copy_(tensor_chunk[i])
            dist.broadcast(sync_tensor, src=0, group=mp_group)
            if (i == tp_rank) and (tensor is not None):
                tensor.data.copy_(sync_tensor)

    def _broadcast_tp_shard_tensor_gate_up(tensor, gate_name, up_name) -> torch.Tensor:
        """broadcast tensor in tp shards across mp_group"""
        nonlocal state_dict
        nonlocal mp_group
        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()

        if torch.distributed.get_rank() == 0:
            gate_weight = state_dict[gate_name]
            up_weight = state_dict[up_name]
            new_gate_up_weight = torch.empty(
                config.intermediate_size * 2, config.hidden_size, dtype=params_dtype, device=get_device_id()
            )
            for i in range(tp_size):
                intermediate_size_tp = config.intermediate_size // tp_size
                gate_weight_tp = gate_weight[i * intermediate_size_tp : (i + 1) * intermediate_size_tp]
                up_weight_tp = up_weight[i * intermediate_size_tp : (i + 1) * intermediate_size_tp]
                new_gate_up_weight[intermediate_size_tp * 2 * i : intermediate_size_tp * 2 * (i + 1)].copy_(
                    torch.cat([gate_weight_tp, up_weight_tp], dim=0)
                )

            tensor_chunk = torch.chunk(new_gate_up_weight, tp_size, dim=0)
            chunk_shape = tensor_chunk[0].shape
        else:
            chunk_shape = None

        obj_list = [chunk_shape]
        dist.broadcast_object_list(obj_list, src=0, group=mp_group)
        chunk_shape = obj_list[0]
        if chunk_shape is None:
            # all or none ranks in the mp_group should reach here
            print_rank_0(f"tp_shard tensor:[{gate_name, up_name}] not in state_dict, skip loading")
            return

        if tensor is None:
            sync_tensor = torch.empty(
                chunk_shape,
                dtype=params_dtype,
                device=get_device_id(),
                requires_grad=False,
            )
        else:
            assert tensor.shape == chunk_shape, (
                f"rank #{torch.distributed.get_rank() == 0:} tensor {gate_name, up_name} shape "
                f"{tensor.shape} != {chunk_shape}"
            )
            sync_tensor = torch.empty_like(tensor, device=get_device_id(), requires_grad=False)

        for i in range(tp_size):
            if torch.distributed.get_rank() == 0:
                sync_tensor.data.copy_(tensor_chunk[i])
            dist.broadcast(sync_tensor, src=0, group=mp_group)
            if (i == tp_rank) and (tensor is not None):
                tensor.data.copy_(sync_tensor)

    def _broadcast_tp_shard_tensor_qkv(tensor, q_name, k_name, v_name) -> torch.Tensor:
        """broadcast tensor in tp shards across mp_group"""
        nonlocal state_dict
        nonlocal mp_group
        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()

        if torch.distributed.get_rank() == 0:
            assert q_name in state_dict and k_name in state_dict and v_name in state_dict
            full_weight_q = state_dict[q_name]
            full_weight_k = state_dict[k_name]
            full_weight_v = state_dict[v_name]

            hidden_size_per_head = config.hidden_size // config.num_attention_heads

            if config.num_key_value_heads >= tp_size:
                q_size_tp = config.hidden_size // tp_size
                kv_size_tp = hidden_size_per_head * config.num_key_value_heads // tp_size
                total_size = q_size_tp + 2 * kv_size_tp
                new_weight_qkv = torch.empty(
                    total_size * tp_size, config.hidden_size, dtype=params_dtype, device=get_device_id()
                )
                for i in range(tp_size):
                    q_part = full_weight_q[i * q_size_tp : (i + 1) * q_size_tp]
                    k_part = full_weight_k[i * kv_size_tp : (i + 1) * kv_size_tp]
                    v_part = full_weight_v[i * kv_size_tp : (i + 1) * kv_size_tp]
                    new_weight_qkv[i * total_size : (i + 1) * total_size].copy_(
                        torch.cat([q_part, k_part, v_part], dim=0)
                    )

            else:
                q_size_tp = config.hidden_size // tp_size
                kv_size_tp = hidden_size_per_head
                total_size = q_size_tp + 2 * kv_size_tp
                new_weight_qkv = torch.empty(
                    total_size * tp_size, config.hidden_size, dtype=params_dtype, device=get_device_id()
                )
                for i in range(tp_size):
                    q_part = full_weight_q[i * q_size_tp : (i + 1) * q_size_tp]
                    start_idx = i * config.num_key_value_heads // tp_size * hidden_size_per_head
                    end_idx = (i * config.num_key_value_heads // tp_size + 1) * hidden_size_per_head
                    k_part = full_weight_k[start_idx:end_idx]
                    v_part = full_weight_v[start_idx:end_idx]
                    new_weight_qkv[i * total_size : (i + 1) * total_size].copy_(
                        torch.cat([q_part, k_part, v_part], dim=0)
                    )

            tensor_chunk = torch.chunk(new_weight_qkv, tp_size, dim=0)
            chunk_shape = tensor_chunk[0].shape
        else:
            chunk_shape = None

        obj_list = [chunk_shape]
        dist.broadcast_object_list(obj_list, src=0, group=mp_group)
        chunk_shape = obj_list[0]
        if chunk_shape is None:
            # all or none ranks in the mp_group should reach here
            print_rank_0(f"tp_shard tensor:[{q_name, k_name, v_name}] not in state_dict, skip loading")
            return

        if tensor is None:
            sync_tensor = torch.empty(
                chunk_shape,
                dtype=params_dtype,
                device=get_device_id(),
                requires_grad=False,
            )
        else:
            assert tensor.shape == chunk_shape, (
                f"rank #{torch.distributed.get_rank()} tensor {q_name} shape {tensor.shape} != {chunk_shape}"
            )
            sync_tensor = torch.empty_like(tensor, device=get_device_id(), requires_grad=False)

        for i in range(tp_size):
            if torch.distributed.get_rank() == 0:
                sync_tensor.data.copy_(tensor_chunk[i])
            dist.broadcast(sync_tensor, src=0, group=mp_group)
            if (i == tp_rank) and (tensor is not None):
                tensor.data.copy_(sync_tensor)

    if dp_rank == 0:
        # Embeddings
        # -------------------
        print_rank_0("loading embeddings...")
        gpt_model_module = _get_gpt_model(models[0])
        embed_tokens_weight = None
        if pp_rank == 0:
            embed_tokens_weight = gpt_model_module.model.embed_tokens.weight
        _broadcast_tp_shard_tensor_vocab(embed_tokens_weight, "model.embed_tokens.weight")

        # Transformer layers
        # -------------------
        layer_map = _megatron_calc_layer_map(config)

        for layer in range(config.num_hidden_layers):
            print_rank_0(f"loading layer #{layer}...")
            layer_name = f"model.layers.{layer}"
            dst_pp_rank, dst_virtual_pp_rank, dst_layer_idx = layer_map[layer]

            gpt_model_module = _get_gpt_model(models[dst_virtual_pp_rank])
            sync_layer = gpt_model_module.model.layers[dst_layer_idx]

            _broadcast_tensor(
                sync_layer.input_layernorm.weight if dst_pp_rank == pp_rank else None,
                f"{layer_name}.input_layernorm.weight",
            )

            _broadcast_tp_shard_tensor_qkv(
                sync_layer.self_attn.qkv_proj.weight if dst_pp_rank == pp_rank else None,
                f"{layer_name}.self_attn.q_proj.weight",
                f"{layer_name}.self_attn.k_proj.weight",
                f"{layer_name}.self_attn.v_proj.weight",
            )

            _broadcast_tp_shard_tensor(
                sync_layer.self_attn.o_proj.weight if dst_pp_rank == pp_rank else None,
                f"{layer_name}.self_attn.o_proj.weight",
                chunk_dim=1,
            )

            _broadcast_tensor(
                sync_layer.post_attention_layernorm.weight if dst_pp_rank == pp_rank else None,
                f"{layer_name}.post_attention_layernorm.weight",
            )

            _broadcast_tp_shard_tensor_gate_up(
                sync_layer.mlp.gate_up_proj.weight if dst_pp_rank == pp_rank else None,
                f"{layer_name}.mlp.gate_proj.weight",
                f"{layer_name}.mlp.up_proj.weight",
            )

            _broadcast_tp_shard_tensor(
                sync_layer.mlp.down_proj.weight if dst_pp_rank == pp_rank else None,
                f"{layer_name}.mlp.down_proj.weight",
                chunk_dim=1,
            )
        # Final Layernorm
        # -------------------
        print_rank_0("loading final layernorm...")
        gpt_model_module = _get_gpt_model(models[-1])
        _broadcast_tensor(
            getattr(gpt_model_module.model.norm, "weight", None),
            "model.norm.weight",
        )

        print_rank_0("loading lm_head...")
        lm_head_weight = None
        if pp_rank + 1 == pp_size:
            lm_head_weight = gpt_model_module.lm_head.weight

        if is_value_model:
            if "lm_head.weight" in state_dict and state_dict["lm_head.weight"].shape[0] == 1:
                _broadcast_tensor(lm_head_weight, "lm_head.weight")
                print_rank_0("load lm_head weight")
            elif "reward_head.weight" in state_dict and state_dict["reward_head.weight"].shape[0] == 1:
                _broadcast_tensor(lm_head_weight, "reward_head.weight")
                print_rank_0("load lm_head from value_head weight")
            else:
                _broadcast_tensor(None, "lm_head.weight")
                print_rank_0("fail to match lm_head in value_model")
        else:
            _broadcast_tp_shard_tensor(lm_head_weight, "lm_head.weight")
    dist.barrier()
    # Broadcast weights inside data parallel groups
    for wrapped_model in wrapped_models:
        broadcast_params(wrapped_model)

    get_torch_device().empty_cache()
    print_rank_0(f"loading megatron ckpt done, time elapsed {time.time() - start_time}s")
