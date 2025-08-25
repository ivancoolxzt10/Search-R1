# 初学者备注：
# 本文件用于加载 Megatron 分布式模型的权重，包含 layer 映射等辅助函数。
# 适用于多卡推理场景，主要面向开发者和分布式系统初学者。
# 典型用法：调用 _megatron_calc_layer_map(config) 获取全局到本地层的映射。
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

import time  # 导入 time 库，用于记录和统计操作耗时

import torch  # 导入 torch 库，PyTorch 是主流深度学习框架
import torch.distributed as dist  # 导入分布式训练相关模块

from verl.utils.device import get_device_id, get_torch_device  # 导入设备相关工具函数


def _megatron_calc_layer_map(config):
    """
    初学者备注：
    该函数用于计算全局层索引到本地层索引的映射，适用于 Megatron 分布式模型。
    输入参数 config 通常为模型配置对象，包含层数等信息。
    返回 layer_map 字典，key 为全局层索引，value 为 (pp_rank, virtual_pp_rank, 层索引)。
    """
    from megatron.core import mpu  # 导入 Megatron 并行工具模块

    print(f"get megatron data parallel size: {mpu.get_data_parallel_world_size()}")  # 打印数据并行规模，便于调试

    pp_size = mpu.get_pipeline_model_parallel_world_size()  # 获取流水线并行总数
    virtual_pp_size = mpu.get_virtual_pipeline_model_parallel_world_size() or 1  # 获取虚拟流水线并行数，默认为1

    layer_map = dict()  # 初始化一个空字典，用于存储全局层索引到本地层索引的映射关系
    num_layers_per_model = config.num_hidden_layers // pp_size // virtual_pp_size  # 计算每个模型分到的层数，分布式推理时每个 rank 只负责部分层
    # 检查层数分配是否正确，确保所有层都被分配且无遗漏
    assert num_layers_per_model * pp_size * virtual_pp_size == config.num_hidden_layers, (
        f"num_layers_per_model: {num_layers_per_model} * pp_size: {pp_size} * virtual_pp_size "
        f"{virtual_pp_size} != config.num_hidden_layers: {config.num_hidden_layers}"
    )

    for pp_rank_idx in range(pp_size):  # 遍历所有流水线 rank（每个 rank 负责一部分层）
        for virtual_pp_rank_idx in range(virtual_pp_size):  # 遍历所有虚拟流水线 rank（用于进一步细分层分配）
            layer_offset = (
                virtual_pp_rank_idx * (config.num_hidden_layers // virtual_pp_size) + pp_rank_idx * num_layers_per_model
            )  # 计算当前 rank 的层偏移量，确定本地层的起始索引
            for layer_idx in range(num_layers_per_model):  # 遍历本地模型的所有层
                layer_map[layer_offset + layer_idx] = (
                    pp_rank_idx,  # 流水线 rank，决定该层属于哪个流水线分组
                    virtual_pp_rank_idx,  # 虚拟流水线 rank，进一步细分分组
                    layer_idx,  # 本地层索引，当前 rank 下的实际层编号
                )  # 将映射关系加入字典，方便后续查找
    return layer_map  # 返回层映射字典，供分布式加载和参数同步使用


def load_state_dict_to_megatron_llama(
    state_dict, wrapped_models, config, params_dtype, is_value_model=False, tie_word_embeddings=False
):
    """Load merged state_dict to sharded Megatron module in training."""
    from megatron.core import DistributedDataParallel as LocalDDP
    from megatron.core import mpu
    from megatron.core.transformer.module import Float16Module
    from torch.nn.parallel import DistributedDataParallel as torchDDP

    from verl.utils.logger import print_rank_0
    from verl.utils.megatron_utils import unwrap_model

    start_time = time.time()  # 记录加载开始时间，便于后续统计耗时

    def _get_gpt_model(model):
        return model  # 返回模型本身，便于后续统一处理

    def fetch_params(module):
        # 遍历模块的所有参数
        for param in module.parameters():
            # 从分布式源 rank 拉取参数数据，保证各进程参数一致
            torch.distributed.fetch(
                param.data, src=mpu.get_data_parallel_src_rank(), group=mpu.get_data_parallel_group()
            )

    dp_rank = mpu.get_data_parallel_rank()  # 获取当前进程的数据并行 rank
    pp_rank = mpu.get_pipeline_model_parallel_rank()  # 获取当前进程的流水线并行 rank
    pp_size = mpu.get_pipeline_model_parallel_world_size()  # 获取流水线并行总数
    virtual_pp_size = mpu.get_virtual_pipeline_model_parallel_world_size() or 1  # 虚拟流水线并行数，默认为1
    mp_group = mpu.get_model_parallel_group()  # 获取模型并行通信组

    # 仅在主进程（rank 0）做断言检查，确保并行配置正确
    if torch.distributed.get_rank() == 0:
        assert mp_group.rank() == 0, f"mp_rank:[{mp_group.rank}] != 0 on rank #0"
        assert pp_rank == 0, f"pp_rank:[{pp_rank}] != 0 on rank #0"
        assert dp_rank == 0, f"dp_rank:[{dp_rank}] != 0 on rank #0"

    # 如果 wrapped_models 不是列表或元组，则转为列表，方便后续处理
    if not isinstance(wrapped_models, list | tuple):
        wrapped_models = list(wrapped_models)

    # 检查模型数量与虚拟流水线并行数一致
    assert len(wrapped_models) == virtual_pp_size
    # 计算每个模型包含的层数
    num_layers_per_model = config.num_hidden_layers // pp_size // virtual_pp_size
    # 检查层数分配是否正确
    assert num_layers_per_model * pp_size * virtual_pp_size == config.num_hidden_layers, (
        f"num_layers_per_model: {num_layers_per_model} * pp_size: {pp_size} * virtual_pp_size "
        f"{virtual_pp_size} != config.num_hidden_layers: {config.num_hidden_layers}"
    )

    models = [None] * len(wrapped_models)  # 初始化模型列表，长度等于虚拟流水线并行数

    # 遍历所有 wrapped_models，解包分布式模型并检查每个模型的层数
    for i, wrapped_model in enumerate(wrapped_models):
        models[i] = unwrap_model(wrapped_model, (torchDDP, LocalDDP, Float16Module))  # 解包分布式模型，获得实际模型对象
        gpt_model_module = _get_gpt_model(models[i])  # 获取 GPT 模型模块，便于后续统一处理
        assert len(gpt_model_module.model.layers) == num_layers_per_model  # 检查每个模型的层数是否正确

    def _fetch_tensor(tensor, name) -> torch.Tensor:
        """fetch tensor"""
        nonlocal state_dict  # 使用外部的 state_dict 字典，存储所有参数
        if tensor is not None:
            tensor.data.copy_(state_dict[name])  # 将 state_dict 中的参数复制到当前张量，实现参数同步

    def _fetch_tp_shard_tensor_vocab(tensor, name, chunk_dim=0, mutate_func=None) -> torch.Tensor:
        """fetch tensor in tp shards"""
        nonlocal state_dict  # 使用外部的 state_dict 字典
        tp_rank = mpu.get_tensor_model_parallel_rank()  # 获取当前进程的张量并行 rank
        tp_size = mpu.get_tensor_model_parallel_world_size()  # 获取张量并行总数
        if name in state_dict:
            full_weight = state_dict[name]  # 获取完整参数张量

            if mutate_func is not None:
                full_weight = mutate_func(full_weight)  # 如果有变换函数则先处理参数
            tensor_chunk = torch.chunk(full_weight, tp_size, dim=chunk_dim)  # 按张量并行数分块，分配到各 rank
            if tensor is not None:
                tensor.data.copy_(tensor_chunk[tp_rank])  # 复制当前 rank 的分块参数到本地张量
        else:
            print(f"tp_shard tensor:[{name}] not in state_dict, skip loading")  # 未找到参数则跳过，提示信息

    def _fetch_tp_shard_tensor(tensor, name, chunk_dim=0, mutate_func=None) -> torch.Tensor:
        """fetch tensor in tp shards"""
        nonlocal state_dict  # 使用外部的 state_dict 字典
        tp_rank = mpu.get_tensor_model_parallel_rank()  # 获取当前进程的张量并行 rank
        tp_size = mpu.get_tensor_model_parallel_world_size()  # 获取张量并行总数
        if name in state_dict:
            full_weight = state_dict[name]  # 获取完整参数张量

            if mutate_func is not None:
                full_weight = mutate_func(full_weight)  # 如果有变换函数则先处理参数
            tensor_chunk = torch.chunk(full_weight, tp_size, dim=chunk_dim)  # 按张量并行数分块，分配到各 rank
            if tensor is not None:
                tensor.data.copy_(tensor_chunk[tp_rank])  # 复制当前 rank 的分块参数到本地张量
        else:
            print(f"tp_shard tensor:[{name}] not in state_dict, skip loading")  # 未找到参数则跳过，提示信息

    def _fetch_tp_shard_tensor_gate_up(tensor, gate_name, up_name) -> torch.Tensor:
        """fetch gate_up tensor in tp shards"""
        nonlocal state_dict  # 使用外部的 state_dict 字典
        nonlocal mp_group  # 使用外部的模型并行通信组
        tp_rank = mpu.get_tensor_model_parallel_rank()  # 获取当前进程的张量并行 rank
        tp_size = mpu.get_tensor_model_parallel_world_size()  # 获取张量并行总数
        if gate_name in state_dict and up_name in state_dict:
            gate_weight = state_dict[gate_name]  # 获取 gate 权重
            up_weight = state_dict[up_name]  # 获取 up 权重
            new_gate_up_weight = torch.empty(
                config.intermediate_size * 2, config.hidden_size, dtype=params_dtype, device=get_device_id()
            )  # 新建一个空张量用于合并 gate 和 up 权重
            for i in range(tp_size):  # 遍历所有张量并行 rank
                intermediate_size_tp = config.intermediate_size // tp_size  # 计算每个 rank 的中间层大小
                gate_weight_tp = gate_weight[i * intermediate_size_tp : (i + 1) * intermediate_size_tp]  # 当前 rank 的 gate 权重
                up_weight_tp = up_weight[i * intermediate_size_tp : (i + 1) * intermediate_size_tp]  # 当前 rank 的 up 权重
                new_gate_up_weight[intermediate_size_tp * 2 * i : intermediate_size_tp * 2 * (i + 1)].copy_(
                    torch.cat([gate_weight_tp, up_weight_tp], dim=0)  # 合并 gate 和 up 权重
                )

            tensor_chunk = torch.chunk(new_gate_up_weight, tp_size, dim=0)  # 按张量并行数分块
            if tensor is not None:
                tensor.data.copy_(tensor_chunk[tp_rank])  # 复制当前 rank 的分块参数到本地张量
        else:
            print(f"tp_shard tensor:[{gate_name}, {up_name}] not in state_dict, skip loading")  # 未找到参数则跳过

    def _fetch_tp_shard_tensor_qkv(tensor, q_name, k_name, v_name) -> torch.Tensor:
        """fetch tensor in tp shards across mp_group"""
        nonlocal state_dict  # 使用外部的 state_dict 字典
        nonlocal mp_group  # 使用外部的模型并行通信组
        tp_rank = mpu.get_tensor_model_parallel_rank()  # 获取当前进程的张量并行 rank
        tp_size = mpu.get_tensor_model_parallel_world_size()  # 获取张量并行总数
        assert q_name in state_dict and k_name in state_dict and v_name in state_dict  # 检查所有参数是否存在
        full_weight_q = state_dict[q_name]  # 获取 Q 权重
        full_weight_k = state_dict[k_name]  # 获取 K 权重
        full_weight_v = state_dict[v_name]  # 获取 V 权重

        hidden_size_per_head = config.hidden_size // config.num_attention_heads  # 计算每个注意力头的隐藏层大小

        if config.num_key_value_heads >= tp_size:  # 如果 key/value 头数大于等于张量并行数，采用标准分块
            q_size_tp = config.hidden_size // tp_size  # 每个 rank 的 Q 权重大小
            kv_size_tp = hidden_size_per_head * config.num_key_value_heads // tp_size  # 每个 rank 的 K/V 权重大小
            total_size = q_size_tp + 2 * kv_size_tp  # 总分块大小
            new_weight_qkv = torch.empty(
                total_size * tp_size, config.hidden_size, dtype=params_dtype, device=get_device_id()
            )  # 新建空张量用于合并 QKV 权重
            for i in range(tp_size):  # 遍历所有张量并行 rank
                q_part = full_weight_q[i * q_size_tp : (i + 1) * q_size_tp]  # 当前 rank 的 Q 权重
                k_part = full_weight_k[i * kv_size_tp : (i + 1) * kv_size_tp]  # 当前 rank 的 K 权重
                v_part = full_weight_v[i * kv_size_tp : (i + 1) * kv_size_tp]  # 当前 rank 的 V 权重
                new_weight_qkv[i * total_size : (i + 1) * total_size].copy_(torch.cat([q_part, k_part, v_part], dim=0))  # 合并 QKV 权重

        else:  # 如果 key/value 头数小于张量并行数，采用特殊分块
            q_size_tp = config.hidden_size // tp_size  # 每个 rank 的 Q 权重大小
            kv_size_tp = hidden_size_per_head  # 每个 rank 的 K/V 权重大小
            total_size = q_size_tp + 2 * kv_size_tp  # 总分块大小
            new_weight_qkv = torch.empty(
                total_size * tp_size, config.hidden_size, dtype=params_dtype, device=get_device_id()
            )  # 新建空张量用于合并 QKV 权重
            for i in range(tp_size):  # 遍历所有张量并行 rank
                q_part = full_weight_q[i * q_size_tp : (i + 1) * q_size_tp]  # 当前 rank 的 Q 权重
                start_idx = i * config.num_key_value_heads // tp_size * hidden_size_per_head  # K/V 权重起始索引
                end_idx = (i * config.num_key_value_heads // tp_size + 1) * hidden_size_per_head  # K/V 权重结束索引
                k_part = full_weight_k[start_idx:end_idx]  # 当前 rank 的 K 权重
                v_part = full_weight_v[start_idx:end_idx]  # 当前 rank 的 V 权重
                new_weight_qkv[i * total_size : (i + 1) * total_size].copy_(torch.cat([q_part, k_part, v_part], dim=0))  # 合并 QKV 权重

        tensor_chunk = torch.chunk(new_weight_qkv, tp_size, dim=0)  # 按张量并行数分块
        if tensor is not None:
            tensor.data.copy_(tensor_chunk[tp_rank])  # 复制当前 rank 的分块参数到本地张量

    # Embeddings
    # -------------------
    print_rank_0("loading embeddings...")
    gpt_model_module = _get_gpt_model(models[0])
    embed_tokens_weight = None
    if pp_rank == 0:
        embed_tokens_weight = gpt_model_module.model.embed_tokens.weight
    _fetch_tp_shard_tensor_vocab(embed_tokens_weight, "model.embed_tokens.weight")

    # Transformer layers
    # -------------------
    layer_map = _megatron_calc_layer_map(config)

    pp_rank = mpu.get_pipeline_model_parallel_rank()
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    num_layer_per_pp = config.num_hidden_layers // pp_size
    vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()

    layer_list = []
    if vpp_size is not None:
        for vpp_rank in range(vpp_size):
            num_layer_vpp_chunk = num_layer_per_pp // vpp_size
            num_layer_this_model = num_layer_vpp_chunk
            offset = vpp_rank * (config.num_hidden_layers // mpu.get_virtual_pipeline_model_parallel_world_size()) + (
                mpu.get_pipeline_model_parallel_rank() * num_layer_vpp_chunk
            )
            layer_list.extend(list(range(offset, offset + num_layer_this_model)))
    else:
        num_layer_this_model = num_layer_per_pp
        offset = pp_rank * num_layer_per_pp
        layer_list.extend(list(range(offset, offset + num_layer_this_model)))

    for layer in layer_list:
        print_rank_0(f"loading layer #{layer}...")
        layer_name = f"model.layers.{layer}"
        dst_pp_rank, dst_virtual_pp_rank, dst_layer_idx = layer_map[layer]

        gpt_model_module = _get_gpt_model(models[dst_virtual_pp_rank])
        sync_layer = gpt_model_module.model.layers[dst_layer_idx]

        _fetch_tensor(
            sync_layer.input_layernorm.weight if dst_pp_rank == pp_rank else None,
            f"{layer_name}.input_layernorm.weight",
        )

        _fetch_tp_shard_tensor_qkv(
            sync_layer.self_attn.qkv_proj.weight if dst_pp_rank == pp_rank else None,
            f"{layer_name}.self_attn.q_proj.weight",
            f"{layer_name}.self_attn.k_proj.weight",
            f"{layer_name}.self_attn.v_proj.weight",
        )

        _fetch_tp_shard_tensor(
            sync_layer.self_attn.o_proj.weight if dst_pp_rank == pp_rank else None,  # 只在目标 pp_rank 上加载 o_proj 权重，o_proj 是注意力输出投影层
            f"{layer_name}.self_attn.o_proj.weight",  # 指定参数名，便于同步和调试
            chunk_dim=1,  # 按维度1分块，适应张量并行，保证每个 rank 只加载自己负责的部分
        )

        _fetch_tensor(
            sync_layer.post_attention_layernorm.weight if dst_pp_rank == pp_rank else None,  # 只在目标 pp_rank 上加载后注意力层归一化权重
            f"{layer_name}.post_attention_layernorm.weight",  # 参数名
        )

        _fetch_tp_shard_tensor_gate_up(
            sync_layer.mlp.gate_up_proj.weight if dst_pp_rank == pp_rank else None,  # 只在目标 pp_rank 上加载 MLP 的 gate_up 权重
            f"{layer_name}.mlp.gate_proj.weight",  # gate 权重参数名
            f"{layer_name}.mlp.up_proj.weight",  # up 权重参数名
        )

        _fetch_tp_shard_tensor(
            sync_layer.mlp.down_proj.weight if dst_pp_rank == pp_rank else None,  # 只在目标 pp_rank 上加载 down_proj 权重
            f"{layer_name}.mlp.down_proj.weight",  # 指定参数名，便于同步
            chunk_dim=1,  # 按维度1分块，适应张量并行
        )
    # Final Layernorm
    # -------------------
    print_rank_0("loading final layernorm...")  # 打印提示，开始加载最终层归一化参数
    gpt_model_module = _get_gpt_model(models[-1])  # 获取最后一个模型（通常是最后一个虚拟流水线分组）
    _fetch_tensor(
        getattr(gpt_model_module.model.norm, "weight", None),  # 获取最终层归一化权重
        "model.norm.weight",  # 参数名
    )

    print_rank_0("loading lm_head...")  # 打印提示，开始加载输出头参数
    if pp_rank + 1 == pp_size:  # 仅在最后一个 pp_rank 上加载 lm_head，保证输出权重只加载一次
        lm_head_weight = gpt_model_module.lm_head.weight  # 获取输出头权重

        if is_value_model:  # 如果是 value model，需特殊处理
            if "lm_head.weight" in state_dict and state_dict["lm_head.weight"].shape[0] == 1:
                _fetch_tensor(lm_head_weight, "lm_head.weight")  # 加载 lm_head 权重
                print_rank_0("load lm_head weight")  # 打印提示
            elif "reward_head.weight" in state_dict and state_dict["reward_head.weight"].shape[0] == 1:
                _fetch_tensor(lm_head_weight, "reward_head.weight")  # 用 value_head 权重加载 lm_head
                print_rank_0("load lm_head from value_head weight")  # 打印提示
            else:
                _fetch_tensor(None, "lm_head.weight")  # 未找到权重则跳过
                print_rank_0("fail to match lm_head in value_model")  # 打印失败提示
        else:
            _fetch_tp_shard_tensor(lm_head_weight, "lm_head.weight")  # 普通模型直接分布式加载 lm_head 权重

    dist.barrier()  # 分布式同步，确保所有进程都完成参数加载
    get_torch_device().empty_cache()  # 清理显存缓存，释放资源，防止内存泄漏
    print_rank_0(f"loading megatron ckpt done, time elapsed {time.time() - start_time}s")  # 打印加载完成和耗时信息
