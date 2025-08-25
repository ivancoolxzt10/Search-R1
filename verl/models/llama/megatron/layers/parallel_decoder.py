# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

from typing import Optional

import torch
from megatron.core import ModelParallelConfig
from torch import nn
from transformers import LlamaConfig

from verl.utils.megatron_utils import TransformerConfig, convert_config

from .parallel_attention import ParallelLlamaAttention, ParallelLlamaAttentionRmPad
from .parallel_mlp import ParallelLlamaMLP
from .parallel_rmsnorm import ParallelLlamaRMSNorm


class ParallelLlamaDecoderLayer(nn.Module):
    """ParallelLlamaDecoderLayer: 这是标准的解码器层实现，
    它处理的是**带填充（Padded）**的数据。这意味着，
    如果一个批次里的句子长短不一，短句子会被填充到和最长的句子一样长。这个版本结构清晰，易于理解。

    标准的并行化 Llama 解码器层。
    原理：这个类将并行的 Attention、MLP 和 RMSNorm 组件组合在一起，
    构建了一个完整的 Transformer 解码器层。它遵循标准的 Pre-Norm 结构，
    即在进入 Attention 和 MLP 之前先进行层归一化。
    """

    def __init__(self, config: LlamaConfig, megatron_config: ModelParallelConfig, layer_idx: int):
        """
        初始化函数。
        :param config: HuggingFace 的 LlamaConfig 对象，包含了模型的所有超参数。
        :param megatron_config: Megatron-LM 的并行配置对象。
        :param layer_idx: 当前是第几层，主要用于调试和记录。
        """
        super().__init__()
        # 将 HuggingFace 的配置和 Megatron 的配置合并成一个内部统一的配置格式
        self.config: TransformerConfig = convert_config(config, megatron_config)
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        # 实例化并行的自注意力模块
        self.self_attn = ParallelLlamaAttention(config=config, megatron_config=megatron_config)

        # 实例化并行的 MLP (前馈网络) 模块
        self.mlp = ParallelLlamaMLP(config, megatron_config=megatron_config)

        # 实例化输入层归一化模块 (在 Attention 之前)
        self.input_layernorm = ParallelLlamaRMSNorm(config, megatron_config)

        # 实例化注意力后层归一化模块 (在 MLP 之前)
        self.post_attention_layernorm = ParallelLlamaRMSNorm(config, megatron_config)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        前向传播函数。
        :param hidden_states: 输入张量，形状为 `(batch, seq_len, embed_dim)`。
        :param attention_mask: 注意力掩码，用于防止 token 看到未来的或填充的 token。
        :param position_ids: 位置 ID，用于 RoPE 计算。
        """

        # 1. 第一个残差连接：保存原始输入
        residual = hidden_states

        # 2. 输入层归一化 (Pre-Norm)
        hidden_states = self.input_layernorm(hidden_states)

        # 关键点：序列并行 (Sequence Parallelism) 的操作被隐藏在并行层内部。
        # 对于标准的张量并行：
        # - ColumnParallelLinear (Attention 和 MLP 的第一层) 的输入是完整的，输出是切分的。
        # - RowParallelLinear (Attention 和 MLP 的第二层) 的输入是切分的，它会在内部执行 All-Reduce 操作，输出是完整的。

        # 3. 自注意力模块
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        # TODO 区域提示：如果启用了序列并行，这里需要一个 reduce_scatter 操作。
        # 原理：序列并行会在序列维度上切分数据。在 Attention 之后，每个 GPU 只有部分序列的结果，
        # 需要通过 reduce_scatter 将不同 GPU 上的结果相加并分发，才能进行后续计算。
        # 不过，在 Megatron 的现代实现中，这个操作通常也被封装在 RowParallelLinear 内部了。

        # 4. 第一个残差加法
        hidden_states = residual + hidden_states

        # 5. 第二个残差连接：保存 Attention 后的结果
        residual = hidden_states

        # 6. 注意力后层归一化 (Pre-Norm for MLP)
        hidden_states = self.post_attention_layernorm(hidden_states)

        # TODO 区域提示：如果启用了序列并行，这里需要一个 all_gather 操作。
        # 原理：MLP 的计算需要完整的 hidden_states，所以在序列并行下，需要先用 all_gather
        # 从所有 GPU 收集完整的 hidden_states，计算完 MLP 后再用 reduce_scatter 切分回去。
        # 同样，这个操作通常也封装在 ColumnParallelLinear 内部。

        # 7. MLP/前馈网络模块
        hidden_states = self.mlp(hidden_states)

        # TODO 区域提示：MLP 之后也需要一个 reduce_scatter。

        # 8. 第二个残差加法
        hidden_states = residual + hidden_states

        outputs = hidden_states

        return outputs


class ParallelLlamaDecoderLayerRmPad(nn.Module):
    """
    使用 FlashAttention 和去填充（RmPad）技术优化的并行解码器层。
    原理：这个类的结构与标准版完全相同，但它实例化的 Attention 模块是 `ParallelLlamaAttentionRmPad`，
    并且它的 `forward` 函数接收的是去填充后的数据和相应的元数据（如 `cu_seqlens`）。
    所有计算都在一个压平的、密集的 token 序列上进行，效率极高。
    """

    def __init__(self, config: LlamaConfig, megatron_config: ModelParallelConfig, layer_idx: int):
        super().__init__()
        self.config: TransformerConfig = convert_config(config, megatron_config)
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        # 核心区别：实例化的是 RmPad 版本的 Attention 模块
        self.self_attn = ParallelLlamaAttentionRmPad(config=config, megatron_config=megatron_config)

        # MLP 和 RMSNorm 层是通用的，可以处理 padded 和 unpadded 数据
        self.mlp = ParallelLlamaMLP(config, megatron_config=megatron_config)
        self.input_layernorm = ParallelLlamaRMSNorm(config, megatron_config)
        self.post_attention_layernorm = ParallelLlamaRMSNorm(config, megatron_config)

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_ids: Optional[torch.LongTensor] = None,  # position_ids 在 FlashAttention 中通常不需要
            sequence_length: int = None,
            indices: torch.Tensor = None,  # 去填充索引，现在也不常用了
            cu_seqlens: int = None,  # 累积序列长度，FlashAttention 的关键输入
            max_seqlen_in_batch: int = None,  # 批次内最大序列长度
    ) -> torch.Tensor:
        """
        前向传播函数。
        :param hidden_states: 输入张量，形状为 `(total_tokens, 1, hidden_dim)`，是去填充后的数据。
        """
        # 1. 第一个残差连接
        residual = hidden_states

        # 2. 输入层归一化
        hidden_states = self.input_layernorm(hidden_states)

        # 3. 自注意力模块
        # 调用的是 RmPad 版本的 Attention forward 函数
        # 内部的张量并行和序列并行通信模式（all-gather -> col + row -> reduce-scatter）被封装好了。
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            sequence_length=sequence_length,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen_in_batch=max_seqlen_in_batch,
        )

        # 4. 第一个残差加法
        hidden_states = residual + hidden_states

        # 5. 第二个残差连接
        residual = hidden_states

        # 6. 注意力后层归一化
        hidden_states = self.post_attention_layernorm(hidden_states)

        # 7. MLP/前馈网络模块
        hidden_states = self.mlp(hidden_states)

        # 8. 第二个残差加法
        hidden_states = residual + hidden_states

        outputs = hidden_states

        return outputs