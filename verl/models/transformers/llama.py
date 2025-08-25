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

# llama.py 兼容 transformers 的 LLaMA 层实现，支持 Flash Attention、Ulysses 序列并行等高级特性。
# 注释详细解释各参数、并行机制、兼容性处理等。

import sys  # 系统相关操作
from typing import Callable, Optional  # 类型提示
import torch  # PyTorch 深度学习框架

if sys.version_info >= (3, 11):
    pass  # 兼容 Python 3.11 及以上版本
else:
    pass  # 兼容低版本 Python

from transformers.cache_utils import Cache  # transformers 缓存工具
from transformers.modeling_flash_attention_utils import _flash_attention_forward  # Flash Attention 前向实现
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb  # 旋转位置编码实现
from transformers.utils import logging  # 日志工具

# 兼容 flash_attn_supports_top_left_mask
from verl.utils.transformers_compat import flash_attn_supports_top_left_mask
from verl.utils.ulysses import (
    gather_heads_scatter_seq,           # Ulysses 并行：头到序列聚合
    gather_seq_scatter_heads,           # Ulysses 并行：序列到头聚合
    get_ulysses_sequence_parallel_world_size,  # 获取 Ulysses 序列并行规模
    validate_ulysses_config,            # 校验 Ulysses 配置
)

logger = logging.get_logger(__name__)  # 获取日志记录器


def llama_flash_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # v4.46 及以后将强制要求
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    """
    适配 transformers 4.47.1，支持 Ulysses 序列并行的 Flash Attention 前向。
    参数:
        self: Llama 层对象
        hidden_states: 输入隐藏状态
        attention_mask: 注意力掩码
        position_ids: 位置索引
        past_key_value: 历史缓存
        output_attentions: 是否输出注意力
        use_cache: 是否使用缓存
        cache_position: 缓存位置
        position_embeddings: 外部位置编码（cos, sin）
        kwargs: 其它参数
    返回:
        (输出张量, 可选注意力, 可选缓存)
    """
    output_attentions = False  # 默认不输出注意力

    bsz, q_len, _ = hidden_states.size()  # 获取 batch size 和序列长度

    query_states = self.q_proj(hidden_states)  # 计算 query
    key_states = self.k_proj(hidden_states)    # 计算 key
    value_states = self.v_proj(hidden_states)  # 计算 value

    # Flash Attention 要求输入 shape: batch_size x seq_length x head_dim x hidden_dim
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # Ulysses 序列并行 AlltoAll 操作
    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()

    if ulysses_sp_size > 1:
        validate_ulysses_config(self.num_heads, ulysses_sp_size)
        # (bsz, n_head, seq_len/n, head_dim) -> (bsz, n_head/n, seq_len, head_dim)
        query_states = gather_seq_scatter_heads(query_states, seq_dim=2, head_dim=1)
        key_states = gather_seq_scatter_heads(key_states, seq_dim=2, head_dim=1)
        value_states = gather_seq_scatter_heads(value_states, seq_dim=2, head_dim=1)

    full_q_len = query_states.size(2)  # 完整序列长度

    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)  # 计算旋转位置编码
    else:
        cos, sin = position_embeddings  # 使用外部位置编码
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)  # 应用旋转位置编码

    if past_key_value is not None:
        # sin 和 cos 是 RoPE 模型特有的; cache_position 用于静态缓存
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # TODO: 这些 transpose 操作效率较低，但 Flash Attention 要求的布局是
    # [batch_size, sequence_length, num_heads, head_dim]。我们需要重构 KV 缓存
    # 以避免许多这样的 transpose/reshape/view 操作。
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    dropout_rate = self.attention_dropout if self.training else 0.0

    # 在 PEFT 中，通常出于训练稳定性考虑，我们将层归一化转换为 float32
    # 因此，输入的隐藏状态会被默默地转换为 float32。我们需要
    # 将它们转换回正确的 dtype，以确保一切按预期工作。
    # 这可能会降低训练和推理速度，因此建议不要将 LayerNorms
    # 转换为 fp32。（LlamaRMSNorm 正确处理了这个问题）

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # 处理模型量化的情况
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to "
            f"the fact you have upcasted embedding or layer norm layers in float32. We will cast back the "
            f"input in {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        full_q_len,
        position_ids=position_ids,
        dropout=dropout_rate,
        sliding_window=getattr(self, "sliding_window", None),
        use_top_left_mask=flash_attn_supports_top_left_mask(),
        is_causal=self.is_causal,
        **kwargs,
    )

    attn_output = attn_output.reshape(bsz, full_q_len, -1, self.head_dim).contiguous()
    ########## AlltoAll for Ulysses ##########
    if ulysses_sp_size > 1:
        attn_output = gather_heads_scatter_seq(attn_output, seq_dim=1, head_dim=2)
    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def llama_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    """
    Adapted from transformers 4.49.0 to support Ulysses sequence parallelism for transformers >= 4.48.0.

    NOTE: This function has been tested only on transformers versions between 4.48.0 and 4.50.0.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.llama.modeling_llama import eager_attention_forward

    bsz, q_len, _ = hidden_states.shape

    query_states = self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    ########## AlltoAll for Ulysses ##########
    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()

    if ulysses_sp_size > 1:
        validate_ulysses_config(self.config.num_attention_heads, ulysses_sp_size)

        query_states = gather_seq_scatter_heads(query_states, seq_dim=2, head_dim=1)
        key_states = gather_seq_scatter_heads(key_states, seq_dim=2, head_dim=1)
        value_states = gather_seq_scatter_heads(value_states, seq_dim=2, head_dim=1)

    full_q_len = query_states.size(2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin 和 cos 是 RoPE 模型特有的; cache_position 用于静态缓存
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
            logger.warning_once(
                "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. "
                "Falling back to eager attention. This warning can be removed using the argument "
                '`attn_implementation="eager"` when loading the model.'
            )
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(bsz, full_q_len, -1, self.head_dim).contiguous()
    ########## AlltoAll for Ulysses ##########
    if ulysses_sp_size > 1:
        attn_output = gather_heads_scatter_seq(attn_output, seq_dim=1, head_dim=2)
    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights
