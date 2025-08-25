# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# 这个文件的代码部分基于 EleutherAI 的 GPT-NeoX 库以及 HuggingFace 的 GPT-NeoX 和 OPT 实现。
# 为了适应 Meta AI 训练的 Llama 模型中微小的架构差异，它在其原始形式上进行了修改。
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 只有在遵守 Apache 2.0 许可证的情况下，你才能使用这个文件。
# 你可以在以下地址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，根据许可证分发的软件是按“原样”提供的，
# 没有任何明示或暗示的保证或条件。
# 请查阅许可证以了解特定语言下的权限和限制。

import math  # 导入Python内置的数学库，用于开方等计算
from typing import Optional  # 导入类型提示工具，Optional表示一个变量可以是某个类型或None

import torch  # PyTorch 核心库
import torch.nn.functional as F  # PyTorch 的函数库，包含 softmax、pad 等操作
from einops import rearrange  # 一个强大的张量操作库，用于直观地重排维度
from flash_attn.layers.rotary import apply_rotary_emb  # 从 flash-attn 库导入专用的 RoPE 应用函数
from megatron.core import ModelParallelConfig, tensor_parallel  # 从 Megatron-LM 导入张量并行配置和核心层
from megatron.core import parallel_state as mpu  # 导入 Megatron-LM 的并行状态管理工具，用于获取并行信息
from torch import nn  # PyTorch 的神经网络模块
from transformers import LlamaConfig  # 从 HuggingFace Transformers 库导入 Llama 的配置类
from transformers.utils import is_flash_attn_2_available  # 检查环境中是否安装了 FlashAttention 2

# 从项目内部模块导入为张量并行定制的线性层
from verl.models.llama.megatron.layers.parallel_linear import QKVParallelLinear
# 从项目内部导入张量并行的辅助工具
from verl.utils.megatron import tensor_parallel as tp_utils


# =================================================================================
# 第一部分：旋转位置编码 (Rotary Positional Embedding - RoPE)
# 原理：RoPE 是一种将位置信息融入到模型的 Attention 机制中的精妙方法。
# 它不是将位置向量“加”到词向量上，而是根据词的位置，将词向量在某些维度上进行“旋转”。
# 位置靠后的词，旋转的角度更大。这种方式能让模型天然地理解词与词之间的相对距离，
# 并且在处理长序列时具有很好的外推性。
# =================================================================================

class LlamaRotaryEmbedding(nn.Module):
    """Llama 模型的标准旋转位置编码实现。"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        """
        初始化函数。
        :param dim: 每个注意力头的维度 (head_dim)。
        :param max_position_embeddings: 模型预设的最大序列长度。
        :param base: RoPE 的基数，一个超参数，通常为 10000。
        :param device: 指定张量所在的设备。
        """
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # RoPE 核心公式：计算旋转频率的倒数。
        # 原理：这个公式为每个维度对（共 dim/2 对）生成一个独特的旋转频率。
        # torch.arange(0, self.dim, 2) -> [0, 2, 4, ..., dim-2]
        # ... / self.dim -> 归一化
        # self.base ** (...) -> 计算频率
        # 1.0 / (...) -> 取倒数
        # 结果是，向量的低维部分旋转得慢（捕捉长距离关系），高维部分旋转得快（捕捉短距离关系）。
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))

        # 将 inv_freq 注册为模型的 buffer。
        # buffer 是模型的状态，但不是需要训练的参数。`persistent=False` 表示它不会被保存在模型的 state_dict 中。
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 为了效率，预先计算并缓存所有可能位置的 cos 和 sin 值。
        # 这避免了在每次前向传播时都重新计算。
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        """
        计算并设置 cos 和 sin 的缓存。
        """
        self.max_seq_len_cached = seq_len  # 记录当前缓存的最大长度
        # 创建一个从 0 到 seq_len-1 的序列，代表位置索引。
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        # 计算每个位置和每个频率的旋转角度 theta = t * inv_freq
        # 原理：torch.einsum("i,j->ij", t, self.inv_freq) 执行了一个外积操作。
        # 结果 `freqs` 是一个 [seq_len, dim/2] 的矩阵，每个元素是对应位置和频率的旋转角度。
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)

        # 将角度复制一份，因为 RoPE 是作用在成对的维度上的，所以 cos 和 sin 的维度需要是 [seq_len, dim]。
        emb = torch.cat((freqs, freqs), dim=-1)

        # 计算并缓存所有位置的 cos 和 sin 值。
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        """
        前向传播函数，返回预计算好的 cos 和 sin 值。
        :param x: 任意输入张量，仅用于获取 device 和 dtype 信息。
        :param seq_len: 当前批次需要的序列长度。
        """
        # 如果请求的序列长度超过了已缓存的最大长度，则重新计算并扩大缓存。
        # 这使得模型可以动态适应比预设 `max_position_embeddings` 更长的输入。
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        # 从缓存中取出所需长度的 cos 和 sin 值并返回。
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """
    LlamaRoPE 的扩展，增加了线性缩放（Linear Scaling）功能。
    原理：这是一种简单的长度外推方法。当输入序列长度超过模型预训练长度时，
    它通过将位置索引 `t` 除以一个缩放因子 `scaling_factor`，将长序列的位置“压缩”到模型熟悉的短序列范围内。
    例如，如果 `scaling_factor=4`，位置 4096 就会被当作位置 1024 来计算 RoPE。
    优点是简单，缺点是可能会损失高频位置信息。
    """

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        # 核心改动：将位置索引 t 除以缩放因子。
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """
    LlamaRoPE 的扩展，增加了动态 NTK 缩放（Dynamic NTK Scaling）功能。
    原理：NTK 缩放是一种更智能的长度外推方法。它认为线性缩放会过度压缩高频信息。
    因此，当序列长度超过预设最大值时，它不是缩放位置索引，而是动态地调整 RoPE 的 `base` 参数。
    这个调整主要影响低频部分，使得模型在处理长距离依赖时表现更好，同时保留了高频（短距离）信息的精度。
    """

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        # 核心改动：如果序列长度超过预设值，就根据 NTK 公式重新计算 `base` 和 `inv_freq`。
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                    (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            # 注意：这里直接覆盖了父类中的 inv_freq buffer
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


# Llama3 的 RoPE 缩放方法，更复杂，效果也更好
class LlamaLlama3ScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """
    Llama 3 模型使用的 RoPE 缩放方法。
    原理：这是一种更复杂的混合缩放策略。它将旋转频率分为三段：高频、中频和低频。
    - 对低频部分（负责长距离依赖），它会进行较大幅度的缩放（除以一个因子，如8）。
    - 对高频部分（负责短距离依赖），它基本不缩放或小幅缩放。
    - 对中频部分，它在这两种缩放策略之间进行平滑的插值过渡。
    这种方法兼顾了长距离依赖的捕捉和短距离细节的保留。
    """

    def __init__(self, dim, config, max_position_embeddings=2048, base=10000, device=None):
        super().__init__(dim, max_position_embeddings, base, device)

        # 从模型配置中读取 Llama3 特有的缩放参数
        self.factor = config.rope_scaling["factor"]
        self.high_freq_factor = config.rope_scaling["high_freq_factor"]
        self.low_freq_factor = config.rope_scaling["low_freq_factor"]
        self.old_context_len = config.rope_scaling["original_max_position_embeddings"]

        # 根据频率的“波长”来判断其属于高、中、低频
        low_freq_wavelen = self.old_context_len / self.low_freq_factor
        high_freq_wavelen = self.old_context_len / self.high_freq_factor
        wavelen = 2 * math.pi / self.inv_freq

        # 对低频部分进行缩放
        inv_freq_llama = torch.where(wavelen > low_freq_wavelen, self.inv_freq / self.factor, self.inv_freq)

        # 对中频部分进行平滑插值
        smooth_factor = (self.old_context_len / wavelen - self.low_freq_factor) / (
                self.high_freq_factor - self.low_freq_factor
        )
        smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / self.factor + smooth_factor * inv_freq_llama
        is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)

        # 合并三段的频率
        inv_freq = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

        # 覆盖父类的 inv_freq
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )


# =================================================================================
# 第二部分：辅助函数
# =================================================================================

def rotate_half(x):
    """
    将输入张量的最后一个维度切成两半，交换它们的位置，并对第一半取反。
    原理：这是在复数域中乘以 i 的高维模拟。RoPE 的数学基础是将向量的每两个维度看作一个复数 z = a + ib，
    对其乘以 e^(i*theta) = cos(theta) + i*sin(theta)。
    展开后 (a+ib)(cos+isin) = (acos - bsin) + i(asin + bcos)。
    这个 `rotate_half` 函数就是为了计算 `-b` 和 `a` 这部分，用于与 `sin` 相乘。
    """
    x1 = x[..., : x.shape[-1] // 2]  # 取前半部分
    x2 = x[..., x.shape[-1] // 2:]  # 取后半部分
    return torch.cat((-x2, x1), dim=-1)  # 对后半部分取反，然后与前半部分拼接


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """
    将旋转位置编码应用到 Query 和 Key 张量上。
    原理：根据 RoPE 的核心公式 `x_rotated = x * cos(theta) + rotate_half(x) * sin(theta)` 来实现。
    """
    # 根据 position_ids 从缓存中取出对应位置的 cos 和 sin 值
    cos = cos[position_ids].unsqueeze(1)  # 形状: [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # 形状: [bs, 1, seq_len, dim]

    # 应用 RoPE 变换
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    为分组查询注意力（GQA）重复 Key 和 Value 张量。
    原理：在 GQA 中，多组查询头（Q）会共享同一组键（K）和值（V）头。
    例如，如果有 32 个 Q 头，但只有 8 个 K/V 头，那么 `n_rep` 就是 4。
    这个函数会将 8 个 K/V 头各自复制 4 次，使得总头数达到 32，从而能与 32 个 Q 头进行匹配计算。
    这比使用 32 个独立的 K/V 头要节省大量显存。
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:  # 如果 n_rep=1, 说明是标准多头注意力，无需重复
        return hidden_states
    # 增加一个维度并扩展（expand），这比 repeat 更高效，因为它不实际复制数据，而是修改 view
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    # 重塑形状，将 num_key_value_heads 和 n_rep 两个维度合并
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# =================================================================================
# 第三部分：标准并行注意力层 (带 Padding)
# =================================================================================

class ParallelLlamaAttention(nn.Module):
    """
    Llama 模型的并行化多头注意力实现。
    原理：这个模块实现了标准的多头/分组查询注意力机制，并深度整合了 Megatron-LM 的张量并行。
    - Q, K, V 的线性投射层 (`qkv_proj`) 是一个列并行层，它的权重矩阵被垂直切分到各个 GPU 上。
    - 输出的线性投射层 (`o_proj`) 是一个行并行层，它的权重矩阵被水平切分到各个 GPU 上。
    这使得单个巨大的 Attention 层可以分布在多个 GPU 上运行，突破了单卡显存瓶颈。
    """

    def __init__(self, config: LlamaConfig, megatron_config: ModelParallelConfig):
        super().__init__()
        self.config = config
        self.megatron_config = megatron_config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads  # GQA 的重复因子
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta  # RoPE 的 base 参数

        # 获取张量并行的 world size (即参与模型并行的 GPU 数量)
        tp_size = mpu.get_tensor_model_parallel_world_size()
        # 断言：确保总头数可以被 GPU 数量整除，否则无法均匀切分
        assert self.num_heads % tp_size == 0
        assert self.num_key_value_heads % tp_size == 0

        # 计算每个 GPU 分片上实际拥有的头数和隐藏层大小
        self.num_heads_per_tp = self.num_heads // tp_size
        self.num_key_value_heads_per_tp = self.num_key_value_heads // tp_size
        self.hidden_size_per_tp = self.hidden_size // tp_size

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError("hidden_size 必须能被 num_heads 整除")

        # 获取用于列并行和行并行的默认参数
        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        row_kwargs = tp_utils.get_default_kwargs_for_row_parallel_linear()

        if megatron_config is not None:
            tp_utils.update_kwargs_with_config(column_kwargs, megatron_config)
            tp_utils.update_kwargs_with_config(row_kwargs, megatron_config)

        # 定义 QKV 投射层。这是一个定制的、高效的并行线性层，一次性计算出 Q, K, V。
        # 它是 ColumnParallelLinear，输入是完整的，输出是切分的。
        self.qkv_proj = QKVParallelLinear(
            input_size=self.hidden_size,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            bias=config.attention_bias,
            **column_kwargs,
        )

        # 计算每个 GPU 上 Q, K, V 切分后的大小
        self.q_size = self.num_heads_per_tp * self.head_dim
        self.k_size = self.num_key_value_heads_per_tp * self.head_dim
        self.v_size = self.num_key_value_heads_per_tp * self.head_dim

        # 定义输出投射层。
        # 它是 RowParallelLinear，输入是切分的，在前向传播时会自动执行 All-Reduce 操作，得到完整的输出。
        self.o_proj = tensor_parallel.RowParallelLinear(
            input_size=self.num_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=config.attention_bias,
            input_is_parallel=True,
            **row_kwargs,
        )

        # 根据配置初始化对应的 RoPE 变体
        self._init_rope()

    def _init_rope(self):
        """根据模型配置初始化正确的 RoPE 实现。"""
        if self.config.rope_scaling is None:
            # 标准 RoPE
            self.rotary_emb = LlamaRotaryEmbedding(...)
        else:
            # 根据 `rope_scaling` 字典中的 `type` 字段选择不同的长度外推方法
            scaling_type = self.config.rope_scaling.get("type") or self.config.rope_scaling.get("rope_type")
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(...)
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(...)
            elif scaling_type == "llama3":
                self.rotary_emb = LlamaLlama3ScalingRotaryEmbedding(...)
            else:
                raise ValueError(f"未知的 RoPE 缩放类型: {scaling_type}")

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """标准注意力的前向传播。"""
        bsz, q_len, _ = hidden_states.size()

        # 1. QKV 投射
        # [bsz, q_len, hidden_size] -> [bsz, q_len, hidden_size_per_tp * (num_q_groups + 2)]
        qkv = self.qkv_proj(hidden_states)[0]
        # 将合并的 QKV 张量切分成独立的 Q, K, V
        query_states, key_states, value_states = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)

        # 2. 变形以匹配多头注意力的计算格式
        # [bsz, q_len, local_hidden_size] -> [bsz, local_num_heads, q_len, head_dim]
        query_states = query_states.view(bsz, q_len, self.num_heads_per_tp, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads_per_tp, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads_per_tp, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]

        # 3. 应用旋转位置编码 (RoPE)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # 4. GQA: 重复 K 和 V
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # 5. 计算注意力分数
        # [bsz, num_heads, q_len, head_dim] @ [bsz, num_heads, head_dim, kv_len] -> [bsz, num_heads, q_len, kv_len]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # 确保形状正确
        if attn_weights.size() != (bsz, self.num_heads_per_tp, q_len, kv_seq_len):
            raise ValueError("注意力权重形状错误")

        # 6. 应用注意力掩码 (Attention Mask)
        # 例如，在自回归任务中，防止 token 看到未来的 token
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # 7. Softmax
        # 将分数转换为概率分布。为了数值稳定性，通常在这里将数据类型提升到 float32。
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # 8. 计算加权和
        # [bsz, num_heads, q_len, kv_len] @ [bsz, num_heads, kv_len, head_dim] -> [bsz, num_heads, q_len, head_dim]
        attn_output = torch.matmul(attn_weights, value_states)

        # 9. 变形回标准格式
        # [bsz, num_heads, q_len, head_dim] -> [bsz, q_len, num_heads, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # [bsz, q_len, num_heads, head_dim] -> [bsz, q_len, local_hidden_size]
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size_per_tp)

        # 10. 输出投射
        # 这里的 RowParallelLinear 会在内部执行 All-Reduce，将所有 GPU 上的结果合并
        attn_output = self.o_proj(attn_output)[0]
        return attn_output


# =================================================================================
# 第四部分：FlashAttention 2 优化的并行注意力层 (去填充)
# 原理：这是一个更高性能的版本。它首先将一个批次中所有序列的有效 token "压平"成一个长向量（去填充），
# 然后调用 FlashAttention 2 的 `flash_attn_varlen_func` 函数。这个函数使用高度优化的 CUDA 内核
# 来计算注意力，无需显式创建注意力矩阵，速度极快且显存占用极低。
# `cu_seqlens` 和 `max_seqlen_in_batch` 等参数就是用来告诉这个内核每个序列的边界在哪里。
# =================================================================================

if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func  # 导入变长序列的 FlashAttention 函数
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # 导入填充/去填充的辅助函数


    # 这是一个为去填充数据应用 RoPE 的旧版实现，新版直接使用 flash-attn 内置的 RoPE 更高效
    def apply_rotary_pos_emb_rmpad(q, k, cos, sin, position_ids, indices, sequence_length):


    # 略...

    # Flash-attn 自带的 RoPE 应用函数，专为去填充的（varlen）数据设计
    def apply_rotary_pos_emb_rmpad_flash(q, k, cos, sin, cu_seqlens, max_seqlen):
        # 调用 flash_attn 库提供的 apply_rotary_emb 函数，它对 unpadded 数据有专门优化
        q_embed = apply_rotary_emb(q, cos, sin, interleaved=False, inplace=False, cu_seqlens=cu_seqlens,
                                   max_seqlen=max_seqlen)
        k_embed = apply_rotary_emb(k, cos, sin, interleaved=False, inplace=False, cu_seqlens=cu_seqlens,
                                   max_seqlen=max_seqlen)
        return q_embed, k_embed


class ParallelLlamaAttentionRmPad(ParallelLlamaAttention):
    """
    使用 FlashAttention-2 和去填充（Remove Padding）技术优化的并行注意力层。
    """

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_ids: Optional[torch.LongTensor] = None,  # 注意：这里的 position_ids 实际上没被使用
            sequence_length: int = None,  # 原始序列长度（未填充）
            indices: torch.Tensor = None,  # 去填充的索引
            cu_seqlens: torch.Tensor = None,  # 累积序列长度，FlashAttention 的关键输入
            max_seqlen_in_batch: int = None,  # 该批次中的最大序列长度
    ):
        # hidden_states 的形状是 [total_tokens, 1, hidden_size_per_tp]
        total_nnz, _, _ = hidden_states.size()  # total_nnz 是批次中所有 token 的总数

        # 如果启用了序列并行，输入 hidden_states 会被填充，这里需要先知道总 token 数
        if self.megatron_config.sequence_parallel:
            total_nnz = total_nnz * mpu.get_tensor_model_parallel_world_size()

        # 1. QKV 投射
        qkv = self.qkv_proj(hidden_states)[0]
        query_states, key_states, value_states = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)

        # 如果启用了序列并行，需要去掉为了并行计算而添加的 padding
        if self.megatron_config.sequence_parallel:
            total_nnz_before_pad = cu_seqlens[-1].item()  # 真正的总 token 数
            query_states = query_states[:total_nnz_before_pad]
            key_states = key_states[:total_nnz_before_pad]
            value_states = value_states[:total_nnz_before_pad]

        # 2. 变形，FlashAttention 输入格式为 [total_tokens, num_heads, head_dim]
        query_states = query_states.view(-1, self.num_heads_per_tp, self.head_dim)
        key_states = key_states.view(-1, self.num_key_value_heads_per_tp, self.head_dim)
        value_states = value_states.view(-1, self.num_key_value_heads_per_tp, self.head_dim)

        # 3. 应用 RoPE
        cos, sin = self.rotary_emb(value_states, seq_len=sequence_length)
        # FlashAttention 的 RoPE 函数只需要 dim/2 的 cos/sin
        cos, sin = cos[:, : cos.shape[1] // 2], sin[:, : sin.shape[1] // 2]
        query_states, key_states = apply_rotary_pos_emb_rmpad_flash(
            query_states, key_states, cos, sin, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen_in_batch
        )

        # 4. GQA: 重复 K 和 V (FlashAttention 内部支持 GQA，所以这里直接传入)
        # 注意：新版 FlashAttention 可以直接处理 GQA，无需手动 repeat_kv
        # key_states = repeat_kv(key_states, self.num_key_value_groups)
        # value_states = repeat_kv(value_states, self.num_key_value_groups)

        # 确保数据类型正确，FlashAttention 通常在 float16/bfloat16 上性能最好
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            query_states = query_states.to(torch.float16)
            key_states = key_states.to(torch.float16)
            value_states = value_states.to(torch.float16)

        # 5. 调用 FlashAttention 核心函数
        attn_output_unpad = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen_in_batch,
            max_seqlen_k=max_seqlen_in_batch,
            dropout_p=0.0,  # 训练时可以设置 dropout
            softmax_scale=None,  # 内部会自动计算 1/sqrt(head_dim)
            causal=True,  # 这是解码器模型，必须是因果的
        )

        # 6. 变形和输出投射
        attn_output_unpad = attn_output_unpad.to(input_dtype)
        attn_output_unpad = attn_output_unpad.reshape(-1, 1, self.hidden_size_per_tp).contiguous()

        # 如果启用了序列并行，需要把之前去掉的 padding 加回来，以便后续的 All-Reduce 操作
        if self.megatron_config.sequence_parallel:
            sequence_parallel_pad = (total_nnz * mpu.get_tensor_model_parallel_world_size()) - total_nnz_before_pad
            attn_output_unpad = F.pad(attn_output_unpad, pad=(0, 0, 0, 0, 0, sequence_parallel_pad))

        # 7. 输出投射
        attn_output_unpad = self.o_proj(attn_output_unpad)[0]
        return attn_output_unpad