# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# (版权和开源许可声明部分...)

import numbers  # 导入 Python 的 numbers 模块，用于进行更通用的数字类型检查

import torch
# 关键导入：从 NVIDIA Apex 库中导入融合的 RMSNorm CUDA 内核
from apex.normalization.fused_layer_norm import fused_rms_norm_affine
from megatron.core import ModelParallelConfig
from torch import nn
from transformers import LlamaConfig

# 从项目内部导入序列并行的辅助工具
from verl.utils.megatron import sequence_parallel as sp_utils


class ParallelLlamaRMSNorm(nn.Module):
    """
    并行化的 Llama RMSNorm 层。
    它等价于 T5 模型中使用的 T5LayerNorm。
    """

    def __init__(self, config: LlamaConfig, megatron_config: ModelParallelConfig):
        """
        初始化函数。
        :param config: HuggingFace 的 LlamaConfig 对象。
        :param megatron_config: Megatron-LM 的并行配置对象。
        """
        super().__init__()

        # 检查 hidden_size 是否是整数类型，以健壮地构造归一化的形状。
        # 原理：使用 `numbers.Integral` 比 `type(x) is int` 更通用，
        # 它可以正确处理 Python 内置 int、NumPy int 等多种整数类型。
        if isinstance(config.hidden_size, numbers.Integral):
            normalized_shape = (config.hidden_size,)
        self.normalized_shape = torch.Size(normalized_shape)

        # 定义可学习的缩放参数 `g` (gain)，在 PyTorch 中称为 weight。
        # 原理：这个参数让模型可以学习在归一化后，是否需要以及如何重新缩放特征的幅度。
        # 它被初始化为 1，这样在训练开始时，这一层不会改变输入的尺度。
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))

        # 从配置中获取一个极小值 epsilon，用于防止除以零。
        # 原理：在计算均方根时，如果输入向量恰好是零向量，分母会为零。
        # 加上这个 epsilon 可以保证数值稳定性。
        self.variance_epsilon = config.rms_norm_eps

        # 如果启用了序列并行，需要特殊处理权重参数。
        if megatron_config.sequence_parallel:
            # 标记这个参数是用于序列并行的。
            # 原理：序列并行会将输入张量在序列维度上切分。RMSNorm 的计算是在
            # hidden_size 维度上进行的，这个维度在所有 GPU 上都是完整的，所以计算本身是局部的。
            # 但是，Megatron 的自动求导和优化器需要知道哪些参数属于哪个并行区域，
            # 以便正确地处理梯度（例如，避免在张量并行组内对这个参数的梯度进行不必要的 All-Reduce）。
            # 这个标记函数就是给 Megatron 框架的一个信号。
            sp_utils.mark_parameter_as_sequence_parallel(self.weight)

    def forward(self, hidden_states):
        """
        前向传播函数。
        :param hidden_states: 输入张量，形状通常是 (batch, seq_len, hidden_size) 或 (total_tokens, hidden_size)。
        """
        # 调用 Apex 提供的、高度优化的融合 CUDA 内核来执行 RMSNorm。
        # 原理：这是本实现高性能的关键。它将多个数学运算合并到一个 GPU Kernel 中执行，
        # 避免了多次从 GPU 显存中读取和写入数据，从而大幅提升了计算速度。
        return fused_rms_norm_affine(
            input=hidden_states,  # 待归一化的输入张量
            weight=self.weight,  # 可学习的缩放参数 g
            normalized_shape=self.normalized_shape,  # 指定在哪个维度上进行归一化，即 (hidden_size,)
            eps=self.variance_epsilon,  # 防止除以零的 epsilon
            memory_efficient=True,  # 启用内存效率更高的实现模式，可能会以微小的计算开销换取更低的内存占用
        )