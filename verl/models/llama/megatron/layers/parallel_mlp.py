# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# (版权和开源许可声明部分...)

# 从 Megatron-LM 导入张量并行核心库
from megatron.core import ModelParallelConfig, tensor_parallel
from megatron.core import parallel_state as mpu  # mpu (Model Parallel Unit) 用于获取并行环境信息
from torch import nn
from transformers.activations import ACT2FN  # 从 HuggingFace Transformers 库导入激活函数字典

# 从项目内部导入为张量并行定制的特殊线性层
from verl.models.llama.megatron.layers.parallel_linear import MergedColumnParallelLinear
# 从项目内部导入张量并行的辅助工具
from verl.utils.megatron import tensor_parallel as tp_utils


class ParallelLlamaMLP(nn.Module):
    """
    Llama 模型的并行化 MLP (多层感知机) / FFN (前馈网络) 层。
    原理：这个模块实现了 Llama 使用的 SwiGLU 激活结构。
    其计算公式为：Output = down_proj(act_fn(gate_proj(x)) * up_proj(x))
    其中 `act_fn` 通常是 SiLU (Sigmoid Linear Unit)。
    为了性能，`gate_proj` 和 `up_proj` 被合并为一个线性层 `gate_up_proj`。
    整个模块通过 Megatron-LM 实现了张量并行。
    """

    def __init__(self, config, megatron_config: ModelParallelConfig = None) -> None:
        """
        初始化函数。
        :param config: HuggingFace 的模型配置对象，包含 hidden_size, intermediate_size 等。
        :param megatron_config: Megatron-LM 的并行配置对象。
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size  # 模型的隐藏层维度
        self.intermediate_size = config.intermediate_size  # MLP 中间层的维度，通常是 hidden_size 的倍数

        # 获取用于 Megatron-LM 列并行和行并行线性层的默认参数
        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        row_kwargs = tp_utils.get_default_kwargs_for_row_parallel_linear()

        # 如果传入了 megatron_config，则用它来更新并行层的参数
        if megatron_config is not None:
            assert column_kwargs.get("config", False), "megatron_config 必须提供给并行层"
            assert row_kwargs.get("config", False), "megatron_config 必须提供给并行层"
            tp_utils.update_kwargs_with_config(row_kwargs, megatron_config)
            tp_utils.update_kwargs_with_config(column_kwargs, megatron_config)

        # 获取张量并行的 world size (即参与模型并行的 GPU 数量)
        tp_size = mpu.get_tensor_model_parallel_world_size()

        # 实例化合并后的 `gate` 和 `up` 投射层。
        # 原理：这是一个列并行层 (ColumnParallelLinear)，它的权重矩阵在输出维度上被切分。
        # 例如，如果 intermediate_size=8192, tp_size=4，那么每个 GPU 上的权重矩阵
        # 形状为 [hidden_size, (8192*2)/4]，输出张量的最后一个维度大小为 4096。
        # `gate_ouput_size` 和 `up_output_size` 分别指定了 gate 和 up 部分的完整输出大小。
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            gate_ouput_size=self.intermediate_size,
            up_output_size=self.intermediate_size,
            bias=False,  # Llama 的 MLP 通常没有偏置项
            gather_output=False,  # 输出保持切分状态，不需要聚合
            skip_bias_add=False,  # 如果有偏置项，是否跳过加法（通常为False）
            **column_kwargs,
        )

        # 计算在当前 GPU 分片上，gate 部分的输出大小。
        # 这是为了后续从合并的输出中正确地切分出 gate 和 up。
        self.gate_size = self.intermediate_size // tp_size

        # 实例化 `down` 投射层。
        # 原理：这是一个行并行层 (RowParallelLinear)。它的权重矩阵在输入维度上被切分。
        # 它的输入 `x` 是切分的（来自上一步的输出），在前向传播时，它会先在各个 GPU 上
        # 对自己持有的那部分输入和权重进行计算，然后通过 All-Reduce 操作将所有 GPU 的结果相加，
        # 最终得到一个完整的、未切分的输出。
        self.down_proj = tensor_parallel.RowParallelLinear(
            input_size=self.intermediate_size,
            output_size=self.hidden_size,
            bias=False,
            input_is_parallel=True,  # 明确告诉该层，它的输入是已经并行切分过的
            skip_bias_add=False,
            **row_kwargs,
        )

        # 从 Transformers 的激活函数字典中获取配置指定的激活函数。
        # Llama 通常使用 "silu" (SiLU)。
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        """
        前向传播函数。
        :param x: 输入张量，形状为 `(..., hidden_size)`。
        """
        # 1. 计算合并的 gate 和 up 投射。
        # 输入 x [..., hidden_size] -> 输出 gate_up [..., intermediate_size*2 / tp_size]
        gate_up = self.gate_up_proj(x)[0]

        # 2. 从合并的输出中切分出 gate 和 up。
        # `gate_up` 的最后一个维度是 `gate_size + up_size`，因为 up_size 和 gate_size 相等，
        # 所以 `gate_size` 恰好是总大小的一半。
        gate, up = gate_up.split(self.gate_size, dim=-1)

        # 3. 执行 SwiGLU 计算。
        # - self.act_fn(gate): 对 gate 部分应用 SiLU 激活函数。
        # - ... * up: 将激活后的 gate 与 up 部分进行逐元素相乘。
        # - self.down_proj(...): 将相乘的结果输入到行并行层。
        # 这一步的输入是切分的 [..., intermediate_size / tp_size]，
        # `down_proj` 内部会执行 All-Reduce，输出是完整的 [..., hidden_size]。
        return self.down_proj(self.act_fn(gate) * up)[0]