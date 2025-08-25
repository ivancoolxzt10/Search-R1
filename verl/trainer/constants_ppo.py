# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# (版权和开源许可声明部分...)

import json  # 导入 JSON 库，用于解析 JSON 格式的字符串
import os  # 导入操作系统库，用于访问环境变量

# 从 Ray 的内部常量中导入一个特定的环境变量名
from ray._private.runtime_env.constants import RAY_JOB_CONFIG_JSON_ENV_VAR

# 原理：RAY_JOB_CONFIG_JSON_ENV_VAR 是 Ray 内部使用的一个环境变量名。当一个 Ray Job 提交时，
# Ray 会将关于这个 Job 的配置信息（包括用户指定的 runtime_env）打包成一个 JSON 字符串，
# 然后设置到这个环境变量中。这样，在 Ray 的工作进程（Worker Process）内部，
# 就可以通过读取这个环境变量来了解当前 Job 的配置。

# 定义一个全局的、推荐的 PPO 训练运行时环境配置字典
PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        # --- Hugging Face Tokenizers 配置 ---
        "TOKENIZERS_PARALLELISM": "true",
        # 中文解释: 启用 Hugging Face Tokenizers 库的并行处理。
        # 原理: 当设置为 "true" 或 "false" 时，可以控制 Tokenizer 是否使用多线程进行快速分词。
        # 启用它可以显著加快数据预处理速度，但有时可能会与 Python 的其他多进程/多线程库（如 Ray）
        # 产生冲突或导致死锁。这里明确启用，表明开发者认为在这种场景下是安全的。

        # --- NVIDIA NCCL 配置 ---
        "NCCL_DEBUG": "WARN",
        # 中文解释: 设置 NVIDIA NCCL (NVIDIA Collective Communication Library) 的日志级别为“警告”。
        # 原理: NCCL 是用于多 GPU/多节点通信的核心库。将其日志级别设为 WARN，可以在通信出现问题时
        # 打印警告信息，帮助诊断分布式训练中的网络或硬件问题，同时又避免了 INFO 级别的大量无关日志。

        # --- VLLM (一个高性能推理引擎) 配置 ---
        "VLLM_LOGGING_LEVEL": "WARN",
        # 中文解释: 设置 VLLM 推理引擎的日志级别为“警告”。
        # 原理: 在 PPO 训练中，Actor 进行 rollout（生成经验）时可能会使用 VLLM 来加速。
        # 将其日志级别设为 WARN 可以保持日志输出的整洁。

        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        # 中文解释: 允许 VLLM 在运行时更新 LoRA 权重。
        # 原理: 在 PPO 训练中，Actor 模型的 LoRA 权重会不断更新。这个环境变量告诉 VLLM 引擎，
        # 它需要支持在不重启服务的情况下，动态地加载和应用新的 LoRA 权重，这是 RLHF 场景下的一个关键功能。

        # --- CUDA 配置 ---
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        # 中文解释: 设置每个 CUDA 设备的最大连接数为 1。
        # 原理: 这是一个用于 CUDA Stream 和事件管理的底层参数。在某些复杂的、多流并发的应用中，
        # 将其设置为 1 可能有助于简化同步、避免某些类型的竞争条件或性能问题。这是一个比较底层的性能调优参数。
    },
}


def get_ppo_ray_runtime_env():
    """
    一个过滤器函数，用于返回一个干净的 PPO Ray 运行时环境。
    目的是为了避免重复设置那些已经被外部环境设置过的环境变量。
    """
    # 尝试从 Ray 的 Job 配置环境变量中获取当前 Job 的 `working_dir`。
    # 原理：这个函数的目标是生成一个“默认”环境，但它不想覆盖用户在提交 Ray Job 时
    # 可能已经显式设置的 `working_dir`。因此，它先去检查 Ray 自身的环境变量，
    # 看看 `working_dir` 是否已经被定义了。
    working_dir = (
        json.loads(os.environ.get(RAY_JOB_CONFIG_JSON_ENV_VAR, "{}"))  # 1. 获取环境变量，如果不存在则返回空 JSON 字符串 '{}'
        .get("runtime_env", {})  # 2. 从解析后的 JSON 中获取 "runtime_env" 字典
        .get("working_dir", None)  # 3. 从 "runtime_env" 字典中获取 "working_dir"
    )

    # 初始化一个新的运行时环境字典
    runtime_env = {
        "env_vars": PPO_RAY_RUNTIME_ENV["env_vars"].copy(),  # 复制一份预定义的推荐环境变量
        # 这是一个条件字典解包的技巧。
        # 原理：如果 `working_dir` 在上面没有被找到 (is None)，那么这个表达式就会
        # 扩展成 `{"working_dir": None}`，并将这个键值对添加到 `runtime_env` 中。
        # 如果 `working_dir` 已经被找到了，那么这个表达式就是空的 `{}`，什么也不添加。
        # 这样做的目的可能是为了确保 `working_dir` 这个键总是存在，即使它的值是 None。
        **({"working_dir": None} if working_dir is None else {}),
    }

    # 遍历推荐环境变量字典的所有键
    for key in list(runtime_env["env_vars"].keys()):
        # 检查这个环境变量是否已经在当前的操作系统环境中被设置了
        if os.environ.get(key) is not None:
            # 如果已经被设置了（例如，用户在启动脚本前手动 `export NCCL_DEBUG=INFO`），
            # 那么就从我们的推荐配置中移除它。
            # 原理：这是一个“尊重用户”的设计。它提供了一组推荐的默认值，但如果用户
            # 已经有自己的设置，就优先使用用户的设置，而不是强制覆盖它。
            # `pop(key, None)` 会安全地移除键，如果键不存在也不会报错。
            runtime_env["env_vars"].pop(key, None)

    # 返回清理和过滤后的运行时环境字典
    return runtime_env