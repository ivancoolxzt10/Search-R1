"""
agent_loop.py

本文件主要实现了 Agent Loop 的核心逻辑，包括：
- 多服务器负载均衡与粘性会话管理（AsyncLLMServerManager）
- Agent Loop 的输入输出结构定义
- Agent Loop 的注册与实例化机制
- 奖励管理器与 Agent Loop 工作线程的异步分布式处理
- Agent Loop 管理器，负责批量分发与性能统计

适合初学者了解分布式 LLM 推理与 RL 训练的整体流程。
"""

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
import asyncio  # 异步编程模块
import heapq    # 堆队列模块，用于负载均衡
import logging  # 日志模块
import os       # 操作系统相关模块
import random   # 随机数模块
from abc import ABC, abstractmethod  # 抽象基类和抽象方法
from typing import Any, Optional    # 类型注解

import hydra  # 配置管理库
import numpy as np  # 数组库
import ray  # 分布式计算框架
import torch  # 深度学习库
from cachetools import LRUCache  # LRU 缓存工具
from omegaconf import DictConfig, OmegaConf  # 配置管理库
from pydantic import BaseModel, ConfigDict  # 数据校验库
from tensordict import TensorDict  # 张量字典工具
from transformers import AutoProcessor, AutoTokenizer  # transformers 库的自动处理器和分词器

from verl.protocol import DataProto  # verl 协议相关的数据结构
from verl.single_controller.ray.base import RayWorkerGroup  # ray worker 组管理类
from verl.trainer.ppo.reward import load_reward_manager  # PPO 奖励管理器加载函数
from verl.utils import hf_processor, hf_tokenizer  # huggingface 处理器和分词器工具
from verl.utils.fs import copy_to_local  # 文件系统工具函数
from verl.utils.model import compute_position_id_with_mask  # 模型相关工具函数
from verl.utils.rollout_trace import RolloutTraceConfig, rollout_trace_attr, rollout_trace_op  # rollout trace 相关工具和装饰器
from verl.workers.rollout.async_server import TokenOutput, async_server_class  # 异步服务相关类和输出结构

logger = logging.getLogger(__file__)  # 获取当前文件的日志记录器
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))  # 设置日志级别，默认 WARN，可通过环境变量修改


class AsyncLLMServerManager:
    """
    管理多个 OpenAI 兼容的 LLM 服务器的类。
    主要功能：
    - 负载均衡：最少请求负载均衡
    - 粘性会话：多轮对话发送到同一服务器，实现自动前缀缓存
    """

    def __init__(self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], max_cache_size: int = 10000):
        """初始化 AsyncLLMServerManager。

        参数:
            config (DictConfig): YAML 配置文件
            server_handles (List[ray.actor.ActorHandle]): OpenAI 兼容的 LLM 服务器 actor 句柄列表
            max_cache_size (int, optional): request_id 到服务器映射的最大缓存数，默认 10000
        """
        self.config = config  # 保存配置
        self.server_handles = server_handles  # 保存服务器句柄列表
        random.shuffle(self.server_handles)  # 随机打乱服务器列表，避免总是选第一个

        # 最少请求负载均衡
        self.weighted_serveres = [[0, (hash(server), server)] for server in server_handles]  # 创建一个带权重的服务器列表，初始权重为 0
        heapq.heapify(self.weighted_serveres)  # 转为堆队列，方便负载均衡

        # LRU 缓存，用于 request_id 到服务器的映射
        self.request_id_to_server = LRUCache(maxsize=max_cache_size)  # 创建 LRU 缓存，最大容量 max_cache_size

    def _choose_server(self, request_id: str) -> ray.actor.ActorHandle:
        # TODO: 实现服务器压力感知的负载均衡
        if request_id in self.request_id_to_server:
            # 如果 request_id 已有映射，直接返回对应服务器
            return self.request_id_to_server[request_id]

        server = self.weighted_serveres[0][1][1]  # 选择权重最小的服务器
        self.weighted_serveres[0][0] += 1  # 增加该服务器的权重
        heapq.heapreplace(self.weighted_serveres, self.weighted_serveres[0])  # 更新堆队列
        self.request_id_to_server[request_id] = server  # 将 request_id 映射到该服务器
        return server  # 返回选中的服务器

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        """根据 prompt ids 生成 token。

        参数:
            request_id (str): 用于粘性会话的请求 id
            prompt_ids (List[int]): prompt 的 token id 列表
            sampling_params (Dict[str, Any]): 采样参数
            image_data (Optional[List[Any]]): 可选的图片数据

        返回:
            TokenOutput: 生成的 token 输出
        """
        server = self._choose_server(request_id)  # 选择服务器
        output = await server.generate.remote(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=image_data,
        )  # 异步调用服务器生成 token
        return output  # 返回生成结果


class AgentLoopMetrics(BaseModel):
    """Agent loop 性能指标类。"""

    generate_sequences: float = 0.0  # 生成序列的耗时
    tool_calls: float = 0.0  # 工具调用的耗时


class AgentLoopOutput(BaseModel):
    """Agent loop 输出结构。"""

    prompt_ids: list[int]  # prompt 的 token id 列表
    response_ids: list[int]  # response 的 token id，包括 LLM 生成和工具响应
    response_mask: list[int]  # response mask，1 表示 LLM 生成，0 表示工具响应
    response_logprobs: Optional[list[float]] = None  # response token 的 log 概率
    multi_modal_data: Optional[dict[str, Any]] = None  # 多模态工具的数据
    reward_score: Optional[float] = None  # 轨迹的奖励分数
    num_turns: int = 0  # 对话轮数，包括用户、助手、工具
    metrics: AgentLoopMetrics  # 辅助性能指标


class _InternalAgentLoopOutput(AgentLoopOutput):
    """带有填充序列的内部 Agent loop 输出结构。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)  # 允许任意类型

    prompt_ids: torch.Tensor  # 填充后的 prompt token id
    response_ids: torch.Tensor  # 填充后的 response token id
    input_ids: torch.Tensor  # 填充后的输入 id（prompt + response）
    position_ids: torch.Tensor  # 填充后的位置 id
    response_mask: torch.Tensor  # 填充后的 response mask
    attention_mask: torch.Tensor  # 填充后的 attention mask
    response_logprobs: Optional[torch.Tensor] = None  # 填充后的 log 概率
    multi_modal_inputs: Optional[dict[str, torch.Tensor]] = None  # 多模态输入（如图片张量）


# 让 hydra.utils.instantiate 能正常工作
class _DummyConfig:
    def __init__(self, config: DictConfig) -> None:
        self.config = config  # 保存配置


class AgentLoopBase(ABC):
    """Agent loop 基类，负责与 OpenAI 兼容的 LLM 服务器和环境交互。"""

    _class_initialized = False  # 类初始化标志

    def __init__(
        self,
        trainer_config: _DummyConfig,
        server_manager: AsyncLLMServerManager,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        **kwargs,
    ):
        """初始化 agent loop，每个样本有自己的 loop 实例。

        参数:
            trainer_config (_DummyConfig): 训练器配置
            server_manager (AsyncLLMServerManager): LLM 服务器管理器
            tokenizer (AutoTokenizer): 分词器
            processor (AutoProcessor): 处理器
        """
        self.init_class(config=trainer_config.config, tokenizer=tokenizer, processor=processor, **kwargs)  # 类级初始化
        self.config = trainer_config.config  # 保存配置
        self.server_manager = server_manager  # 保存服务器管理器
        self.tokenizer = tokenizer  # 保存分词器
        self.processor = processor  # 保存处理器
        self.loop = asyncio.get_running_loop()  # 获取当前异步事件循环

    @classmethod
    def init_class(cls, config: DictConfig, tokenizer: AutoTokenizer, processor: AutoProcessor, **kwargs):
        """用于做只需一次的重初始化工作，所有实例共享。

        参数:
            config (DictConfig): 训练器配置
            tokenizer (AutoTokenizer): 分词器
            processor (AutoProcessor): 处理器
            **kwargs: 其他配置参数
        """
        if cls._class_initialized:
            return  # 如果已经初始化过，直接返回
        cls._class_initialized = True  # 标记为已初始化

    @abstractmethod
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """运行 agent loop，与 LLM 服务器和环境交互。

        参数:
            sampling_params (Dict[str, Any]): LLM 采样参数
            **kwargs: 数据集字段

        返回:
            AgentLoopOutput: Agent loop 输出
        """
        raise NotImplementedError


"""Agent loop registry: key is agent_name, value is a dict of agent loop config
used by hydra.utils.instantiate to initialize agent loop instance.

https://hydra.cc/docs/advanced/instantiate_objects/overview/
"""
_agent_loop_registry: dict[str, dict] = {}


def register(agent_name: str):
    """注册 agent loop 类。

    参数:
        agent_name (str): agent 名称
    """

    def decorator(subclass: type[AgentLoopBase]) -> type[AgentLoopBase]:
        fqdn = f"{subclass.__module__}.{subclass.__qualname__}"
        _agent_loop_registry[agent_name] = {"_target_": fqdn}  # 将 agent 名称和类的完全限定名注册到字典
        return subclass

    return decorator


@ray.remote(num_cpus=1)
class RewardManagerWorker:
    """奖励管理器工作线程，异步计算奖励分数，与 agent loop 重叠执行。"""

    def __init__(self, config: DictConfig, local_path: str) -> None:
        tokenizer = hf_tokenizer(local_path, trust_remote_code=True)  # 加载分词器
        self.reward_manager = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )  # 加载奖励管理器
        self.loop = asyncio.get_event_loop()  # 获取事件循环

    async def compute_score(self, output: AgentLoopOutput, kwargs: dict) -> float:
        """计算 agent loop 输出的奖励分数。

        注意：由于 `reward_manager.__call__` 是阻塞函数，因此在线程池中运行它，以便并行计算多个样本。

        参数:
            output (AgentLoopOutput): Agent loop 输出
            kwargs (dict): 数据集字段

        返回:
            float: 奖励分数
        """
        prompts = torch.tensor(output.prompt_ids, dtype=torch.long).unsqueeze(0)  # 转为张量并增加一个维度
        responses = torch.tensor(output.response_ids, dtype=torch.long).unsqueeze(0)  # 转为张量并增加一个维度
        attention_mask = torch.ones((1, prompts.shape[1] + responses.shape[1]), dtype=torch.long)  # 全 1 的 attention mask
        batch = TensorDict(
            {
                "prompts": prompts,  # [1, prompt_length]
                "responses": responses,  # [1, response_length]
                "attention_mask": attention_mask,  # [1, prompt_length + response_length]
            },
            batch_size=1,
        )
        non_tensor_batch = {
            **{k: np.array([v]) for k, v in kwargs.items()},
            "__num_turns__": np.array([output.num_turns]),
        }
        data = DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
        )
        reward_tensor = await self.loop.run_in_executor(
            None,
            self.reward_manager,
            data,
        )  # 在线程池中运行奖励管理器
        return reward_tensor.sum(dim=-1).item()  # 返回奖励分数


@ray.remote
class AgentLoopWorker:
    """Agent loop 工作线程，处理一批消息并在 agent loop 中运行每个消息。"""

    def __init__(self, config: DictConfig, server_handles: list[ray.actor.ActorHandle]):
        """初始化 agent loop 管理器。

        参数:
            config (DictConfig): YAML 配置
            server_handles (List[ray.actor.ActorHandle]): OpenAI 兼容的 LLM 服务器 actor 句柄列表
        """
        self.config = config  # 保存配置
        self.server_manager = AsyncLLMServerManager(config, server_handles)  # 初始化服务器管理器

        model_path = config.actor_rollout_ref.model.path  # 模型路径
        self.model_name = "/".join(model_path.split("/")[-2:])  # 模型名称
        local_path = copy_to_local(config.actor_rollout_ref.model.path)  # 复制模型到本地
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)  # 加载分词器
        self.processor = hf_processor(local_path, trust_remote_code=True)  # 加载处理器

        agent_loop_config_path = config.actor_rollout_ref.rollout.agent.agent_loop_config_path  # agent loop 配置路径
        if agent_loop_config_path:
            agent_loop_configs = OmegaConf.load(agent_loop_config_path)  # 加载 agent loop 配置
            for agent_loop_config in agent_loop_configs:
                _agent_loop_registry[agent_loop_config.name] = agent_loop_config  # 注册 agent loop 配置
        if self.config.actor_rollout_ref.model.get("custom_chat_template", None) is not None:
            # 如果有自定义聊天模板，设置分词器和处理器的聊天模板
            if self.processor is not None:
                self.processor.chat_template = self.config.actor_rollout_ref.model.custom_chat_template
            self.tokenizer.chat_template = self.config.actor_rollout_ref.model.custom_chat_template

        self.reward_manager_worker = RewardManagerWorker.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(),
                soft=False,
            ),
        ).remote(self.config, local_path)  # 初始化奖励管理器工作线程

        trace_config = self.config.actor_rollout_ref.rollout.get("trace", {})
        RolloutTraceConfig.init(
            self.config.trainer.project_name,
            self.config.trainer.experiment_name,
            trace_config.get("backend"),
            trace_config.get("token2text", False),
        )  # 初始化 RolloutTraceConfig

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """通过 agent loop 生成序列。

        参数:
            batch (DataProto): 输入批次

        返回:
            DataProto: 输出批次，包含生成的序列和其他信息
        """
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )  # 采样参数

        # 验证时覆盖采样参数
        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["temperature"] = config.val_kwargs.temperature

        # 默认认为是单轮对话 agent
        if "agent_name" not in batch.non_tensor_batch:
            batch.non_tensor_batch["agent_name"] = np.array(["single_turn_agent"] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index.tolist(), batch.meta_info.get("validate", False)
        )  # 获取轨迹信息

        tasks = []
        for i in range(len(batch)):
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            # 为每个样本创建异步任务
            tasks.append(asyncio.create_task(self._run_agent_loop(sampling_params, trajectory_info[i], **kwargs)))
        outputs = await asyncio.gather(*tasks)  # 等待所有任务完成

        output = self._postprocess(outputs)  # 后处理
        return output

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        **kwargs,
    ) -> _InternalAgentLoopOutput:
        """运行单个 agent loop。

        参数:
            sampling_params (Dict[str, Any]): LLM 采样参数
            trajectory (Dict[str, Any]): 轨迹信息
            agent_name (str): agent 名称
            **kwargs: 其他参数

        返回:
            _InternalAgentLoopOutput: 内部输出，包括填充后的序列和其他信息
        """
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
        ):
            assert agent_name in _agent_loop_registry, (
                f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
            )

            agent_loop_config = _agent_loop_registry[agent_name]
            agent_loop = hydra.utils.instantiate(
                config=agent_loop_config,
                trainer_config=_DummyConfig(config=self.config),
                server_manager=self.server_manager,
                tokenizer=self.tokenizer,
                processor=self.processor,
            )  # 实例化 agent loop
            output: AgentLoopOutput = await agent_loop.run(sampling_params, **kwargs)  # 运行 agent loop

            # 有些 AgentLoop 可能已经计算了奖励分数，例如 SWE-agent。
            if output.reward_score is None and not self.config.reward_model.enable:
                # 如果奖励分数未计算且奖励模型未启用，使用奖励管理器计算分数
                output.reward_score = await self.reward_manager_worker.compute_score.remote(output, kwargs)

            # NOTE: 与 vllm_rollout_spmd.py 中的 generate_sequences 批处理版本保持一致
            # prompt_ids: 左侧用零填充 (e.g., [0,0,0,0,1,2,3,4])
            # response_ids: 右侧用零填充 (e.g., [5,6,7,8,0,0,0,0])
            # input_ids: prompt + response 的连接
            # Mask:
            # 例如，如果提示是 [1,2,3,4] 而响应是 [5,6,7,(tool start)8,9(tool end),10,11,12]
            # - prompt_attention_mask: 填充为 0，真实 token 为 1
            #   e.g., [0,0,0,0,1,1,1,1]
            # - response_attention_mask: 填充为 0，真实 token 为 1
            #   e.g., [1,1,1,1,1,1,1,1,1,1,1,0,0,0,0]
            # attention_mask: prompt 和 response attention_mask 的连接
            #   e.g., [0,0,0,0,1,1,1,1(prompt),1,1,1,1,1,1,1,1,1,1,1,0,0,0,0(response)]
            # - response_mask: LLM 生成的 token 为 1，工具响应/填充 token 为 0
            #   e.g., [1,1,1,1,1,1,1,(tool start),0,0(tool end),1,1,0,0,0,0]
            # - position_ids: 真实 token 的顺序位置，从 0 开始
            #   e.g., [0,0,0,0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,0,0,0,0]

            self.tokenizer.padding_side = "left"
            prompt_output = self.tokenizer.pad(
                {"input_ids": output.prompt_ids},
                padding="max_length",
                max_length=self.config.actor_rollout_ref.rollout.prompt_length,
                return_tensors="pt",
                return_attention_mask=True,
            )  # 填充 prompt_ids
            if prompt_output["input_ids"].dim() == 1:
                prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
                prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

            self.tokenizer.padding_side = "right"
            response_output = self.tokenizer.pad(
                {"input_ids": output.response_ids},
                padding="max_length",
                max_length=self.config.actor_rollout_ref.rollout.response_length,
                return_tensors="pt",
                return_attention_mask=True,
            )  # 填充 response_ids
            if response_output["input_ids"].dim() == 1:
                response_output["input_ids"] = response_output["input_ids"].unsqueeze(0)
                response_output["attention_mask"] = response_output["attention_mask"].unsqueeze(0)

            response_mask_output = self.tokenizer.pad(
                {"input_ids": output.response_mask},
                padding="max_length",
                max_length=self.config.actor_rollout_ref.rollout.response_length,
                return_tensors="pt",
                return_attention_mask=False,
            )  # 填充 response_mask
            if response_mask_output["input_ids"].dim() == 1:
                response_mask_output["input_ids"] = response_mask_output["input_ids"].unsqueeze(0)

            response_logprobs = None
            if output.response_logprobs is not None:
                pad_size = self.config.actor_rollout_ref.rollout.response_length - len(output.response_logprobs)
                response_logprobs = torch.tensor(output.response_logprobs + [0.0] * pad_size).unsqueeze(0)

            response_mask = response_mask_output["input_ids"] * response_output["attention_mask"]
            attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)
            input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)

            # 处理多模态输入和 position_ids 计算
            # 目前仅支持 Qwen2VLImageProcessor 进行多模态处理
            # TODO: 支持其他多模态输入
            multi_modal_inputs = None
            if (
                self.processor is not None
                and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__
            ):
                from verl.models.transformers.qwen2_vl import get_rope_index

                images = output.multi_modal_data.get("image", None)
                current_text = self.tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)
                multi_modal_inputs = self.processor(text=[current_text], images=images, return_tensors="pt")
                multi_modal_inputs.pop("input_ids", None)
                multi_modal_inputs.pop("attention_mask", None)

                # We must use dict(multi_modal_inputs) to convert BatchFeature values to a new dict
                # because np.array() only keeps the keys for BatchFeature.
                multi_modal_inputs = dict(multi_modal_inputs)

                image_grid_thw = multi_modal_inputs.get("image_grid_thw")
                video_grid_thw = multi_modal_inputs.get("video_grid_thw")
                second_per_grid_ts = multi_modal_inputs.get("second_per_grid_ts")

                position_ids = get_rope_index(
                    self.processor,
                    input_ids=input_ids.squeeze(0),
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask.squeeze(0),
                ).unsqueeze(0)  # (1, 3, seq_len)
            else:
                position_ids = compute_position_id_with_mask(attention_mask)  # (1, seq_len)

            return _InternalAgentLoopOutput(
                prompt_ids=prompt_output["input_ids"],
                response_ids=response_output["input_ids"],
                input_ids=input_ids,
                position_ids=position_ids,
                response_mask=response_mask,
                attention_mask=attention_mask,
                response_logprobs=response_logprobs,
                multi_modal_inputs=multi_modal_inputs,
                multi_modal_data=output.multi_modal_data,
                reward_score=output.reward_score,
                num_turns=output.num_turns,
                metrics=output.metrics,
            )

    def _postprocess(self, inputs: list[_InternalAgentLoopOutput]) -> DataProto:
        """处理来自 _run_agent_loop 的填充输出，并将它们合并为一个批次。"""
        # 将列表转换回张量并堆叠以创建一个批次。
        prompt_ids = torch.cat([input.prompt_ids for input in inputs], dim=0)
        response_ids = torch.cat([input.response_ids for input in inputs], dim=0)
        response_mask = torch.cat([input.response_mask for input in inputs], dim=0)
        attention_mask = torch.cat([input.attention_mask for input in inputs], dim=0)
        input_ids = torch.cat([input.input_ids for input in inputs], dim=0)
        position_ids = torch.cat([input.position_ids for input in inputs], dim=0)
        optional_outputs = {}
        if inputs[0].response_logprobs is not None:
            optional_outputs["rollout_log_probs"] = torch.cat([input.response_logprobs for input in inputs], dim=0)

        batch = TensorDict(
            {
                "prompts": prompt_ids,  # [bsz, prompt_length]
                "responses": response_ids,  # [bsz, response_length]
                "response_mask": response_mask,  # [bsz, response_length]
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                # position_ids: [bsz, 3, prompt_length + response_length] or [bsz, prompt_length + response_length]
                "position_ids": position_ids,
                **optional_outputs,
            },
            batch_size=len(inputs),
        )

        scores = [input.reward_score for input in inputs]
        if all(score is not None for score in scores):
            prompt_length = prompt_ids.size(1)
            response_length = attention_mask[:, prompt_length:].sum(dim=1) - 1
            rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
            rm_scores[torch.arange(response_mask.size(0)), response_length] = torch.tensor(scores, dtype=torch.float32)
            batch["rm_scores"] = rm_scores

        non_tensor_batch = {
            "__num_turns__": np.array([input.num_turns for input in inputs], dtype=np.int32),
        }

        # 如果有样本包含多模态输入，则将 multi_modal_inputs 添加到 non_tensor_batch
        multi_modal_inputs_list = [input.multi_modal_inputs for input in inputs]
        if any(mmi is not None for mmi in multi_modal_inputs_list):
            non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs_list, dtype=object)

        metrics = [input.metrics.model_dump() for input in inputs]
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info={"metrics": metrics})


async def get_trajectory_info(step, index, validate):
    """获取轨迹信息。

    参数:
        step (int): 全局步骤
        index (list): 数据存储索引
        validate (bool): 是否为验证步骤

    返回:
        list: 轨迹信息
    """
    trajectory_info = []
    rollout_n = 0
    for i in range(len(index)):
        if i > 0 and index[i - 1] == index[i]:
            rollout_n += 1
        else:
            rollout_n = 0
        trajectory_info.append({"step": step, "sample_index": index[i], "rollout_n": rollout_n, "validate": validate})
    return trajectory_info


class AgentLoopManager:
    """
    Agent loop 管理器，管理一组 agent loop 工作线程。
    主要功能：
    - 初始化和管理分布式 LLM 服务器
    - 初始化和分配 agent loop worker
    - 批量分发推理任务，收集和合并结果
    - 统计性能指标，支持唤醒/休眠服务器
    """

    def __init__(self, config: DictConfig, worker_group: RayWorkerGroup):
        """
        初始化 agent loop 管理器。
        参数:
            config (DictConfig): 训练器配置
            worker_group (RayWorkerGroup): ActorRolloutRef 工作组
        """
        self.config = config  # 保存配置
        self.worker_group = worker_group  # 保存工作组

        self._initialize_llm_servers()  # 初始化 LLM 服务器（分布式推理服务）
        self._init_agent_loop_workers()  # 初始化 agent loop 工作线程（分布式推理 worker）

        self.sleep()  # 启动后默认让所有服务器休眠，节省资源

    def _initialize_llm_servers(self):
        """
        初始化 LLM 服务器。
        包括分布式推理的 TP/DP 参数计算、节点分配、服务器实例启动与重试、引擎初始化。
        """
        self.rollout_tp_size = self.config.actor_rollout_ref.rollout.tensor_model_parallel_size  # 张量并行数
        self.rollout_dp_size = self.worker_group.world_size // self.rollout_tp_size  # 数据并行数

        workers_info = ray.get(
            [
                worker.__ray_call__.remote(lambda self: ray.get_runtime_context().get_node_id())
                for worker in self.worker_group.workers
            ]
        )  # 获取所有 worker 的 node_id
        assert len(workers_info) == self.worker_group.world_size  # 校验 worker 数量

        self.async_llm_servers = [None] * self.rollout_dp_size  # 初始化服务器句柄列表
        self.server_addresses = [None] * self.rollout_dp_size  # 初始化服务器地址列表

        # 判断是否使用自定义异步服务器类
        if self.config.actor_rollout_ref.rollout.agent.custom_async_server:
            server_class = async_server_class(
                rollout_backend=self.config.actor_rollout_ref.rollout.name,
                rollout_backend_module=self.config.actor_rollout_ref.rollout.agent.custom_async_server.path,
                rollout_backend_class=self.config.actor_rollout_ref.rollout.agent.custom_async_server.name,
            )
        else:
            server_class = async_server_class(rollout_backend=self.config.actor_rollout_ref.rollout.name)

        # 启动所有服务器实例，如果地址已被占用则重启。
        unready_dp_ranks = set(range(self.rollout_dp_size))  # 未就绪的 DP rank 集合
        while len(unready_dp_ranks) > 0:
            servers = {
                rollout_dp_rank: server_class.options(
                    # 确保 AsyncvLLMServer 与其对应的工作线程共址
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=workers_info[rollout_dp_rank * self.rollout_tp_size],
                        # 通过 TP/DP 计算，找到该 DP rank 对应的物理节点 id
                        soft=False,  # 强约束，必须分配到指定节点
                    ),
                    name=f"async_llm_server_{rollout_dp_rank}",  # 给每个服务器实例命名，方便管理和调试
                ).remote(self.config, self.rollout_dp_size, rollout_dp_rank, self.worker_group.name_prefix)
                # 启动远程 Ray actor，传递必要参数
                for rollout_dp_rank in unready_dp_ranks  # 对所有未就绪的 DP rank 批量启动服务器
            }

            for rollout_dp_rank, server in servers.items():
                try:
                    address = ray.get(server.get_server_address.remote())  # 获取服务器地址
                    self.server_addresses[rollout_dp_rank] = address  # 保存地址
                    self.async_llm_servers[rollout_dp_rank] = server  # 保存服务器句柄
                    unready_dp_ranks.remove(rollout_dp_rank)  # 标记为已就绪
                except Exception:
                    ray.kill(server)  # 启动失败则杀死重启
                    print(f"rollout server {rollout_dp_rank} failed, maybe address already in use, restarting...")

        # 所有服务器实例均已准备就绪，初始化 AsyncLLM 引擎。
        ray.get([server.init_engine.remote() for server in self.async_llm_servers])  # 初始化引擎

    def _init_agent_loop_workers(self):
        """
        初始化 agent loop 工作线程。
        按 worker 数量轮询分配到所有可用节点，创建 AgentLoopWorker 实例。
        """
        self.agent_loop_workers = []  # 工作线程列表
        num_workers = self.config.actor_rollout_ref.rollout.agent.num_workers  # worker 数量

        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]  # 获取所有可用节点 id
        for i in range(num_workers):
            node_id = node_ids[i % len(node_ids)]  # 轮询分配节点
            self.agent_loop_workers.append(
                AgentLoopWorker.options(
                    name=f"agent_loop_worker_{i}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=True
                    ),
                ).remote(self.config, self.async_llm_servers)
            )  # 创建并启动远程 worker

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """
        拆分输入批次并分发给 agent loop 工作线程。
        支持推理前唤醒服务器，推理后休眠服务器，批量收集和合并结果，统计性能指标。
        参数:
            prompts (DataProto): 输入批次
        返回:
            DataProto: 输出批次
        """
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.wake_up()  # 推理前唤醒服务器
        chunkes = prompts.chunk(len(self.agent_loop_workers))  # 按 worker 数量拆分输入
        outputs = ray.get(
            [
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
            ]
        )  # 并发分发到各 worker 并收集结果
        output = DataProto.concat(outputs)  # 合并所有 worker 的输出
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()  # 推理后休眠服务器

        metrics = [output.meta_info["metrics"] for output in outputs]  # 收集所有样本的性能指标
        timing = self._performance_metrics(metrics, output)  # 统计性能指标

        output.meta_info = {"timing": timing}  # 写入 meta_info
        return output  # 返回最终批次

    def _performance_metrics(self, metrics: list[list[dict[str, str]]], output: DataProto) -> dict[str, float]:
        """
        计算性能指标。
        包括所有样本的生成耗时、工具调用耗时，统计 min/max/mean，记录最慢样本详细信息。
        """
        timing = {}
        t_generate_sequences = np.array([metric["generate_sequences"] for chunk in metrics for metric in chunk])  # 生成耗时
        t_tool_calls = np.array([metric["tool_calls"] for chunk in metrics for metric in chunk])  # 工具调用耗时
        timing["agent_loop/generate_sequences/min"] = t_generate_sequences.min()
        timing["agent_loop/generate_sequences/max"] = t_generate_sequences.max()
        timing["agent_loop/generate_sequences/mean"] = t_generate_sequences.mean()
        timing["agent_loop/tool_calls/min"] = t_tool_calls.min()
        timing["agent_loop/tool_calls/max"] = t_tool_calls.max()
        timing["agent_loop/tool_calls/mean"] = t_tool_calls.mean()

        slowest = np.argmax(t_generate_sequences + t_tool_calls)  # 找出最慢样本
        attention_mask = output.batch["attention_mask"][slowest]
        prompt_length = output.batch["prompts"].shape[1]
        timing["agent_loop/slowest/generate_sequences"] = t_generate_sequences[slowest]
        timing["agent_loop/slowest/tool_calls"] = t_tool_calls[slowest]
        timing["agent_loop/slowest/prompt_length"] = attention_mask[:prompt_length].sum().item()
        timing["agent_loop/slowest/response_length"] = attention_mask[prompt_length:].sum().item()

        return timing  # 返回性能统计结果

    def wake_up(self):
        """
        唤醒所有 rollout 服务器实例。
        """
        ray.get([server.wake_up.remote() for server in self.async_llm_servers])  # 批量唤醒

    def sleep(self):
        """
        使所有 rollout 服务器实例休眠。
        """
        ray.get([server.sleep.remote() for server in self.async_llm_servers])  # 批量休眠
