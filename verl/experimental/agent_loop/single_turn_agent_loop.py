"""
single_turn_agent_loop.py

本文件实现了单轮对话 Agent Loop（SingleTurnAgentLoop），用于处理只包含一轮问答的场景。
主要功能：
- 继承 AgentLoopBase，实现 run 方法，处理单轮对话输入与输出
- 适合初学者理解最基础的 LLM 推理流程
- 包含性能计时、唯一请求 id 生成等基础功能

适合初学者快速上手 Agent Loop 的最简用法。
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
import logging  # 日志模块
import os       # 操作系统相关模块
from typing import Any  # 类型注解
from uuid import uuid4  # 生成唯一请求 id

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register  # 导入基类和注册器
from verl.utils.profiler import simple_timer  # 性能计时工具

logger = logging.getLogger(__file__)  # 获取当前文件的日志记录器
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))  # 设置日志级别


@register("single_turn_agent")  # 注册为单轮对话 agent
class SingleTurnAgentLoop(AgentLoopBase):
    """只做单轮对话的简单 agent loop。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # 调用基类初始化
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length  # prompt 最大长度
        self.response_length = self.config.actor_rollout_ref.rollout.response_length  # response 最大长度
        self.apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})  # 聊天模板参数

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])  # 获取原始对话消息

        metrics = {}  # 性能指标
        request_id = uuid4().hex  # 生成唯一请求 id
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
            ),
        )  # 异步生成 prompt 的 token id

        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params
            )  # 异步生成 response
        response_mask = [1] * len(output.token_ids)  # 响应全部为 LLM 生成

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,  # prompt 的 token id
            response_ids=output.token_ids[: self.response_length],  # response 的 token id，截断到最大长度
            response_mask=response_mask[: self.response_length],  # 响应 mask
            response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,  # log 概率
            multi_modal_data={},  # 多模态数据，单轮对话为空
            num_turns=2,  # 对话轮数，用户+助手
            metrics=metrics,  # 性能指标
        )
        return output  # 返回输出结构
