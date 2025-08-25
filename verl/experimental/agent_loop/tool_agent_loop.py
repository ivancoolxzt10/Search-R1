"""
tool_agent_loop.py

本文件实现了工具增强型 Agent Loop（ToolAgentLoop），用于支持 LLM 结合外部工具的多轮推理。
主要功能：
- 继承 AgentLoopBase，实现 run 方法，支持工具调用、函数解析、工具响应等
- 适合初学者学习 LLM 与工具协作的推理流程
- 包含工具注册、工具调用解析、性能计时等功能

适合初学者理解 LLM+Tool 的高级推理场景。
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
import copy     # 深拷贝工具
import json     # JSON 处理模块
import logging  # 日志模块
import os       # 操作系统相关模块
from typing import Any  # 类型注解
from uuid import uuid4  # 生成唯一请求 id

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register  # 导入基类和注册器
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser  # 工具调用解析器
from verl.tools.schemas import ToolResponse  # 工具响应结构
from verl.tools.utils.tool_registry import initialize_tools_from_config  # 工具初始化函数
from verl.utils.profiler import simple_timer  # 性能计时工具
from verl.utils.rollout_trace import rollout_trace_op  # rollout trace 装饰器

logger = logging.getLogger(__file__)  # 获取当前文件的日志记录器
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))  # 设置日志级别


@register("tool_agent")  # 注册为工具 agent
class ToolAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return  # 已初始化则直接返回
        cls._class_initialized = True
        print("Performing class-level ToolAgentLoop initialization")

        # 从配置文件初始化工具
        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns  # 用户最大轮数
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns  # 助手最大轮数
        cls.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls  # 最大并行工具调用数
        cls.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length  # 工具响应最大长度
        cls.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side  # 工具响应截断方向
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path  # 工具配置路径
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []  # 初始化工具列表
        cls.tools = {tool.name: tool for tool in tool_list}  # 工具字典
        cls.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]  # 工具 schema
        cls.tool_parser = ToolParser.get_tool_parser(config.actor_rollout_ref.rollout.multi_turn.format, cls.tokenizer)  # 工具解析器
        print(f"Initialized tools: {cls.tools}")

        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})  # 聊天模板参数
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length  # prompt 最大长度
        cls.response_length = config.actor_rollout_ref.rollout.response_length  # response 最大长度
        cls.system_prompt = tokenizer.apply_chat_template(
            [{}], add_generation_prompt=False, tokenize=True, **cls.apply_chat_template_kwargs
        )  # 系统 prompt

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])  # 获取原始对话消息
        image_data = copy.deepcopy(kwargs.get("multi_modal_data", {}).get("image", None))  # 深拷贝图片数据
        metrics = {}  # 性能指标
        request_id = uuid4().hex  # 生成唯一请求 id
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    messages,
                    tools=self.tool_schemas,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )  # 处理器生成原始 prompt
            model_inputs = self.processor(text=[raw_prompt], images=image_data, return_tensors="pt")  # 处理器生成模型输入
            prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()  # 获取 prompt 的 token id
        else:
            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    messages,
                    tools=self.tool_schemas,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )  # 分词器生成 prompt 的 token id
        response_mask, response_logprobs = [], []  # 响应 mask 和 log 概率
        tools_kwargs = kwargs.get("tools_kwargs", {})  # 工具参数

        user_turns, assistant_turns = 0, 0  # 用户和助手轮数
        while True:
            with simple_timer("generate_sequences", metrics):
                output = await self.server_manager.generate(
                    request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params, image_data=image_data
                )  # 异步生成 response
            response_ids = output.token_ids  # 响应 token id
            prompt_ids += response_ids  # 拼接到 prompt
            response_mask += [1] * len(response_ids)  # 响应 mask
            if output.log_probs:
                response_logprobs += output.log_probs  # 响应 log 概率
            assistant_turns += 1  # 助手轮数加一

            # 达到最大响应长度
            if len(response_mask) >= self.response_length:
                break

            # 达到最大助手轮数
            if self.max_assistant_turns and assistant_turns >= self.max_assistant_turns:
                break

            # 达到最大用户轮数
            if self.max_user_turns and user_turns >= self.max_user_turns:
                break

            # 没有工具调用
            _, tool_calls = await self.tool_parser.extract_tool_calls(response_ids)
            if not tool_calls:
                break

            # 工具调用
            tasks = []
            for tool_call in tool_calls[: self.max_parallel_calls]:
                tasks.append(self._call_tool(tool_call, tools_kwargs))  # 异步调用工具
            with simple_timer("tool_calls", metrics):
                tool_responses = await asyncio.gather(*tasks)  # 等待所有工具响应
            if any(isinstance(item, Exception) for item in tool_responses):
                break

            # 处理工具响应消息和多模态数据
            tool_messages = []
            new_images_this_turn = []
            for tool_response in tool_responses:
                # 构建工具响应消息
                if tool_response.image or tool_response.video:
                    # 多模态内容
                    content = []
                    if tool_response.image:
                        content.append({"type": "image"})
                    if tool_response.video:
                        content.append({"type": "video"})
                    if tool_response.text:
                        content.append({"type": "text", "text": tool_response.text})
                    message = {"role": "tool", "content": content}
                else:
                    # 纯文本内容
                    message = {"role": "tool", "content": tool_response.text or ""}

                tool_messages.append(message)

                # 处理图片数据
                if tool_response.image:
                    if image_data is None:
                        image_data = []
                    elif not isinstance(image_data, list):
                        image_data = [image_data]

                    # 添加新图片数据
                    if isinstance(tool_response.image, list):
                        image_data.extend(tool_response.image)
                        new_images_this_turn.extend(tool_response.image)
                    else:
                        image_data.append(tool_response.image)
                        new_images_this_turn.append(tool_response.image)

                # 处理视频数据
                if tool_response.video:
                    # 当前不支持视频，抛出异常
                    logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                    raise NotImplementedError(
                        "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                    )

            # 拼接工具响应 token id
            if self.processor is not None:
                raw_tool_response = await self.loop.run_in_executor(
                    None,
                    lambda messages=tool_messages: self.processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
                    ),
                )
                # 只用本轮新图片处理工具响应
                current_images = new_images_this_turn if new_images_this_turn else None
                model_inputs = self.processor(text=[raw_tool_response], images=current_images, return_tensors="pt")
                tool_response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
            else:
                tool_response_ids = await self.loop.run_in_executor(
                    None,
                    lambda messages=tool_messages: self.tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
                    ),
                )
            tool_response_ids = tool_response_ids[len(self.system_prompt) :]

            # NOTE: 最后一个回合不应该是用户回合，否则 EOS token 的奖励无法在 GAE 中传播到前一个 token。
            if len(response_mask) + len(tool_response_ids) >= self.response_length:
                break

            prompt_ids += tool_response_ids
            response_mask += [0] * len(tool_response_ids)
            if response_logprobs:
                response_logprobs += [0.0] * len(tool_response_ids)
            user_turns += 1

        response_ids = prompt_ids[-len(response_mask) :]
        prompt_ids = prompt_ids[: len(prompt_ids) - len(response_mask)]

        multi_modal_data = {"image": image_data} if image_data is not None else {}

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            num_turns=user_turns + assistant_turns + 1,
            metrics=metrics,
        )
        return output

    async def _call_tool(self, tool_call: FunctionCall, tools_kwargs: dict[str, Any]) -> ToolResponse:
        """调用工具并返回工具响应."""
        tool, instance_id = None, None
        try:
            # TODO: 将格式错误的 tool_call 附加到提示中：无效的函数名或参数
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            tool = self.tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            tool_execution_response, _, _ = await tool.execute(instance_id, tool_args)
        except Exception as e:
            logger.warning(f"Error when executing tool: {e}")
            return ToolResponse(
                text=f"Error when executing tool: {e}",
            )
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        # 根据工具执行结果创建 ToolResponse
        tool_response_kwargs = {"text": tool_response_text}

        # 如果存在多媒体数据，则添加
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        return ToolResponse(**tool_response_kwargs)
