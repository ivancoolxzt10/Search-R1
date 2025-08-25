"""
tool_parser.py

本文件实现了工具调用解析相关的类和方法。
主要功能：
- 定义 FunctionCall 数据结构，规范工具调用参数
- 实现 ToolParser，负责从 LLM 输出中解析工具调用信息
- 适合初学者理解 LLM 工具调用的解析流程

适合初学者学习 LLM 工具调用的结构化解析与处理。
"""

# Copyright 2024 Bytedance Ltd. and/or its affiliates
# 版权声明，表明该文件归 Bytedance 所有
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 按照 Apache 2.0 协议授权
# you may not use this file except in compliance with the License.
# 只有遵循协议才能使用本文件
# You may obtain a copy of the License at
# 可在以下网址获取协议全文
#
#     http://www.apache.org/licenses/LICENSE-2.0
# 协议网址
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 本文件按“原样”分发
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 不提供任何明示或暗示的担保
# See the License for the specific language governing permissions and
# 具体权限请查阅协议
# limitations under the License.
# 协议中的限制条款
import asyncio  # 异步编程模块
import json     # JSON 处理模块
import logging  # 日志模块
import os       # 操作系统相关模块
from abc import ABC, abstractmethod  # 抽象基类和抽象方法

import regex as re  # 正则表达式库，支持多行匹配
from pydantic import BaseModel  # 数据校验库

from verl.utils.rollout_trace import rollout_trace_op  # rollout trace 装饰器

logger = logging.getLogger(__file__)  # 获取当前文件的日志记录器
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))  # 设置日志级别


class FunctionCall(BaseModel):
    arguments: str  # 函数调用参数，模型生成的 JSON 字符串
    """
    arguments 字段用于存储模型生成的函数调用参数，通常为 JSON 格式。
    注意：模型可能生成无效 JSON 或多余参数，实际调用前需校验。
    """

    name: str  # 函数名
    """name 字段用于存储要调用的函数名。"""


class ToolParser(ABC):
    _registry: dict[str, type["ToolParser"]] = {}  # 工具解析器注册表

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer  # 保存分词器

    @abstractmethod
    async def extract_tool_calls(self, responses_ids: list[int]) -> tuple[str, list[FunctionCall]]:
        """从响应 token id 中提取工具调用。

        参数:
            responses_ids (List[int]): 响应的 token id 列表。

        返回:
            Tuple[str, List[FunctionCall]]: 剩余文本和提取的工具调用列表。
        """
        raise NotImplementedError

    @classmethod
    def get_tool_parser(cls, name: str, tokenizer):
        if name not in cls._registry:
            raise ValueError(f"Unknown tool parser: {name}")
        return cls._registry[name](tokenizer)  # 根据名称获取工具解析器实例

    @classmethod
    def register(cls, name: str):
        def decorator(subclass: type[ToolParser]) -> type[ToolParser]:
            cls._registry[name] = subclass  # 注册工具解析器子类
            return subclass

        return decorator


@ToolParser.register("hermes")  # 注册 hermes 工具解析器
class HermesToolParser(ToolParser):
    """适配自 vllm 项目的 hermes 工具解析器。"""

    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer)

        self.tool_call_start_token: str = "<tool_call>"  # 工具调用起始标记
        self.tool_call_end_token: str = "</tool_call>"  # 工具调用结束标记
        self.tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)  # 匹配工具调用内容

    @rollout_trace_op
    async def extract_tool_calls(self, responses_ids: list[int]) -> tuple[str, list[FunctionCall]]:
        loop = asyncio.get_running_loop()  # 获取当前事件循环
        text = await loop.run_in_executor(None, self.tokenizer.decode, responses_ids)  # 异步解码 token id 为文本
        if self.tool_call_start_token not in text or self.tool_call_end_token not in text:
            return text, []  # 没有工具调用则返回原文本和空列表

        matches = self.tool_call_regex.findall(text)  # 匹配所有工具调用内容
        function_calls = []
        for match in matches:
            try:
                function_call = json.loads(match)  # 解析工具调用 JSON
                name, arguments = function_call["name"], function_call["arguments"]
                function_calls.append(FunctionCall(name=name, arguments=json.dumps(arguments, ensure_ascii=False)))  # 构造 FunctionCall
            except Exception as e:
                logger.error(f"Failed to decode tool call: {e}")  # 解析失败则记录错误

        # 剩余文本，去除工具调用内容
        content = self.tool_call_regex.sub("", text)

        return content, function_calls  # 返回剩余文本和工具调用列表
