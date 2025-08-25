"""
gsm8k_interaction.py

本文件实现了 GSM8K 数据集的交互逻辑，适用于数学推理和问答任务。
主要功能：
- 继承 BaseInteraction，实现与 GSM8K 数据集相关的交互流程
- 支持交互实例的启动、助手响应生成、分数计算、交互结束
- 适合初学者理解 RL/LLM 交互式评测与奖励机制

每一行代码均有详细中文备注，便于初学者理解。
"""
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
# 版权声明，表明该文件归 Bytedance、SGLang Team、ModelBest 所有
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

import logging  # 日志模块
import os       # 操作系统相关模块
from typing import Any, Optional  # 类型注解
from uuid import uuid4  # 生成唯一实例 id

from verl.utils.reward_score import gsm8k  # GSM8K 奖励分数计算工具

from .base import BaseInteraction  # 导入交互基类

logger = logging.getLogger(__name__)  # 获取当前模块日志记录器
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))  # 设置日志级别


class Gsm8kInteraction(BaseInteraction):
    """
    GSM8K 数据集交互示例。
    - start_interaction: 为轨迹启动交互实例。
    - generate_response: 生成助手响应。
    - calculate_score: 计算交互分数。
    - finalize_interaction: 完成交互实例。
    """

    def __init__(self, config: dict):
        super().__init__(config)  # 初始化基类，传入配置
        self._instance_dict = {}  # 用于存储每个交互实例的状态

    async def start_interaction(
        self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs
    ) -> str:
        # 启动交互实例，记录 ground_truth
        if instance_id is None:
            instance_id = str(uuid4())  # 生成唯一实例 id
        self._instance_dict[instance_id] = {
            "response": "",  # 当前响应内容
            "ground_truth": ground_truth,  # 标准答案
            "reward": 0.0,  # 当前分数
        }
        return instance_id  # 返回实例 id

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict]:
        # 生成助手响应，并根据分数决定是否终止交互
        content = ""  # 初始化响应内容
        for i in range(len(messages) - 1, -1, -1):  # 从最后一条消息向前遍历
            item = messages[i]
            if item.get("role") == "assistant":  # 找到助手角色的消息
                content = item.get("content")  # 获取助手回复内容
                break  # 找到后跳出循环

        self._instance_dict[instance_id]["response"] = content  # 保存响应内容到实例状态

        reward = await self.calculate_score(instance_id)  # 计算分数
        if reward == 1.0:
            response = "Your response is correct!"  # 正确则提示
            should_terminate_sequence = True  # 终止交互
        else:
            response = "Your response is incorrect! You need to reflect on your answer and try again."  # 错误则提示
            should_terminate_sequence = False  # 继续交互

        return should_terminate_sequence, response, reward, {}  # 返回交互结果（是否终止、提示语、分数、附加信息）

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        # 调用 GSM8K 工具计算分数
        return gsm8k.compute_score(
            self._instance_dict[instance_id]["response"],  # 助手回复内容
            self._instance_dict[instance_id]["ground_truth"],  # 标准答案
            method="strict",  # 严格匹配
            format_score=0.0,  # 格式分数（未用）
            score=1.0,  # 满分分值
        )

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        # 删除交互实例，释放资源
        del self._instance_dict[instance_id]
