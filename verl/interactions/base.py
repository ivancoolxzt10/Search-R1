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
# 本文件仅用于声明模块和版权，无实际代码逻辑。
from typing import Any, Optional
from uuid import uuid4


class BaseInteraction:
    def __init__(self, config: dict[str, Any]):
        """
        初始化交互基类实例

        Args:
            config: 包含配置参数的字典，至少应包含一个 "name" 键
        """
        self.config = config
        # 从配置中获取名称，默认为 "interaction_agent"
        self.name: str = config.get("name", "interaction_agent")  # More general agent default role name

    async def start_interaction(self, instance_id: Optional[str] = None, **kwargs) -> str:
        """创建一个工具实例。

        Args:
            instance_id: 工具的实例 ID，若为 None 则自动生成一个新的 ID

        Returns:
            工具的实例 ID
        """
        if instance_id is None:
            # 如果未提供实例 ID，则生成一个新的 UUID 作为实例 ID
            return str(uuid4())
        else:
            # 如果提供了实例 ID，则直接返回
            return instance_id

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict[str, Any]]:  # More clear response generation method
        """
        生成当前交互回合的响应。
        返回一个元组，包含：
        - should_terminate_sequence (bool): 如果为 True，表示交互序列应该结束。
        - response_content (str): 响应的文本内容。
        - current_turn_score (float): 当前回合/响应的分数。
        - additional_data (dict): 任何额外的信息或元数据。
        """
        should_terminate_sequence: bool = False  # if True, end rollout
        response_content: str = "Your current result seems acceptable."
        current_turn_score: float = 0.8
        additional_data: dict[str, Any] = {}
        return should_terminate_sequence, response_content, current_turn_score, additional_data

    async def calculate_score(self) -> float:  # More clear score calculation method
        """
        计算当前交互的分数，
        可能考虑部分暴露和上下文任务切换等方面。
        应在每个回合结束时调用
        """
        # ...implement the logic to calculate turn-level score...
        score = 0.0
        return score

    async def finalize_interaction(self) -> None:  # More clear interaction end and resource release method
        """
        完成交互会话并释放任何相关的状态或资源。
        模拟：释放状态
        """
        # ...implement the logic to release state...
        pass
