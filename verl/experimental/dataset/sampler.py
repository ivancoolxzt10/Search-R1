# Copyright 2025 Amazon.com Inc and/or its affiliates
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
from abc import abstractmethod  # 抽象方法装饰器
from collections.abc import Sized  # 支持 len() 的类型

from omegaconf import DictConfig  # 配置管理库
from torch.utils.data import Sampler  # PyTorch 数据采样器基类

from verl import DataProto  # verl 数据协议


class AbstractSampler(Sampler[int]):
    """自定义采样器的抽象接口。"""

    @abstractmethod
    def __init__(
        self,
        data_source: Sized,  # 数据源，支持 len()
        data_config: DictConfig,  # 数据相关配置
    ):
        pass  # 抽象方法，无实现


class AbstractCurriculumSampler(AbstractSampler):
    """用于课程学习的采样器接口（实验性）。"""

    @abstractmethod
    def update(self, batch: DataProto) -> None:
        pass  # 抽象方法，更新采样策略
