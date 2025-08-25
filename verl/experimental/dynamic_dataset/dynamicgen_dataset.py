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
"""
Dataset class that enables dynamic data generation strategies between iterations of training.
This class extends RLHFDataset and uses an AbstractDataGen instance to generate data.

This is especially useful in settings where proposer model generates new tasks based
on rollout data.
"""

import logging  # 日志模块
from abc import ABC, abstractmethod  # 抽象基类和抽象方法
from typing import Optional  # 类型注解

import datasets  # HuggingFace datasets 库
from omegaconf import DictConfig  # 配置管理库
from torch.utils.data import Dataset  # PyTorch 数据集基类
from transformers import PreTrainedTokenizer, ProcessorMixin  # 分词器和处理器基类

from verl import DataProto  # verl 数据协议
from verl.utils.dataset import RLHFDataset  # RLHF 数据集基类
from verl.utils.import_utils import load_extern_type  # 动态加载外部类型工具

logger = logging.getLogger(__name__)  # 获取当前模块日志记录器


class AbstractDataGenerator(ABC):
    def __init__(self, config: DictConfig):
        self.config = config  # 保存配置

    @abstractmethod
    def generate(self, dataset: Dataset) -> datasets.Dataset:
        """
        子类必须实现 generate 方法。
        参数:
            dataset: 用于生成新数据的数据集。
        返回:
            由子类实现的处理结果。
        """
        pass  # 抽象方法，无实现


class MockDataGenerator(AbstractDataGenerator):
    """
    一个无操作的数据生成器，仅重新添加第一个数据点。
    适用于占位和测试。
    """

    def __init__(self, config: DictConfig = None):
        super().__init__(config)

    def generate(self, dataset: Dataset) -> datasets.Dataset:
        print("MockDataGenerator: No operation performed on the dataset.")
        return dataset.dataframe.select([0])  # 仅返回第一个数据点


class DynamicGenDataset(RLHFDataset):
    """
    使用数据生成策略处理数据的数据集类。
    继承 RLHFDataset，使用 AbstractDataGen 实例生成数据。
    """

    def __init__(
        self,
        data_files: str | list[str],  # 数据文件路径或列表
        tokenizer: PreTrainedTokenizer,  # 分词器
        config: DictConfig,  # 配置
        processor: Optional[ProcessorMixin] = None,  # 可选处理器
    ):
        super().__init__(data_files, tokenizer, config, processor)  # 初始化父类
        self.datagen: AbstractDataGenerator = config.datagen  # 数据生成器配置
        assert "datagen" in config and config.datagen.get("path", None) is not None, (
            f"datagen path is not set in config: {config}"
        )  # 检查 datagen 路径
        # 动态加载自定义数据生成器类
        datagen_cls = load_extern_type(config.datagen.path, config.datagen.name)

        # 校验自定义数据生成器类是否继承 AbstractDataGenerator
        abs_cls = AbstractDataGenerator
        if not issubclass(datagen_cls, abs_cls):
            raise TypeError(
                f"The custom datagen class '{config.datagen.name}' from '{config.datagen.path}'"
                + " must inherit from {abs_cls}"
            )

        self.data_generator = datagen_cls(config.datagen)  # 实例化数据生成器
        self.on_batch_end()  # 初始化时生成一次数据

    def append_dataframe(self, new_dataframe: datasets.Dataset):
        new_dataframe = self.maybe_filter_out_long_prompts(new_dataframe)  # 过滤过长的 prompt
        self.dataframe = datasets.concatenate_datasets([self.dataframe, new_dataframe])  # 合并数据集

        logger.info(f"new dataset len: {len(self.dataframe)}")  # 打印新数据集长度

    def on_batch_end(self, batch: DataProto) -> None:
        """
        使用数据生成策略生成新数据。
        注意：该方法用于每个训练 batch 结束后动态改变数据集。
        """
        new_data = self.data_generator.generate(self)  # 生成新数据
        self.append_dataframe(new_data)  # 添加到数据集
