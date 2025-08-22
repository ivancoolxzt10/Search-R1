# Copyright 2024 Bytedance Ltd. and/or its affiliates
# 版权声明，禁止未授权使用。
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 遵循 Apache 2.0 协议。
# you may not use this file except in compliance with the License.
# 未遵循协议不得使用本文件。
# You may obtain a copy of the License at
# 可在以下网址获取协议：
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the nq dataset to parquet format
NQ数据集预处理为parquet格式。
"""

import re  # 正则表达式库，用于文本处理
import os  # 操作系统相关库，处理文件路径和文件操作
import datasets  # HuggingFace数据集库，用于加载和处理公开数据集

from verl.utils.hdfs_io import copy, makedirs  # HDFS文件操作工具，copy用于文件复制，makedirs用于创建目录
import argparse  # 命令行参数解析库，解析用户输入的参数

# 功能说明：生成模型输入前缀，拼接问题，指导模型回答格式。
def make_prefix(dp, template_type):
    """
    根据模板类型生成模型输入前缀，指导模型如何推理和回答。
    参数：
        dp: dict，包含问题文本等信息的数据样本
        template_type: str，模板类型（如 'base'）
    返回：
        prefix: str，拼接好的模型输入前缀
    """
    question = dp['question']  # 获取问题文本

    # NOTE: also need to change reward_score/countdown.py
    # 注意：reward_score/countdown.py 也需同步修改
    if template_type == 'base':
        """适用于所有基础模型的输入模板"""
        # 构造模型输入前缀，指导模型如何推理和回答
        prefix = f"""Answer the given question. \
You should first have a reasoning process in mind and then provides the answer. \
Show your reasoning in <think> </think> tags and return the final answer in <answer> </answer> tags, for example <answer> Beijing </answer>. \
Question: {question}\n"""
    else:
        raise NotImplementedError    # 其他模板类型未实现，抛出异常
    return prefix  # 返回拼接好的前缀

# 功能说明：返回一个处理函数，对每条样本进行格式化，补全问号，生成模型输入和标准答案，附加分割和索引信息。
def make_map_fn(split):
    """
    生成用于数据集map操作的处理函数。
    参数：
        split: str，数据集划分（如 'train' 或 'test'）
    返回：
        process_fn: function，处理单条样本的函数
    """
    def process_fn(example, idx):
        """
        处理单条数据样本，补全问号，生成模型输入和标准答案，附加分割和索引信息。
        参数：
            example: dict，原始数据样本
            idx: int，样本索引
        返回：
            data: dict，格式化后的样本
        """
        example['question'] = example['question'].strip()  # 去除问题文本首尾空格
        if example['question'][-1] != '?':
            example['question'] += '?'  # 确保问题文本以问号结尾
        question = make_prefix(example, template_type=args.template_type)  # 构造模型输入前缀
        solution = {
            "target": example['golden_answers'],  # 正确答案
        }

        data = {
            "data_source": data_source,  # 数据来源标识
            "prompt": [{
                "role": "user",
                "content": question,  # 用户提问内容
            }],
            "ability": "fact-reasoning",  # 推理能力标识
            "reward_model": {
                "style": "rule",
                "ground_truth": solution  # 奖励模型的真实标签
            },
            "extra_info": {
                'split': split,  # 数据集划分（训练集/测试集）
                'index': idx,  # 数据样本索引
            }
        }
        return data  # 返回格式化后的样本

    return process_fn  # 返回处理函数

# 主流程入口，负责参数解析、数据加载、样本格式化、保存和分发。
if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # 创建参数解析器
    parser.add_argument('--local_dir', default='./data/nq', help='本地数据存储路径')  # 本地数据存储路径参数
    parser.add_argument('--hdfs_dir', default=None, help='HDFS数据存储路径')  # HDFS数据存储路径参数
    parser.add_argument('--template_type', type=str, default='base', help='输入模板类型')  # 模板类型参数

    args = parser.parse_args()  # 解析命令行参数

    data_source = 'nq'  # 数据源标识，固定为'nq'

    # 加载NQ数据集，包含训练集和测试集
    dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'nq')  # 加载nq数据集

    train_dataset = dataset['train']  # 获取训练集
    test_dataset = dataset['test']  # 获取测试集

    # 对训练集和测试集应用格式化处理函数
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)  # 训练集格式化
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)  # 测试集格式化

    local_dir = args.local_dir  # 本地存储路径
    hdfs_dir = args.hdfs_dir    # HDFS存储路径（可选）

    # 保存训练集和测试集为parquet文件，便于后续分布式处理和高效读取
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))  # 保存训练集为parquet文件
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))  # 保存测试集为parquet文件

    # 如指定HDFS路径，可将本地文件上传至HDFS（此处未实现，可根据需求补充）
    # if hdfs_dir:
    #     makedirs(hdfs_dir)
    #     copy(os.path.join(local_dir, 'train.parquet'), os.path.join(hdfs_dir, 'train.parquet'))
    #     copy(os.path.join(local_dir, 'test.parquet'), os.path.join(hdfs_dir, 'test.parquet'))
