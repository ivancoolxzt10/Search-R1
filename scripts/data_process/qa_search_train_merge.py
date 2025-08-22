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
"""
Preprocess the QA dataset to parquet format
"""

import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse


# 功能说明：生成带有搜索指令的模型输入前缀，指导模型推理和搜索流程。
def make_prefix(dp, template_type):
    question = dp['question']  # 获取问题文本

    # NOTE: also need to change reward_score/countdown.py
    # 注意：reward_score/countdown.py 也需同步修改
    if template_type == 'base':
        """This works for any base model"""
        # 构造模型输入前缀，包含推理和搜索指令
        prefix = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""
    else:
        raise NotImplementedError  # 其他模板类型未实现
    return prefix  # 返回拼接好的前缀


# 功能说明：返回一个处理函数，对每条样本进行格式化，补全问号，生成模型输入和标准答案，附加分割和索引信息。
def make_map_fn(split):

    def process_fn(example, idx):
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
                'split': split,  # 数据集划分（训练集）
                'index': idx,  # 数据样本索引
            }
        }
        return data  # 返回格式化后的样本

    return process_fn  # 返回处理函数


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/nq_search')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base')
    parser.add_argument('--data_sources', default='nq')

    args = parser.parse_args()

    # data_source = 'nq'
    data_sources = args.data_sources.split(',')
    all_dataset = []

    for data_source in data_sources:

        dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', data_source)

        train_dataset = dataset['train']

        # add a row to each data item that represents a unique id
        train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)  # 对训练集应用格式化函数
        all_dataset.append(train_dataset)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    all_train_dataset = datasets.concatenate_datasets(all_dataset)  # 合并所有数据源的训练集
    train_parquet_path = os.path.join(local_dir, 'train.parquet')
    all_train_dataset.to_parquet(train_parquet_path)  # 保存合并后的训练集为parquet文件

    if hdfs_dir is not None:
        makedirs(hdfs_dir)  # 创建HDFS目标目录

        copy(src=local_dir, dst=hdfs_dir)  # 将数据从本地目录复制到HDFS

# 文件整体功能总结：
# 本脚本支持多数据源，将各类QA训练集处理为统一格式，补全问题文本，生成标准输入输出结构，
# 合并后保存为高效的parquet文件，支持分发到HDFS，便于后续模型训练。
