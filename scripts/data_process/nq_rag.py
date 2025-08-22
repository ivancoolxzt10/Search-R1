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
Preprocess the nq dataset to parquet format
NQ数据集预处理为RAG任务格式，拼接检索上下文。
"""

import re  # 正则表达式库，用于文本处理
import os  # 操作系统相关库，处理文件路径和文件操作
import json  # JSON解析库，用于读取检索缓存和语料库
import datasets  # HuggingFace数据集库，用于加载和处理公开数据集

from verl.utils.hdfs_io import copy, makedirs  # HDFS文件操作工具，copy用于文件复制，makedirs用于创建目录
import argparse  # 命令行参数解析库，解析用户输入的参数

# 功能说明：生成带有检索上下文的模型输入前缀，指导模型推理和回答。
def make_prefix(dp, template_type):
    """
    根据模板类型生成模型输入前缀，包含问题和检索到的上下文。
    参数：
        dp: dict，包含问题文本和上下文信息的数据样本
        template_type: str，模板类型（如 'base'）
    返回：
        prefix: str，拼接好的模型输入前缀
    """
    question = dp['question']  # 获取问题文本
    context = dp['context']    # 获取检索到的上下文

    # NOTE: also need to change reward_score/countdown.py
    # 注意：reward_score/countdown.py 也需同步修改
    if template_type == 'base':
        """适用于所有基础模型的输入模板"""
        # 构造模型输入前缀，指导模型如何推理和回答
        prefix = f"""Answer the given question with some potentially useful context. \
You should analyze the question carefully, evaluate the given context (which may or may not be useful), and then generate an accurate and well-reasoned response. \
You should first have a reasoning process in mind and then provides the answer. \
Show your reasoning in <think> </think> tags and return the final answer in <answer> </answer> tags, for example <answer> Beijing </answer>. \
Question: {question} Context: {context} \n"""
    else:
        raise NotImplementedError    # 其他模板类型未实现，抛出异常
    return prefix  # 返回拼接好的前缀

# 功能说明：将检索结果格式化为可读字符串，便于模型输入。
def format_reference(retrieval_result):
    """
    将检索到的文档列表格式化为字符串，包含标题和正文。
    参数：
        retrieval_result: list，检索到的文档列表，每个元素为文档字典
    返回：
        format_reference: str，格式化后的文档字符串
    """
    format_reference = ''  # 初始化格式化字符串
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item['contents']  # 获取文档内容
        title = content.split("\n")[0]  # 文档标题为内容的第一行
        text = "\n".join(content.split("\n")[1:])  # 文档正文为内容的其余部分
        # 按照"Doc {idx+1}(Title: {title}) {text}\n"格式拼接文档信息
        format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

    return format_reference  # 返回格式化后的文档字符串

# 主流程入口，负责参数解析、数据加载、检索缓存和语料库读取、样本拼接上下文。
if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # 创建参数解析器
    parser.add_argument('--local_dir', default='./data/nq_rag', help='本地数据存储路径')  # 本地数据存储路径参数
    parser.add_argument('--hdfs_dir', default=None, help='HDFS数据存储路径')  # HDFS数据存储路径参数
    parser.add_argument('--template_type', type=str, default='base', help='输入模板类型')  # 模板类型参数
    parser.add_argument('--topk', type=int, default=3, help='每个问题检索的文档数量')  # topk参数
    parser.add_argument('--corpus_path', type=str, default='/home/peterjin/mnt/data/retrieval-corpus/wiki-18.jsonl', help='语料库路径')  # 语料库路径参数
    parser.add_argument('--train_retrieval_cache', type=str, default='/home/peterjin/rag_retrieval_cache/nq/e5_train_retrieval_cache_2048.json', help='训练集检索缓存路径')  # 训练集检索缓存参数
    parser.add_argument('--test_retrieval_cache', type=str, default='/home/peterjin/rag_retrieval_cache/nq/e5_test_retrieval_cache_10000.json', help='测试集检索缓存路径')  # 测试集检索缓存参数

    args = parser.parse_args()  # 解析命令行参数

    data_source = 'nq'  # 数据源标识，固定为'nq'

    # 加载NQ数据集，包含训练集和测试集
    dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'nq')  # 加载nq数据集

    train_dataset = dataset['train']  # 获取训练集
    test_dataset = dataset['test']  # 获取测试集

    # 读取检索缓存，合并训练集和测试集检索结果
    print('reading retrieval cache...')
    retrieval_cache = json.load(open(args.train_retrieval_cache))  # 读取训练集检索缓存
    # test_retrieval_cache = json.load(open(args.test_retrieval_cache))
    retrieval_cache.update(json.load(open(args.test_retrieval_cache)))  # 合并测试集检索缓存

    # 读取语料库，构建id到文档内容的映射
    print('reading corpus...')
    corpus = {}  # 用于存储所有语料库文档，key为文档id
    with open(args.corpus_path) as f:
        readin = f.readlines()  # 逐行读取语料库文件
        for line in readin:
            tmp = json.loads(line)  # 解析每行JSON数据为字典
            corpus[tmp['id']] = tmp  # 以文档id为键存储文档数据，方便后续检索

    # 为每个数据样本添加检索到的上下文信息
    def add_context(example):
        """
        根据检索缓存为样本添加context字段，拼接topk个相关文档。
        参数：
            example: dict，原始数据样本
        返回：
            example: dict，添加了context字段的样本
        """
        # 检索缓存中存储了每个问题对应的相关文档id，取topk个文档
        example['context'] = format_reference([corpus[docs["id"]] for docs in retrieval_cache[example['question']][:args.topk]])
        return example
    
    train_dataset = train_dataset.map(function=add_context)  # 对训练集每条样本添加context字段
    test_dataset = test_dataset.map(function=add_context)  # 对测试集每条样本添加context字段

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        # 返回一个处理函数，用于格式化每条样本
        def process_fn(example, idx):
            example['question'] = example['question'].strip()  # 去除问题文本首尾空格
            if example['question'][-1] != '?':
                example['question'] += '?'  # 确保问题文本以问号结尾
            question = make_prefix(example, template_type=args.template_type)  # 构造模型输入前缀
            solution = {
                "target": example['golden_answers'],  # 标准答案
            }

            data = {
                "data_source": data_source,  # 数据来源标识
                "prompt": [{
                    "role": "user",
                    "content": question,  # 用户输入内容
                }],
                "ability": "fact-reasoning",  # 能力类型：事实推理
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution  # 奖励模型的真实标签
                },
                "extra_info": {
                    'split': split,  # 数据集划分（train/test）
                    'index': idx,  # 样本索引
                }
            }
            return data  # 返回格式化后的样本

        return process_fn  # 返回处理函数

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)  # 为训练集添加唯一id
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)  # 为测试集添加唯一id

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))  # 将训练集保存为parquet格式
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))  # 将测试集保存为parquet格式

    if hdfs_dir is not None:
        makedirs(hdfs_dir)  # 创建HDFS目标目录

        copy(src=local_dir, dst=hdfs_dir)  # 将数据从本地目录复制到HDFS
