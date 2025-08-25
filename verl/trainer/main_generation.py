# Copyright 2024 Bytedance Ltd. and/or its affiliates  # 版权声明，标明归属和年份
#
# Licensed under the Apache License, Version 2.0 (the "License");  # 采用 Apache 2.0 开源协议
# you may not use this file except in compliance with the License.  # 使用需遵守协议条款
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
Generate responses given a dataset of prompts  # 文件功能说明：根据数据集批量生成响应
"""

import os  # 导入 os 用于环境变量和文件操作

import hydra  # 配置管理工具
import numpy as np  # 数值计算库
import ray  # 分布式任务调度

os.environ["NCCL_DEBUG"] = "WARN"  # 设置 NCCL 调试级别为 WARN，减少分布式训练时的日志噪音
os.environ["TOKENIZERS_PARALLELISM"] = "true"  # 允许 tokenizer 并行处理，加速分词
# os.environ['TORCH_COMPILE_DISABLE'] = '1'  # 可选：禁用 torch.compile

from pprint import pprint  # 美化打印

import pandas as pd  # 数据处理库
from omegaconf import OmegaConf  # 配置对象

from verl import DataProto  # 数据协议对象
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto  # 数据填充与去填充工具
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup  # Ray 分布式工具
from verl.utils import hf_tokenizer  # Huggingface 分词器工具
from verl.utils.fs import copy_to_local  # 文件系统工具
from verl.utils.hdfs_io import makedirs  # HDFS 目录创建工具
from verl.utils.model import compute_position_id_with_mask  # 位置编码计算工具
from verl.workers.fsdp_workers import ActorRolloutRefWorker  # 分布式推理工作器


@hydra.main(config_path="config", config_name="generation", version_base=None)  # Hydra 主入口，加载配置
def main(config):
    run_generation(config)  # 执行生成主流程


def run_generation(config) -> None:
    if not ray.is_initialized():  # 判断 Ray 是否已初始化
        # this is for local ray cluster
        default_runtime_env = {"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}}  # 默认环境变量
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})  # 获取 Ray 初始化参数
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})  # 获取运行环境参数
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)  # 合并环境变量
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})  # 构造 Ray 初始化参数
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))  # 初始化 Ray 集群

    ray.get(main_task.remote(config))  # 分发主任务到 Ray 集群


@ray.remote(num_cpus=1)  # Ray 远程任务，分配 1 个 CPU
def main_task(config):
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values  # 打印解析后的配置
    OmegaConf.resolve(config)  # 解析配置中的符号值

    local_path = copy_to_local(config.model.path)  # 模型文件复制到本地
    trust_remote_code = config.data.get("trust_remote_code", False)  # 是否信任远程代码
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)  # 构建分词器

    if config.rollout.temperature == 0.0:
        assert config.data.n_samples == 1, "When temperature=0, n_samples must be 1."  # 温度为 0 时只允许采样 1 个
    assert config.data.n_samples >= 1, "n_samples should always >= 1"  # 采样数必须大于等于 1

    # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
    dataset = pd.read_parquet(config.data.path)  # 读取 parquet 格式数据集
    chat_lst = dataset[config.data.prompt_key].tolist()  # 获取 prompt 列

    chat_lst = [chat.tolist() for chat in chat_lst]  # 转换为列表格式

    tokenizer.padding_side = "left"  # 设置分词器填充方向为左
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # 如果没有 pad_token，则使用 eos_token

    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role="rollout")  # 构造 Ray 远程类
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)  # 构造资源池
    wg = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=ray_cls_with_init,
        device_name=config.trainer.device,
    )  # 构造分布式工作组
    wg.init_model()  # 初始化模型

    total_samples = len(dataset)  # 样本总数
    config_batch_size = config.data.batch_size  # 批次大小
    apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})  # chat 模板参数
    num_batch = -(-total_samples // config_batch_size)  # 计算批次数（向上取整）
    output_lst = [[] for _ in range(config.data.n_samples)]  # 初始化输出列表

    for batch_idx in range(num_batch):  # 遍历每个批次
        print(f"[{batch_idx + 1}/{num_batch}] Start to process.")
        batch_chat_lst = chat_lst[batch_idx * config_batch_size : (batch_idx + 1) * config_batch_size]  # 获取当前批次数据
        inputs = tokenizer.apply_chat_template(
            batch_chat_lst,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            max_length=config.rollout.prompt_length,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
            **apply_chat_template_kwargs,
        )  # 应用聊天模板，准备模型输入
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        position_ids = compute_position_id_with_mask(attention_mask)  # 计算位置编码
        batch_dict = {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}

        data = DataProto.from_dict(batch_dict)  # 构造数据协议对象
        data_padded, pad_size = pad_dataproto_to_divisor(data, wg.world_size)  # 数据填充至整除指定值

        # START TO GENERATE FOR n_samples TIMES
        print(f"[{batch_idx + 1}/{num_batch}] Start to generate.")
        for n_sample in range(config.data.n_samples):
            output_padded = wg.generate_sequences(data_padded)  # 生成序列
            output = unpad_dataproto(output_padded, pad_size=pad_size)  # 去除填充

            output_texts = []
            for i in range(len(output)):
                data_item = output[i]
                prompt_length = data_item.batch["prompts"].shape[-1]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = data_item.batch["responses"][:valid_response_length]
                response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)  # 解码生成的 ID 为文本
                output_texts.append(response_str)

            output_lst[n_sample].extend(output_texts)  # 收集每个样本的输出

    # convert output_lst from (n_samples, n_data) to (n_data, n_sampels)
    output_lst = np.array(output_lst, dtype=object)
    output_lst = np.transpose(output_lst, axes=(1, 0)).tolist()  # 转置输出列表，便于与原数据结合

    # add to the data frame
    dataset["responses"] = output_lst  # 将生成的响应添加到数据集中

    # write to a new parquet
    output_dir = os.path.dirname(config.data.output_path)
    makedirs(output_dir, exist_ok=True)  # 创建输出目录
    dataset.to_parquet(config.data.output_path)  # 将数据集写入 parquet 文件


if __name__ == "__main__":  # 脚本入口
    main()
