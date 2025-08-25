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
Offline evaluate the performance of a generated file using reward model and ground truth verifier.  # 文件功能说明：离线评估生成结果性能
The input is a parquet file that contains N generated sequences and (optional) the ground truth.
"""

from collections import defaultdict  # 用于分组统计

import hydra  # 配置管理工具
import numpy as np  # 数值计算库
import pandas as pd  # 数据处理库
import ray  # 分布式任务调度
from omegaconf import OmegaConf  # 配置对象
from tqdm import tqdm  # 进度条显示

from verl.trainer.ppo.reward import get_custom_reward_fn  # 获取自定义奖励函数
from verl.utils.fs import copy_to_local  # 文件系统工具


@ray.remote  # Ray 远程任务装饰器，实现分布式并行处理
def process_item(reward_fn, data_source, response_lst, reward_data):
    ground_truth = reward_data["ground_truth"]  # 获取当前样本的真实标签
    score_lst = [reward_fn(data_source, r, ground_truth) for r in response_lst]  # 对每个生成结果评分
    return data_source, np.mean(score_lst)  # 返回数据源和平均分


@hydra.main(config_path="config", config_name="evaluation", version_base=None)  # Hydra 主入口，加载配置
def main(config):
    local_path = copy_to_local(config.data.path, use_shm=config.data.get("use_shm", False))  # 数据文件复制到本地
    dataset = pd.read_parquet(local_path)  # 读取 parquet 格式数据
    responses = dataset[config.data.response_key]  # 获取生成结果列
    data_sources = dataset[config.data.data_source_key]  # 获取数据源列
    reward_model_data = dataset[config.data.reward_model_key]  # 获取奖励模型相关数据

    total = len(dataset)  # 样本总数

    # Initialize Ray
    if not ray.is_initialized():  # 判断 Ray 是否已初始化
        ray.init(**OmegaConf.to_container(config.ray_kwargs.get("ray_init", {})))  # 初始化 Ray 集群

    # evaluate test_score based on data source
    data_source_reward = defaultdict(list)  # 用于统计每个数据源的分数
    compute_score = get_custom_reward_fn(config)  # 获取自定义评分函数

    # Create remote tasks
    remote_tasks = [
        process_item.remote(compute_score, data_sources[i], responses[i], reward_model_data[i]) for i in range(total)
    ]  # 创建分布式评分任务

    # Process results as they come in
    with tqdm(total=total) as pbar:  # 进度条显示
        while len(remote_tasks) > 0:
            # Use ray.wait to get completed tasks
            done_ids, remote_tasks = ray.wait(remote_tasks)  # 获取已完成任务
            for result_id in done_ids:
                data_source, score = ray.get(result_id)  # 获取任务结果
                data_source_reward[data_source].append(score)  # 记录分数
                pbar.update(1)  # 更新进度条

    metric_dict = {}  # 结果字典
    for data_source, rewards in data_source_reward.items():
        metric_dict[f"test_score/{data_source}"] = np.mean(rewards)  # 统计每个数据源的平均分

    print(metric_dict)  # 输出评估结果


if __name__ == "__main__":
    main()  # 程序入口，执行主流程
