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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.  # 文件说明：主入口不与 ray_trainer 合并，便于复用
"""

import os  # 导入 os 用于环境变量和文件操作
import socket  # 导入 socket 用于获取主机名等网络相关操作

import hydra  # 导入 Hydra，一个强大的应用配置管理框架
import ray  # 导入 Ray，用于分布式计算
from omegaconf import OmegaConf  # Hydra 使用的配置对象库

from verl.experimental.dataset.sampler import AbstractSampler  # 导入课程学习采样器的抽象基类，用于类型检查
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env  # 获取 PPO 训练所需的默认 Ray 运行时环境
from verl.trainer.ppo.ray_trainer import RayPPOTrainer  # 导入核心的 PPO 训练器
from verl.trainer.ppo.reward import load_reward_manager  # 导入加载奖励函数管理器的函数
from verl.utils.device import is_cuda_available  # 检查 CUDA (GPU) 是否可用
from verl.utils.import_utils import load_extern_type  # 动态从路径加载类/函数的工具


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)  # Hydra 主入口，加载配置
def main(config):
    """
    使用 Hydra 配置管理的 PPO 训练主入口点。
    原理：`@hydra.main` 装饰器会自动读取 `config` 目录下的 `ppo_trainer.yaml` 文件，
    将其解析成一个 OmegaConf 对象 `config`，并传递给 `main` 函数。
    这使得我们可以通过命令行轻松地覆盖配置文件中的任何参数，例如 `python main.py trainer.nnodes=4`。
    """
    run_ppo(config)  # 执行 PPO 训练主流程


def run_ppo(config) -> None:
    """
    初始化 Ray 集群并运行分布式的 PPO 训练过程。
    """
    # 检查 Ray 是否已经初始化
    if not ray.is_initialized():  # 判断 Ray 是否已初始化
        # 如果没有，则根据配置初始化一个本地或多节点的 Ray 集群
        default_runtime_env = get_ppo_ray_runtime_env()  # 获取默认的运行时环境（包含一些推荐的环境变量设置）
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})  # 从配置文件中获取用户自定义的 ray 初始化参数
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})  # 获取运行环境参数
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)  # 将默认和用户的运行时环境配置合并，用户配置会覆盖默认配置
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})  # 更新 ray_init_kwargs，使用合并后的 runtime_env
        print(f"ray init kwargs: {ray_init_kwargs}") # 打印最终的 Ray 初始化参数，便于调试
        # 使用最终的参数初始化 Ray。OmegaConf.to_container 将配置对象转换为 Python 的原生字典
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    # 根据是否启用性能分析工具 (nsys) 来创建 TaskRunner Actor
    if (
        is_cuda_available
        and config.global_profiler.tool == "nsys"
        and config.global_profiler.get("steps") is not None
        and len(config.global_profiler.get("steps", [])) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available

        assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"  # 确保 NVTX 工具可用，这是 nsys 进行代码范围标记所必需的
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )  # 获取 nsight 配置
        # 创建一个带有特定运行时环境的 TaskRunner Actor，以便 nsys 可以附加到它
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        # 创建一个普通的远程 TaskRunner 实例 (Actor)
        # .remote() 是 Ray 的语法，用于在集群的某个节点上异步地创建一个类的实例
        runner = TaskRunner.remote()
    # 远程执行 TaskRunner 实例的 run 方法，并等待它完成
    # `runner.run.remote(config)` 会返回一个 future 对象 (ObjectRef)
    # `ray.get()` 会阻塞主进程，直到这个远程任务执行完毕
    ray.get(runner.run.remote(config))

    # (可选功能) 生成 Ray 的时间线追踪文件，用于性能分析
    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)  # 获取性能分析文件路径
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)  # 保存 Ray 的任务调度和执行时间线


@ray.remote(num_cpus=1)  # 这是一个 Ray Actor 定义。`num_cpus=1` 告诉 Ray 调度器为这个 Actor 分配一个 CPU 核心
class TaskRunner:
    """
    一个 Ray 远程类，用于执行分布式的 PPO 训练任务。
    原理：这个类封装了主要的训练设置逻辑。把它变成一个 Actor，意味着它的所有方法都将在
    一个独立的、远程的进程中执行，从而将繁重的设置工作从主进程中分离出来。
    """

    def __init__(self):
        # 初始化两个字典，用于存储角色到工作者类和资源池的映射
        self.role_worker_mapping = {} # 例如: Role.ActorRollout -> ActorRolloutRefWorker class
        self.mapping = {} # 例如: Role.ActorRollout -> "global_pool"

    def add_actor_rollout_worker(self, config):
        """根据 actor 的并行策略 (strategy) 添加 Actor-Rollout 工作者。"""
        from verl.single_controller.ray import RayWorkerGroup
        # 根据配置是 fsdp 还是 megatron，从不同的模块导入相应的 Worker 类
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker
            # 根据 rollout 模式是同步还是异步，选择不同的 Worker 实现
            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker
            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import Role
        # 将 Role 枚举和实际的 Worker 类关联起来
        # ray.remote() 将普通的 Python 类转换为 Ray Actor 类，这样它就可以被远程实例化
        self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)

        return actor_rollout_cls, ray_worker_group_cls

    def add_critic_worker(self, config):
        """添加 Critic 工作者到角色映射中。"""
        # 同样，根据并行策略选择不同的 CriticWorker 实现
        if config.critic.strategy in {"fsdp", "fsdp2"}:
            use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
            # 这里还处理了一个新旧实现切换的逻辑
            if use_legacy_worker_impl in ["auto", "enable"]:
                from verl.workers.fsdp_workers import CriticWorker
            elif use_legacy_worker_impl == "disable":
                from verl.workers.roles import CriticWorker
                print("Using new worker implementation")
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")

        elif config.critic.strategy == "megatron":
            from verl.workers.megatron_workers import CriticWorker

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import Role
        self.role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)

    def init_resource_pool_mgr(self, config):
        """初始化资源池管理器。"""
        from verl.trainer.ppo.ray_trainer import Role
        global_pool_id = "global_pool"
        # 定义一个名为 "global_pool" 的资源池
        # 它的规格是 [每个节点的GPU数] 重复 nnodes 次
        # 例如，4个节点，每个节点8卡，就是 [8, 8, 8, 8]
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        # 将所有角色都映射到这个全局资源池
        self.mapping[Role.ActorRollout] = global_pool_id
        self.mapping[Role.Critic] = global_pool_id
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager
        # 创建资源池管理器实例
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)
        return resource_pool_manager

    def add_reward_model_worker(self, config):
        """如果启用了奖励模型，则添加相应的 Reward Model 工作者。"""
        from verl.trainer.ppo.ray_trainer import Role
        if config.reward_model.enable:
            if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            self.role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            self.mapping[Role.RewardModel] = "global_pool"

    def add_ref_policy_worker(self, config, ref_policy_cls):
        """如果使用了 KL 散度损失或奖励，则添加参考策略 (Reference Policy) 工作者。"""
        from verl.trainer.ppo.ray_trainer import Role
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            self.role_worker_mapping[Role.RefPolicy] = ray.remote(ref_policy_cls)
            self.mapping[Role.RefPolicy] = "global_pool"

    def run(self, config):
        """
        执行主要的 PPO 训练工作流。
        """
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True)) # 打印解析后的完整配置，便于调试
        OmegaConf.resolve(config) # 确保配置中的所有变量引用都被解析

        # 从远程存储（如HDFS）下载模型权重到本地
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        # 实例化 Tokenizer 和 Processor (用于多模态)
        from verl.utils import hf_processor, hf_tokenizer
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # 按照配置添加各种工作者
        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_worker(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        # 加载训练和验证用的奖励函数管理器
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )

        # 初始化资源池管理器
        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl.utils.dataset.rl_dataset import collate_fn

        # 创建训练和验证数据集，以及训练数据采样器
        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor, is_train=True)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor, is_train=False)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # 实例化核心的 PPO 训练器
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping, # 传入角色->Worker类的映射
            resource_pool_manager=resource_pool_manager, # 传入资源管理器
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn, # 传入奖励函数
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        # 初始化训练器中的所有分布式工作者 (Actor)
        trainer.init_workers()
        # 开始训练循环
        trainer.fit()


def create_rl_dataset(data_paths, data_config, tokenizer, processor, is_train=True):
    """一个工厂函数，用于创建数据集。"""
    from torch.utils.data import Dataset
    from verl.utils.dataset.rl_dataset import RLHFDataset

    # 检查配置中是否指定了自定义的数据集类
    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        # 动态加载自定义类
        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(f"自定义数据集类 '{data_config.custom_cls.name}' 必须继承自 torch.utils.data.Dataset")
    # 检查是否配置了动态数据生成
    elif "datagen" in data_config and data_config.datagen.get("path", None) is not None and is_train:
        from verl.utils.dataset.dynamicgen_dataset import DynamicGenDataset
        dataset_cls = DynamicGenDataset
        print("Using DynamicGenDataset for data generation.")
    else:
        # 使用默认的 RLHF 数据集类
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    # 实例化并返回数据集对象
    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )
    return dataset


def create_rl_sampler(data_config, dataset):
    """一个工厂函数，用于创建数据采样器。"""
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    # 检查是否配置了自定义的课程学习 (curriculum learning) 采样器
    if data_config.sampler is not None and data_config.sampler.get("class_path", None) is not None:
        curriculum_class = load_extern_type(
            data_config.sampler.class_path,
            data_config.sampler.class_name,
        )
        sampler = curriculum_class(
            data_source=dataset,
            data_config=data_config,
        )
        assert isinstance(sampler, AbstractSampler)
        # 课程学习采样器不能和多进程数据加载同时使用，因为多进程会预取数据，破坏课程顺序
        assert data_config.get("dataloader_num_workers", 8) == 0, (
            "使用课程学习时，dataloader_num_workers 必须为 0，以防止数据被缓存。"
        )
    # 如果配置了 shuffle=True，则使用随机采样器
    elif data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    # 否则，使用顺序采样器
    else:
        sampler = SequentialSampler(data_source=dataset)
    return sampler


# Python 的标准入口点。当直接运行此脚本时，`main()` 函数会被调用。
if __name__ == "__main__":
    main()