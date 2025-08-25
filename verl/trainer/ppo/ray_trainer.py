# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# (版权和开源许可声明...)
"""
使用基于 Ray 的单控制器实现的 PPO 训练器。
这个训练器支持与 huggingface 兼容的模型无关的模型初始化。
"""

# 导入标准库和第三方库
import json
import os
import uuid  # 用于生成全局唯一标识符 (UUID)，常用于追踪数据样本
import warnings  # 用于发出警告信息
from collections import defaultdict  # 提供带默认值的字典
from copy import deepcopy  # 用于创建对象的深拷贝
from dataclasses import dataclass, field  # 用于创建数据类
from enum import Enum  # 用于定义枚举类型，使代码更清晰、安全
from pprint import pprint  # 用于美观地打印 Python 对象
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict  # Hydra 配置管理库
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader  # 支持状态保存和恢复的数据加载器，对断点续训至关重要
from tqdm import tqdm  # 用于显示进度条，提升用户体验

# 从项目内部导入模块
from verl import DataProto  # 项目自定义的数据传输协议/容器，用于在不同组件间传递结构化数据
from verl.experimental.dataset.sampler import AbstractCurriculumSampler  # 课程学习采样器的抽象基类
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto  # 数据填充/去填充的辅助工具
from verl.single_controller.base import Worker  # 所有分布式工作者的基类
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup  # Ray 分布式核心组件
from verl.single_controller.ray.base import create_colocated_worker_cls  # 用于将多个工作者并置到同一个进程的工具
from verl.trainer.config import AlgoConfig  # 算法配置的数据类
from verl.trainer.ppo import core_algos  # PPO 核心算法实现
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss  # 优势估计器枚举和损失聚合函数
from verl.trainer.ppo.metric_utils import (  # 指标计算相关的工具函数
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async  # 奖励计算函数（同步和异步版本）
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi  # 检查点管理工具
from verl.utils.config import omega_conf_to_dataclass  # 配置对象转换工具
from verl.utils.debug import marked_timer  # 性能计时装饰器/上下文管理器
from verl.utils.metric import reduce_metrics  # 分布式环境下聚合指标的工具
from verl.utils.rollout_skip import RolloutSkip  # 用于跳过 Rollout 阶段的调试工具
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance  # 序列长度平衡算法
from verl.utils.torch_functional import masked_mean  # 带掩码的均值计算函数
from verl.utils.tracking import ValidationGenerationsLogger  # 用于记录验证生成结果的日志器

WorkerType = type[Worker]  # 为 Worker 的类类型创建一个类型别名，方便类型注解


class Role(Enum):
    """
    定义分布式训练中的不同角色。
    原理：使用枚举（Enum）来表示角色，相比于直接使用字符串，可以提供编译时检查，减少拼写错误，
    并让代码意图更清晰。
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2  # 将 Actor（策略模型）和 Rollout（经验生成）合并为一个角色
    Critic = 3  # 评论家（价值模型）
    RefPolicy = 4  # 参考策略（通常是 SFT 模型，用于 KL 散度计算）
    RewardModel = 5  # 奖励模型
    ActorRolloutRef = 6  # 将 Actor, Rollout, RefPolicy 合并为一个角色


@dataclass
class ResourcePoolManager:
    """
    管理 Ray 资源池的类。
    原理：将资源定义、创建和分配的逻辑封装在一个类中，实现了关注点分离。
    这使得主训练器的代码不必关心底层的资源分配细节，只需声明需要哪些资源即可。
    """
    # 资源池规格，例如: {"gpu_pool": [8, 8]} 表示一个名为 gpu_pool 的池，包含2个节点，每个节点8个GPU
    resource_pool_spec: dict[str, list[int]]
    # 角色到资源池名称的映射，例如: {Role.ActorRollout: "gpu_pool"}
    mapping: dict[Role, str]
    # 用于存储已创建的 RayResourcePool 对象的字典，由 default_factory 自动初始化为空字典
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """根据规格创建 Ray 资源池。"""
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count=1 表示一个资源池内的所有工作者组（WorkerGroups）将被合并到同一个进程集合中。
            # 这对于 FSDP 这种需要所有进程紧密协作的后端是推荐的。
            # 对于 Megatron，可能需要 >1 的值来为不同模型使用不同的工作者组。
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()  # 检查集群资源是否满足要求

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """根据角色获取其对应的资源池对象。"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """计算并返回此管理器配置的总 GPU 数量。"""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """检查 Ray 集群的可用资源是否能满足定义的资源池规格。"""
        # 获取 Ray 集群中每个节点的可用资源信息
        node_available_resources = ray.state.available_resources_per_node()
        # 提取每个节点的可用 GPU 数量
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # 检查总 GPU 数量是否满足要求
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = self.get_n_gpus()
        if total_available_gpus < total_required_gpus:
            raise ValueError(f"总可用 GPU 数 {total_available_gpus} 少于总需求 GPU 数 {total_required_gpus}")

        # 逐个检查每个资源池是否能被满足（这是一个简化的贪心匹配算法）
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(f"资源池 {resource_pool_name} 的需求无法在当前 Ray 集群中被满足")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """
    将 KL 惩罚项应用到 token 级别的奖励上。
    原理：在 RLHF 中，为防止 PPO 策略（Actor）过度优化奖励而偏离原始 SFT 模型（RefPolicy）太远，
    引入了 KL 散度作为正则化项。这个函数计算每个 token 的 KL 散度，并从 token 级别的奖励中减去它
    （乘以一个自适应系数 beta），从而激励 Actor 在追求高奖励的同时，保持与原始模型的风格和事实一致性。
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # 计算当前策略和参考策略之间的 KL 散度
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)
    kld = kld * response_mask  # 只考虑 response 部分的 KL
    beta = kl_ctrl.value  # 获取当前自适应的 KL 系数

    # 从原始奖励中减去 KL 惩罚
    token_level_rewards = token_level_scores - beta * kld

    # 计算当前批次的平均 KL，并用它来更新 KL 控制器
    current_kl = masked_mean(kld, mask=response_mask, axis=-1).mean().item()
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    # 返回相关的指标用于日志记录
    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}
    return data, metrics


def compute_response_mask(data: DataProto):
    """
    计算序列中回答（response）部分的掩码。
    原理：在 Transformer 的输入中，prompt 和 response 是拼接在一起的。但在计算损失、奖励或 KL 散度时，
    我们通常只关心模型生成的部分（回答）。此函数从完整的注意力掩码中，根据 response 的长度，
    精确地提取出只覆盖回答部分的掩码。
    """
    response_length = data.batch["responses"].size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
        data: DataProto,
        adv_estimator: AdvantageEstimator,
        gamma: float = 1.0,
        lam: float = 1.0,
        num_repeat: int = 1,
        norm_adv_by_std_in_grpo: bool = True,
        config: Optional[AlgoConfig] = None,
) -> DataProto:
    """
    计算优势函数。
    原理：优势函数 A(s, a) = Q(s, a) - V(s) 是策略梯度方法的核心，它衡量了在某个状态下，
    采取某个动作比平均水平（由价值函数 V(s) 衡量）要好多少。这个函数是一个分发器，
    根据配置文件中指定的优势估计器类型（如 GAE, GRPO），调用相应的核心算法来计算优势值和回报值（returns）。
    这些计算通常不涉及大型模型运算，因此在主控制器（Driver）上执行是高效的。
    """
    # 如果 data 中没有 response_mask，先计算它
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)

    # 根据配置选择不同的优势估计算法
    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(data, config.pf_ppo.reweight_method,
                                                           config.pf_ppo.weight_pow)
    elif adv_estimator == AdvantageEstimator.GRPO:
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # 其他自定义优势估计器
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """使用 Ray 的分布式 PPO 训练器。"""

    def __init__(
            self,
            config,
            tokenizer,
            role_worker_mapping: dict[Role, WorkerType],
            resource_pool_manager: ResourcePoolManager,
            ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
            processor=None,
            reward_fn=None,
            val_reward_fn=None,
            train_dataset: Optional[Dataset] = None,
            val_dataset: Optional[Dataset] = None,
            collate_fn=None,
            train_sampler: Optional[Sampler] = None,
            device_name=None,
    ):
        """
        初始化函数。
        原理：这个函数在主进程（Driver）中执行，负责接收所有配置和组件，并进行初始化设置。
        它本身不进行繁重的计算，而是作为总指挥，为后续的分布式训练做好万全准备。
        """
        # 保存传入的配置和组件
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "目前只支持混合引擎模式"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"角色映射中缺少 ActorRollout: {role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager

        # 根据角色映射判断是否启用某些功能
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name or self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # 如果使用了 LoRA，那么参考策略就是不带 LoRA 的基础 Actor 模型
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # 定义 KL 控制器（如果启用）
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        # 判断是否使用 Critic
        if config.critic.enable is not None:
            self.use_critic = bool(config.critic.enable)
        elif self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True  # GAE 算法必须使用 Critic
        else:
            warnings.warn("已禁用 Critic，因为优势估计器不是 GAE。如果这不是预期行为，请设置 critic.enable=True",
                          stacklevel=2)
            self.use_critic = False

        self._validate_config()  # 执行配置校验
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)  # 创建数据加载器

    def _validate_config(self):
        """
        校验配置文件的合法性和自洽性。
        原理：这是一个重要的防御性编程实践。它会检查各种参数组合是否合理，
        例如，全局批量大小是否能被微批次大小和 GPU 数量整除，互斥的参数是否被同时设置等。
        在训练开始前发现配置错误，可以节省大量的调试时间和计算资源。
        """
        config = self.config
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            if config.actor_rollout_ref.actor.strategy == "megatron":
                model_parallel_size = (
                        config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size
                        * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
                )
                assert n_gpus % (
                            model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0
                megatron_dp = n_gpus // (
                            model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size)
                minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
            else:
                minimal_bsz = n_gpus

            real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
            assert real_train_batch_size % minimal_bsz == 0, f"实际训练批大小 ({real_train_batch_size}) 必须能被最小批大小 ({minimal_bsz}) 整除"

        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            """校验互斥的微批次大小配置选项。"""
            settings = {
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"在配置 [{name}] 中，请至少设置 '{name}.{param}' 或 '{name}.{param_per_gpu}' 其中之一。")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"在配置 [{name}] 中，您同时设置了 '{name}.{param}' 和 '{name}.{param_per_gpu}'。请移除前者，因为它已弃用。")

        actor_config = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
        actor_config.validate(n_gpus, config.data.train_batch_size, config.actor_rollout_ref.model)

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            if self.use_reference_policy:
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(
                config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model"
            )

        if self.config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("注意：您同时启用了奖励内 KL 和 KL 损失。")

        if self.use_critic:
            critic_config = omega_conf_to_dataclass(config.critic)
            critic_config.validate(n_gpus, config.data.train_batch_size)

        if config.data.get("val_batch_size") is not None:
            print("警告：val_batch_size 已弃用。验证数据集将作为一个整体批次发送给推理引擎。")

        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, "启用采样时，验证生成温度应大于0"

        print("[validate_config] 所有配置检查均已通过！")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        创建训练和验证数据加载器。
        原理：使用 `StatefulDataLoader`，这种加载器可以保存和恢复其内部状态（例如，
        已经迭代到数据集的哪个位置），这对于从检查点无缝恢复训练至关重要。
        同时，它会根据配置计算出总的训练步数，并更新到配置对象中，供学习率调度器等组件使用。
        """
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer,
                                              self.processor)
        if val_dataset is None:
            val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer,
                                            self.processor)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn
            collate_fn = default_collate_fn

        num_workers = self.config.data.dataloader_num_workers

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size or len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "训练数据加载器为空！"
        assert len(self.val_dataloader) >= 1, "验证数据加载器为空！"

        print(f"训练数据加载器大小: {len(self.train_dataloader)}, 验证数据加载器大小: {len(self.val_dataloader)}")

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps
        else:
            total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        self.total_training_steps = total_training_steps
        print(f"总训练步数: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"警告：无法在配置中设置 total_training_steps。结构可能缺失？错误: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """
        将生成的样本（输入、输出、分数等）以 JSONL 格式保存到文件。
        原理：用于详细的离线分析和调试。将模型在验证集或 rollout 过程中的具体表现保存下来，
        有助于深入理解模型的行为、优点和失败案例。
        """
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {"input": inputs, "output": outputs, "gts": gts, "score": scores, "step": [self.global_steps] * n}
        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = [json.dumps({k: v[i] for k, v in base_data.items()}, ensure_ascii=False) for i in range(n)]

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"已将生成样本转储到 {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """
        将验证样本记录到配置的日志系统（如 WandB）。
        原理：在 WandB 等实验跟踪工具中创建一个交互式表格，展示模型的输入、输出和得分。
        这使得用户可以在网页上直观地、定性地评估模型的生成质量，而不仅仅依赖于数值指标。
        """
        generations_to_log = self.config.trainer.log_val_generations
        if generations_to_log == 0:
            return

        samples = sorted(list(zip(inputs, outputs, scores, strict=True)), key=lambda x: x[0])
        rng = np.random.RandomState(42)
        rng.shuffle(samples)
        samples = samples[:generations_to_log]
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        """
        从完整的批次数据中，提取出生成（generation）所需的部分。
        原理：模型的 `generate` 函数通常只需要输入的 `prompt`（即 `input_ids` 等），
        而不需要 `labels` 或其他用于计算损失的信息。这个函数通过 `pop` 操作，
        分离出生成所需的数据，同时保留其他数据以备后续使用，提高了数据处理的清晰度。
        """
        reward_model_keys = {"data_source", "reward_model", "extra_info", "uid"} & batch.non_tensor_batch.keys()
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(batch_keys=batch_keys_to_pop, non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop))

        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)
        return gen_batch

    def _validate(self):
        """
        执行一次完整的验证流程。
        原理：这是一个标准的模型评估流程，用于在训练过程中周期性地检查模型在未见过的
        验证集上的性能，以监控模型是否过拟合，并选择最佳的模型检查点。
        """
        data_source_lst, reward_extra_infos_dict = [], defaultdict(list)
        sample_inputs, sample_outputs, sample_gts, sample_scores, sample_turns = [], [], [], [], []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            test_batch = test_batch.repeat(self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in
                           test_batch.batch["input_ids"]]
            sample_inputs.extend(input_texts)
            sample_gts.extend(
                [item.non_tensor_batch.get("reward_model", {}).get("ground_truth") for item in test_batch])

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"验证生成批次元信息: {test_gen_batch.meta_info}")

            size_divisor = self.actor_rollout_wg.world_size if not self.async_rollout_mode else self.config.actor_rollout_ref.rollout.agent.num_workers
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)

            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print("验证生成结束")

            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in
                            test_output_gen_batch.batch["responses"]]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            if self.val_reward_fn is None:
                raise ValueError("验证需要提供 val_reward_fn。")
            result = self.val_reward_fn(test_batch, return_dict=True)
            scores = result["reward_tensor"].sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * len(scores)))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        if val_data_dir := self.config.trainer.get("validation_data_dir"):
            self._dump_generations(inputs=sample_inputs, outputs=sample_outputs, gts=sample_gts, scores=sample_scores,
                                   reward_extra_infos_dict=reward_extra_infos_dict, dump_path=val_data_dir)

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"指标 {key_info} 长度与样本数不匹配"

        data_sources = np.concatenate(data_source_lst, axis=0)
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            # ... 格式化指标字典 ...
            pass

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """
        初始化所有的分布式工作者。
        原理：这是分布式系统启动的核心步骤。控制器（Driver）在这里根据配置，
        通过 Ray API 在集群中远程地创建和初始化所有计算单元（Actor、Critic等）。
        这个过程是惰性的，直到这里被调用，实际的 GPU 资源才会被占用。
        """
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref, role="actor_rollout")
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        if self.use_rm:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        all_wg = {}
        wg_kwargs = {}
        if self.config.trainer.ray_wait_register_center_timeout is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if self.config.global_profiler.steps is not None:
            wg_kwargs["profile_steps"] = self.config.global_profiler.steps
            if self.config.global_profiler.tool == "nsys":
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    self.config.global_profiler.global_tool_config.nsys.worker_nsight_options)
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls,
                                                **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager
            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(config=self.config, worker_group=self.actor_rollout_wg)

    def _save_checkpoint(self):
        """
        保存训练检查点。
        原理：这是一个分布式的检查点保存流程。控制器（Driver）负责协调，但实际的
        模型权重保存操作由各个工作者组（`WorkerGroup`）并行执行。每个工作者组只保存
        自己持有的模型分片。控制器还负责保存全局状态，如数据加载器的状态和当前的全局步数。
        """
        from verl.utils.fs import local_mkdir_safe
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f"global_step_{self.global_steps}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")
        actor_remote_path = os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}",
                                         "actor") if self.config.trainer.default_hdfs_dir else None

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "警告：remove_previous_ckpt_in_save 已弃用，请改用 max_actor_ckpt_to_keep=1 和 max_critic_ckpt_to_keep=1。")
        max_actor_ckpt_to_keep = self.config.trainer.get(
            "max_actor_ckpt_to_keep") if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get(
            "max_critic_ckpt_to_keep") if not remove_previous_ckpt_in_save else 1

        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps,
                                              max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}",
                                              "critic") if self.config.trainer.default_hdfs_dir else None
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps,
                                           max_ckpt_to_keep=max_critic_ckpt_to_keep)

        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        torch.save(self.train_dataloader.state_dict(), dataloader_local_path)

        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                           "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        """
        从检查点恢复训练。
        原理：与保存相对应，控制器首先根据恢复策略找到正确的检查点文件夹，
        然后远程调用所有工作者组的 `load_checkpoint` 方法来并行加载它们各自的模型分片。
        同时，在控制器本地加载数据加载器等全局状态，以实现完全无缝的训练恢复。
        """
        if self.config.trainer.resume_mode == "disable":
            return 0

        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("从 HDFS 加载尚未实现")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                checkpoint_folder = os.path.join(os.getcwd(), checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)

        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("从头开始训练")
                return 0
        elif self.config.trainer.resume_mode == "resume_path":
            global_step_folder = self.config.trainer.resume_from_path
            if not os.path.isabs(global_step_folder):
                global_step_folder = os.path.join(os.getcwd(), global_step_folder)

        print(f"从检查点文件夹加载: {global_step_folder}")
        self.global_steps = int(global_step_folder.split("global_step_")[-1])
        print(f"设置全局步数为 {self.global_steps}")

        self.actor_rollout_wg.load_checkpoint(os.path.join(global_step_folder, "actor"),
                                              del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        if self.use_critic:
            self.critic_wg.load_checkpoint(os.path.join(global_step_folder, "critic"),
                                           del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            self.train_dataloader.load_state_dict(torch.load(dataloader_local_path, weights_only=False))
        else:
            print(f"警告: 在 {dataloader_local_path} 未找到数据加载器状态，将从头开始加载数据")

    def _start_profiling(self, do_profile: bool):
        """如果启用性能分析，则启动所有工作者组的分析器。"""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool):
        """如果启用性能分析，则停止所有工作者组的分析器。"""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """
        通过重排批次数据来平衡各个数据并行（DP）副本的 token 负载。
        原理：在数据并行中，如果不同 GPU 处理的序列长度差异很大，会导致某些 GPU 先完成计算而其他 GPU 仍在工作，
        造成“木桶效应”。这个函数通过一个排序和分区算法，重新排列批次中的样本顺序，
        使得最终分发给每个 DP 副本的子批次的总 token 数大致相等，从而最大化并行效率。
        """
        global_seqlen_lst = batch.batch["attention_mask"].view(batch.batch.batch_size[0], -1).sum(-1).tolist()
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=self.actor_rollout_wg.world_size,
                                                              equal_size=True)
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        PPO 的主训练循环。
        这是整个脚本的核心，精确地编排了 PPO 算法的复杂数据流。
        """
        # 导入并初始化日志记录器，例如 WandB
        from verl.utils.tracking import Tracking
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        # 尝试从检查点加载，如果成功则返回上次的全局步数，否则从 0 开始
        self.global_steps = self._load_checkpoint() or 0

        # 如果配置了验证奖励函数，并且设置了在训练前进行验证，则执行一次验证
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"验证指标为空: {val_metrics=}"  # 确保验证有结果
            pprint(f"初始验证指标: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)  # 记录初始验证结果
            # 如果是仅验证模式，则直接返回，不进入训练循环
            if self.config.trainer.get("val_only", False):
                return

        # (调试功能) 如果配置了跳过 rollout，则包装生成函数以使用缓存数据
        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # 初始化 TQDM 进度条，用于可视化训练进度
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="训练进度")

        # 将全局步数加 1，因为训练是从第 1 步开始的
        self.global_steps += 1

        # 初始化一些状态变量
        last_val_metrics, self.max_steps_duration = None, 0
        # 初始化性能分析的状态标志
        prev_step_profile = False
        curr_step_profile = self.global_steps in self.config.global_profiler.get("steps", [])
        next_step_profile = False

        # --- 主训练循环开始 ---
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics, timing_raw = {}, {}  # 初始化用于存储本步骤指标和计时的字典

                # 如果当前步骤需要性能分析，则启动它
                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        # 根据是否连续分析的配置，决定是否启动
                        curr_step_profile and not prev_step_profile if self.config.global_profiler.profile_continuous_steps else curr_step_profile
                    )

                # --- 1. 数据准备阶段 ---
                batch = DataProto.from_single_dict(batch_dict)  # 将数据字典转换为项目内部的 DataProto 格式
                # 为批次中的每个样本分配一个唯一的ID，用于后续跨异步操作的追踪
                batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                         dtype=object)
                # 从完整批次中提取出生成（Rollout）所需的部分（主要是 prompt）
                gen_batch = self._get_gen_batch(batch)
                # 将当前全局步数添加到元信息中，便于 worker 内部使用
                gen_batch.meta_info["global_steps"] = self.global_steps
                # 根据配置重复 prompt，用于 n-best 生成
                gen_batch = gen_batch.repeat(self.config.actor_rollout_ref.rollout.n, interleave=True)
                # 检查是否是最后一个训练步骤
                is_last_step = self.global_steps >= self.total_training_steps

                # --- 2. PPO 核心数据流 ---
                # 使用计时器记录整个步骤的总耗时
                with marked_timer("step", timing_raw):
                    # --- 2a. Rollout 阶段 (生成经验) ---
                    with marked_timer("gen", timing_raw, color="red"):
                        # 根据是否是异步模式，调用不同的生成接口
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        # 从返回结果中提取并保存详细的计时信息
                        timing_raw.update(gen_batch_output.meta_info.pop("timing", {}))

                    # --- 2b. (可选) REMAX 基线生成 ---
                    # REMAX 是一种优势估计算法，需要一个确定性生成的 baseline 奖励
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("REMAX 优势估计需要提供 reward_fn。")
                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False  # 关闭采样，进行确定性生成
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(
                                gen_baseline_batch) if not self.async_rollout_mode else self.async_rollout_manager.generate_sequences(
                                gen_baseline_batch)
                            # 将 baseline 生成结果合并，计算其奖励，然后移除
                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch).sum(dim=-1)
                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                            batch.batch["reward_baselines"] = reward_baseline_tensor
                            del gen_baseline_batch, gen_baseline_output  # 清理内存

                    # --- 2c. 数据整合与预处理 ---
                    # 将 prompt 重复，以匹配重复生成的 response
                    batch = batch.repeat(self.config.actor_rollout_ref.rollout.n, interleave=True)
                    # 将生成的 response 合并到主数据对象中
                    batch = batch.union(gen_batch_output)
                    # 计算 response 部分的掩码，用于后续的损失和指标计算
                    if "response_mask" not in batch.batch:
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    # 如果启用了 token 平衡，则对批次进行重排
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # 统计总 token 数，用于计算吞吐量
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # --- 2d. 评估阶段 (计算奖励、旧概率、价值等) ---
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # 如果有奖励模型 (Reward Model)，则远程调用它计算分数
                        if self.use_rm:
                            batch = batch.union(self.rm_wg.compute_rm_score(batch))
                        # 调用奖励函数 (Reward Function) 计算最终奖励
                        if self.config.reward_model.launch_reward_fn_async:
                            # 异步模式：立即返回一个 future 对象
                            future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                        else:
                            # 同步模式：阻塞直到奖励计算完成
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        # 远程调用 Actor 计算生成序列时所用策略的对数概率 (log_probs)
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        # 计算并记录熵，用于鼓励探索
                        entropys = old_log_prob.batch.pop("entropys")
                        metrics["actor/entropy"] = agg_loss(entropys, batch.batch["response_mask"],
                                                            self.config.actor_rollout_ref.actor.loss_agg_mode).detach().item()
                        batch = batch.union(old_log_prob)
                        # (调试功能) 检查 actor 计算的 log_prob 与 rollout 时的是否一致
                        if "rollout_log_probs" in batch.batch:
                            from verl.utils.debug.metrics import calculate_debug_metrics
                            metrics.update(calculate_debug_metrics(batch))

                    if self.use_reference_policy:
                        with marked_timer("ref", timing_raw, color="olive"):
                            # 远程调用参考策略计算对数概率，用于 KL 散度计算
                            batch = batch.union(self.ref_policy_wg.compute_ref_log_prob(
                                batch) if not self.ref_in_actor else self.actor_rollout_wg.compute_ref_log_prob(batch))

                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            # 远程调用 Critic 计算每个 token 的价值 (Value)
                            batch = batch.union(self.critic_wg.compute_values(batch))

                    # --- 2e. 优势计算阶段 (在 Driver 上进行) ---
                    with marked_timer("adv", timing_raw, color="brown"):
                        # 如果是异步奖励，现在等待结果返回
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        # 将奖励 tensor 和额外信息存入数据对象
                        batch.batch["token_level_scores"] = reward_tensor
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # 如果配置了 KL 惩罚，则应用它来调整奖励
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl_in_reward,
                                                                 self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            # 否则，最终奖励就等于原始分数
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # 调用核心算法计算优势函数和回报值
                        batch = compute_advantage(batch, self.config.algorithm.adv_estimator,
                                                  self.config.algorithm.gamma, self.config.algorithm.lam,
                                                  self.config.actor_rollout_ref.rollout.n,
                                                  self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
                                                  self.config.algorithm)

                    # --- 2f. 模型更新阶段 ---
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            # 远程调用 Critic 进行一次或多次梯度更新
                            critic_output = self.critic_wg.update_critic(batch)
                            # 收集并记录 Critic 的损失等指标
                            metrics.update(reduce_metrics(critic_output.meta_info["metrics"]))

                    # 如果 Critic 预热期已过，则开始更新 Actor
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            # 远程调用 Actor 进行一次或多次梯度更新
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                            # 收集并记录 Actor 的损失、KL散度等指标
                            metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

                    # --- 2g. (可选) 保存 Rollout 样本 ---
                    if rollout_data_dir := self.config.trainer.get("rollout_data_dir"):
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            # 解码并准备数据
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth") for item in
                                          batch]
                            if "request_id" in batch.non_tensor_batch:
                                reward_extra_infos_dict.setdefault("request_id",
                                                                   batch.non_tensor_batch["request_id"].tolist())
                            # 调用函数将样本保存为 JSONL 文件
                            self._dump_generations(inputs, outputs, sample_gts, scores, reward_extra_infos_dict,
                                                   rollout_data_dir)

                # --- 3. 验证、保存与日志 ---
                # 定期执行验证
                if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # 检查是否需要保存检查点（基于步数频率、是否是最后一步、或 ESI 实例是否即将到期）
                esi_close_to_expiration = should_save_ckpt_esi(self.max_steps_duration,
                                                               self.config.trainer.esi_redundant_time)
                if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration):
                    if esi_close_to_expiration:
                        print("强制保存检查点：ESI 实例即将到期。")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                # 如果需要，停止性能分析
                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (self.global_steps + 1) in self.config.global_profiler.get("steps", [])
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile if self.config.global_profiler.profile_continuous_steps else curr_step_profile)
                    prev_step_profile, curr_step_profile = curr_step_profile, next_step_profile

                # 更新单步最大耗时，用于 ESI 保存决策
                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # --- 4. 指标收集与记录 ---
                # 收集训练过程、数据、计时和吞吐量等各类指标
                metrics.update({"training/global_step": self.global_steps, "training/epoch": epoch})
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw,
                                                          n_gpus=self.resource_pool_manager.get_n_gpus()))

                # 如果使用了课程学习采样器，则更新其状态
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # 将所有指标记录到日志系统
                logger.log(data=metrics, step=self.global_steps)

                # 更新进度条
                progress_bar.update(1)
                self.global_steps += 1

                # (调试功能) 如果启用了内存分析，则转储内存快照
                if self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory":
                    self.actor_rollout_wg.dump_memory_snapshot(tag=f"post_update_step{self.global_steps}",
                                                               sub_dir=f"step{self.global_steps}")

                # 如果达到总训练步数，则结束训练
                if is_last_step:
                    pprint(f"最终验证指标: {last_val_metrics}")
                    progress_bar.close()
                    return

                # (实验性功能) 为动态数据集提供钩子，在每个批次结束后更新数据集
                if hasattr(self.train_dataset, "on_batch_end"):
                    self.train_dataset.on_batch_end(batch=batch)