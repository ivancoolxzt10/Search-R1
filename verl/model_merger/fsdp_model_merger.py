# Copyright 2024 Bytedance Ltd. and/or its affiliates
# 版权声明，表明该文件归 Bytedance 所有
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
import json  # JSON 处理模块
import os    # 操作系统相关模块
from concurrent.futures import ThreadPoolExecutor  # 线程池并行加载
from pathlib import Path  # 路径处理工具

import numpy as np  # 数组库
import torch  # 深度学习库
from torch.distributed._tensor import Placement, Shard  # 分布式张量分片工具

try:
    # 兼容 torch 2.5+
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

from tqdm import tqdm  # 进度条工具

from .base_model_merger import BaseModelMerger  # 导入模型合并基类


class FSDPModelMerger(BaseModelMerger):
    """
    FSDP（Fully Sharded Data Parallel）模型 checkpoint 合并器。
    支持 FSDP、FSDP+DDP、DTensor 等分布式分片方式，将分片参数合并为 HuggingFace 格式。
    主要功能：自动检测分片数、并行加载分片、合并参数、支持 DTensor。
    """

    def _get_world_size(self) -> int:
        # 功能：获取分布式训练的 world_size（即总进程数/GPU数）。
        # 原理：FSDP 训练通常会保存一个配置文件 fsdp_config.json，其中记录了 world_size。
        # 这个函数就是去读取这个文件并返回该值。这是确定需要加载多少个分片文件的第一步。
        config_path = Path(self.config.local_dir) / "fsdp_config.json"  # 构造配置文件路径
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)  # 读取 JSON 配置
        world_size = config.get("world_size", None)  # 获取 world_size
        if world_size is None:
            raise ValueError("World size not found in the config file.")  # 如果没有则报错
        return world_size  # 返回 world_size

    def _load_rank_zero_state_dict(self, world_size: int) -> dict:
        # 功能：加载 rank_0 进程保存的那个分片文件。
        # 原理：这个文件作为“样本”，用来分析整个模型的分片方式。无需一开始就加载所有文件，先分析一个就够了。
        return torch.load(
            Path(self.config.local_dir) / f"model_world_size_{world_size}_rank_0.pt",
            map_location="cpu",
            weights_only=False,
        )  # 加载 rank_0 的分片权重

    def _extract_device_mesh_info(self, state_dict: dict, world_size: int) -> tuple[np.ndarray, tuple[str, ...]]:
        # 功能：从 rank_0 的 state_dict 中提取设备网格（device mesh）和分片信息。
        # 原理：
        # 它随便取一个参数的权重 (weight)。
        # 检查这个 weight是不是 DTensor 类型。
        # 如果是 DTensor，DTensor 对象内部就保存了 device_mesh（描述了 GPU 的拓扑结构，比如 (ddp_dim, fsdp_dim)) 和 mesh_dim_names（维度名称）。
        # 代码直接从对象中读取这些权威信息。
        # 如果不是 DTensor，说明是比较早期的 FSDP checkpoint。代码会假设一个最简单的一维 FSDP 网格，大小就是 world_size。
        pivot_key = sorted(list(state_dict.keys()))[0]  # 取第一个参数名
        weight = state_dict[pivot_key]  # 获取权重
        if isinstance(weight, DTensor):
            device_mesh = weight.device_mesh  # 获取 DTensor 的设备网格
            mesh = device_mesh.mesh  # 网格形状
            mesh_dim_names = device_mesh.mesh_dim_names  # 网格维度名称
        else:
            mesh = np.array([world_size], dtype=np.int64)  # 假设一维 FSDP 网格
            mesh_dim_names = ("fsdp",)  # 维度名称为 fsdp
        return mesh, mesh_dim_names  # 返回网格信息

    def _calculate_shard_configuration(
        self, mesh: np.ndarray, mesh_dim_names: tuple[str, ...]
    ) -> tuple[int, tuple[int, ...]]:
        # 功能：根据设备网格信息，计算总分片数和网格形状。
        # 原理：这是一个简单的辅助函数。对于纯 FSDP，总分片数就是 world_size。
        # 对于 FSDP+DDP 混合并行，总分片数是 fsdp 维度的大小。目前它还不支持张量并行（TP），所以相关逻辑是占位符。
        assert mesh_dim_names in (("fsdp",), ("ddp", "fsdp")), f"Unsupported mesh_dim_names {mesh_dim_names}"

        if "tp" in mesh_dim_names:
            # TODO: "tp" 尚未支持
            total_shards = mesh.shape[-1] * mesh.shape[-2]
            mesh_shape = (mesh.shape[-2], mesh.shape[-1])
        else:
            total_shards = mesh.shape[-1]
            mesh_shape = (mesh.shape[-1],)

        return total_shards, mesh_shape

    def _merge_by_placement(self, tensors: list[torch.Tensor], placement: Placement) -> torch.Tensor:
        """根据 DTensor 分片方式合并张量。"""
        if placement.is_replicate():
            return tensors[0]
        elif placement.is_partial():
            raise NotImplementedError("Partial placement is not supported yet")
        elif placement.is_shard():
            return torch.cat(tensors, dim=placement.dim).contiguous()

        raise NotImplementedError(f"Unsupported placement: {placement}")

    def _load_and_merge_state_dicts(
        self, world_size: int, total_shards: int, mesh_shape: tuple[int, ...], mesh_dim_names: tuple[str, ...]
    ) -> dict[str, torch.Tensor]:
        """并行加载所有分片并合并为完整 state dict。"""
        model_state_dict_lst = [None] * total_shards

        def process_one_shard(rank: int, model_state_dict_lst: list):
            model_path = Path(self.config.local_dir) / f"model_world_size_{world_size}_rank_{rank}.pt"
            state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
            model_state_dict_lst[rank] = state_dict
            return state_dict

        with ThreadPoolExecutor(max_workers=min(32, os.cpu_count())) as executor:
            futures = [executor.submit(process_one_shard, rank, model_state_dict_lst) for rank in range(total_shards)]
            for future in tqdm(futures, desc=f"Loading {total_shards} FSDP shards", total=total_shards):
                future.result()

        # 合并所有分片的 state dict
        state_dict = {}
        param_placements: dict[str, list] = {}

        for key in set(model_state_dict_lst[0].keys()):
            state_dict[key] = []
            for model_state_shard in model_state_dict_lst:
                tensor = model_state_shard.pop(key)
                if isinstance(tensor, DTensor):
                    state_dict[key].append(tensor._local_tensor.bfloat16())

                    placements = tuple(tensor.placements)
                    # dp 维度的 replicated 可忽略
                    if mesh_dim_names[0] in ("dp", "ddp"):
                        placements = placements[1:]

                    if key not in param_placements:
                        param_placements[key] = placements
                    else:
                        assert param_placements[key] == placements
                else:
                    state_dict[key].append(tensor.bfloat16())

        del model_state_dict_lst

        # 合并张量
        for key in sorted(state_dict):
            if not isinstance(state_dict[key], list):
                print(f"No need to merge key {key}")
                continue
            if key in param_placements:
                placements: tuple[Shard] = param_placements[key]
                if len(mesh_shape) == 1:
                    # 一维分片，FSDP 无 TP
                    assert len(placements) == 1
                    shards = state_dict[key]
                    state_dict[key] = self._merge_by_placement(shards, placements[0])
                else:
                    # 二维分片，FSDP + TP
                    raise NotImplementedError("FSDP + TP is not supported yet")
            else:
                state_dict[key] = torch.cat(state_dict[key], dim=0)

        return state_dict

    def merge_and_save(self):
        world_size = self._get_world_size()
        rank_zero_state_dict = self._load_rank_zero_state_dict(world_size)

        mesh, mesh_dim_names = self._extract_device_mesh_info(rank_zero_state_dict, world_size)
        print(f"Got device mesh {mesh}, mesh_dim_names {mesh_dim_names}")

        total_shards, mesh_shape = self._calculate_shard_configuration(mesh, mesh_dim_names)
        print(f"Processing model shards with {total_shards} {mesh_shape} in total")

        merged_state_dict = self._load_and_merge_state_dicts(world_size, total_shards, mesh_shape, mesh_dim_names)

        if self.config.operation == "test":
            if not self.config.test_hf_dir:
                raise ValueError("test_hf_dir must be provided for test operation")
            self._validate_state_dict(merged_state_dict)
        elif self.config.operation == "merge":
            self.save_hf_model_and_tokenizer(merged_state_dict)
            if self.config.hf_upload:
                self.upload_to_huggingface()
        else:
            raise ValueError(f"Unknown operation: {self.config.operation}")

    def _validate_state_dict(self, state_dict: dict[str, torch.Tensor]):
        # 功能：校验合并后的参数字典与 HuggingFace 标准模型是否完全一致。
        # 原理：检查参数名、shape、dtype、数值精度，确保无误。
        auto_model_class = self.get_transformers_auto_model_class()  # 获取 transformers 自动模型类

        hf_model = auto_model_class.from_pretrained(self.config.test_hf_dir, torch_dtype=torch.bfloat16)  # 加载 HF 标准模型
        hf_state_dict = hf_model.state_dict()  # 获取标准模型参数字典
        del hf_model  # 释放内存

        hf_model_keys = set(hf_state_dict.keys())  # 标准模型参数名集合
        collected_keys = set(state_dict.keys())  # 合并后参数名集合

        missing_keys = hf_model_keys - collected_keys  # 检查缺失参数
        assert len(missing_keys) == 0, f"Missing keys in collected state dict: {list(sorted(missing_keys))}"  # 有缺失则报错

        extra_keys = collected_keys - hf_model_keys  # 检查多余参数
        assert len(extra_keys) == 0, f"Extra keys in collected state dict: {list(sorted(extra_keys))}"  # 有多余则报错

        for key in hf_model_keys:
            hf_shape = hf_state_dict[key].shape  # 标准参数 shape
            collected_shape = state_dict[key].shape  # 合并后参数 shape
            assert hf_shape == collected_shape, (
                f"Shape mismatch for key '{key}': original {hf_shape} vs collected {collected_shape}"
            )  # shape 不一致则报错

            hf_dtype = hf_state_dict[key].dtype  # 标准参数 dtype
            collected_dtype = state_dict[key].dtype  # 合并后参数 dtype
            assert hf_dtype == collected_dtype, (
                f"Dtype mismatch for key '{key}': original {hf_dtype} vs collected {collected_dtype}"
            )  # dtype 不一致则报错

            torch.testing.assert_close(hf_state_dict[key], state_dict[key], atol=1e-6, rtol=1e-6)  # 数值精度校验

        print("FSDP checks passed: The merged state_dict matches the hf model saved by FSDPCheckpointManager.")  # 校验通过提示

    def cleanup(self):
        """Cleanup temporary files if needed."""
        # FSDP merger does not create temporary files, so no cleanup is needed.
        pass  # 无需清理，接口占位
