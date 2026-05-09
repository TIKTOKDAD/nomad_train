# ============================================================
# Navigation data module - dataset/DataLoader/LMDB orchestration
# ============================================================
# 本文件负责把配置中的 datasets 字段落到真实数据管线：
# 1. 为每个数据集 split 构建 NavigationDataset
# 2. 在 DDP 前检查或强制只读 LMDB，避免多进程重复构建缓存
# 3. 合并训练集、构建 train/test DataLoader，并写回可视化所需 dataset_metadata

from copy import deepcopy
from typing import Dict, Iterable

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import transforms

from training_base.core.dataloader import loader_kwargs, workers_per_rank
from training_base.core.runtime import RuntimeContext, barrier, is_torchrun, seed_worker
from training_base.data.batch import navigation_collate
from training_base.data.lmdb_cache import check_lmdb_cache_ready
from training_base.data.navigation_dataset import NavigationDataset
from training_base.data.navigation_factory import (
    build_lmdb_cache_config,
    build_navigation_dataset_spec,
    load_navigation_data_config,
)
from training_base.data.sampling import (
    EpochAwareDataset,
    EpochAwareSampler,
    _seeded_generator,
    build_epoch_sampler,
    stable_subset_indices,
)
from training_base.registry import data_module_registry


def apply_train_subset(dataset: Dataset, *, subset_fraction: float, seed: int) -> tuple[Dataset, int, int]:
    # train_subset 用于快速冒烟或小比例训练；返回子集大小和原始大小供日志记录
    original_size = len(dataset)
    if float(subset_fraction) < 1.0:
        indices = stable_subset_indices(original_size, subset_fraction, int(seed))
        dataset = Subset(dataset, indices)
    return dataset, len(dataset), original_size


def resolve_train_batch_size(runtime: dict, *, distributed: bool, world_size: int) -> tuple[int, int]:
    # global_batch_size 优先；未配置时沿用旧字段 batch_size
    configured_global_batch_size = runtime.get("global_batch_size")
    global_batch_size = int(runtime["batch_size"] if configured_global_batch_size is None else configured_global_batch_size)
    per_device_batch_size = global_batch_size
    if distributed:
        # DDP 下每个 rank 拿 global_batch_size/world_size，保持总 batch 语义不变
        if per_device_batch_size % world_size != 0:
            raise ValueError(
                f"全局 batch_size={per_device_batch_size} 必须能被 world_size={world_size} 整除，以保持 DDP batch 语义。"
            )
        per_device_batch_size = per_device_batch_size // world_size
    return global_batch_size, per_device_batch_size


def resolve_data_runtime(
    runtime: dict,
    *,
    distributed: bool,
    world_size: int,
    train_subset_size: int,
    train_subset_total_size: int,
    train_num_workers: int,
) -> dict:
    # 将数据运行时的派生字段集中计算，再写回 runtime 供日志和 Trainer 使用
    global_batch_size, per_device_batch_size = resolve_train_batch_size(
        runtime,
        distributed=distributed,
        world_size=world_size,
    )
    return {
        "global_batch_size": global_batch_size,
        "batch_size": global_batch_size,
        "per_device_batch_size": per_device_batch_size,
        "train_subset_size": int(train_subset_size),
        "train_subset_total_size": int(train_subset_total_size),
        "num_workers_per_rank": int(train_num_workers),
        "eval_batch_size": int(runtime.get("eval_batch_size", global_batch_size)),
    }


# 根据配置生成所有数据集分割信息
def configured_dataset_splits(config) -> Iterable[dict]:
    # 这些字段决定索引文件名和 LMDB 完整性标记，必须与 NavigationDataset 初始化保持一致
    context_type = config["data"].get("context_type", "temporal")
    distance = config["data"]["distance"]
    for dataset_name, data_config in config["data"]["datasets"].items():
        # 数据集级配置可以覆盖默认 waypoint_spacing/end_slack
        waypoint_spacing = data_config.get("waypoint_spacing", 1)
        end_slack = data_config.get("end_slack", 0)
        for split in ("train", "test"):
            if split not in data_config:
                continue
            # yield 一个扁平字典，便于缓存检查函数不关心原始 YAML 层级
            yield {
                "dataset_name": dataset_name,
                "split": split,
                "data_split_folder": data_config[split],
                "waypoint_spacing": waypoint_spacing,
                "end_slack": end_slack,
                "context_type": context_type,
                "context_size": config["data"]["context_size"],
                "min_dist_cat": distance["min_dist_cat"],
                "max_dist_cat": distance["max_dist_cat"],
            }


# 将 LMDB 校验错误格式化为可读字符串
def format_lmdb_cache_errors(errors) -> str:
    lines = []
    for item in errors:
        lines.append(f"- {item['dataset_name']}:{item['split']} ({item['data_split_folder']})")
        for problem in item["problems"]:
            lines.append(f"  * {problem}")
    return "\n".join(lines)


# 批量检查所有数据分割的 LMDB 缓存是否完整
def check_lmdb_caches_ready(config):
    errors = []
    for split_config in configured_dataset_splits(config):
        # check_lmdb_cache_ready 会同时检查索引 pkl、LMDB 目录和 complete.json 标记
        ready, problems = check_lmdb_cache_ready(
            data_split_folder=split_config["data_split_folder"],
            dataset_name=split_config["dataset_name"],
            min_dist_cat=split_config["min_dist_cat"],
            max_dist_cat=split_config["max_dist_cat"],
            waypoint_spacing=split_config["waypoint_spacing"],
            context_type=split_config["context_type"],
            context_size=split_config["context_size"],
            end_slack=split_config["end_slack"],
        )
        if not ready:
            errors.append(
                {
                    "dataset_name": split_config["dataset_name"],
                    "split": split_config["split"],
                    "data_split_folder": split_config["data_split_folder"],
                    "problems": problems,
                }
            )
    return errors


# 汇总单个数据集的元信息（用于日志与可视化）
def navigation_dataset_metadata(dataset: NavigationDataset) -> dict:
    data_config = dataset.data_config or {}
    return {
        # name/dataset_name 两个字段都保留，兼容不同可视化器读取习惯
        "name": dataset.dataset_name,
        "dataset_name": dataset.dataset_name,
        "dataset_index": int(dataset.dataset_index),
        # metric_scale = metric_waypoint_spacing * waypoint_spacing，用于把归一化轨迹还原成米
        "metric_waypoint_spacing": float(data_config.get("metric_waypoint_spacing", 1.0)),
        "waypoint_spacing": int(dataset.waypoint_spacing),
        "metric_scale": float(dataset.metric_scale),
        # camera_metrics 只用于投影视觉化，不参与训练标签计算
        "camera_metrics": deepcopy(data_config.get("camera_metrics", {})),
    }


# 分布式训练前要求 LMDB 缓存已就绪
def require_lmdb_ready_before_ddp(config) -> None:
    runtime = config["runtime"]
    if not bool(runtime.get("require_lmdb_ready_for_ddp", True)):
        return
    errors = check_lmdb_caches_ready(config)
    if errors:
        # DDP 下每个 rank 同时构建 LMDB 风险很高，因此要求先单进程 build-lmdb-only
        raise RuntimeError(
            "DDP 训练要求所有 LMDB 缓存已经预构建且完整。\n"
            "请先用单进程运行:\n"
            "  python -m training_base.cli -c training_base/configs/nomad_retrain.yaml --build-lmdb-only\n"
            "如果之前的构建被中断，请追加:\n"
            "  --rebuild-incomplete-lmdb\n\n"
            f"LMDB 问题:\n{format_lmdb_cache_errors(errors)}"
        )


# torchrun + DDP 场景下的前置检查
def preflight_navigation_data(config) -> None:
    # 只有 torchrun 分布式启动时才做强制 LMDB preflight；单卡/单进程允许 auto 构建
    if is_torchrun() and bool(config["runtime"].get("distributed", True)):
        require_lmdb_ready_before_ddp(config)


# 仅构建 LMDB 缓存的快捷入口
def build_data_module(config, context):
    module_name = str(config.get("data", {}).get("module_name", "navigation")).lower()
    return data_module_registry.build(module_name, config, context)


def handle_build_lmdb_only(config, context) -> bool:
    if not bool(config["runtime"].get("build_lmdb_only", False)):
        return False
    datamodule = build_data_module(config, context)
    datamodule.setup(build_lmdb_only=True)
    return True


# 数据模块：构建数据集与 DataLoader，并处理 LMDB 缓存策略
@data_module_registry.register("navigation")
class NavigationDataModule:
    """Build navigation datasets and DataLoaders with DDP-safe LMDB semantics."""

    # 初始化数据模块并设置图像归一化变换
    def __init__(self, config, context: RuntimeContext) -> None:
        self.config = config
        self.context = context
        # Dataset 返回 [0,1] 图像；训练前统一做 ImageNet normalize，匹配 EfficientNet/MobileNet 预训练分布
        self.transform = transforms.Compose(
            [
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.train_loader = None
        self.test_dataloaders: Dict[str, DataLoader] = {}
        self.train_sampler = None
        self.train_dataset = None
        self.loader_summary = None

    def set_train_epoch(self, epoch: int) -> None:
        if self.train_sampler is not None and hasattr(self.train_sampler, "set_epoch"):
            self.train_sampler.set_epoch(epoch)
        if self.train_dataset is not None and hasattr(self.train_dataset, "set_epoch"):
            self.train_dataset.set_epoch(epoch)

    def set_eval_epoch(self, epoch: int) -> None:
        for loader in self.test_dataloaders.values():
            sampler = getattr(loader, "sampler", None)
            dataset = getattr(loader, "dataset", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            if dataset is not None and hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)

    # 决定 LMDB 缓存模式（build/read/auto）
    def _lmdb_cache_mode(self, build_lmdb_only: bool) -> str:
        runtime = self.config["runtime"]
        mode = str(runtime.get("lmdb_cache_mode", "auto")).lower()
        if build_lmdb_only:
            # 命令行 --build-lmdb-only 明确要求只构建缓存，不进入训练
            return "build"
        if self.context.distributed and bool(runtime.get("require_lmdb_ready_for_ddp", True)):
            # DDP 训练阶段只读缓存，避免多进程抢写同一个 LMDB
            return "read"
        return mode

    def _prepare_data_defaults(self) -> None:
        data = self.config["data"]
        # 旧配置可能缺少这两个字段，在进入 Dataset 前补齐
        data.setdefault("context_type", "temporal")
        data.setdefault("clip_goals", False)
        data.setdefault("module_name", "navigation")
        data.setdefault("obs_type", "image")
        data.setdefault("goal_type", "image")

    # 构建训练与测试数据集，并记录元数据
    def _build_datasets(self, build_lmdb_only: bool):
        config = self.config
        data = config["data"]
        runtime = config["runtime"]
        train_dataset = []
        test_datasets = {}
        dataset_metadata = {}
        dataset_metadata_by_name = {}
        distributed_eval = bool(runtime.get("distributed_eval", False))
        lmdb_cache_mode = self._lmdb_cache_mode(build_lmdb_only)
        all_data_config = load_navigation_data_config(data.get("data_config_path"))
        cache_config = build_lmdb_cache_config(runtime, cache_mode=lmdb_cache_mode)

        # 非主进程等待主进程完成缓存或准备
        if self.context.distributed and not self.context.is_main_process:
            barrier()

        for dataset_name, data_config in data["datasets"].items():
            # 对缺省字段进行默认补齐
            # 这里直接 setdefault 到 config 副本，后续日志打印能看到最终生效值
            data_config.setdefault("negative_mining", True)
            data_config.setdefault("goals_per_obs", 1)
            data_config.setdefault("end_slack", 0)
            data_config.setdefault("waypoint_spacing", 1)

            for split in ("train", "test"):
                # 非主进程在非分布式评估时跳过测试集构建
                if self.context.distributed and not self.context.is_main_process and split != "train" and not distributed_eval:
                    continue
                if split not in data_config:
                    continue

                # 构建单个数据集实例
                # 传入的距离/action 配置分别控制目标采样距离桶和动作损失有效区间
                dataset = NavigationDataset(
                    spec=build_navigation_dataset_spec(
                        data=data,
                        dataset_config=data_config,
                        dataset_name=dataset_name,
                        split=split,
                        all_data_config=all_data_config,
                    ),
                    cache_config=cache_config,
                )
                metadata = navigation_dataset_metadata(dataset)
                # 按 dataset_index 和 dataset_name 各存一份，方便 batch 级查找或人工读日志
                dataset_metadata[str(dataset.dataset_index)] = metadata
                dataset_metadata_by_name[dataset.dataset_name] = metadata
                if split == "train":
                    train_dataset.append(dataset)
                else:
                    test_datasets[f"{dataset_name}_{split}"] = dataset

        data["dataset_metadata"] = dataset_metadata
        data["dataset_metadata_by_name"] = dataset_metadata_by_name
        # 主进程完成后唤醒其他进程
        if self.context.distributed and self.context.is_main_process:
            barrier()
        return train_dataset, test_datasets

    def _build_train_dataset(self, train_datasets) -> tuple[Dataset, int, int]:
        runtime = self.config["runtime"]
        if not train_datasets:
            raise RuntimeError("未配置任何训练数据集，请检查 data.datasets 中的 train split。")
        # 多数据集训练先 concat，再按稳定随机子集裁剪，保证比例应用在整体训练集上
        dataset, subset_size, total_size = apply_train_subset(
            ConcatDataset(train_datasets),
            subset_fraction=float(runtime.get("train_subset", 1.0)),
            seed=int(runtime.get("seed", 0)),
        )
        # 包装成 epoch-aware dataset，使每个 __getitem__ 能感知当前 epoch/index
        dataset = EpochAwareDataset(dataset, seed=int(runtime.get("seed", 0)))
        self.train_dataset = dataset
        return dataset, subset_size, total_size

    def _build_train_loader(self, *, train_dataset: Dataset, train_batch_size: int, train_num_workers: int, pin_memory: bool, prefetch_factor: int):
        runtime = self.config["runtime"]
        # sampler 负责 shuffle/DDP 分片，并把 epoch 一起传给 dataset
        self.train_sampler = build_epoch_sampler(
            train_dataset,
            distributed=self.context.distributed,
            world_size=self.context.world_size,
            rank=self.context.rank,
            seed=int(runtime.get("seed", 0)),
        )
        persistent_workers = bool(runtime.get("persistent_workers", train_num_workers > 0))
        return DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            # shuffle 由 sampler 控制，不能同时传 shuffle=True
            shuffle=False,
            sampler=self.train_sampler,
            drop_last=False,
            worker_init_fn=seed_worker,
            generator=_seeded_generator(int(runtime.get("seed", 0))),
            collate_fn=navigation_collate,
            **loader_kwargs(train_num_workers, pin_memory, prefetch_factor, persistent_workers),
        )

    def _build_eval_loaders(self, *, test_datasets, test_num_workers: int, pin_memory: bool, prefetch_factor: int) -> None:
        runtime = self.config["runtime"]
        distributed_eval = bool(runtime.get("distributed_eval", False))
        for dataset_type, dataset in test_datasets.items():
            if self.context.distributed and distributed_eval:
                # 分布式评估时手动按 rank 切片；之后 MetricStore.reduce_distributed 聚合指标
                dataset = Subset(dataset, list(range(self.context.rank, len(dataset), self.context.world_size)))
            # 评估使用独立 seed 区间，避免和训练采样上下文重叠
            eval_seed = int(runtime.get("seed", 0)) + 10_000
            dataset = EpochAwareDataset(dataset, seed=eval_seed)
            # 评估不打乱顺序，但仍通过 EpochAwareSampler 传递 epoch
            test_sampler = EpochAwareSampler(dataset, shuffle=False, seed=eval_seed)
            self.test_dataloaders[dataset_type] = DataLoader(
                dataset,
                batch_size=runtime["eval_batch_size"],
                shuffle=False,
                sampler=test_sampler,
                drop_last=False,
                worker_init_fn=seed_worker,
                generator=_seeded_generator(int(runtime.get("seed", 0)), offset=10_000),
                collate_fn=navigation_collate,
                **loader_kwargs(
                    test_num_workers,
                    pin_memory,
                    prefetch_factor,
                    bool(runtime.get("test_persistent_workers", test_num_workers > 0)),
                ),
            )

    def _log_loader_summary(self, *, train_num_workers: int, test_num_workers: int, pin_memory: bool) -> None:
        if not self.context.is_main_process:
            return
        runtime = self.config["runtime"]
        self.loader_summary = (
            "导航 DataLoader 配置: "
            f"global_batch_size={runtime['global_batch_size']}, "
            f"per_device_batch_size={runtime['per_device_batch_size']}, "
            f"eval_batch_size={runtime['eval_batch_size']}, "
            f"train_subset_size={runtime.get('train_subset_size')}/{runtime.get('train_subset_total_size')}, "
            f"train_num_workers_per_rank={train_num_workers}, "
            f"test_num_workers_per_rank={test_num_workers}, "
            f"distributed_eval={bool(runtime.get('distributed_eval', False))}, "
            f"pin_memory={pin_memory}"
        )

    # 初始化 DataLoader，支持仅构建缓存模式
    def setup(self, build_lmdb_only: bool = False) -> None:
        runtime = self.config["runtime"]
        self._prepare_data_defaults()
        train_datasets, test_datasets = self._build_datasets(build_lmdb_only)
        if build_lmdb_only:
            # build-lmdb-only 只需要触发 Dataset 初始化和缓存构建，不创建 DataLoader
            return

        train_dataset, train_subset_size, train_subset_total_size = self._build_train_dataset(train_datasets)
        train_num_workers = workers_per_rank(runtime, self.context.distributed, self.context.world_size, train=True)
        # 计算有效 batch/workers 等字段并写回 runtime，确保 Trainer/日志看到的是实际值
        effective_runtime = resolve_data_runtime(
            runtime,
            distributed=self.context.distributed,
            world_size=self.context.world_size,
            train_subset_size=train_subset_size,
            train_subset_total_size=train_subset_total_size,
            train_num_workers=train_num_workers,
        )
        runtime.update(effective_runtime)

        # DataLoader 性能参数集中从 runtime 读取，方便在 YAML 中统一调参
        pin_memory = bool(runtime.get("pin_memory", torch.cuda.is_available()))
        prefetch_factor = int(runtime.get("prefetch_factor", 2))
        self.train_loader = self._build_train_loader(
            train_dataset=train_dataset,
            train_batch_size=runtime["per_device_batch_size"],
            train_num_workers=train_num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
        )

        distributed_eval = bool(runtime.get("distributed_eval", False))
        # 评估 worker 可以独立配置，避免测试阶段和训练阶段吞吐需求绑定
        test_num_workers = workers_per_rank(runtime, self.context.distributed and distributed_eval, self.context.world_size, train=False)
        runtime["test_num_workers_per_rank"] = int(test_num_workers)
        self._log_loader_summary(train_num_workers=train_num_workers, test_num_workers=test_num_workers, pin_memory=pin_memory)
        self._build_eval_loaders(
            test_datasets=test_datasets,
            test_num_workers=test_num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
        )
