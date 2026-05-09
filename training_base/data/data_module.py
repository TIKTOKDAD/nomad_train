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
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from training_base.core.dataloader import loader_kwargs, workers_per_rank
from training_base.core.runtime import RuntimeContext, barrier, is_torchrun, seed_worker
from training_base.data.batch import navigation_collate
from training_base.data.labeling import sample_context
from training_base.data.lmdb_cache import check_lmdb_cache_ready
from training_base.data.navigation_dataset import NavigationDataset


def _seeded_generator(seed: int, offset: int = 0) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed) + int(offset))
    return generator


def stable_subset_indices(dataset_size: int, subset_fraction: float, seed: int) -> list:
    if dataset_size <= 0:
        return []
    subset_size = max(1, int(dataset_size * float(subset_fraction)))
    subset_size = min(subset_size, dataset_size)
    return torch.randperm(dataset_size, generator=_seeded_generator(seed))[:subset_size].tolist()


class EpochAwareDataset(Dataset):
    """Wrap a dataset so each item receives seed/epoch/index sampling context."""

    def __init__(self, dataset: Dataset, *, seed: int) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        epoch = self.epoch
        sample_index = index
        if isinstance(index, tuple) and len(index) == 2:
            epoch, sample_index = index
        with sample_context(seed=self.seed, epoch=int(epoch), index=int(sample_index)):
            return self.dataset[int(sample_index)]


class EpochAwareSampler(Sampler):
    """Sequential or shuffled sampler that carries epoch into dataset indexing."""

    def __init__(self, data_source: Dataset, *, shuffle: bool, seed: int, offset: int = 0) -> None:
        self.data_source = data_source
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.offset = int(offset)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        dataset_size = len(self.data_source)
        if self.shuffle:
            indices = torch.randperm(dataset_size, generator=_seeded_generator(self.seed, self.offset + self.epoch)).tolist()
        else:
            indices = list(range(dataset_size))
        return iter((self.epoch, int(index)) for index in indices)

    def __len__(self) -> int:
        return len(self.data_source)


class EpochAwareDistributedSampler(DistributedSampler):
    """DistributedSampler variant that also passes epoch to the dataset."""

    def __iter__(self):
        return iter((self.epoch, int(index)) for index in super().__iter__())


def apply_train_subset(dataset: Dataset, runtime: dict) -> Dataset:
    original_size = len(dataset)
    subset_fraction = float(runtime.get("train_subset", 1.0))
    if subset_fraction < 1.0:
        indices = stable_subset_indices(original_size, subset_fraction, int(runtime.get("seed", 0)))
        dataset = Subset(dataset, indices)
        runtime["train_subset_size"] = len(indices)
    else:
        runtime["train_subset_size"] = len(dataset)
    runtime["train_subset_total_size"] = original_size
    return dataset


def resolve_train_batch_size(runtime: dict, *, distributed: bool, world_size: int) -> tuple[int, int]:
    configured_global_batch_size = runtime.get("global_batch_size")
    global_batch_size = int(runtime["batch_size"] if configured_global_batch_size is None else configured_global_batch_size)
    per_device_batch_size = global_batch_size
    if distributed:
        if per_device_batch_size % world_size != 0:
            raise ValueError(
                f"全局 batch_size={per_device_batch_size} 必须能被 world_size={world_size} 整除，以保持 DDP batch 语义。"
            )
        per_device_batch_size = per_device_batch_size // world_size
    runtime["global_batch_size"] = global_batch_size
    runtime["batch_size"] = global_batch_size
    runtime["per_device_batch_size"] = per_device_batch_size
    return global_batch_size, per_device_batch_size


def build_epoch_sampler(dataset: Dataset, *, distributed: bool, world_size: int, rank: int, seed: int):
    if distributed:
        return EpochAwareDistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=int(seed),
            drop_last=False,
        )
    return EpochAwareSampler(dataset, shuffle=True, seed=int(seed))


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
def handle_build_lmdb_only(config, context) -> bool:
    if not bool(config["runtime"].get("build_lmdb_only", False)):
        return False
    datamodule = NavigationDataModule(config, context)
    datamodule.setup(build_lmdb_only=True)
    return True


# 数据模块：构建数据集与 DataLoader，并处理 LMDB 缓存策略
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
                    data_folder=data_config["data_folder"],
                    data_split_folder=data_config[split],
                    dataset_name=dataset_name,
                    image_size=data["image_size"],
                    waypoint_spacing=data_config["waypoint_spacing"],
                    min_dist_cat=data["distance"]["min_dist_cat"],
                    max_dist_cat=data["distance"]["max_dist_cat"],
                    min_action_distance=data["action"]["min_dist_cat"],
                    max_action_distance=data["action"]["max_dist_cat"],
                    negative_mining=data_config["negative_mining"],
                    len_traj_pred=data["len_traj_pred"],
                    learn_angle=data["learn_angle"],
                    context_size=data["context_size"],
                    context_type=data.get("context_type", "temporal"),
                    end_slack=data_config["end_slack"],
                    goals_per_obs=data_config["goals_per_obs"],
                    normalize=data["normalize"],
                    goal_type=data["goal_type"],
                    lmdb_lock=bool(runtime.get("lmdb_lock", False)),
                    lmdb_readahead=bool(runtime.get("lmdb_readahead", False)),
                    lmdb_meminit=bool(runtime.get("lmdb_meminit", False)),
                    lmdb_max_readers=int(runtime.get("lmdb_max_readers", 512)),
                    lmdb_cache_mode=lmdb_cache_mode,
                    rebuild_incomplete_lmdb=bool(runtime.get("rebuild_incomplete_lmdb", False)),
                    data_config_path=data.get("data_config_path"),
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

    # 初始化 DataLoader，支持仅构建缓存模式
    def setup(self, build_lmdb_only: bool = False) -> None:
        config = self.config
        runtime = config["runtime"]
        data = config["data"]
        # 旧配置可能缺少这两个字段，在进入 Dataset 前补齐
        data.setdefault("context_type", "temporal")
        data.setdefault("clip_goals", False)

        train_datasets, test_datasets = self._build_datasets(build_lmdb_only)
        if build_lmdb_only:
            return

        # 合并多个训练数据集为一个 ConcatDataset
        if not train_datasets:
            raise RuntimeError("未配置任何训练数据集，请检查 data.datasets 中的 train split。")
        train_dataset = apply_train_subset(ConcatDataset(train_datasets), runtime)
        train_dataset = EpochAwareDataset(train_dataset, seed=int(runtime.get("seed", 0)))
        self.train_dataset = train_dataset
        # global_batch_size 若设置，代表跨所有 DDP rank 的总 batch；否则沿用 runtime.batch_size
        global_batch_size, train_batch_size = resolve_train_batch_size(
            runtime,
            distributed=self.context.distributed,
            world_size=self.context.world_size,
        )
        self.train_sampler = build_epoch_sampler(
            train_dataset,
            distributed=self.context.distributed,
            world_size=self.context.world_size,
            rank=self.context.rank,
            seed=int(runtime.get("seed", 0)),
        )

        # DataLoader 性能参数集中从 runtime 读取，方便在 YAML 中统一调参
        pin_memory = bool(runtime.get("pin_memory", torch.cuda.is_available()))
        prefetch_factor = int(runtime.get("prefetch_factor", 2))
        train_num_workers = workers_per_rank(runtime, self.context.distributed, self.context.world_size, train=True)
        runtime["num_workers_per_rank"] = train_num_workers
        persistent_workers = bool(runtime.get("persistent_workers", train_num_workers > 0))

        # 训练 DataLoader
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            drop_last=False,
            worker_init_fn=seed_worker,
            generator=_seeded_generator(int(runtime.get("seed", 0))),
            collate_fn=navigation_collate,
            **loader_kwargs(train_num_workers, pin_memory, prefetch_factor, persistent_workers),
        )

        runtime.setdefault("eval_batch_size", runtime["batch_size"])
        distributed_eval = bool(runtime.get("distributed_eval", False))
        # 评估 worker 可以独立配置，避免测试阶段和训练阶段吞吐需求绑定
        test_num_workers = workers_per_rank(runtime, self.context.distributed and distributed_eval, self.context.world_size, train=False)

        if self.context.is_main_process:
            print(
                "导航 DataLoader 配置: "
                f"global_batch_size={global_batch_size}, "
                f"per_device_batch_size={train_batch_size}, "
                f"eval_batch_size={runtime['eval_batch_size']}, "
                f"train_subset_size={runtime.get('train_subset_size')}/{runtime.get('train_subset_total_size')}, "
                f"train_num_workers_per_rank={train_num_workers}, "
                f"test_num_workers={test_num_workers}, "
                f"distributed_eval={distributed_eval}, "
                f"pin_memory={pin_memory}"
            )

        # 构建测试集 DataLoader
        for dataset_type, dataset in test_datasets.items():
            if self.context.distributed and distributed_eval:
                # 分布式评估时手动按 rank 切片；之后 MetricStore.reduce_distributed 聚合指标
                dataset = Subset(dataset, list(range(self.context.rank, len(dataset), self.context.world_size)))
            eval_seed = int(runtime.get("seed", 0)) + 10_000
            dataset = EpochAwareDataset(dataset, seed=eval_seed)
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
