from copy import deepcopy
from typing import Dict, Iterable

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from training_base.core.dataloader import loader_kwargs, workers_per_rank
from training_base.core.runtime import RuntimeContext, barrier, is_torchrun, seed_worker
from training_base.data.batch import navigation_collate
from training_base.data.navigation_dataset import NavigationDataset, check_lmdb_cache_ready


def configured_dataset_splits(config) -> Iterable[dict]:
    context_type = config["data"].get("context_type", "temporal")
    distance = config["data"]["distance"]
    for dataset_name, data_config in config["data"]["datasets"].items():
        waypoint_spacing = data_config.get("waypoint_spacing", 1)
        end_slack = data_config.get("end_slack", 0)
        for split in ("train", "test"):
            if split not in data_config:
                continue
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


def format_lmdb_cache_errors(errors) -> str:
    lines = []
    for item in errors:
        lines.append(f"- {item['dataset_name']}:{item['split']} ({item['data_split_folder']})")
        for problem in item["problems"]:
            lines.append(f"  * {problem}")
    return "\n".join(lines)


def check_lmdb_caches_ready(config):
    errors = []
    for split_config in configured_dataset_splits(config):
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


def navigation_dataset_metadata(dataset: NavigationDataset) -> dict:
    data_config = dataset.data_config or {}
    return {
        "name": dataset.dataset_name,
        "dataset_name": dataset.dataset_name,
        "dataset_index": int(dataset.dataset_index),
        "metric_waypoint_spacing": float(data_config.get("metric_waypoint_spacing", 1.0)),
        "waypoint_spacing": int(dataset.waypoint_spacing),
        "metric_scale": float(dataset.metric_scale),
        "camera_metrics": deepcopy(data_config.get("camera_metrics", {})),
    }


def require_lmdb_ready_before_ddp(config) -> None:
    runtime = config["runtime"]
    if not bool(runtime.get("require_lmdb_ready_for_ddp", True)):
        return
    errors = check_lmdb_caches_ready(config)
    if errors:
        raise RuntimeError(
            "DDP training requires all LMDB caches to be prebuilt and complete.\n"
            "Run this first with a single process:\n"
            "  python -m training_base.cli -c training_base/configs/nomad_retrain.yaml --build-lmdb-only\n"
            "If a previous build was interrupted, add:\n"
            "  --rebuild-incomplete-lmdb\n\n"
            f"LMDB problems:\n{format_lmdb_cache_errors(errors)}"
        )


def preflight_navigation_data(config) -> None:
    if is_torchrun() and bool(config["runtime"].get("distributed", True)):
        require_lmdb_ready_before_ddp(config)


def handle_build_lmdb_only(config, context) -> bool:
    if not bool(config["runtime"].get("build_lmdb_only", False)):
        return False
    datamodule = NavigationDataModule(config, context)
    datamodule.setup(build_lmdb_only=True)
    return True


class NavigationDataModule:
    """Build navigation datasets and DataLoaders with DDP-safe LMDB semantics."""

    def __init__(self, config, context: RuntimeContext) -> None:
        self.config = config
        self.context = context
        self.transform = transforms.Compose(
            [
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.train_loader = None
        self.test_dataloaders: Dict[str, DataLoader] = {}
        self.train_sampler = None

    def _lmdb_cache_mode(self, build_lmdb_only: bool) -> str:
        runtime = self.config["runtime"]
        mode = str(runtime.get("lmdb_cache_mode", "auto")).lower()
        if build_lmdb_only:
            return "build"
        if self.context.distributed and bool(runtime.get("require_lmdb_ready_for_ddp", True)):
            return "read"
        return mode

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

        if self.context.distributed and not self.context.is_main_process:
            barrier()

        for dataset_name, data_config in data["datasets"].items():
            data_config.setdefault("negative_mining", True)
            data_config.setdefault("goals_per_obs", 1)
            data_config.setdefault("end_slack", 0)
            data_config.setdefault("waypoint_spacing", 1)

            for split in ("train", "test"):
                if self.context.distributed and not self.context.is_main_process and split != "train" and not distributed_eval:
                    continue
                if split not in data_config:
                    continue

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
                )
                metadata = navigation_dataset_metadata(dataset)
                dataset_metadata[str(dataset.dataset_index)] = metadata
                dataset_metadata_by_name[dataset.dataset_name] = metadata
                if split == "train":
                    train_dataset.append(dataset)
                else:
                    test_datasets[f"{dataset_name}_{split}"] = dataset

        data["dataset_metadata"] = dataset_metadata
        data["dataset_metadata_by_name"] = dataset_metadata_by_name
        if self.context.distributed and self.context.is_main_process:
            barrier()
        return train_dataset, test_datasets

    def setup(self, build_lmdb_only: bool = False) -> None:
        config = self.config
        runtime = config["runtime"]
        data = config["data"]
        data.setdefault("context_type", "temporal")
        data.setdefault("clip_goals", False)

        train_datasets, test_datasets = self._build_datasets(build_lmdb_only)
        if build_lmdb_only:
            return

        train_dataset = ConcatDataset(train_datasets)
        configured_global_batch_size = runtime.get("global_batch_size")
        global_batch_size = int(runtime["batch_size"] if configured_global_batch_size is None else configured_global_batch_size)
        runtime["global_batch_size"] = global_batch_size
        runtime["batch_size"] = global_batch_size

        train_batch_size = global_batch_size
        if self.context.distributed:
            if train_batch_size % self.context.world_size != 0:
                raise ValueError(
                    f"Global batch_size={train_batch_size} must be divisible by "
                    f"world_size={self.context.world_size} to preserve DDP batch semantics."
                )
            train_batch_size = train_batch_size // self.context.world_size
            self.train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.context.world_size,
                rank=self.context.rank,
                shuffle=True,
                drop_last=False,
            )
        runtime["per_device_batch_size"] = train_batch_size

        pin_memory = bool(runtime.get("pin_memory", torch.cuda.is_available()))
        prefetch_factor = int(runtime.get("prefetch_factor", 2))
        train_num_workers = workers_per_rank(runtime, self.context.distributed, self.context.world_size, train=True)
        runtime["num_workers_per_rank"] = train_num_workers
        persistent_workers = bool(runtime.get("persistent_workers", train_num_workers > 0))

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=self.train_sampler is None,
            sampler=self.train_sampler,
            drop_last=False,
            worker_init_fn=seed_worker,
            collate_fn=navigation_collate,
            **loader_kwargs(train_num_workers, pin_memory, prefetch_factor, persistent_workers),
        )

        runtime.setdefault("eval_batch_size", runtime["batch_size"])
        distributed_eval = bool(runtime.get("distributed_eval", False))
        test_num_workers = workers_per_rank(runtime, self.context.distributed and distributed_eval, self.context.world_size, train=False)

        if self.context.is_main_process:
            print(
                "Navigation DataLoader config: "
                f"global_batch_size={global_batch_size}, "
                f"per_device_batch_size={train_batch_size}, "
                f"eval_batch_size={runtime['eval_batch_size']}, "
                f"train_num_workers_per_rank={train_num_workers}, "
                f"test_num_workers={test_num_workers}, "
                f"distributed_eval={distributed_eval}, "
                f"pin_memory={pin_memory}"
            )

        for dataset_type, dataset in test_datasets.items():
            if self.context.distributed and distributed_eval:
                dataset = Subset(dataset, list(range(self.context.rank, len(dataset), self.context.world_size)))
            self.test_dataloaders[dataset_type] = DataLoader(
                dataset,
                batch_size=runtime["eval_batch_size"],
                shuffle=False,
                drop_last=False,
                worker_init_fn=seed_worker,
                collate_fn=navigation_collate,
                **loader_kwargs(
                    test_num_workers,
                    pin_memory,
                    prefetch_factor,
                    bool(runtime.get("test_persistent_workers", test_num_workers > 0)),
                ),
            )
