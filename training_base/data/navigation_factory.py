# ============================================================
# Navigation dataset factory - config to dataset spec
# ============================================================
# 本文件把 YAML/runtime 字段翻译成 NavigationDatasetSpec，集中处理默认值、
# 数据集元信息读取和 LMDB 参数，保持 DataModule 和 Dataset 边界清楚。

from copy import deepcopy
import os
from typing import Dict

import yaml

from training_base.core.image_size import as_width_height
from training_base.data.navigation_spec import (
    LmdbCacheConfig,
    NavigationActionConfig,
    NavigationContextConfig,
    NavigationDatasetMetadata,
    NavigationDatasetSpec,
    NavigationDistanceConfig,
)


def resolve_navigation_data_config_path(path: str = None) -> str:
    config_path = path or os.path.join(os.path.dirname(__file__), "data_config.yaml")
    return config_path if os.path.isabs(config_path) else os.path.abspath(config_path)


def load_navigation_data_config(path: str = None) -> Dict[str, dict]:
    config_path = resolve_navigation_data_config_path(path)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_lmdb_cache_config(runtime: dict, *, cache_mode: str) -> LmdbCacheConfig:
    return LmdbCacheConfig(
        lock=bool(runtime.get("lmdb_lock", False)),
        readahead=bool(runtime.get("lmdb_readahead", False)),
        meminit=bool(runtime.get("lmdb_meminit", False)),
        max_readers=int(runtime.get("lmdb_max_readers", 512)),
        map_size=int(runtime.get("lmdb_map_size", 2 ** 40)),
        cache_mode=str(cache_mode).lower(),
        rebuild_incomplete=bool(runtime.get("rebuild_incomplete_lmdb", False)),
    )


def build_navigation_metadata(
    *,
    all_data_config: Dict[str, dict],
    dataset_name: str,
    waypoint_spacing: int,
    data_config_path: str = None,
) -> NavigationDatasetMetadata:
    if dataset_name not in all_data_config:
        source = resolve_navigation_data_config_path(data_config_path)
        raise KeyError(f"在数据集配置 {source} 中找不到数据集 {dataset_name}")
    dataset_names = sorted(all_data_config.keys())
    data_config = deepcopy(all_data_config[dataset_name])
    metric_scale = float(data_config.get("metric_waypoint_spacing", 1.0)) * float(waypoint_spacing)
    return NavigationDatasetMetadata(
        dataset_index=dataset_names.index(dataset_name),
        data_config=data_config,
        metric_scale=metric_scale,
    )


def build_navigation_dataset_spec(
    *,
    data: dict,
    dataset_config: dict,
    dataset_name: str,
    split: str,
    all_data_config: Dict[str, dict],
) -> NavigationDatasetSpec:
    waypoint_spacing = int(dataset_config.get("waypoint_spacing", 1))
    return NavigationDatasetSpec(
        data_folder=dataset_config["data_folder"],
        data_split_folder=dataset_config[split],
        dataset_name=dataset_name,
        image_size=as_width_height(data["image_size"], "data.image_size"),
        waypoint_spacing=waypoint_spacing,
        distance=NavigationDistanceConfig(
            min_dist_cat=int(data["distance"]["min_dist_cat"]),
            max_dist_cat=int(data["distance"]["max_dist_cat"]),
        ),
        action=NavigationActionConfig(
            min_dist_cat=int(data["action"]["min_dist_cat"]),
            max_dist_cat=int(data["action"]["max_dist_cat"]),
        ),
        negative_mining=bool(dataset_config.get("negative_mining", True)),
        len_traj_pred=int(data["len_traj_pred"]),
        learn_angle=bool(data["learn_angle"]),
        context=NavigationContextConfig(
            context_type=str(data.get("context_type", "temporal")).lower(),
            context_size=int(data["context_size"]),
        ),
        end_slack=int(dataset_config.get("end_slack", 0)),
        goals_per_obs=int(dataset_config.get("goals_per_obs", 1)),
        normalize=bool(data["normalize"]),
        obs_type=str(data.get("obs_type", "image")).lower(),
        goal_type=str(data.get("goal_type", "image")).lower(),
        image_aspect_ratio=float(data.get("image_aspect_ratio", 4 / 3)),
        goal_sampling=data.get("goal_sampling"),
        metadata=build_navigation_metadata(
            all_data_config=all_data_config,
            dataset_name=dataset_name,
            waypoint_spacing=waypoint_spacing,
            data_config_path=data.get("data_config_path"),
        ),
    )
