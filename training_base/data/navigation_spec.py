# ============================================================
# Navigation dataset specs - typed construction contract
# ============================================================
# DataModule 负责把 YAML/runtime 翻译成这些 dataclass；NavigationDataset
# 只消费已经整理好的 spec，避免构造函数继续膨胀。

from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class NavigationDistanceConfig:
    min_dist_cat: int
    max_dist_cat: int


@dataclass(frozen=True)
class NavigationActionConfig:
    min_dist_cat: int
    max_dist_cat: int


@dataclass(frozen=True)
class NavigationContextConfig:
    context_type: str
    context_size: int


@dataclass(frozen=True)
class NavigationDatasetMetadata:
    dataset_index: int
    data_config: Dict[str, Any]
    metric_scale: float


@dataclass(frozen=True)
class LmdbCacheConfig:
    lock: bool = False
    readahead: bool = False
    meminit: bool = False
    max_readers: int = 512
    map_size: int = 2 ** 40
    cache_mode: str = "auto"
    rebuild_incomplete: bool = False


@dataclass(frozen=True)
class NavigationDatasetSpec:
    data_folder: str
    data_split_folder: str
    dataset_name: str
    image_size: Tuple[int, int]
    waypoint_spacing: int
    distance: NavigationDistanceConfig
    action: NavigationActionConfig
    negative_mining: bool
    len_traj_pred: int
    learn_angle: bool
    context: NavigationContextConfig
    end_slack: int
    goals_per_obs: int
    normalize: bool
    obs_type: str
    goal_type: str
    image_aspect_ratio: float
    goal_sampling: dict
    metadata: NavigationDatasetMetadata
