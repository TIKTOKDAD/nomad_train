# ============================================================
# Config loading - YAML merge, validation, and CLI overrides
# ============================================================
# 本文件负责训练配置的入口处理：
# 1. 读取 defaults.yaml 和用户 YAML
# 2. 深度合并配置，用户字段覆盖默认字段
# 3. 校验必需顶层字段，并把 registry key 统一小写
# 4. 将 CLI 开关写回 runtime 配置

import argparse
import os
import shutil
from copy import deepcopy
from typing import Any, Dict, Sequence

import yaml

from training_base.core.image_size import as_width_height


# 配置文件必须包含的顶层字段
REQUIRED_SECTIONS = (
    "runtime",
    "data",
    "model",
    "objective",
    "optimizer",
    "scheduler",
    "metrics",
    "logging",
    "visualization",
    "callbacks",
    "algorithm",
)


LEGACY_LOGGING_FIELD_PATHS = (
    ("metric_log_freq", ("train", "metrics", "freq")),
    ("heavy_metric_log_freq", ("train", "behavior", "freq")),
    ("heavy_metric_start_step", ("train", "behavior", "start_step")),
    ("image_log_freq", ("media", "train", "freq")),
    ("image_start_step", ("media", "train", "start_step")),
    ("perf_log_freq", ("runtime", "perf", "freq")),
    ("optim_log_freq", ("train", "optim", "freq")),
    ("param_norm_log_freq", ("train", "param_norm", "freq")),
    ("eval_freq", ("eval", "schedule", "freq")),
    ("eval_fraction", ("eval", "schedule", "fraction")),
    ("by_global_step", ("step", "by_global_step")),
    ("first_step", ("step", "first_step")),
)


LEGACY_RUNTIME_LOGGING_FIELD_PATHS = (
    ("eval_freq", ("eval", "schedule", "freq")),
    ("eval_fraction", ("eval", "schedule", "fraction")),
)


LOGGING_SCHEDULE_PATHS = (
    ("logging.train.metrics", ("logging", "train", "metrics")),
    ("logging.train.behavior", ("logging", "train", "behavior")),
    ("logging.train.optim", ("logging", "train", "optim")),
    ("logging.train.param_norm", ("logging", "train", "param_norm")),
    ("logging.eval.schedule", ("logging", "eval", "schedule")),
    ("logging.eval.behavior", ("logging", "eval", "behavior")),
    ("logging.media.train", ("logging", "media", "train")),
    ("logging.media.eval", ("logging", "media", "eval")),
    ("logging.runtime.perf", ("logging", "runtime", "perf")),
    ("logging.system.gpu", ("logging", "system", "gpu")),
)

LOGGING_UNITS = {"step", "epoch", "eval"}


def _set_nested_if_missing(mapping: Dict[str, Any], path: Sequence[str], value: Any) -> None:
    current = mapping
    for key in path[:-1]:
        child = current.get(key)
        if not isinstance(child, dict):
            child = {}
            current[key] = child
        current = child
    current.setdefault(path[-1], value)


def _pop_nested(mapping: Dict[str, Any], path: Sequence[str]) -> Any:
    current = mapping
    for key in path[:-1]:
        child = current.get(key)
        if not isinstance(child, dict):
            return None
        current = child
    return current.pop(path[-1], None)


def _nested_get(mapping: Dict[str, Any], path: Sequence[str]) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _prune_empty_dicts(mapping: Dict[str, Any]) -> None:
    for key, value in list(mapping.items()):
        if isinstance(value, dict):
            _prune_empty_dicts(value)
            if not value:
                mapping.pop(key)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _validate_positive_int(value: Any, field_name: str) -> None:
    if int(value) <= 0:
        raise ValueError(f"{field_name} 必须是正整数，实际为 {value!r}")


def _validate_non_negative_int(value: Any, field_name: str) -> None:
    if int(value) < 0:
        raise ValueError(f"{field_name} 必须是非负整数，实际为 {value!r}")


def _validate_logging_schedules(config: Dict[str, Any]) -> None:
    for field_name, path in LOGGING_SCHEDULE_PATHS:
        section = _nested_get(config, path)
        if section is None:
            continue
        if not isinstance(section, dict):
            raise TypeError(f"{field_name} 必须是映射/字典")
        if "unit" in section and str(section["unit"]).lower() not in LOGGING_UNITS:
            raise ValueError(f"{field_name}.unit 必须是 step、epoch 或 eval 之一，实际为 {section['unit']!r}")
        if "freq" in section:
            _validate_non_negative_int(section["freq"], f"{field_name}.freq")
        if "start_step" in section:
            _validate_non_negative_int(section["start_step"], f"{field_name}.start_step")


def _move_nested_fields(mapping: Dict[str, Any], source: Sequence[str], target: Sequence[str], fields: Sequence[str], default_unit: str = "step") -> None:
    moved = False
    for field in fields:
        value = _pop_nested(mapping, tuple(source) + (field,))
        if value is not None:
            _set_nested_if_missing(mapping, tuple(target) + (field,), value)
            moved = True
    if moved and _nested_get(mapping, tuple(target) + ("unit",)) is None:
        _set_nested_if_missing(mapping, tuple(target) + ("unit",), default_unit)


def _upgrade_nested_logging_config(logging_config: Dict[str, Any]) -> None:
    # 旧版 nested runtime/optim 属于训练健康指标，迁到 train/optim 和 train/model 分区
    _move_nested_fields(logging_config, ("runtime", "optim"), ("train", "optim"), ("freq", "unit"))
    _move_nested_fields(logging_config, ("runtime", "param_norm"), ("train", "param_norm"), ("freq", "unit"))

    # 旧版 eval.behavior_every_eval 是布尔保底策略；新版用 unit=eval 的频率表达
    eval_every = logging_config.pop("eval_heavy_every_eval", None)
    nested_eval = logging_config.get("eval")
    if isinstance(nested_eval, dict):
        nested_value = nested_eval.pop("behavior_every_eval", None)
        if eval_every is None:
            eval_every = nested_value
    if eval_every is not None:
        _set_nested_if_missing(logging_config, ("eval", "behavior", "freq"), 1 if _coerce_bool(eval_every) else 0)
        _set_nested_if_missing(logging_config, ("eval", "behavior", "unit"), "eval")

    # 旧版 GPU 显存开关挂在 runtime.perf 下；新版 system.gpu 可独立控制
    include_gpu_memory = _pop_nested(logging_config, ("runtime", "perf", "include_gpu_memory"))
    if include_gpu_memory is not None:
        _set_nested_if_missing(logging_config, ("system", "gpu", "enabled"), _coerce_bool(include_gpu_memory))
        perf_freq = _nested_get(logging_config, ("runtime", "perf", "freq"))
        if perf_freq is not None:
            _set_nested_if_missing(logging_config, ("system", "gpu", "freq"), perf_freq)
        _set_nested_if_missing(logging_config, ("system", "gpu", "unit"), "step")

    _prune_empty_dicts(logging_config)


def upgrade_logging_config(config: Dict[str, Any]) -> Dict[str, Any]:
    # 兼容旧版 logging/runtime 字段：加载时迁移到按 W&B 板块分组的新结构
    upgraded = deepcopy(config)
    logging_config = upgraded.get("logging")
    if not isinstance(logging_config, dict):
        return upgraded
    for legacy_key, path in LEGACY_LOGGING_FIELD_PATHS:
        if legacy_key in logging_config:
            _set_nested_if_missing(logging_config, path, logging_config.pop(legacy_key))
    runtime_config = upgraded.get("runtime")
    if isinstance(runtime_config, dict):
        for legacy_key, path in LEGACY_RUNTIME_LOGGING_FIELD_PATHS:
            if legacy_key in runtime_config:
                _set_nested_if_missing(logging_config, path, runtime_config.pop(legacy_key))
    _upgrade_nested_logging_config(logging_config)
    return upgraded


def upgrade_goal_sampling_config(config: Dict[str, Any]) -> Dict[str, Any]:
    upgraded = deepcopy(config)
    data = upgraded.get("data")
    if not isinstance(data, dict):
        return upgraded
    goal_sampling = data.setdefault("goal_sampling", {})
    negative = goal_sampling.setdefault("negative", {})
    if "enabled" not in negative:
        legacy_values = [
            bool(dataset_config["negative_mining"])
            for dataset_config in data.get("datasets", {}).values()
            if isinstance(dataset_config, dict) and "negative_mining" in dataset_config
        ]
        negative["enabled"] = all(legacy_values) if legacy_values else True
    negative.setdefault("policy", "offset_zero")
    negative.setdefault("distance_label", "max_dist_cat")
    for dataset_config in data.get("datasets", {}).values():
        if isinstance(dataset_config, dict) and "negative_mining" in dataset_config:
            dataset_config.setdefault("negative_mining", dataset_config["negative_mining"])
    return upgraded


# 读取 YAML 配置文件，空文件返回空字典
def load_yaml(path: str) -> Dict[str, Any]:
    # 所有配置文件都按 UTF-8 读取，保证中文注释不会影响解析
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


# 深度合并配置：用户配置覆盖默认配置
def deep_merge(default_config: Dict[str, Any], user_config: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(default_config)
    for key, value in user_config.items():
        # losses 通常是 objective 内的完整子配置，显式写时整体覆盖，避免默认损失残留
        if key == "losses" and isinstance(value, dict):
            merged[key] = deepcopy(value)
            continue
        # dict + dict 递归合并；其他类型直接用用户值替换默认值
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


# 校验配置结构与关键字段是否存在
def validate_config(config: Dict[str, Any]) -> None:
    # 顶层 section 缺失会导致训练中后段才报错，这里提前失败更容易定位
    missing = [section for section in REQUIRED_SECTIONS if section not in config]
    if missing:
        raise KeyError(f"缺少必需的配置区块: {', '.join(missing)}")
    for section in REQUIRED_SECTIONS:
        # callbacks 设计成列表，因为同一训练可以启用多个回调
        if section == "callbacks":
            if not isinstance(config[section], list):
                raise TypeError("callbacks 必须是列表")
        elif not isinstance(config[section], dict):
            raise TypeError(f"{section} 必须是映射/字典")
    if not config["algorithm"].get("name"):
        raise KeyError("必须配置 algorithm.name")
    if not config["model"].get("name"):
        raise KeyError("必须配置 model.name")
    if not config["objective"].get("name"):
        raise KeyError("必须配置 objective.name")

    runtime = config["runtime"]
    data = config["data"]
    logging_config = config["logging"]

    _validate_positive_int(runtime.get("epochs", 0), "runtime.epochs")
    _validate_positive_int(runtime.get("batch_size", 0), "runtime.batch_size")
    _validate_positive_int(runtime.get("eval_batch_size", runtime.get("batch_size", 0)), "runtime.eval_batch_size")
    _validate_non_negative_int(runtime.get("num_workers", 0), "runtime.num_workers")
    _validate_non_negative_int(runtime.get("test_num_workers", runtime.get("num_workers", 0)), "runtime.test_num_workers")

    train_subset = float(runtime.get("train_subset", 1.0))
    if not (0.0 < train_subset <= 1.0):
        raise ValueError(f"runtime.train_subset 必须在 (0, 1] 范围内，实际为 {train_subset}")

    global_batch_size = runtime.get("global_batch_size")
    if global_batch_size is not None:
        _validate_positive_int(global_batch_size, "runtime.global_batch_size")

    if str(runtime.get("amp_dtype", "fp16")).lower() not in {"fp16", "bf16"}:
        raise ValueError("runtime.amp_dtype 必须是 fp16 或 bf16")
    if str(runtime.get("lmdb_cache_mode", "auto")).lower() not in {"auto", "read", "build"}:
        raise ValueError("runtime.lmdb_cache_mode 必须是 auto、read 或 build 之一")
    _validate_positive_int(runtime.get("lmdb_map_size", 2 ** 40), "runtime.lmdb_map_size")

    context_type = str(data.get("context_type", "temporal")).lower()
    if context_type != "temporal":
        raise ValueError("data.context_type 当前只支持 temporal")
    obs_type = str(data.get("obs_type", "image")).lower()
    goal_type = str(data.get("goal_type", "image")).lower()
    if obs_type != "image":
        raise ValueError("data.obs_type 当前只支持 image；新模态请注册新的 data.module_name")
    if goal_type != "image":
        raise ValueError("data.goal_type 当前只支持 image；新模态请注册新的 data.module_name")
    data["module_name"] = str(data.get("module_name", "navigation")).lower()
    as_width_height(data.get("image_size"), "data.image_size")
    if float(data.get("image_aspect_ratio", 4 / 3)) <= 0:
        raise ValueError("data.image_aspect_ratio 必须大于 0")
    distance = data.get("distance", {})
    action = data.get("action", {})
    if int(distance.get("min_dist_cat", 0)) > int(distance.get("max_dist_cat", 0)):
        raise ValueError("data.distance.min_dist_cat 不能大于 max_dist_cat")
    if int(action.get("min_dist_cat", 0)) > int(action.get("max_dist_cat", 0)):
        raise ValueError("data.action.min_dist_cat 不能大于 max_dist_cat")
    goal_negative = data.get("goal_sampling", {}).get("negative", {})
    if str(goal_negative.get("policy", "offset_zero")).lower() not in {"offset_zero"}:
        raise ValueError("data.goal_sampling.negative.policy 当前只支持 offset_zero")
    if str(goal_negative.get("distance_label", "max_dist_cat")).lower() not in {"max_dist_cat", "minus_one"}:
        raise ValueError("data.goal_sampling.negative.distance_label 必须是 max_dist_cat 或 minus_one")
    as_width_height(config["visualization"].get("image_size", [160, 120]), "visualization.image_size")
    if not config["optimizer"].get("name"):
        raise KeyError("必须配置 optimizer.name")
    if "lr" not in config["optimizer"]:
        raise KeyError("必须配置 optimizer.lr")
    if config.get("scheduler") and "name" not in config["scheduler"]:
        raise KeyError("必须配置 scheduler.name")
    if bool(runtime.get("validate_dataset_paths", False)):
        for dataset_name, dataset_config in data.get("datasets", {}).items():
            for field in ("data_folder", "train", "test"):
                if field in dataset_config and not os.path.exists(dataset_config[field]):
                    raise FileNotFoundError(f"data.datasets.{dataset_name}.{field} 不存在: {dataset_config[field]}")
    for sink in logging_config.get("sinks", []):
        if not isinstance(sink, dict) or not sink.get("name"):
            raise KeyError("logging.sinks 每项都必须配置 name")
        if str(sink.get("name")).lower() == "wandb" and bool(sink.get("enabled", True)):
            project = sink.get("project", config["runtime"].get("project_name"))
            if not project:
                raise KeyError("启用 W&B sink 时必须配置 project 或 runtime.project_name")

    eval_fraction_value = _nested_get(logging_config, ("eval", "schedule", "fraction"))
    eval_fraction = float(1.0 if eval_fraction_value is None else eval_fraction_value)
    if not (0.0 < eval_fraction <= 1.0):
        raise ValueError(f"logging.eval.schedule.fraction 必须在 (0, 1] 范围内，实际为 {eval_fraction}")
    _validate_logging_schedules(config)


# 规范化配置：关键名称统一小写，并执行校验
def normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    config = upgrade_goal_sampling_config(deepcopy(config))
    # registry key 大小写统一，避免 YAML 中写 GNM/gnm 导致查找不一致
    config["algorithm"]["name"] = str(config["algorithm"]["name"]).lower()
    config["model"]["name"] = str(config["model"]["name"]).lower()
    config["objective"]["name"] = str(config["objective"]["name"]).lower()
    config["optimizer"]["name"] = str(config["optimizer"]["name"]).lower()
    config["data"]["context_type"] = str(config["data"].get("context_type", "temporal")).lower()
    config["data"]["module_name"] = str(config["data"].get("module_name", "navigation")).lower()
    config["data"]["obs_type"] = str(config["data"].get("obs_type", "image")).lower()
    config["data"]["goal_type"] = str(config["data"].get("goal_type", "image")).lower()
    config["data"]["goal_sampling"]["negative"]["policy"] = str(
        config["data"]["goal_sampling"]["negative"].get("policy", "offset_zero")
    ).lower()
    config["data"]["goal_sampling"]["negative"]["distance_label"] = str(
        config["data"]["goal_sampling"]["negative"].get("distance_label", "max_dist_cat")
    ).lower()
    config["runtime"]["amp_dtype"] = str(config["runtime"].get("amp_dtype", "fp16")).lower()
    config["runtime"]["lmdb_cache_mode"] = str(config["runtime"].get("lmdb_cache_mode", "auto")).lower()
    if config.get("scheduler") and config["scheduler"].get("name") is not None:
        config["scheduler"]["name"] = str(config["scheduler"]["name"]).lower()
    validate_config(config)
    return config


# 加载默认配置与用户配置并合并
def load_config(default_path: str, user_path: str) -> Dict[str, Any]:
    default_config = upgrade_goal_sampling_config(upgrade_logging_config(load_yaml(default_path)))
    user_config = upgrade_goal_sampling_config(upgrade_logging_config(load_yaml(user_path)))
    config = deep_merge(default_config, user_config)
    # 记录用户配置路径，日志 sink 可上传/保存该文件以便复现实验
    config["config_path"] = user_path
    return normalize_config(config)


def safe_config_for_logging(config: Dict[str, Any]) -> Dict[str, Any]:
    clean = deepcopy(config)
    for sink in clean.get("logging", {}).get("sinks", []):
        sink.pop("full_config", None)
    return clean


def prepare_logging_config(config: Dict[str, Any]) -> None:
    full_config = safe_config_for_logging(config)
    for sink in config["logging"].get("sinks", []):
        sink["project"] = config["runtime"]["project_name"]
        sink["run_name"] = config["runtime"]["run_name"]
        sink["config_path"] = config.get("config_path")
        sink["config_artifact_paths"] = config["runtime"].get("config_artifact_paths", {})
        sink["full_config"] = full_config


def run_config_artifact_paths(project_folder: str) -> Dict[str, str]:
    return {
        "resolved": os.path.join(project_folder, "config.resolved.yaml"),
        "user": os.path.join(project_folder, "config.user.yaml"),
    }


def save_run_configs(config: Dict[str, Any], project_folder: str) -> Dict[str, str]:
    paths = dict(config.get("runtime", {}).get("config_artifact_paths") or run_config_artifact_paths(project_folder))
    os.makedirs(project_folder, exist_ok=True)

    with open(paths["resolved"], "w", encoding="utf-8") as f:
        yaml.safe_dump(safe_config_for_logging(config), f, allow_unicode=True, sort_keys=False)

    user_config_path = config.get("config_path")
    if user_config_path and os.path.exists(user_config_path):
        shutil.copyfile(user_config_path, paths["user"])
    else:
        paths.pop("user", None)
    return paths


# 构建命令行参数解析器
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="视觉导航训练基座")
    # -c/--config 指向用户配置，defaults.yaml 会自动作为基础配置合并
    parser.add_argument(
        "--config",
        "-c",
        default="nomad_retrain.yaml",
        type=str,
        help="训练配置文件路径。",
    )
    parser.add_argument(
        "--build-lmdb-only",
        action="store_true",
        help="用单进程构建所有已配置的 LMDB 缓存，然后在训练前退出。",
    )
    parser.add_argument(
        "--rebuild-incomplete-lmdb",
        action="store_true",
        help="在 --build-lmdb-only 模式下删除不完整或未校验的 LMDB 缓存并重建。",
    )
    return parser


# 将命令行选项写回配置结构
def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    config = deepcopy(config)
    # CLI 开关优先级高于 YAML，方便临时执行缓存构建或重建
    if args.build_lmdb_only:
        config["runtime"]["build_lmdb_only"] = True
    if args.rebuild_incomplete_lmdb:
        config["runtime"]["rebuild_incomplete_lmdb"] = True
    return config
