# ============================================================
# Config loading - YAML merge, validation, and CLI overrides
# ============================================================
# 本文件负责训练配置的入口处理：
# 1. 读取 defaults.yaml 和用户 YAML
# 2. 深度合并配置，用户字段覆盖默认字段
# 3. 校验必需顶层字段，并把 registry key 统一小写
# 4. 将 CLI 开关写回 runtime 配置

import argparse
from copy import deepcopy
from typing import Any, Dict

import yaml


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
        raise KeyError(f"Missing required config sections: {', '.join(missing)}")
    for section in REQUIRED_SECTIONS:
        # callbacks 设计成列表，因为同一训练可以启用多个回调
        if section == "callbacks":
            if not isinstance(config[section], list):
                raise TypeError("callbacks must be a list")
        elif not isinstance(config[section], dict):
            raise TypeError(f"{section} must be a mapping")
    if not config["algorithm"].get("name"):
        raise KeyError("algorithm.name is required")
    if not config["model"].get("name"):
        raise KeyError("model.name is required")
    if not config["objective"].get("name"):
        raise KeyError("objective.name is required")


# 规范化配置：关键名称统一小写，并执行校验
def normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    config = deepcopy(config)
    # registry key 大小写统一，避免 YAML 中写 GNM/gnm 导致查找不一致
    config["algorithm"]["name"] = str(config["algorithm"]["name"]).lower()
    config["model"]["name"] = str(config["model"]["name"]).lower()
    config["objective"]["name"] = str(config["objective"]["name"]).lower()
    config["optimizer"]["name"] = str(config["optimizer"]["name"]).lower()
    if config.get("scheduler") and config["scheduler"].get("name") is not None:
        config["scheduler"]["name"] = str(config["scheduler"]["name"]).lower()
    validate_config(config)
    return config


# 加载默认配置与用户配置并合并
def load_config(default_path: str, user_path: str) -> Dict[str, Any]:
    default_config = load_yaml(default_path)
    user_config = load_yaml(user_path)
    config = deep_merge(default_config, user_config)
    # 记录用户配置路径，日志 sink 可上传/保存该文件以便复现实验
    config["config_path"] = user_path
    return normalize_config(config)


# 构建命令行参数解析器
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Navigation training base")
    # -c/--config 指向用户配置，defaults.yaml 会自动作为基础配置合并
    parser.add_argument(
        "--config",
        "-c",
        default="nomad_retrain.yaml",
        type=str,
        help="Path to the training config file.",
    )
    parser.add_argument(
        "--build-lmdb-only",
        action="store_true",
        help="Build all configured LMDB caches with one process, then exit before training.",
    )
    parser.add_argument(
        "--rebuild-incomplete-lmdb",
        action="store_true",
        help="Remove incomplete/unverified LMDB caches and rebuild them during --build-lmdb-only.",
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
