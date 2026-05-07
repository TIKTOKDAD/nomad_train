import argparse
from copy import deepcopy
from typing import Any, Dict

import yaml


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


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def deep_merge(default_config: Dict[str, Any], user_config: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(default_config)
    for key, value in user_config.items():
        if key == "losses" and isinstance(value, dict):
            merged[key] = deepcopy(value)
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def validate_config(config: Dict[str, Any]) -> None:
    missing = [section for section in REQUIRED_SECTIONS if section not in config]
    if missing:
        raise KeyError(f"Missing required config sections: {', '.join(missing)}")
    for section in REQUIRED_SECTIONS:
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


def normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    config = deepcopy(config)
    config["algorithm"]["name"] = str(config["algorithm"]["name"]).lower()
    config["model"]["name"] = str(config["model"]["name"]).lower()
    config["objective"]["name"] = str(config["objective"]["name"]).lower()
    config["optimizer"]["name"] = str(config["optimizer"]["name"]).lower()
    if config.get("scheduler") and config["scheduler"].get("name") is not None:
        config["scheduler"]["name"] = str(config["scheduler"]["name"]).lower()
    validate_config(config)
    return config


def load_config(default_path: str, user_path: str) -> Dict[str, Any]:
    default_config = load_yaml(default_path)
    user_config = load_yaml(user_path)
    config = deep_merge(default_config, user_config)
    config["config_path"] = user_path
    return normalize_config(config)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Navigation training base")
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


def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    config = deepcopy(config)
    if args.build_lmdb_only:
        config["runtime"]["build_lmdb_only"] = True
    if args.rebuild_incomplete_lmdb:
        config["runtime"]["rebuild_incomplete_lmdb"] = True
    return config
