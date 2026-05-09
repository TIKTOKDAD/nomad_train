# ============================================================
# Checkpoint utilities - save, load, resume, and legacy remap
# ============================================================

from dataclasses import dataclass
import os
import random
import warnings
from typing import Any, Dict, Optional

import numpy as np
import torch


CHECKPOINT_SCHEMA_VERSION = 1


@dataclass
class ResumeState:
    current_epoch: int = 0
    latest_checkpoint: Optional[Dict[str, Any]] = None
    load_project_folder: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


def capture_rng_state() -> Dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(rng_state: Optional[Dict[str, Any]], *, path: str = "<checkpoint>") -> bool:
    if not rng_state:
        warnings.warn(
            f"检查点 {path} 不包含 rng_state；将继续训练，但无法保证严格复现。",
            RuntimeWarning,
            stacklevel=2,
        )
        return False

    try:
        python_state = rng_state.get("python")
        numpy_state = rng_state.get("numpy")
        torch_state = rng_state.get("torch")
        cuda_state = rng_state.get("cuda")

        if python_state is not None:
            random.setstate(python_state)
        if numpy_state is not None:
            np.random.set_state(numpy_state)
        if torch_state is not None:
            torch.set_rng_state(torch_state)
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_state)
    except Exception as exc:
        warnings.warn(
            f"恢复检查点 {path} 的随机数状态失败：{exc}；将继续训练。",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    return True


def strip_module_prefix(state_dict: dict) -> dict:
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        return state_dict
    if all(key.startswith("module.") for key in state_dict.keys()):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def extract_model_state(checkpoint: dict) -> dict:
    loaded_model = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    if isinstance(loaded_model, dict):
        state_dict = loaded_model
    elif hasattr(loaded_model, "module"):
        state_dict = loaded_model.module.state_dict()
    elif hasattr(loaded_model, "state_dict"):
        state_dict = loaded_model.state_dict()
    else:
        raise TypeError(f"不支持的检查点载荷类型: {type(loaded_model)!r}")
    return strip_module_prefix(state_dict)


def _format_keys(keys, limit: int = 20) -> str:
    if not keys:
        return "<无>"
    preview = list(keys[:limit])
    suffix = "" if len(keys) <= limit else f" ... (另有 {len(keys) - limit} 个)"
    return ", ".join(preview) + suffix


def _remap_key_prefix(key: str, mappings) -> str:
    for old_prefix, new_prefix in mappings:
        if key.startswith(old_prefix):
            return f"{new_prefix}{key[len(old_prefix):]}"
    return key


def remap_legacy_state_dict(model_name: Optional[str], state_dict: dict) -> dict:
    if not model_name or not isinstance(state_dict, dict):
        return state_dict
    model_name = str(model_name).lower()
    mappings = {
        "gnm": (
            ("obs_mobilenet.", "encoder.obs_mobilenet."),
            ("compress_observation.", "encoder.compress_observation."),
            ("goal_mobilenet.", "encoder.goal_mobilenet."),
            ("compress_goal.", "encoder.compress_goal."),
            ("linear_layers.", "encoder.fusion."),
            ("dist_predictor.0.", "head.dist_predictor."),
            ("action_predictor.0.", "head.action_predictor."),
        ),
        "vint": (
            ("obs_encoder.", "encoder.obs_encoder."),
            ("goal_encoder.", "encoder.goal_encoder."),
            ("compress_obs_enc.", "encoder.compress_obs_enc."),
            ("compress_goal_enc.", "encoder.compress_goal_enc."),
            ("decoder.", "encoder.decoder."),
            ("dist_predictor.0.", "head.dist_predictor."),
            ("action_predictor.0.", "head.action_predictor."),
        ),
        "nomad": (
            ("noise_pred_net.", "diffusion_model."),
            ("dist_pred_net.", "distance_predictor."),
        ),
    }.get(model_name)
    if not mappings:
        return state_dict

    remapped = {}
    changed = []
    for key, value in state_dict.items():
        new_key = _remap_key_prefix(key, mappings)
        remapped[new_key] = value
        if new_key != key:
            changed.append((key, new_key))
    if changed:
        preview = ", ".join(f"{old} -> {new}" for old, new in changed[:8])
        suffix = "" if len(changed) <= 8 else f" ... (另有 {len(changed) - 8} 个)"
        print(f"已应用旧版 {model_name} 检查点的键重映射：{preview}{suffix}")
    return remapped


def report_state_key_differences(incompatible, label: str = "检查点模型状态") -> None:
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if missing or unexpected:
        print(
            f"{label} 加载后存在键差异：\n"
            f"  缺失键: {_format_keys(missing)}\n"
            f"  多余键: {_format_keys(unexpected)}"
        )


def load_model_state(model, checkpoint: dict, *, strict: bool = False, model_name: Optional[str] = None):
    state_dict = remap_legacy_state_dict(model_name, extract_model_state(checkpoint))
    incompatible = model.load_state_dict(state_dict, strict=strict)
    report_state_key_differences(incompatible)
    return incompatible


def find_latest_checkpoint(project_folder: str) -> str:
    return os.path.join(project_folder, "latest.pth")


def _looks_like_legacy_model_state(payload) -> bool:
    if not isinstance(payload, dict) or len(payload) == 0:
        return False
    return all(torch.is_tensor(value) for value in payload.values())


def _looks_like_training_checkpoint(payload) -> bool:
    if not isinstance(payload, dict):
        return False
    training_keys = {
        "checkpoint_schema_version",
        "model",
        "optimizer",
        "scheduler",
        "algorithm_state",
        "callback_state",
        "epoch",
        "global_step",
        "eval_summaries",
    }
    return any(key in payload for key in training_keys)


def validate_training_checkpoint(payload, path: str = "<checkpoint>") -> None:
    if not _looks_like_training_checkpoint(payload):
        return
    required = ("model", "epoch", "global_step")
    missing = [key for key in required if key not in payload]
    if missing:
        raise RuntimeError(
            f"检查点 {path} 看起来是 training_base 训练状态，但缺少必要字段: {', '.join(missing)}。"
            "如果这是旧版模型权重，请只把它当作模型 state_dict 加载，不要当作完整训练恢复点。"
        )


def load_checkpoint(path: str, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    if _looks_like_training_checkpoint(payload):
        validate_training_checkpoint(payload, path)
    elif isinstance(payload, dict) and not _looks_like_legacy_model_state(payload):
        raise RuntimeError(
            f"检查点 {path} 既不像有效的 training_base 训练状态，也不像旧版模型 state_dict。"
        )
    return payload


def _checkpoint_backup_path(path: str) -> str:
    root, ext = os.path.splitext(path)
    return f"{root}.backup{ext}"


def atomic_torch_save(payload, path: str, *, backup_existing: bool = False) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        torch.save(payload, tmp_path)
        if backup_existing and os.path.exists(path):
            os.replace(path, _checkpoint_backup_path(path))
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def save_checkpoint(
    path: str,
    *,
    epoch: int,
    global_step: int,
    model,
    optimizer,
    scheduler,
    algorithm_state,
    callback_state,
    config,
    eval_summaries=None,
    rng_state: Optional[Dict[str, Any]] = None,
) -> None:
    from training_base.core.native_utils import unwrap_model

    payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "epoch": epoch,
        "global_step": global_step,
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "algorithm_state": algorithm_state or {},
        "callback_state": callback_state or {},
        "config": config,
        "eval_summaries": eval_summaries or {},
        "rng_state": rng_state if rng_state is not None else capture_rng_state(),
    }
    backup_existing = os.path.basename(path).endswith("latest.pth")
    atomic_torch_save(payload, path, backup_existing=backup_existing)
