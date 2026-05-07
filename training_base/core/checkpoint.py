import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


CHECKPOINT_SCHEMA_VERSION = 1


@dataclass
class ResumeState:
    current_epoch: int = 0
    latest_checkpoint: Optional[Dict[str, Any]] = None
    load_project_folder: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


def strip_module_prefix(state_dict: dict) -> dict:
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        return state_dict
    if all(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
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
        raise TypeError(f"Unsupported checkpoint payload: {type(loaded_model)!r}")
    return strip_module_prefix(state_dict)


def _format_keys(keys, limit: int = 20) -> str:
    if not keys:
        return "<none>"
    preview = list(keys[:limit])
    suffix = "" if len(keys) <= limit else f" ... (+{len(keys) - limit} more)"
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
        suffix = "" if len(changed) <= 8 else f" ... (+{len(changed) - 8} more)"
        print(f"Applied legacy {model_name} checkpoint key remap: {preview}{suffix}")
    return remapped


def report_state_key_differences(incompatible, label: str = "Checkpoint model state") -> None:
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if missing or unexpected:
        print(
            f"{label} loaded with key differences:\n"
            f"  missing_keys: {_format_keys(missing)}\n"
            f"  unexpected_keys: {_format_keys(unexpected)}"
        )


def load_model_state(model, checkpoint: dict, *, strict: bool = False, model_name: Optional[str] = None):
    state_dict = remap_legacy_state_dict(model_name, extract_model_state(checkpoint))
    incompatible = model.load_state_dict(state_dict, strict=strict)
    report_state_key_differences(incompatible)
    return incompatible


def find_latest_checkpoint(project_folder: str) -> str:
    return os.path.join(project_folder, "latest.pth")


def load_checkpoint(path: str, device):
    return torch.load(path, map_location=device)


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
    }
    torch.save(payload, path)
