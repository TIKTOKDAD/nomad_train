# ============================================================
# Checkpoint utilities - save, load, and legacy key remapping
# ============================================================
# 本文件处理训练断点相关逻辑：
# 1. 统一保存模型、优化器、调度器、算法状态和配置
# 2. 从新旧 checkpoint 格式中提取模型 state_dict
# 3. 对历史 GNM/ViNT/NoMaD 权重做 key 前缀迁移，降低旧实验恢复成本

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


# 检查点结构版本号，用于向后兼容
CHECKPOINT_SCHEMA_VERSION = 1


# 断点恢复状态：保存当前 epoch 与路径信息
@dataclass
class ResumeState:
    # current_epoch 是下一次训练循环开始的 epoch
    current_epoch: int = 0
    # latest_checkpoint 保存完整 checkpoint，算法可读取其中的额外状态
    latest_checkpoint: Optional[Dict[str, Any]] = None
    # load_project_folder 记录恢复来源，便于日志和调试
    load_project_folder: Optional[str] = None
    # extra 用于算法自定义恢复信息，如 EMA 权重或 global_step
    extra: Optional[Dict[str, Any]] = None


# 去除 DDP 包装导致的 "module." 前缀
def strip_module_prefix(state_dict: dict) -> dict:
    # 只有所有 key 都带 module. 时才整体裁掉，避免混合 key 被错误处理
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        return state_dict
    if all(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


# 从检查点中提取模型 state_dict（兼容多种格式）
def extract_model_state(checkpoint: dict) -> dict:
    # 新格式 checkpoint["model"] 是 state_dict；旧格式可能直接保存模型或 DDP 包装模型
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


# 格式化键列表，避免打印过长
def _format_keys(keys, limit: int = 20) -> str:
    if not keys:
        return "<无>"
    preview = list(keys[:limit])
    suffix = "" if len(keys) <= limit else f" ... (另有 {len(keys) - limit} 个)"
    return ", ".join(preview) + suffix


# 根据映射规则替换前缀，兼容旧权重命名
def _remap_key_prefix(key: str, mappings) -> str:
    # mappings 按顺序匹配第一个旧前缀，避免一个 key 被重复改写
    for old_prefix, new_prefix in mappings:
        if key.startswith(old_prefix):
            return f"{new_prefix}{key[len(old_prefix):]}"
    return key


# 为旧版本模型权重做键名重映射
def remap_legacy_state_dict(model_name: Optional[str], state_dict: dict) -> dict:
    if not model_name or not isinstance(state_dict, dict):
        return state_dict
    model_name = str(model_name).lower()
    # 这些映射对应训练基座重构后的模块路径变化，只改 key，不改 tensor 内容
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
        print(f"已应用旧版 {model_name} 检查点的键重映射: {preview}{suffix}")
    return remapped


# 打印加载过程中 missing/unexpected keys
def report_state_key_differences(incompatible, label: str = "检查点模型状态") -> None:
    # strict=False 恢复时缺失/多余 key 不会中断训练，但必须打印出来让用户可见
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if missing or unexpected:
        print(
            f"{label} 加载后存在键差异:\n"
            f"  缺失键: {_format_keys(missing)}\n"
            f"  多余键: {_format_keys(unexpected)}"
        )


# 加载模型参数，并输出键差异信息
def load_model_state(model, checkpoint: dict, *, strict: bool = False, model_name: Optional[str] = None):
    # extract -> remap -> load 三步分开，方便定位是格式问题还是 key 命名问题
    state_dict = remap_legacy_state_dict(model_name, extract_model_state(checkpoint))
    incompatible = model.load_state_dict(state_dict, strict=strict)
    report_state_key_differences(incompatible)
    return incompatible


# 约定的最新检查点路径
def find_latest_checkpoint(project_folder: str) -> str:
    return os.path.join(project_folder, "latest.pth")


# 加载检查点到指定设备
def load_checkpoint(path: str, device):
    return torch.load(path, map_location=device)


# 保存训练检查点（模型/优化器/调度器/自定义状态）
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

    # 只保存 unwrap 后的模型参数，避免 DDP 外壳进入 checkpoint
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
