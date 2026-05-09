# ============================================================
# Checkpoint utilities - save, load, resume, and legacy remap
# ============================================================
# 本文件集中处理训练状态的保存与恢复：
# 1. 支持完整 training_base checkpoint 和旧版裸 state_dict
# 2. 保存/恢复模型、优化器、调度器、回调、算法状态、随机数状态和 GradScaler
# 3. 对旧版 GNM/ViNT/NoMaD 权重提供显式开关控制的 key 重映射

from dataclasses import dataclass
import logging
import os
import random
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


CHECKPOINT_SCHEMA_VERSION = 1
LOGGER = logging.getLogger(__name__)


# 训练恢复结果：Trainer/Algorithm 根据这里的信息决定从哪里继续
@dataclass
class ResumeState:
    # 下一轮要开始的 epoch；完整训练恢复时通常等于 checkpoint epoch + 1
    current_epoch: int = 0
    # 原始 checkpoint payload，供算法读取 EMA 等私有状态
    latest_checkpoint: Optional[Dict[str, Any]] = None
    # 被恢复 run 的目录，用于查找旧版拆分保存的 ema/optimizer/scheduler 文件
    load_project_folder: Optional[str] = None
    # 附加信息，如 global_step、checkpoint_path、恢复策略开关
    extra: Optional[Dict[str, Any]] = None


# 捕获 Python/NumPy/PyTorch 随机数状态，保证断点恢复可复现
def capture_rng_state() -> Dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


# 恢复随机数状态；旧 checkpoint 缺失时只警告不中断训练
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


# 去掉 DDP/DataParallel 保存时自动添加的 module. 前缀
def strip_module_prefix(state_dict: dict) -> dict:
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        return state_dict
    if all(key.startswith("module.") for key in state_dict.keys()):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


# 从多种 checkpoint 载荷形态中抽取模型 state_dict
def extract_model_state(checkpoint: dict) -> dict:
    # 完整训练 checkpoint 使用 "model" 字段；旧权重可能本身就是 state_dict
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


# 格式化 key 列表，避免错误信息一次打印过长
def _format_keys(keys, limit: int = 20) -> str:
    if not keys:
        return "<无>"
    preview = list(keys[:limit])
    suffix = "" if len(keys) <= limit else f" ... (另有 {len(keys) - limit} 个)"
    return ", ".join(preview) + suffix


# 将匹配到的旧前缀替换成新模块结构的前缀
def _remap_key_prefix(key: str, mappings) -> str:
    for old_prefix, new_prefix in mappings:
        if key.startswith(old_prefix):
            return f"{new_prefix}{key[len(old_prefix):]}"
    return key


# 旧工程权重到 training_base 模块命名的显式重映射
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
        # 只重命名前缀，张量本身不做任何变换
        new_key = _remap_key_prefix(key, mappings)
        remapped[new_key] = value
        if new_key != key:
            changed.append((key, new_key))
    if changed:
        preview = ", ".join(f"{old} -> {new}" for old, new in changed[:8])
        suffix = "" if len(changed) <= 8 else f" ... (另有 {len(changed) - 8} 个)"
        LOGGER.info("已应用旧版 %s 检查点的键重映射：%s%s", model_name, preview, suffix)
    return remapped


# 打印 load_state_dict 返回的缺失/多余键
def report_state_key_differences(incompatible, label: str = "检查点模型状态") -> None:
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if missing or unexpected:
        LOGGER.warning(
            "%s 加载后存在键差异：\n  缺失键: %s\n  多余键: %s",
            label,
            _format_keys(missing),
            _format_keys(unexpected),
        )


# 加载模型权重，并按配置选择 strict 与旧 key remap 策略
def load_model_state(
    model,
    checkpoint: dict,
    *,
    strict: bool = False,
    model_name: Optional[str] = None,
    allow_legacy_remap: bool = False,
):
    state_dict = extract_model_state(checkpoint)
    if allow_legacy_remap:
        # 旧版重映射需要用户显式开启，避免把不兼容权重误当作可恢复权重
        state_dict = remap_legacy_state_dict(model_name, state_dict)
    incompatible = model.load_state_dict(state_dict, strict=strict)
    report_state_key_differences(incompatible)
    return incompatible


# latest.pth 的统一路径约定
def find_latest_checkpoint(project_folder: str) -> str:
    return os.path.join(project_folder, "latest.pth")


# 根据 runtime.load_checkpoint_path 或 runtime.load_run 解析恢复路径
def resolve_resume_checkpoint_path(config: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    runtime = config.get("runtime", {})
    load_checkpoint_path = runtime.get("load_checkpoint_path")
    load_run = runtime.get("load_run")
    if load_checkpoint_path and load_run:
        raise ValueError("runtime.load_checkpoint_path 和 runtime.load_run 只能配置一个")
    if load_checkpoint_path:
        checkpoint_path = str(load_checkpoint_path)
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.abspath(checkpoint_path)
        return checkpoint_path, os.path.dirname(checkpoint_path)
    if load_run:
        log_root = str(runtime.get("log_root", "logs"))
        load_project_folder = os.path.join(log_root, str(load_run))
        return find_latest_checkpoint(load_project_folder), load_project_folder
    return None, None


# 旧版裸模型权重通常是 key->Tensor 的字典
def _looks_like_legacy_model_state(payload) -> bool:
    if not isinstance(payload, dict) or len(payload) == 0:
        return False
    return all(torch.is_tensor(value) for value in payload.values())


# 判断载荷是否像完整 training_base 训练检查点
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


# 对看起来像训练检查点但缺关键字段的文件提前报错
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


# 从磁盘加载 checkpoint，并区分完整训练状态与旧版裸 state_dict
def load_checkpoint(path: str, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    if _looks_like_training_checkpoint(payload):
        validate_training_checkpoint(payload, path)
    elif isinstance(payload, dict) and not _looks_like_legacy_model_state(payload):
        raise RuntimeError(
            f"检查点 {path} 既不像有效的 training_base 训练状态，也不像旧版模型 state_dict。"
        )
    return payload


# 解析恢复策略：完整训练恢复默认 strict，裸权重默认非 strict
def _resume_flags(config: Dict[str, Any], checkpoint: dict) -> Tuple[bool, bool]:
    runtime = config.get("runtime", {})
    default_strict = True if _looks_like_training_checkpoint(checkpoint) else False
    strict = bool(runtime.get("resume_strict", default_strict))
    allow_legacy_remap = bool(runtime.get("allow_legacy_weight_remap", False))
    return strict, allow_legacy_remap


# 加载完整训练恢复所需的模型/优化器/调度器状态
def load_training_resume(
    *,
    model,
    optimizer,
    scheduler,
    config: Dict[str, Any],
    device,
    model_name: Optional[str] = None,
) -> ResumeState:
    checkpoint_path, load_project_folder = resolve_resume_checkpoint_path(config)
    if not checkpoint_path:
        # 没配置恢复路径时返回空状态，训练从头开始
        return ResumeState(extra={"global_step": 0})

    LOGGER.info("正在从检查点恢复训练: %s", checkpoint_path)
    latest_checkpoint = load_checkpoint(checkpoint_path, device)
    strict, allow_legacy_remap = _resume_flags(config, latest_checkpoint)
    load_model_state(
        model,
        latest_checkpoint,
        strict=strict,
        model_name=model_name,
        allow_legacy_remap=allow_legacy_remap,
    )

    is_training_checkpoint = _looks_like_training_checkpoint(latest_checkpoint)
    # 完整训练检查点从下一轮继续；裸权重只加载模型，epoch 从 0 开始
    current_epoch = latest_checkpoint.get("epoch", -1) + 1 if is_training_checkpoint else 0
    optimizer_state = latest_checkpoint.get("optimizer", None) if is_training_checkpoint else None
    scheduler_state = latest_checkpoint.get("scheduler", None) if is_training_checkpoint else None
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    global_step = latest_checkpoint.get("global_step", 0) if is_training_checkpoint else 0
    LOGGER.info("从第 %s 轮继续训练", current_epoch)
    return ResumeState(
        current_epoch=current_epoch,
        latest_checkpoint=latest_checkpoint,
        load_project_folder=load_project_folder,
        extra={
            "global_step": global_step,
            "checkpoint_path": checkpoint_path,
            "resume_strict": strict,
            "allow_legacy_weight_remap": allow_legacy_remap,
        },
    )


# latest.pth 覆盖前的备份文件名
def _checkpoint_backup_path(path: str) -> str:
    root, ext = os.path.splitext(path)
    return f"{root}.backup{ext}"


# 原子保存 torch 文件，避免中断时留下损坏的目标文件
def atomic_torch_save(payload, path: str, *, backup_existing: bool = False) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        # 先写临时文件，成功后再替换正式路径
        torch.save(payload, tmp_path)
        if backup_existing and os.path.exists(path):
            os.replace(path, _checkpoint_backup_path(path))
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


# 保存完整训练状态
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
    grad_scaler=None,
) -> None:
    from training_base.core.native_utils import unwrap_model

    grad_scaler_state = None
    if grad_scaler is not None and getattr(grad_scaler, "is_enabled", lambda: False)():
        # AMP 开启且 GradScaler 有效时保存缩放器状态，恢复后避免重新 warmup
        grad_scaler_state = grad_scaler.state_dict()

    # payload 尽量自包含：训练、评估、随机性和回调状态都放在同一个文件里
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
        "grad_scaler": grad_scaler_state,
    }
    # latest.pth 会频繁覆盖，保留一个 backup 便于从异常写入中恢复
    backup_existing = os.path.basename(path).endswith("latest.pth")
    atomic_torch_save(payload, path, backup_existing=backup_existing)
