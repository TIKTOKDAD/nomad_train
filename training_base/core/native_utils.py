from contextlib import nullcontext
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def distributed_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def distributed_barrier() -> None:
    if distributed_ready():
        dist.barrier()


def rank0_tqdm_enabled(use_tqdm: bool) -> bool:
    if not use_tqdm:
        return False
    return (not distributed_ready()) or dist.get_rank() == 0


def amp_dtype(amp_dtype_name: str):
    if str(amp_dtype_name).lower() == "bf16":
        return torch.bfloat16
    return torch.float16


def autocast(device: torch.device, enabled: bool, amp_dtype_name: str):
    if enabled and device.type == "cuda":
        return torch.cuda.amp.autocast(dtype=amp_dtype(amp_dtype_name))
    return nullcontext()


def make_grad_scaler(device: torch.device, amp_enabled: bool, use_grad_scaler: bool):
    enabled = bool(amp_enabled and use_grad_scaler and device.type == "cuda")
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _clip_gradients(model, optimizer, grad_scaler, clip_config) -> None:
    clip_config = clip_config or {}
    if model is None or not bool(clip_config.get("enabled", False)):
        return
    if grad_scaler is not None and grad_scaler.is_enabled():
        grad_scaler.unscale_(optimizer)
    params = [
        param
        for param in unwrap_model(model).parameters()
        if param.requires_grad and param.grad is not None
    ]
    if not params:
        return
    mode = str(clip_config.get("mode", "norm")).lower()
    if mode == "value":
        if "value" in clip_config:
            clip_value = float(clip_config["value"])
        elif "max_value" in clip_config:
            clip_value = float(clip_config["max_value"])
        else:
            clip_value = float(clip_config["max_norm"])
        torch.nn.utils.clip_grad_value_(params, clip_value)
        return
    if mode != "norm":
        raise ValueError(f"Unsupported gradient_clip.mode '{mode}'. Expected 'norm' or 'value'.")
    torch.nn.utils.clip_grad_norm_(
        params,
        max_norm=float(clip_config["max_norm"]),
        norm_type=float(clip_config.get("norm_type", 2.0)),
    )


def scale_backward_step(loss, optimizer, grad_scaler=None, *, model=None, clip_config=None) -> None:
    optimizer.zero_grad(set_to_none=True)
    if grad_scaler is not None and grad_scaler.is_enabled():
        grad_scaler.scale(loss).backward()
        _clip_gradients(model, optimizer, grad_scaler, clip_config)
        grad_scaler.step(optimizer)
        grad_scaler.update()
    else:
        loss.backward()
        _clip_gradients(model, optimizer, grad_scaler, clip_config)
        optimizer.step()


def event_step(epoch: int, num_batches: int, batch_idx: int, log_by_global_step: bool) -> int:
    if log_by_global_step:
        return epoch * num_batches + batch_idx
    return batch_idx


def should_log_event(
    freq: int,
    epoch: int,
    num_batches: int,
    batch_idx: int,
    log_by_global_step: bool = True,
    start_step: int = 0,
    log_first_step: bool = False,
) -> bool:
    if int(freq) == 0:
        return False
    step = event_step(epoch, num_batches, batch_idx, log_by_global_step)
    if not log_first_step and step == 0:
        return False
    if step < start_step:
        return False
    return step % int(freq) == 0


def should_save_epoch(epoch: int, freq: int) -> bool:
    return int(freq) > 0 and (epoch + 1) % int(freq) == 0


def reduce_loggers(loggers: Dict[str, object], device: torch.device) -> None:
    if not distributed_ready():
        return
    for logger in loggers.values():
        values = [float(v) for v in logger.data if not np.isnan(v)]
        local_sum = float(np.sum(values)) if values else 0.0
        local_count = float(len(values))
        payload = torch.tensor([local_sum, local_count], device=device, dtype=torch.float64)
        dist.all_reduce(payload, op=dist.ReduceOp.SUM)
        if payload[1].item() > 0:
            logger.data = [(payload[0] / payload[1]).item()]
        else:
            logger.data = []


def broadcast_main_value(value):
    if not distributed_ready():
        return value
    obj = [value if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(obj, src=0)
    return obj[0]
