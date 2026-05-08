# ============================================================
# Native training utilities - AMP, gradient steps, logging gates
# ============================================================
# 本文件放置训练循环中高频调用的小工具：
# 1. AMP autocast / GradScaler / 梯度裁剪 / optimizer.step
# 2. 日志与保存事件的频率判断
# 3. 分布式日志数值聚合和主进程广播

from contextlib import nullcontext
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist


# 解除 DDP 包装，获取原始模型
def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


# 判断分布式通信是否已就绪
def distributed_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


# 分布式同步屏障
def distributed_barrier() -> None:
    if distributed_ready():
        dist.barrier()


# 仅主进程启用 tqdm，避免多进程输出混乱
def rank0_tqdm_enabled(use_tqdm: bool) -> bool:
    if not use_tqdm:
        return False
    return (not distributed_ready()) or dist.get_rank() == 0


# 根据字符串返回 AMP 使用的 dtype
def amp_dtype(amp_dtype_name: str):
    # 默认 fp16；显式写 bf16 时返回 bfloat16
    if str(amp_dtype_name).lower() == "bf16":
        return torch.bfloat16
    return torch.float16


# AMP autocast 上下文（仅 CUDA 生效）
def autocast(device: torch.device, enabled: bool, amp_dtype_name: str):
    # CPU 或未启用 AMP 时返回 nullcontext，调用方可以统一使用 with autocast(...)
    if enabled and device.type == "cuda":
        return torch.cuda.amp.autocast(dtype=amp_dtype(amp_dtype_name))
    return nullcontext()


# 创建梯度缩放器（防止 AMP 下溢）
def make_grad_scaler(device: torch.device, amp_enabled: bool, use_grad_scaler: bool):
    # bf16 通常不需要 scaler，但这里由配置 use_grad_scaler 控制，保持显式
    enabled = bool(amp_enabled and use_grad_scaler and device.type == "cuda")
    return torch.cuda.amp.GradScaler(enabled=enabled)


# 执行梯度裁剪（支持 norm/value 两种模式）
def _clip_gradients(model, optimizer, grad_scaler, clip_config) -> None:
    clip_config = clip_config or {}
    if model is None or not bool(clip_config.get("enabled", False)):
        return
    # AMP 下必须先 unscale 梯度，再做真实梯度裁剪
    if grad_scaler is not None and grad_scaler.is_enabled():
        grad_scaler.unscale_(optimizer)
    # 只收集有梯度且需要训练的参数，避免冻结参数或 None grad 干扰裁剪
    params = [
        param
        for param in unwrap_model(model).parameters()
        if param.requires_grad and param.grad is not None
    ]
    if not params:
        return
    mode = str(clip_config.get("mode", "norm")).lower()
    if mode == "value":
        # value 模式按元素截断梯度值，兼容 value/max_value/max_norm 三种字段名
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
    # norm 模式按整体范数裁剪，是训练配置的默认语义
    torch.nn.utils.clip_grad_norm_(
        params,
        max_norm=float(clip_config["max_norm"]),
        norm_type=float(clip_config.get("norm_type", 2.0)),
    )


# 反向传播 + 优化器更新，兼容 AMP 与梯度裁剪
def scale_backward_step(loss, optimizer, grad_scaler=None, *, model=None, clip_config=None) -> None:
    # set_to_none=True 可减少显存写入，并让 PyTorch 对 None grad 做优化
    optimizer.zero_grad(set_to_none=True)
    if grad_scaler is not None and grad_scaler.is_enabled():
        # AMP 分支：scale loss -> backward -> unscale/clip -> scaler.step -> scaler.update
        grad_scaler.scale(loss).backward()
        _clip_gradients(model, optimizer, grad_scaler, clip_config)
        grad_scaler.step(optimizer)
        grad_scaler.update()
    else:
        # 普通精度分支：直接 backward/clip/step
        loss.backward()
        _clip_gradients(model, optimizer, grad_scaler, clip_config)
        optimizer.step()


# 计算日志事件步数（按 global_step 或 batch_idx）
def event_step(epoch: int, num_batches: int, batch_idx: int, log_by_global_step: bool) -> int:
    if log_by_global_step:
        return epoch * num_batches + batch_idx
    return batch_idx


# 判断是否触发日志事件（频率/起始步/首步控制）
def should_log_event(
    freq: int,
    epoch: int,
    num_batches: int,
    batch_idx: int,
    log_by_global_step: bool = True,
    start_step: int = 0,
    log_first_step: bool = False,
) -> bool:
    # freq=0 表示禁用该事件
    if int(freq) == 0:
        return False
    step = event_step(epoch, num_batches, batch_idx, log_by_global_step)
    if not log_first_step and step == 0:
        return False
    if step < start_step:
        return False
    return step % int(freq) == 0


# 判断是否保存当前 epoch 的检查点
def should_save_epoch(epoch: int, freq: int) -> bool:
    return int(freq) > 0 and (epoch + 1) % int(freq) == 0


# 在分布式场景下聚合各进程的日志数值
def reduce_loggers(loggers: Dict[str, object], device: torch.device) -> None:
    if not distributed_ready():
        return
    for logger in loggers.values():
        # 每个进程先得到本地 sum/count，再 all_reduce 得到全局平均
        values = [float(v) for v in logger.data if not np.isnan(v)]
        local_sum = float(np.sum(values)) if values else 0.0
        local_count = float(len(values))
        payload = torch.tensor([local_sum, local_count], device=device, dtype=torch.float64)
        dist.all_reduce(payload, op=dist.ReduceOp.SUM)
        if payload[1].item() > 0:
            logger.data = [(payload[0] / payload[1]).item()]
        else:
            logger.data = []


# 广播主进程的值到其他进程
def broadcast_main_value(value):
    if not distributed_ready():
        return value
    # 用 object_list 保留 Python 对象类型，常用于路径/字符串等轻量值
    obj = [value if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(obj, src=0)
    return obj[0]
