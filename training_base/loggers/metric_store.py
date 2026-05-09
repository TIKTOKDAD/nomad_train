# ============================================================
# Metric store - sliding metrics and distributed aggregation
# ============================================================
# 本文件管理训练/评估指标的临时缓存：
# 1. update 写入当前 step 的标量日志
# 2. latest/average 为 Recorder 提供不同聚合视图
# 3. reduce_distributed 在 DDP 评估时合并各 rank 的指标均值

from collections import defaultdict
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist


# 将张量/数值统一转换为 float
def _to_float(value) -> float:
    # 日志只保存 Python float，避免持有计算图或 GPU 张量
    if isinstance(value, torch.Tensor):
        return value.detach().float().item()
    return float(value)


def reduce_metric_logs_distributed(logs: Dict[str, object], device) -> Dict[str, float]:
    reduced = {}
    local = {}
    for key, value in logs.items():
        if value is None:
            local[key] = None
            continue
        number = _to_float(value)
        local[key] = number if not np.isnan(number) else None
    if not (dist.is_available() and dist.is_initialized()):
        return {key: value for key, value in local.items() if value is not None}
    for key in sorted(local):
        value = local[key]
        payload = torch.tensor([0.0 if value is None else value, 0.0 if value is None else 1.0], device=device, dtype=torch.float64)
        dist.all_reduce(payload, op=dist.ReduceOp.SUM)
        if payload[1].item() > 0:
            reduced[key] = (payload[0] / payload[1]).item()
    return reduced


# 指标缓存：支持滑动窗口统计与分布式聚合
class MetricStore:
    # window_size 控制 display_latest 的移动平均窗口
    def __init__(self, window_size: int = 10) -> None:
        self.window_size = max(int(window_size), 1)
        self.data = defaultdict(list)

    # 写入新日志值
    def update(self, logs: Dict[str, object]) -> None:
        for key, value in logs.items():
            if value is None:
                continue
            number = _to_float(value)
            # NaN 指标不进入窗口，避免 display/average 被污染
            if not np.isnan(number):
                self.data[key].append(number)

    # 获取最新值
    def latest(self, prefix: str = "") -> Dict[str, float]:
        return {f"{prefix}{key}": values[-1] for key, values in self.data.items() if values}

    # 获取平均值
    def average(self, prefix: str = "") -> Dict[str, float]:
        return {f"{prefix}{key}": float(np.mean(values)) for key, values in self.data.items() if values}

    # 格式化输出最新值与移动平均
    def display_latest(self) -> str:
        parts = []
        for key, values in self.data.items():
            if not values:
                continue
            moving = float(np.mean(values[-self.window_size :]))
            parts.append(f"{key}: {values[-1]:.4f} ({self.window_size}pt moving_avg: {moving:.4f})")
        return " | ".join(parts)

    # 分布式聚合（均值）
    def reduce_distributed(self, device) -> None:
        if not (dist.is_available() and dist.is_initialized()):
            return
        for key, values in self.data.items():
            # 本地 sum/count -> all_reduce -> 全局平均，适合各 rank batch 数不同的评估
            local_sum = float(np.sum(values)) if values else 0.0
            local_count = float(len(values))
            payload = torch.tensor([local_sum, local_count], device=device, dtype=torch.float64)
            dist.all_reduce(payload, op=dist.ReduceOp.SUM)
            if payload[1].item() > 0:
                self.data[key] = [(payload[0] / payload[1]).item()]
            else:
                self.data[key] = []
