from collections import defaultdict
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist


def _to_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return value.detach().float().item()
    return float(value)


class MetricStore:
    def __init__(self, window_size: int = 10) -> None:
        self.window_size = max(int(window_size), 1)
        self.data = defaultdict(list)

    def update(self, logs: Dict[str, object]) -> None:
        for key, value in logs.items():
            if value is None:
                continue
            number = _to_float(value)
            if not np.isnan(number):
                self.data[key].append(number)

    def latest(self, prefix: str = "") -> Dict[str, float]:
        return {f"{prefix}{key}": values[-1] for key, values in self.data.items() if values}

    def average(self, prefix: str = "") -> Dict[str, float]:
        return {f"{prefix}{key}": float(np.mean(values)) for key, values in self.data.items() if values}

    def display_latest(self) -> str:
        parts = []
        for key, values in self.data.items():
            if not values:
                continue
            moving = float(np.mean(values[-self.window_size :]))
            parts.append(f"{key}: {values[-1]:.4f} ({self.window_size}pt moving_avg: {moving:.4f})")
        return " | ".join(parts)

    def reduce_distributed(self, device) -> None:
        if not (dist.is_available() and dist.is_initialized()):
            return
        for key, values in self.data.items():
            local_sum = float(np.sum(values)) if values else 0.0
            local_count = float(len(values))
            payload = torch.tensor([local_sum, local_count], device=device, dtype=torch.float64)
            dist.all_reduce(payload, op=dist.ReduceOp.SUM)
            if payload[1].item() > 0:
                self.data[key] = [(payload[0] / payload[1]).item()]
            else:
                self.data[key] = []
