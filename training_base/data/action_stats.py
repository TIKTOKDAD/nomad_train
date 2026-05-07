import os
from typing import Optional

import numpy as np
import torch
import yaml


DEFAULT_DATA_CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "data_config.yaml"))
_ACTION_STATS_TENSOR_CACHE = {}


def _coerce_action_stats(stats) -> dict:
    if not isinstance(stats, dict) or "min" not in stats or "max" not in stats:
        raise ValueError("action_stats must contain 'min' and 'max' entries.")
    action_stats = {
        "min": np.asarray(stats["min"], dtype=np.float32),
        "max": np.asarray(stats["max"], dtype=np.float32),
    }
    if action_stats["min"].shape != action_stats["max"].shape:
        raise ValueError(f"action_stats min/max shapes differ: {action_stats['min'].shape} != {action_stats['max'].shape}")
    if not np.all(np.isfinite(action_stats["min"])) or not np.all(np.isfinite(action_stats["max"])):
        raise ValueError("action_stats min/max must be finite.")
    return action_stats


def load_action_stats(config_stats: Optional[dict] = None, data_config_path: str = DEFAULT_DATA_CONFIG_PATH) -> dict:
    if config_stats is not None:
        return _coerce_action_stats(config_stats)
    with open(data_config_path, "r", encoding="utf-8-sig") as f:
        data_config = yaml.safe_load(f)
    return _coerce_action_stats(data_config["action_stats"])


def _stats_to_tensor(stats, device, dtype):
    cache_key = (id(stats), torch.device(device).type, torch.device(device).index, str(dtype))
    cached = _ACTION_STATS_TENSOR_CACHE.get(cache_key)
    if cached is None:
        cached = {
            "min": torch.as_tensor(stats["min"], device=device, dtype=dtype),
            "max": torch.as_tensor(stats["max"], device=device, dtype=dtype),
        }
        _ACTION_STATS_TENSOR_CACHE[cache_key] = cached
    return cached


def normalize_data_torch(data: torch.Tensor, stats: dict):
    stats = _stats_to_tensor(stats, data.device, data.dtype)
    ndata = (data - stats["min"]) / (stats["max"] - stats["min"])
    return ndata * 2 - 1


def unnormalize_data_torch(ndata: torch.Tensor, stats: dict):
    stats = _stats_to_tensor(stats, ndata.device, ndata.dtype)
    data = (ndata + 1) / 2
    return data * (stats["max"] - stats["min"]) + stats["min"]


def get_delta_torch(actions: torch.Tensor):
    zero_action = torch.zeros(actions.shape[0], 1, actions.shape[-1], device=actions.device, dtype=actions.dtype)
    ex_actions = torch.cat([zero_action, actions], dim=1)
    return ex_actions[:, 1:] - ex_actions[:, :-1]


def get_action_torch(diffusion_output: torch.Tensor, action_stats: dict):
    ndeltas = diffusion_output.reshape(diffusion_output.shape[0], -1, 2)
    deltas = unnormalize_data_torch(ndeltas, action_stats)
    return torch.cumsum(deltas, dim=1)

