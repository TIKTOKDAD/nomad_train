# ============================================================
# Action stats - normalization helpers for diffusion actions
# ============================================================
# 本文件处理 NoMaD 扩散动作的 min/max 归一化：
# 1. 从 data_config.yaml 或 objective 配置读取 action_stats
# 2. 将绝对航点轨迹转换为相邻增量，再归一化到 [-1, 1]
# 3. 将扩散模型输出反归一化并累积回绝对航点轨迹

import os
from typing import Optional

import numpy as np
import torch
import yaml


# 默认数据配置路径与动作统计缓存（避免重复张量化）
DEFAULT_DATA_CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "data_config.yaml"))
# 缓存 key 绑定 stats 对象、device 和 dtype，避免训练循环里反复创建同样的 min/max 张量
_ACTION_STATS_TENSOR_CACHE = {}


# 校验并标准化动作统计信息
def _coerce_action_stats(stats) -> dict:
    # action_stats 必须包含同形状的 min/max，例如二维动作为 [dx, dy]
    if not isinstance(stats, dict) or "min" not in stats or "max" not in stats:
        raise ValueError("action_stats 必须包含 'min' 和 'max' 字段。")
    action_stats = {
        "min": np.asarray(stats["min"], dtype=np.float32),
        "max": np.asarray(stats["max"], dtype=np.float32),
    }
    if action_stats["min"].shape != action_stats["max"].shape:
        raise ValueError(f"action_stats 的 min/max 形状不一致: {action_stats['min'].shape} != {action_stats['max'].shape}")
    if not np.all(np.isfinite(action_stats["min"])) or not np.all(np.isfinite(action_stats["max"])):
        raise ValueError("action_stats 的 min/max 必须是有限数值。")
    return action_stats


# 从配置或文件加载动作统计（min/max）
def load_action_stats(config_stats: Optional[dict] = None, data_config_path: str = DEFAULT_DATA_CONFIG_PATH) -> dict:
    # objective 中显式写 action_stats 时优先使用，便于针对特定数据集覆盖全局统计
    if config_stats is not None:
        return _coerce_action_stats(config_stats)
    # utf-8-sig 兼容带 BOM 的 YAML 文件
    with open(data_config_path, "r", encoding="utf-8-sig") as f:
        data_config = yaml.safe_load(f)
    return _coerce_action_stats(data_config["action_stats"])


# 将统计信息转换为指定设备/类型的张量，并缓存
def _stats_to_tensor(stats, device, dtype):
    # id(stats) 用于区分不同配置来源；device.index 区分多 GPU 场景
    cache_key = (id(stats), torch.device(device).type, torch.device(device).index, str(dtype))
    cached = _ACTION_STATS_TENSOR_CACHE.get(cache_key)
    if cached is None:
        cached = {
            "min": torch.as_tensor(stats["min"], device=device, dtype=dtype),
            "max": torch.as_tensor(stats["max"], device=device, dtype=dtype),
        }
        _ACTION_STATS_TENSOR_CACHE[cache_key] = cached
    return cached


# 将数据按 min/max 归一化到 [-1, 1]
def normalize_data_torch(data: torch.Tensor, stats: dict):
    stats = _stats_to_tensor(stats, data.device, data.dtype)
    # 先映射到 [0,1]，再线性映射到 [-1,1]，匹配扩散模型训练目标
    ndata = (data - stats["min"]) / (stats["max"] - stats["min"])
    return ndata * 2 - 1


# 从 [-1, 1] 反归一化回原始量纲
def unnormalize_data_torch(ndata: torch.Tensor, stats: dict):
    stats = _stats_to_tensor(stats, ndata.device, ndata.dtype)
    # 与 normalize_data_torch 完全相反，用于把采样结果恢复到真实动作增量
    data = (ndata + 1) / 2
    return data * (stats["max"] - stats["min"]) + stats["min"]


# 将绝对动作序列转换为相邻增量序列
def get_delta_torch(actions: torch.Tensor):
    # 在序列前补一个零动作，使第一个增量等于第一个绝对航点
    zero_action = torch.zeros(actions.shape[0], 1, actions.shape[-1], device=actions.device, dtype=actions.dtype)
    ex_actions = torch.cat([zero_action, actions], dim=1)
    return ex_actions[:, 1:] - ex_actions[:, :-1]


# 将扩散模型输出的归一化增量还原为动作轨迹
def get_action_torch(diffusion_output: torch.Tensor, action_stats: dict):
    # 扩散模型只预测二维位移增量，不包含朝向；reshape 保证 [B, T, 2]
    ndeltas = diffusion_output.reshape(diffusion_output.shape[0], -1, 2)
    deltas = unnormalize_data_torch(ndeltas, action_stats)
    # 累积增量得到局部坐标系下的绝对航点
    return torch.cumsum(deltas, dim=1)
