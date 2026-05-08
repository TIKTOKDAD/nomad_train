# ============================================================
# Action metrics - supervised waypoint quality measurements
# ============================================================
# 本文件提供轻量动作指标：
# 1. waypoint_mse 衡量轨迹数值误差
# 2. waypoint_cosine 衡量逐航点方向相似度
# 3. flattened_waypoint_cosine 衡量整条轨迹形状方向相似度

import torch
import torch.nn.functional as F

from training_base.losses import action_reduce


# 路径点 MSE（按 mask 归约）
def waypoint_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # reduction="none" 保留 [B,T,D]，再由 action_reduce 按 mask 做样本级归约
    return action_reduce(F.mse_loss(pred, target, reduction="none"), mask)


# 路径点方向余弦相似度
def waypoint_cosine(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # 只比较位置分量 x/y，不比较角度分量
    return action_reduce(F.cosine_similarity(pred[:, :, :2], target[:, :, :2], dim=-1), mask)


# 展平后的路径点余弦相似度（跨时间步）
def flattened_waypoint_cosine(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # 将所有时间步拼成一个长向量，强调整条轨迹的总体方向
    return action_reduce(
        F.cosine_similarity(
            torch.flatten(pred[:, :, :2], start_dim=1),
            torch.flatten(target[:, :, :2], start_dim=1),
            dim=-1,
        ),
        mask,
    )
