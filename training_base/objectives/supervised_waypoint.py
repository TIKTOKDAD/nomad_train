# ============================================================
# Supervised waypoint objective - distance and trajectory losses
# ============================================================
# 本文件计算 GNM/ViNT 的监督训练损失：
# 1. dist_loss 回归观测到目标的离散距离
# 2. action_loss 回归局部坐标系下的未来航点
# 3. action_mask 控制哪些样本参与动作损失，负样本不学习动作轨迹

from typing import Dict

import torch
import torch.nn.functional as F

from training_base.losses import action_reduce, get_configured_loss
from training_base.registry import objective_registry


# 注册监督航点目标函数
@objective_registry.register("supervised_waypoint")
class SupervisedWaypointObjective:
    # 初始化损失权重与各项损失函数
    def __init__(self, config) -> None:
        # alpha 控制距离损失与动作损失的权重；距离项额外乘 1e-2 保持历史尺度
        self.alpha = float(config.get("alpha", 0.5))
        losses = config.get("losses", {})
        self.distance_loss = get_configured_loss(losses, "distance", "mse")
        self.action_loss = get_configured_loss(losses, "action", "mse")

    # 计算距离/动作损失与相似度指标
    def __call__(
        self,
        *,
        dist_label: torch.Tensor,
        action_label: torch.Tensor,
        dist_pred: torch.Tensor,
        action_pred: torch.Tensor,
        learn_angle: bool,
        action_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # 距离损失
        # dist_pred [B,1] squeeze 后与 dist_label [B] 对齐
        dist_loss = self.distance_loss(dist_pred.squeeze(-1), dist_label.float())
        if action_pred.shape != action_label.shape:
            raise ValueError(f"{action_pred.shape} != {action_label.shape}")

        # 动作损失（按 mask 归约）
        # action_reduce 会先按 T/D 求均值，再按 action_mask 对 batch 归约
        action_loss = action_reduce(self.action_loss(action_pred, action_label, reduction="none"), action_mask)
        # 单航点方向相似度：每个时间步的 (x,y) 向量方向
        action_waypts_cos_sim = action_reduce(
            F.cosine_similarity(action_pred[:, :, :2], action_label[:, :, :2], dim=-1),
            action_mask,
        )
        # 整条轨迹方向相似度：把所有时间步展平后比较整体形状
        multi_action_waypts_cos_sim = action_reduce(
            F.cosine_similarity(
                torch.flatten(action_pred[:, :, :2], start_dim=1),
                torch.flatten(action_label[:, :, :2], start_dim=1),
                dim=-1,
            ),
            action_mask,
        )
        results = {
            "dist_loss": dist_loss,
            "action_loss": action_loss,
            "action_waypts_cos_sim": action_waypts_cos_sim,
            "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim,
        }
        # 如学习角度，额外计算朝向相似度
        if learn_angle:
            # action_pred/action_label 的后两维是 cos/sin 朝向向量
            results["action_orien_cos_sim"] = action_reduce(
                F.cosine_similarity(action_pred[:, :, 2:], action_label[:, :, 2:], dim=-1),
                action_mask,
            )
            results["multi_action_orien_cos_sim"] = action_reduce(
                F.cosine_similarity(
                    torch.flatten(action_pred[:, :, 2:], start_dim=1),
                    torch.flatten(action_label[:, :, 2:], start_dim=1),
                    dim=-1,
                ),
                action_mask,
            )
        # 总损失：距离损失 + 动作损失
        # 这里保留原 NoMaD/ViNT 代码的距离损失缩放：alpha * 1e-2 * dist_loss
        results["total_loss"] = self.alpha * 1e-2 * dist_loss + (1 - self.alpha) * action_loss
        return results
