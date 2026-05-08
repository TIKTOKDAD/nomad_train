# ============================================================
# NoMaD diffusion objective - distance plus denoising loss
# ============================================================
# 本文件计算 NoMaD 训练/评估目标：
# 1. 随机采样 goal mask，训练模型在有目标和无目标条件下都能编码视觉条件
# 2. 将绝对动作航点转换为归一化动作增量，作为扩散模型的 denoising 目标
# 3. 同时优化距离预测损失和扩散噪声预测损失

from typing import Dict

import numpy as np
import torch

from training_base.data.action_stats import get_delta_torch, load_action_stats, normalize_data_torch
from training_base.losses import action_reduce, get_configured_loss
from training_base.registry import objective_registry


# 采样目标 mask（用于无目标条件训练）
def sample_goal_mask(batch_size: int, goal_mask_prob: float, device: torch.device) -> torch.Tensor:
    # clip 概率到 [0,1]，避免 YAML 写错时采样行为失控
    goal_mask_prob = float(np.clip(float(goal_mask_prob), 0.0, 1.0))
    # 返回 long tensor，后续作为 index_select 的索引：0=no mask，1=goal mask
    return (torch.rand((batch_size,), device=device) < goal_mask_prob).long()


# 注册 NoMaD 的扩散目标函数
@objective_registry.register("nomad_diffusion")
class NoMaDDiffusionObjective:
    # 读取超参与损失函数配置
    def __init__(self, config) -> None:
        self.goal_mask_prob = float(config["goal_mask_prob"])
        # alpha 控制距离损失占比；扩散损失权重为 1-alpha
        self.alpha = float(config["alpha"])
        self.distance_mask_mode = str(config.get("distance_mask_mode", "per_sample")).lower()
        if self.distance_mask_mode not in {"per_sample", "legacy_scalar"}:
            raise ValueError("objective.distance_mask_mode 必须是 per_sample 或 legacy_scalar 之一")
        self.action_stats = load_action_stats(config.get("action_stats"))
        losses = config.get("losses", {})
        # distance/diffusion 两个子损失可分别配置，但默认都用 MSE
        self.distance_loss = get_configured_loss(losses, "distance", "mse")
        self.diffusion_loss = get_configured_loss(losses, "diffusion", "mse")

    # 训练损失：距离预测 + 扩散噪声预测
    def train_losses(
        self,
        *,
        model,
        noise_scheduler,
        batch_obs_images: torch.Tensor,
        batch_goal_images: torch.Tensor,
        actions: torch.Tensor,
        distance: torch.Tensor,
        action_mask: torch.Tensor,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        batch_size = actions.shape[0]
        # 采样 goal mask，并编码观测/目标条件
        goal_mask = sample_goal_mask(batch_size, self.goal_mask_prob, device)
        # obsgoal_cond 是扩散模型的 global condition，也是距离预测器输入
        obsgoal_cond = model.encode_vision(batch_obs_images, batch_goal_images, goal_mask)

        # 将绝对动作转为增量并归一化
        # Dataset 提供的是局部坐标系绝对航点；扩散模型学习相邻增量更稳定
        deltas = get_delta_torch(actions)
        naction = normalize_data_torch(deltas, self.action_stats)
        if naction.shape[-1] != 2:
            raise ValueError("action 维度必须为 2")

        # 距离损失按可见目标进行加权
        dist_pred = model.predict_distance(obsgoal_cond)
        visible_goal = 1 - goal_mask.float()
        if self.distance_mask_mode == "legacy_scalar":
            # legacy_scalar 先求全局标量损失再乘 mask，保留旧实现兼容
            dist_loss = self.distance_loss(dist_pred.squeeze(-1), distance)
            dist_loss = (dist_loss * visible_goal).mean() / (1e-2 + visible_goal.mean())
        else:
            # per_sample 先保留每个样本的距离损失，再只对可见目标样本归约
            raw_dist_loss = self.distance_loss(dist_pred.squeeze(-1), distance, reduction="none")
            dist_loss = (raw_dist_loss * visible_goal).mean() / (1e-2 + visible_goal.mean())

        # 扩散噪声预测损失
        # 标准 DDPM 训练：随机采样 timestep，把干净动作加噪，再预测噪声 epsilon
        noise = torch.randn_like(naction)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device).long()
        noisy_action = noise_scheduler.add_noise(naction, noise, timesteps)
        noise_pred = model.predict_noise(noisy_action, timesteps, obsgoal_cond)

        # action_mask 排除负样本/距离不合适样本的动作扩散损失
        diffusion_loss = action_reduce(self.diffusion_loss(noise_pred, noise, reduction="none"), action_mask)
        total_loss = self.alpha * dist_loss + (1 - self.alpha) * diffusion_loss
        return {
            "loss": total_loss,
            "total_loss": total_loss,
            "dist_loss": dist_loss,
            "diffusion_loss": diffusion_loss,
            "noise_pred": noise_pred,
            "noise": noise,
            "goal_mask": goal_mask,
        }

    # 评估损失：比较不同 mask 策略
    def eval_losses(
        self,
        *,
        model,
        noise_scheduler,
        batch_obs_images: torch.Tensor,
        batch_goal_images: torch.Tensor,
        actions: torch.Tensor,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        batch_size = actions.shape[0]
        # 三种条件分别评估：训练式随机 mask、完全有目标、完全无目标
        rand_goal_mask = sample_goal_mask(batch_size, self.goal_mask_prob, device)
        goal_mask = torch.ones_like(rand_goal_mask).long()
        no_mask = torch.zeros_like(rand_goal_mask).long()

        # 不同 mask 下视觉条件不同，但使用同一批 noisy action/noise 进行公平比较
        rand_mask_cond = model.encode_vision(batch_obs_images, batch_goal_images, rand_goal_mask)
        obsgoal_cond = model.encode_vision(batch_obs_images, batch_goal_images, no_mask).flatten(start_dim=1)
        goal_mask_cond = model.encode_vision(batch_obs_images, batch_goal_images, goal_mask)

        # 评估阶段同样从绝对动作转增量并归一化
        deltas = get_delta_torch(actions)
        naction = normalize_data_torch(deltas, self.action_stats)
        noise = torch.randn_like(naction)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device).long()
        noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)

        rand_mask_noise_pred = model.predict_noise(noisy_actions, timesteps, rand_mask_cond)
        no_mask_noise_pred = model.predict_noise(noisy_actions, timesteps, obsgoal_cond)
        goal_mask_noise_pred = model.predict_noise(noisy_actions, timesteps, goal_mask_cond)
        return {
            "rand_mask_loss": self.diffusion_loss(rand_mask_noise_pred, noise),
            "no_mask_loss": self.diffusion_loss(no_mask_noise_pred, noise),
            "goal_mask_loss": self.diffusion_loss(goal_mask_noise_pred, noise),
        }
