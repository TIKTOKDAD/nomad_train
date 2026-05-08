# ============================================================
# NoMaD behavior metrics - sampled action quality evaluation
# ============================================================
# 本文件计算 NoMaD 的重指标：
# 1. 通过反向扩散采样生成无目标条件 uc_actions 和有目标条件 gc_actions
# 2. 比较采样动作与标签动作的 MSE/余弦相似度
# 3. 同时记录有目标条件下的距离预测误差

from typing import Dict

import torch

from training_base.data.action_stats import get_action_torch, load_action_stats
from training_base.core.native_utils import unwrap_model
from training_base.metrics.action_metrics import flattened_waypoint_cosine, waypoint_cosine, waypoint_mse
from training_base.registry import loss_registry, metric_registry


# 采样模型输出：无条件/有条件动作，以及距离预测
def model_output(
    *,
    model,
    noise_scheduler,
    batch_obs_images: torch.Tensor,
    batch_goal_images: torch.Tensor,
    pred_horizon: int,
    action_dim: int,
    num_samples: int,
    device: torch.device,
    action_stats=None,
):
    model = unwrap_model(model)
    # 准备动作统计与条件编码
    action_stats = load_action_stats(action_stats)
    # goal_mask=1 表示屏蔽目标图，得到 unconditional/exploration 条件
    goal_mask = torch.ones((batch_goal_images.shape[0],), dtype=torch.long, device=device)
    obs_cond = model.encode_vision(batch_obs_images, batch_goal_images, goal_mask)
    # 每个样本重复 num_samples 次，以便一次性采样多条动作轨迹
    obs_cond = obs_cond.repeat_interleave(num_samples, dim=0)

    # 无条件采样
    # 从标准高斯初始化动作序列，然后沿 scheduler.timesteps 逐步去噪
    diffusion_output = torch.randn((len(obs_cond), pred_horizon, action_dim), device=device)
    for k in noise_scheduler.timesteps[:]:
        noise_pred = model.predict_noise(
            diffusion_output,
            k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
            obs_cond,
        )
        diffusion_output = noise_scheduler.step(model_output=noise_pred, timestep=k, sample=diffusion_output).prev_sample
    uc_actions = get_action_torch(diffusion_output, action_stats)

    # no_mask=0 表示使用目标图，得到 goal-conditioned navigation 条件
    no_mask = torch.zeros((batch_goal_images.shape[0],), dtype=torch.long, device=device)
    obsgoal_cond = model.encode_vision(batch_obs_images, batch_goal_images, no_mask)
    obsgoal_cond = obsgoal_cond.repeat_interleave(num_samples, dim=0)

    # 有条件采样
    # 和无条件采样流程一致，只替换视觉条件
    diffusion_output = torch.randn((len(obsgoal_cond), pred_horizon, action_dim), device=device)
    for k in noise_scheduler.timesteps[:]:
        noise_pred = model.predict_noise(
            diffusion_output,
            k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
            obsgoal_cond,
        )
        diffusion_output = noise_scheduler.step(model_output=noise_pred, timestep=k, sample=diffusion_output).prev_sample
    gc_actions = get_action_torch(diffusion_output, action_stats)
    # 距离预测只对有目标条件有意义
    gc_distance = model.predict_distance(obsgoal_cond.flatten(start_dim=1))
    return {"uc_actions": uc_actions, "gc_actions": gc_actions, "gc_distance": gc_distance}


# 注册 NoMaD 行为指标
@metric_registry.register("nomad_behavior")
def compute_nomad_behavior_metrics(
    *,
    model,
    noise_scheduler,
    batch_obs_images: torch.Tensor,
    batch_goal_images: torch.Tensor,
    batch_dist_label: torch.Tensor,
    batch_action_label: torch.Tensor,
    device: torch.device,
    action_mask: torch.Tensor,
    action_stats=None,
) -> Dict[str, torch.Tensor]:
    # 生成动作与距离预测
    # num_samples=1 表示指标只评估一条采样轨迹；可视化器会传更大的采样数
    output = model_output(
        model=model,
        noise_scheduler=noise_scheduler,
        batch_obs_images=batch_obs_images,
        batch_goal_images=batch_goal_images,
        pred_horizon=batch_action_label.shape[1],
        action_dim=batch_action_label.shape[2],
        num_samples=1,
        device=device,
        action_stats=action_stats,
    )
    uc_actions = output["uc_actions"]
    gc_actions = output["gc_actions"]
    gc_distance = output["gc_distance"]

    if uc_actions.shape != batch_action_label.shape:
        raise ValueError(f"无条件动作形状不匹配: {uc_actions.shape} != {batch_action_label.shape}")
    if gc_actions.shape != batch_action_label.shape:
        raise ValueError(f"有目标条件动作形状不匹配: {gc_actions.shape} != {batch_action_label.shape}")

    # 距离回归损失
    distance_loss = loss_registry.get("mse")
    # uc=unconditioned/action without goal，gc=goal conditioned/action with goal
    return {
        "uc_action_loss": waypoint_mse(uc_actions, batch_action_label, action_mask),
        "uc_action_waypts_cos_sim": waypoint_cosine(uc_actions, batch_action_label, action_mask),
        "uc_multi_action_waypts_cos_sim": flattened_waypoint_cosine(uc_actions, batch_action_label, action_mask),
        "gc_dist_loss": distance_loss(gc_distance, batch_dist_label.unsqueeze(-1)),
        "gc_action_loss": waypoint_mse(gc_actions, batch_action_label, action_mask),
        "gc_action_waypts_cos_sim": waypoint_cosine(gc_actions, batch_action_label, action_mask),
        "gc_multi_action_waypts_cos_sim": flattened_waypoint_cosine(gc_actions, batch_action_label, action_mask),
    }
