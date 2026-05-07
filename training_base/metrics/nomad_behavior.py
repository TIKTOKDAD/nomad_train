from typing import Dict

import torch

from training_base.data.action_stats import get_action_torch, load_action_stats
from training_base.metrics.action_metrics import flattened_waypoint_cosine, waypoint_cosine, waypoint_mse
from training_base.registry import loss_registry, metric_registry


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
    action_stats = load_action_stats(action_stats)
    goal_mask = torch.ones((batch_goal_images.shape[0],), dtype=torch.long, device=device)
    obs_cond = model.encode_vision(batch_obs_images, batch_goal_images, goal_mask)
    obs_cond = obs_cond.repeat_interleave(num_samples, dim=0)

    diffusion_output = torch.randn((len(obs_cond), pred_horizon, action_dim), device=device)
    for k in noise_scheduler.timesteps[:]:
        noise_pred = model.predict_noise(
            diffusion_output,
            k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
            obs_cond,
        )
        diffusion_output = noise_scheduler.step(model_output=noise_pred, timestep=k, sample=diffusion_output).prev_sample
    uc_actions = get_action_torch(diffusion_output, action_stats)

    no_mask = torch.zeros((batch_goal_images.shape[0],), dtype=torch.long, device=device)
    obsgoal_cond = model.encode_vision(batch_obs_images, batch_goal_images, no_mask)
    obsgoal_cond = obsgoal_cond.repeat_interleave(num_samples, dim=0)

    diffusion_output = torch.randn((len(obsgoal_cond), pred_horizon, action_dim), device=device)
    for k in noise_scheduler.timesteps[:]:
        noise_pred = model.predict_noise(
            diffusion_output,
            k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
            obsgoal_cond,
        )
        diffusion_output = noise_scheduler.step(model_output=noise_pred, timestep=k, sample=diffusion_output).prev_sample
    gc_actions = get_action_torch(diffusion_output, action_stats)
    gc_distance = model.predict_distance(obsgoal_cond.flatten(start_dim=1))
    return {"uc_actions": uc_actions, "gc_actions": gc_actions, "gc_distance": gc_distance}


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
        raise ValueError(f"{uc_actions.shape} != {batch_action_label.shape}")
    if gc_actions.shape != batch_action_label.shape:
        raise ValueError(f"{gc_actions.shape} != {batch_action_label.shape}")

    distance_loss = loss_registry.get("mse")
    return {
        "uc_action_loss": waypoint_mse(uc_actions, batch_action_label, action_mask),
        "uc_action_waypts_cos_sim": waypoint_cosine(uc_actions, batch_action_label, action_mask),
        "uc_multi_action_waypts_cos_sim": flattened_waypoint_cosine(uc_actions, batch_action_label, action_mask),
        "gc_dist_loss": distance_loss(gc_distance, batch_dist_label.unsqueeze(-1)),
        "gc_action_loss": waypoint_mse(gc_actions, batch_action_label, action_mask),
        "gc_action_waypts_cos_sim": waypoint_cosine(gc_actions, batch_action_label, action_mask),
        "gc_multi_action_waypts_cos_sim": flattened_waypoint_cosine(gc_actions, batch_action_label, action_mask),
    }
