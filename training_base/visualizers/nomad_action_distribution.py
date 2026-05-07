import os

import torch

from training_base.visualizers.trajectory import save_navigation_plot
from training_base.metrics.nomad_behavior import model_output
from training_base.registry import visualizer_registry


@visualizer_registry.register("nomad_action_distribution")
class NoMaDActionDistributionVisualizer:
    def __init__(self, config) -> None:
        self.config = config

    def __call__(
        self,
        *,
        recorder,
        model,
        noise_scheduler,
        batch_obs_images,
        batch_goal_images,
        batch_viz_obs_images,
        batch_viz_goal_images,
        batch_action_label,
        batch_dist_label,
        goal_pos,
        metric_scale,
        device,
        mode,
        project_folder,
        epoch,
        num_images_log,
        num_action_samples_log,
        action_stats=None,
    ) -> None:
        count = min(int(num_images_log), batch_action_label.shape[0])
        if count <= 0:
            return
        num_samples = max(int(num_action_samples_log), 1)
        with torch.inference_mode():
            output = model_output(
                model=model,
                noise_scheduler=noise_scheduler,
                batch_obs_images=batch_obs_images[:count],
                batch_goal_images=batch_goal_images[:count],
                pred_horizon=batch_action_label.shape[1],
                action_dim=batch_action_label.shape[2],
                num_samples=num_samples,
                device=device,
                action_stats=action_stats,
            )
        image_payload = []
        folder = os.path.join(project_folder, "visualizations", mode)
        for i in range(count):
            start = i * num_samples
            stop = start + num_samples
            gc_samples = output["gc_actions"][start:stop]
            uc_samples = output["uc_actions"][start:stop]
            pred_samples = torch.cat([gc_samples, uc_samples], dim=0)
            pred = gc_samples[0]
            dist_pred = output["gc_distance"][start]
            path = os.path.join(folder, f"nomad_epoch_{epoch}_{i}.png")
            save_navigation_plot(
                path,
                label=batch_action_label[i],
                pred=pred,
                pred_samples=pred_samples,
                obs_image=batch_viz_obs_images[i],
                goal_image=batch_viz_goal_images[i],
                goal_pos=goal_pos[i],
                dist_pred=dist_pred,
                dist_label=batch_dist_label[i],
                normalized=False,
                metric_scale=metric_scale[i],
                title=f"{mode} epoch {epoch}",
            )
            image_payload.append(recorder.image(path))
        if image_payload:
            recorder.log_images({f"{mode}/nomad_action_samples": image_payload}, commit=False)
