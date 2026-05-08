# ============================================================
# NoMaD action distribution visualizer - sampled diffusion plots
# ============================================================
# 本文件负责 NoMaD 的可视化：
# 1. 用 model_output 采样 goal-conditioned 和 unconditioned 动作分布
# 2. 把采样轨迹、标签轨迹、观测图、目标图画到同一张图
# 3. 将生成图片交给 Recorder 写入 W&B/其他日志 sink

import os

import torch

from training_base.visualizers.trajectory import save_navigation_plot
from training_base.metrics.nomad_behavior import model_output
from training_base.registry import visualizer_registry


# 注册 NoMaD 动作分布可视化器
@visualizer_registry.register("nomad_action_distribution")
class NoMaDActionDistributionVisualizer:
    # 保存可视化配置
    def __init__(self, config) -> None:
        self.config = config

    # 生成采样轨迹图并记录到日志
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
        dataset_index,
        metric_scale,
        dataset_metadata,
        device,
        mode,
        project_folder,
        epoch,
        num_images_log,
        num_action_samples_log,
        action_stats=None,
    ) -> None:
        # 控制输出的样本数量
        count = min(int(num_images_log), batch_action_label.shape[0])
        if count <= 0:
            return
        num_samples = max(int(num_action_samples_log), 1)
        # 模型采样：无梯度推理
        # count 控制选几个样本，num_samples 控制每个样本采几条轨迹
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
            # model_output 对每个样本 repeat_interleave，因此第 i 个样本对应连续 num_samples 条轨迹
            start = i * num_samples
            stop = start + num_samples
            gc_samples = output["gc_actions"][start:stop]
            uc_samples = output["uc_actions"][start:stop]
            # 取第一条有目标条件采样作为主预测轨迹，其余作为分布背景
            pred = gc_samples[0]
            dist_pred = output["gc_distance"][start:stop]
            path = os.path.join(folder, f"nomad_epoch_{epoch}_{i}.png")
            save_navigation_plot(
                path,
                label=batch_action_label[i],
                pred=pred,
                sample_groups=[
                    {
                        "label": "gc sample",
                        "samples": gc_samples,
                        "color": "tab:green",
                        "alpha": 0.28,
                    },
                    {
                        "label": "uc sample",
                        "samples": uc_samples,
                        "color": "tab:red",
                        "alpha": 0.22,
                    },
                ],
                obs_image=batch_viz_obs_images[i],
                goal_image=batch_viz_goal_images[i],
                goal_pos=goal_pos[i],
                dist_pred=dist_pred,
                dist_label=batch_dist_label[i],
                normalized=False,
                dataset_index=dataset_index[i],
                metric_scale=metric_scale[i],
                dataset_metadata=dataset_metadata,
                title=f"{mode} epoch {epoch}",
            )
            image_payload.append(recorder.image(path))
        if image_payload:
            recorder.log_images({f"{mode}/nomad_action_samples": image_payload}, commit=False)
