# ============================================================
# Supervised waypoint visualizer - prediction vs label plots
# ============================================================
# 本文件负责 GNM/ViNT 的可视化：
# 1. 每次日志抽取若干样本
# 2. 绘制预测轨迹、标签轨迹、目标位置和距离文本
# 3. 训练时可复用 latest 文件名，评估时按 epoch/batch 命名保留历史

import os

from training_base.visualizers.trajectory import save_navigation_plot
from training_base.registry import visualizer_registry


# 注册监督航点可视化器
@visualizer_registry.register("supervised_waypoint")
class SupervisedWaypointVisualizer:
    # 保存可视化配置
    def __init__(self, config) -> None:
        self.config = config

    # 生成轨迹对比图并写入日志
    def __call__(
        self,
        *,
        recorder,
        mode,
        batch_idx,
        epoch,
        num_batches,
        normalized,
        project_folder,
        num_images_log,
        obs_image,
        goal_image,
        action_pred,
        action_label,
        dist_pred,
        dist_label,
        goal_pos,
        dataset_index,
        metric_scale,
        dataset_metadata,
        use_latest,
    ) -> None:
        del num_batches
        image_payload = []
        # num_images_log 防止一个 batch 生成过多图片
        count = min(int(num_images_log), action_label.shape[0])
        folder = os.path.join(project_folder, "visualizations", mode)
        for i in range(count):
            # train 模式通常覆盖 latest，eval 模式保留 epoch/batch/i 便于回看
            suffix = "latest" if use_latest else f"epoch_{epoch}_batch_{batch_idx}_{i}"
            path = os.path.join(folder, f"waypoints_{suffix}.png")
            save_navigation_plot(
                path,
                label=action_label[i],
                pred=action_pred[i],
                obs_image=obs_image[i],
                goal_image=goal_image[i],
                goal_pos=goal_pos[i],
                dist_pred=dist_pred[i],
                dist_label=dist_label[i],
                normalized=bool(normalized),
                dataset_index=dataset_index[i],
                metric_scale=metric_scale[i],
                dataset_metadata=dataset_metadata,
                title=f"{mode} {epoch}:{batch_idx}",
            )
            image_payload.append(recorder.image(path))
        if image_payload:
            recorder.log_images({f"{mode}/waypoint_prediction": image_payload}, commit=False)
