import os

from training_base.visualizers.trajectory import save_navigation_plot
from training_base.registry import visualizer_registry


@visualizer_registry.register("supervised_waypoint")
class SupervisedWaypointVisualizer:
    def __init__(self, config) -> None:
        self.config = config

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
        use_latest,
    ) -> None:
        del num_batches
        image_payload = []
        count = min(int(num_images_log), action_label.shape[0])
        folder = os.path.join(project_folder, "visualizations", mode)
        for i in range(count):
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
                title=f"{mode} {epoch}:{batch_idx}",
            )
            image_payload.append(recorder.image(path))
        if image_payload:
            recorder.log_images({f"{mode}/waypoint_prediction": image_payload}, commit=False)
