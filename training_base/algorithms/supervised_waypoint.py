import os
from typing import Dict

import torch
import torchvision.transforms.functional as TF

from training_base.algorithms.base import Algorithm, StepResult
from training_base.core.checkpoint import ResumeState, find_latest_checkpoint, load_checkpoint, load_model_state
from training_base.data.batch import split_and_transform_obs, transform_goal
from training_base.data.data_utils import VISUALIZATION_IMAGE_SIZE
from training_base.models import build_model
from training_base.registry import metric_registry, objective_registry, visualizer_registry


class SupervisedWaypointAlgorithm(Algorithm):
    def build_model(self, config):
        result = build_model(config)
        return result.model, result.extras

    def build_objective(self, config):
        return objective_registry.build(config["objective"]["name"], config["objective"])

    def prepare_resume(self, model, optimizer, scheduler, config, device) -> ResumeState:
        load_run = config["runtime"].get("load_run")
        if not load_run:
            return ResumeState(extra={})

        load_project_folder = os.path.join("logs", load_run)
        print("Loading model from ", load_project_folder)
        latest_checkpoint = load_checkpoint(find_latest_checkpoint(load_project_folder), device)
        load_model_state(model, latest_checkpoint, strict=False)

        current_epoch = latest_checkpoint.get("epoch", -1) + 1 if isinstance(latest_checkpoint, dict) else 0
        optimizer_state = latest_checkpoint.get("optimizer", None) if isinstance(latest_checkpoint, dict) else None
        scheduler_state = latest_checkpoint.get("scheduler", None) if isinstance(latest_checkpoint, dict) else None
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        if scheduler is not None and scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
        print(f"Resuming training from epoch {current_epoch}")
        global_step = latest_checkpoint.get("global_step", 0) if isinstance(latest_checkpoint, dict) else 0
        return ResumeState(current_epoch=current_epoch, latest_checkpoint=latest_checkpoint, load_project_folder=load_project_folder, extra={"global_step": global_step})

    def prepare_batch(self, batch, transform, device, mode: str, should_log_images: bool):
        obs_images = torch.split(batch.obs_image, 3, dim=1)
        viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE) if should_log_images else None
        viz_goal_image = TF.resize(batch.goal_image, VISUALIZATION_IMAGE_SIZE) if should_log_images else None
        return {
            "obs_image": split_and_transform_obs(batch.obs_image, transform, device),
            "goal_image": transform_goal(batch.goal_image, transform, device),
            "dist_label": batch.distance.to(device, non_blocking=True),
            "action_label": batch.actions.to(device, non_blocking=True),
            "action_mask": batch.action_mask.to(device, non_blocking=True),
            "goal_pos": batch.goal_pos,
            "dataset_index": batch.dataset_index,
            "metric_scale": batch.metric_scale,
            "viz_obs_image": viz_obs_image,
            "viz_goal_image": viz_goal_image,
        }

    def _step(self, model, prepared, state, config) -> StepResult:
        dist_pred, action_pred = model(prepared["obs_image"], prepared["goal_image"])
        objective = state["objective"]
        losses = objective(
            dist_label=prepared["dist_label"],
            action_label=prepared["action_label"],
            dist_pred=dist_pred,
            action_pred=action_pred,
            learn_angle=bool(config["data"]["learn_angle"]),
            action_mask=prepared["action_mask"],
        )
        return StepResult(
            loss=losses["total_loss"],
            logs=losses,
            batch_size=int(prepared["obs_image"].shape[0]),
            extras={"dist_pred": dist_pred, "action_pred": action_pred},
        )

    def train_step(self, model, prepared, state, config) -> StepResult:
        return self._step(model, prepared, state, config)

    def eval_step(self, model, prepared, state, config) -> StepResult:
        return self._step(model, prepared, state, config)

    def light_metrics(self, model, prepared, result, state, config, mode: str):
        del model, state
        entries = config.get("metrics", {}).get("train" if mode == "train" else "eval", [])
        logs = {}
        for metric_config in entries:
            metric_config = dict(metric_config)
            if not bool(metric_config.pop("enabled", True)):
                continue
            metric_name = metric_config.pop("name")
            log_name = metric_config.pop("log_name", metric_name)
            metric = metric_registry.get(metric_name)
            logs[log_name] = metric(
                result.extras["action_pred"],
                prepared["action_label"],
                prepared["action_mask"],
                **metric_config,
            )
        return logs

    def visualize(self, *, model, prepared, result, state, config, mode, project_folder, epoch, batch_idx, num_batches, recorder) -> None:
        if prepared["viz_obs_image"] is None or prepared["viz_goal_image"] is None:
            return
        for visualizer_config in self.visualization_configs(config, mode, "supervised_waypoint"):
            visualizer_name = visualizer_config.pop("name")
            visualizer = visualizer_registry.build(visualizer_name, visualizer_config)
            visualizer(
                recorder=recorder,
                mode=mode,
                batch_idx=batch_idx,
                epoch=epoch,
                num_batches=num_batches,
                normalized=bool(config["data"]["normalize"]),
                project_folder=project_folder,
                num_images_log=int(visualizer_config.get("num_images_log", config["visualization"].get("num_images_log", 8))),
                obs_image=prepared["viz_obs_image"],
                goal_image=prepared["viz_goal_image"],
                action_pred=result.extras["action_pred"],
                action_label=prepared["action_label"],
                dist_pred=result.extras["dist_pred"],
                dist_label=prepared["dist_label"],
                goal_pos=prepared["goal_pos"],
                dataset_index=prepared["dataset_index"],
                metric_scale=prepared["metric_scale"],
                use_latest=mode == "train",
            )

    def primary_metric(self, eval_summaries: Dict[str, Dict[str, float]]) -> float:
        values = [
            metrics["total_loss"]
            for metrics in eval_summaries.values()
            if "total_loss" in metrics and metrics["total_loss"] == metrics["total_loss"]
        ]
        return sum(values) / len(values) if values else float("nan")
