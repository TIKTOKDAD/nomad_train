# ============================================================
# NoMaD algorithm - diffusion navigation training logic
# ============================================================

import os
from dataclasses import dataclass

import torch
import torchvision.transforms.functional as TF

from training_base.algorithms.base import Algorithm, StepResult
from training_base.core.checkpoint import (
    ResumeState,
    find_latest_checkpoint,
    load_checkpoint,
    load_model_state,
    remap_legacy_state_dict,
    report_state_key_differences,
    strip_module_prefix,
)
from training_base.core.native_utils import unwrap_model
from training_base.data.batch import split_and_transform_obs, transform_goal
from training_base.data.data_utils import VISUALIZATION_IMAGE_SIZE
from training_base.models import build_model
from training_base.registry import algorithm_registry, metric_registry, objective_registry, visualizer_registry


@dataclass
class NoMaDState:
    noise_scheduler: object
    ema_model: object
    objective: object


@algorithm_registry.register("nomad")
class NoMaDAlgorithm(Algorithm):
    name = "nomad"

    def build_model(self, config):
        result = build_model(config)
        return result.model, result.extras

    def build_objective(self, config):
        objective_config = dict(config["objective"])
        objective_config.setdefault("data_config_path", config.get("data", {}).get("data_config_path"))
        return objective_registry.build(objective_config["name"], objective_config)

    def prepare_resume(self, model, optimizer, scheduler, config, device) -> ResumeState:
        load_run = config["runtime"].get("load_run")
        if not load_run:
            return ResumeState(extra={"ema_state_dict": None})

        load_project_folder = os.path.join("logs", load_run)
        checkpoint_path = find_latest_checkpoint(load_project_folder)
        print("正在从以下目录加载模型:", load_project_folder)
        latest_checkpoint = load_checkpoint(checkpoint_path, device)
        load_model_state(model, latest_checkpoint, strict=False, model_name=config["model"]["name"])

        current_epoch = latest_checkpoint.get("epoch", -1) + 1 if isinstance(latest_checkpoint, dict) else 0
        if current_epoch == 0:
            epoch_ids = []
            for filename in os.listdir(load_project_folder):
                stem, ext = os.path.splitext(filename)
                if ext == ".pth" and stem.isdigit():
                    epoch_ids.append(int(stem))
            if epoch_ids:
                current_epoch = max(epoch_ids) + 1

        algorithm_state = latest_checkpoint.get("algorithm_state", {}) if isinstance(latest_checkpoint, dict) else {}
        resume_ema_state = algorithm_state.get("ema_model", None)
        if resume_ema_state is None and isinstance(latest_checkpoint, dict):
            resume_ema_state = latest_checkpoint.get("ema_model", None)
        ema_latest_path = os.path.join(load_project_folder, "ema_latest.pth")
        if resume_ema_state is None and os.path.exists(ema_latest_path):
            resume_ema_state = torch.load(ema_latest_path, map_location=device)

        optimizer_state = latest_checkpoint.get("optimizer", None) if isinstance(latest_checkpoint, dict) else None
        scheduler_state = latest_checkpoint.get("scheduler", None) if isinstance(latest_checkpoint, dict) else None
        if optimizer_state is None:
            optimizer_latest_path = os.path.join(load_project_folder, "optimizer_latest.pth")
            if os.path.exists(optimizer_latest_path):
                optimizer_state = torch.load(optimizer_latest_path, map_location=device)
        if scheduler_state is None:
            scheduler_latest_path = os.path.join(load_project_folder, "scheduler_latest.pth")
            if os.path.exists(scheduler_latest_path):
                scheduler_state = torch.load(scheduler_latest_path, map_location=device)
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state if isinstance(optimizer_state, dict) else optimizer_state.state_dict())
        if scheduler is not None and scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state if isinstance(scheduler_state, dict) else scheduler_state.state_dict())

        print(f"从第 {current_epoch} 轮继续训练")
        global_step = latest_checkpoint.get("global_step", 0) if isinstance(latest_checkpoint, dict) else 0
        return ResumeState(
            current_epoch=current_epoch,
            latest_checkpoint=latest_checkpoint,
            load_project_folder=load_project_folder,
            extra={"ema_state_dict": resume_ema_state, "global_step": global_step},
        )

    def create_state(self, model, model_extras, objective, config, device, resume_state: ResumeState):
        from diffusers.training_utils import EMAModel

        ema_config = config.get("algorithm", {}).get("ema", {})
        ema_model = None
        if bool(ema_config.get("enabled", True)):
            ema_model = EMAModel(model=unwrap_model(model), power=float(ema_config.get("power", 0.75)))
            resume_ema = (resume_state.extra or {}).get("ema_state_dict")
            if resume_ema is not None:
                ema_state = remap_legacy_state_dict("nomad", strip_module_prefix(resume_ema))
                incompatible = ema_model.averaged_model.load_state_dict(ema_state, strict=False)
                report_state_key_differences(incompatible, label="NoMaD EMA 状态")
        return NoMaDState(noise_scheduler=model_extras["noise_scheduler"], ema_model=ema_model, objective=objective)

    def prepare_batch(self, batch, transform, device, mode: str, should_log_images: bool):
        obs_images = torch.split(batch.obs_image, 3, dim=1)
        viz_obs = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1]) if should_log_images else None
        viz_goal = TF.resize(batch.goal_image, VISUALIZATION_IMAGE_SIZE[::-1]) if should_log_images else None
        return {
            "obs": split_and_transform_obs(batch.obs_image, transform, device),
            "goal": transform_goal(batch.goal_image, transform, device),
            "actions": batch.actions.to(device, non_blocking=True),
            "distance": batch.distance.float().to(device, non_blocking=True),
            "action_mask": batch.action_mask.float().to(device, non_blocking=True),
            "goal_pos": batch.goal_pos,
            "dataset_index": batch.dataset_index,
            "metric_scale": batch.metric_scale,
            "viz_obs": viz_obs,
            "viz_goal": viz_goal,
        }

    def train_step(self, model, prepared, state: NoMaDState, config) -> StepResult:
        losses = state.objective.train_losses(
            model=model,
            noise_scheduler=state.noise_scheduler,
            batch_obs_images=prepared["obs"],
            batch_goal_images=prepared["goal"],
            actions=prepared["actions"],
            distance=prepared["distance"],
            action_mask=prepared["action_mask"],
            device=prepared["obs"].device,
        )
        return StepResult(
            loss=losses["loss"],
            logs={
                "total_loss": losses["total_loss"],
                "dist_loss": losses["dist_loss"],
                "diffusion_loss": losses["diffusion_loss"],
            },
            batch_size=int(prepared["obs"].shape[0]),
            extras=losses,
        )

    def eval_step(self, model, prepared, state: NoMaDState, config) -> StepResult:
        eval_losses = state.objective.eval_losses(
            model=model,
            noise_scheduler=state.noise_scheduler,
            batch_obs_images=prepared["obs"],
            batch_goal_images=prepared["goal"],
            actions=prepared["actions"],
            device=prepared["obs"].device,
        )
        logs = {
            "diffusion_eval_loss_random_masking": eval_losses["rand_mask_loss"],
            "diffusion_eval_loss_no_masking": eval_losses["no_mask_loss"],
            "diffusion_eval_loss_goal_masking": eval_losses["goal_mask_loss"],
            "total_loss": eval_losses["rand_mask_loss"],
        }
        return StepResult(loss=None, logs=logs, batch_size=int(prepared["obs"].shape[0]), extras=eval_losses)

    def model_for_eval(self, model, state: NoMaDState):
        if state.ema_model is None:
            return unwrap_model(model)
        return state.ema_model.averaged_model

    def after_optimizer_step(self, model, state: NoMaDState, config) -> None:
        if state.ema_model is not None:
            state.ema_model.step(unwrap_model(model))

    def heavy_metrics(self, model, prepared, state: NoMaDState, config, mode: str):
        logs = {}
        for metric_config in config.get("metrics", {}).get("heavy", []):
            metric_config = dict(metric_config)
            if not bool(metric_config.pop("enabled", True)):
                continue
            metric_name = metric_config.pop("name")
            metric = metric_registry.get(metric_name)
            logs.update(
                metric(
                    model=model,
                    noise_scheduler=state.noise_scheduler,
                    batch_obs_images=prepared["obs"],
                    batch_goal_images=prepared["goal"],
                    batch_dist_label=prepared["distance"],
                    batch_action_label=prepared["actions"],
                    device=prepared["obs"].device,
                    action_mask=prepared["action_mask"],
                    action_stats=state.objective.action_stats,
                    **metric_config,
                )
            )
        return logs

    def visualize(self, *, model, prepared, result, state: NoMaDState, config, mode, project_folder, epoch, batch_idx, num_batches, global_step, recorder) -> None:
        if prepared["viz_obs"] is None or prepared["viz_goal"] is None:
            return
        for visualizer_config in self.visualization_configs(config, mode, "nomad_action_distribution"):
            visualizer_name = visualizer_config.pop("name")
            visualizer = visualizer_registry.build(visualizer_name, visualizer_config)
            visualizer(
                recorder=recorder,
                model=model,
                noise_scheduler=state.noise_scheduler,
                batch_obs_images=prepared["obs"],
                batch_goal_images=prepared["goal"],
                batch_viz_obs_images=prepared["viz_obs"],
                batch_viz_goal_images=prepared["viz_goal"],
                batch_action_label=prepared["actions"],
                batch_dist_label=prepared["distance"],
                goal_pos=prepared["goal_pos"],
                dataset_index=prepared["dataset_index"],
                metric_scale=prepared["metric_scale"],
                dataset_metadata=config["data"].get("dataset_metadata", {}),
                device=prepared["obs"].device,
                mode=mode,
                project_folder=project_folder,
                epoch=epoch,
                global_step=global_step,
                num_images_log=int(visualizer_config.get("num_images_log", config["visualization"].get("num_images_log", 8))),
                num_action_samples_log=int(
                    visualizer_config.get("num_action_samples_log", config["visualization"].get("num_action_samples_log", 30))
                ),
                action_stats=state.objective.action_stats,
            )

    def state_dict(self, state: NoMaDState):
        if state.ema_model is None:
            return {}
        return {"ema_model": state.ema_model.averaged_model.state_dict()}

    def step_scheduler(self, scheduler, eval_summaries, config) -> None:
        if scheduler is not None:
            scheduler.step()
