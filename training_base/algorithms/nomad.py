# ============================================================
# NoMaD algorithm - diffusion navigation training logic
# ============================================================
# 本文件实现 NoMaD 的算法层逻辑：
# 1. 构建视觉编码器、扩散模型、距离头以及扩散噪声调度器
# 2. 训练时同时优化距离分类/回归损失和扩散噪声预测损失
# 3. 维护 EMA 模型，用于评估与可视化时获得更平滑的预测

import os
from dataclasses import dataclass

import torch
import torchvision.transforms.functional as TF

from training_base.algorithms.base import Algorithm, StepResult
from training_base.core.checkpoint import (
    ResumeState,
    load_training_resume,
    remap_legacy_state_dict,
    report_state_key_differences,
    strip_module_prefix,
)
from training_base.core.native_utils import unwrap_model
from training_base.data.batch import split_and_transform_obs, transform_goal
from training_base.data.data_utils import VISUALIZATION_IMAGE_SIZE
from training_base.models import build_model
from training_base.registry import algorithm_registry, metric_registry, objective_registry, visualizer_registry


# NoMaD 算法运行状态：Trainer 会把它传回各个 step/hook
@dataclass
class NoMaDState:
    # diffusers 风格的噪声调度器，负责加噪/反向采样时间步
    noise_scheduler: object
    # EMA 模型可选；启用时评估优先使用 averaged_model
    ema_model: object
    # NoMaD objective，封装扩散损失、距离损失和动作统计
    objective: object


# NoMaD 算法注册入口
@algorithm_registry.register("nomad")
class NoMaDAlgorithm(Algorithm):
    name = "nomad"

    # 构建模型并取出 noise_scheduler 等 extras
    def build_model(self, config):
        result = build_model(config)
        return result.model, result.extras

    # 构建 NoMaD objective；data_config_path 用于读取动作归一化统计
    def build_objective(self, config):
        objective_config = dict(config["objective"])
        objective_config.setdefault("data_config_path", config.get("data", {}).get("data_config_path"))
        return objective_registry.build(objective_config["name"], objective_config)

    # 恢复模型、优化器、调度器以及 NoMaD 额外的 EMA 状态
    def prepare_resume(self, model, optimizer, scheduler, config, device) -> ResumeState:
        resume_state = load_training_resume(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            device=device,
            model_name=config["model"]["name"],
        )
        if resume_state.latest_checkpoint is None:
            # 从头训练时没有 EMA 权重可恢复，仍写入统一字段便于 create_state 读取
            resume_state.extra = {"ema_state_dict": None, **(resume_state.extra or {})}
            return resume_state

        latest_checkpoint = resume_state.latest_checkpoint
        load_project_folder = resume_state.load_project_folder
        current_epoch = resume_state.current_epoch
        if current_epoch == 0:
            # 兼容旧版只有 epoch 文件、latest 中 epoch 字段不完整的 run 目录
            epoch_ids = []
            if load_project_folder and os.path.isdir(load_project_folder):
                for filename in os.listdir(load_project_folder):
                    stem, ext = os.path.splitext(filename)
                    if ext == ".pth" and stem.isdigit():
                        epoch_ids.append(int(stem))
            if epoch_ids:
                current_epoch = max(epoch_ids) + 1

        # 新版 checkpoint 把 EMA 放在 algorithm_state，旧版可能放顶层或单独 ema_latest.pth
        algorithm_state = latest_checkpoint.get("algorithm_state", {}) if isinstance(latest_checkpoint, dict) else {}
        resume_ema_state = algorithm_state.get("ema_model", None)
        if resume_ema_state is None and isinstance(latest_checkpoint, dict):
            resume_ema_state = latest_checkpoint.get("ema_model", None)
        ema_latest_path = os.path.join(load_project_folder, "ema_latest.pth") if load_project_folder else None
        if resume_ema_state is None and ema_latest_path and os.path.exists(ema_latest_path):
            resume_ema_state = torch.load(ema_latest_path, map_location=device)

        # 兼容旧版把 optimizer/scheduler 单独保存的目录结构
        optimizer_state = latest_checkpoint.get("optimizer", None) if isinstance(latest_checkpoint, dict) else None
        scheduler_state = latest_checkpoint.get("scheduler", None) if isinstance(latest_checkpoint, dict) else None
        if optimizer_state is None and load_project_folder:
            optimizer_latest_path = os.path.join(load_project_folder, "optimizer_latest.pth")
            if os.path.exists(optimizer_latest_path):
                optimizer_state = torch.load(optimizer_latest_path, map_location=device)
        if scheduler_state is None and load_project_folder:
            scheduler_latest_path = os.path.join(load_project_folder, "scheduler_latest.pth")
            if os.path.exists(scheduler_latest_path):
                scheduler_state = torch.load(scheduler_latest_path, map_location=device)
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state if isinstance(optimizer_state, dict) else optimizer_state.state_dict())
        if scheduler is not None and scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state if isinstance(scheduler_state, dict) else scheduler_state.state_dict())

        extra = dict(resume_state.extra or {})
        extra["ema_state_dict"] = resume_ema_state
        return ResumeState(
            current_epoch=current_epoch,
            latest_checkpoint=latest_checkpoint,
            load_project_folder=load_project_folder,
            extra=extra,
        )

    # 创建 NoMaDState，并按配置启用/恢复 EMA
    def create_state(self, model, model_extras, objective, config, device, resume_state: ResumeState):
        from diffusers.training_utils import EMAModel

        ema_config = config.get("algorithm", {}).get("ema", {})
        ema_model = None
        if bool(ema_config.get("enabled", True)):
            # EMA 跟踪 unwrap 后的真实模型，避免 DDP 包装层进入 averaged_model
            ema_model = EMAModel(model=unwrap_model(model), power=float(ema_config.get("power", 0.75)))
            resume_ema = (resume_state.extra or {}).get("ema_state_dict")
            if resume_ema is not None:
                # EMA 权重可能来自 DDP 或旧版命名，加载前统一整理 key
                ema_state = strip_module_prefix(resume_ema)
                if bool(config.get("runtime", {}).get("allow_legacy_weight_remap", False)):
                    ema_state = remap_legacy_state_dict("nomad", ema_state)
                incompatible = ema_model.averaged_model.load_state_dict(ema_state, strict=False)
                report_state_key_differences(incompatible, label="NoMaD EMA 状态")
        return NoMaDState(noise_scheduler=model_extras["noise_scheduler"], ema_model=ema_model, objective=objective)

    # 准备 NoMaD batch：图像归一化、张量搬设备、可视化图保留原尺度
    def prepare_batch(self, batch, transform, device, mode: str, should_log_images: bool, config=None):
        obs_images = torch.split(batch.obs_image, 3, dim=1)
        viz_size = tuple((config or {}).get("visualization", {}).get("image_size", VISUALIZATION_IMAGE_SIZE))
        viz_obs = TF.resize(obs_images[-1], viz_size[::-1]) if should_log_images else None
        viz_goal = TF.resize(batch.goal_image, viz_size[::-1]) if should_log_images else None
        return {
            # NoMaD 视觉编码器仍接收按通道拼接的历史观测
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

    # 单步训练：计算距离损失和扩散噪声预测损失
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

    # 评估时分别测试随机 mask、无 mask、goal mask 三种扩散条件
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

    # 评估优先使用 EMA 模型，没有 EMA 时回退到当前模型
    def model_for_eval(self, model, state: NoMaDState):
        if state.ema_model is None:
            return unwrap_model(model)
        return state.ema_model.averaged_model

    # 优化器更新后推进 EMA
    def after_optimizer_step(self, model, state: NoMaDState, config) -> None:
        if state.ema_model is not None:
            state.ema_model.step(unwrap_model(model))

    # 计算 NoMaD 较重的行为指标，如动作分布采样指标
    def heavy_metrics(self, model, prepared, state: NoMaDState, config, mode: str):
        logs = {}
        for metric_config in config.get("metrics", {}).get("heavy", []):
            metric_config = dict(metric_config)
            if not bool(metric_config.pop("enabled", True)):
                continue
            # heavy metric 通常会运行扩散采样，开销较大，因此由 Trainer 的 schedule 控制频率
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

    # 生成 NoMaD 动作分布可视化
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

    # checkpoint 中保存 EMA 权重；主模型权重由 core.checkpoint 统一保存
    def state_dict(self, state: NoMaDState):
        if state.ema_model is None:
            return {}
        return {"ema_model": state.ema_model.averaged_model.state_dict()}

    # NoMaD 默认使用按 epoch 前进的 scheduler，不依赖验证集主指标
    def step_scheduler(self, scheduler, eval_summaries, config) -> None:
        if scheduler is not None:
            scheduler.step()
