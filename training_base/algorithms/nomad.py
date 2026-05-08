# ============================================================
# NoMaD algorithm - diffusion navigation training logic
# ============================================================
# 本文件承接 NoMaD 的特殊训练流程：
# 1. 维护扩散噪声调度器、EMA 模型和 NoMaD objective
# 2. 训练时采样 goal mask，并优化距离预测 + 动作扩散噪声预测
# 3. 评估/重指标/可视化时复用 EMA 模型，观察条件/无条件动作分布

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


# NoMaD 算法的运行时状态
@dataclass
class NoMaDState:
    # diffusion scheduler 控制加噪/去噪时间步，不是 nn.Module 参数
    noise_scheduler: object
    # ema_model 保存滑动平均权重，评估时通常比即时权重更稳定
    ema_model: object
    # objective 保存损失函数、action_stats、goal mask 概率等训练超参
    objective: object


# 注册 NoMaD 算法
@algorithm_registry.register("nomad")
class NoMaDAlgorithm(Algorithm):
    name = "nomad"

    # 构建模型并返回额外对象（如噪声调度器）
    def build_model(self, config):
        result = build_model(config)
        return result.model, result.extras

    # 构建损失目标
    def build_objective(self, config):
        return objective_registry.build(config["objective"]["name"], config["objective"])

    # 从检查点恢复模型/优化器/EMA 状态
    def prepare_resume(self, model, optimizer, scheduler, config, device) -> ResumeState:
        # 若未指定恢复目录，直接返回空状态
        load_run = config["runtime"].get("load_run")
        if not load_run:
            return ResumeState(extra={"ema_state_dict": None})

        # 读取最新检查点与辅助状态
        load_project_folder = os.path.join("logs", load_run)
        print("正在从以下目录加载模型: ", load_project_folder)
        latest_checkpoint = load_checkpoint(find_latest_checkpoint(load_project_folder), device)
        load_model_state(model, latest_checkpoint, strict=False, model_name=config["model"]["name"])

        # 优先读取 checkpoint 内记录的 epoch；若旧 checkpoint 没写 epoch，则扫描数字命名权重兜底
        current_epoch = latest_checkpoint.get("epoch", -1) + 1 if isinstance(latest_checkpoint, dict) else 0
        if current_epoch == 0:
            epoch_ids = []
            for filename in os.listdir(load_project_folder):
                stem, ext = os.path.splitext(filename)
                if ext == ".pth" and stem.isdigit():
                    epoch_ids.append(int(stem))
            if epoch_ids:
                current_epoch = max(epoch_ids) + 1

        # EMA 与优化器/调度器状态（兼容不同存储方式）
        algorithm_state = latest_checkpoint.get("algorithm_state", {}) if isinstance(latest_checkpoint, dict) else {}
        # 新格式保存在 algorithm_state.ema_model，旧格式可能直接保存在 checkpoint 顶层或独立 ema_latest.pth
        resume_ema_state = algorithm_state.get("ema_model", None)
        if resume_ema_state is None and isinstance(latest_checkpoint, dict):
            resume_ema_state = latest_checkpoint.get("ema_model", None)
        ema_latest_path = os.path.join(load_project_folder, "ema_latest.pth")
        if resume_ema_state is None and os.path.exists(ema_latest_path):
            resume_ema_state = torch.load(ema_latest_path, map_location=device)

        optimizer_state = latest_checkpoint.get("optimizer", None) if isinstance(latest_checkpoint, dict) else None
        scheduler_state = latest_checkpoint.get("scheduler", None) if isinstance(latest_checkpoint, dict) else None
        # 为兼容历史训练脚本，优化器/调度器也支持独立 latest 文件
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
        # 返回恢复状态，包含 EMA 权重与 global_step
        return ResumeState(
            current_epoch=current_epoch,
            latest_checkpoint=latest_checkpoint,
            load_project_folder=load_project_folder,
            extra={"ema_state_dict": resume_ema_state, "global_step": global_step},
        )

    # 创建噪声调度器与 EMA 模型等状态
    def create_state(self, model, model_extras, objective, config, device, resume_state: ResumeState):
        from diffusers.training_utils import EMAModel

        # 依据配置决定是否启用 EMA
        ema_config = config.get("algorithm", {}).get("ema", {})
        ema_model = None
        if bool(ema_config.get("enabled", True)):
            # unwrap_model 取出 DDP 内部模型，EMA 只跟踪真实参数而不是 DDP 外壳
            ema_model = EMAModel(model=unwrap_model(model), power=float(ema_config.get("power", 0.75)))
            resume_ema = (resume_state.extra or {}).get("ema_state_dict")
            if resume_ema is not None:
                # remap_legacy_state_dict 处理旧命名，例如 noise_pred_net -> diffusion_model
                ema_state = remap_legacy_state_dict("nomad", strip_module_prefix(resume_ema))
                incompatible = ema_model.averaged_model.load_state_dict(ema_state, strict=False)
                report_state_key_differences(incompatible, label="NoMaD EMA 状态")
        return NoMaDState(noise_scheduler=model_extras["noise_scheduler"], ema_model=ema_model, objective=objective)

    # 数据预处理：图像变换、可视化缩放、张量迁移
    def prepare_batch(self, batch, transform, device, mode: str, should_log_images: bool):
        # obs_image 是 [B, 3*(context+1), H, W]；最后一帧用于日志观测图
        obs_images = torch.split(batch.obs_image, 3, dim=1)
        viz_obs = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1]) if should_log_images else None
        viz_goal = TF.resize(batch.goal_image, VISUALIZATION_IMAGE_SIZE[::-1]) if should_log_images else None
        return {
            # NoMaD 的视觉编码器仍然需要多帧观测拼接张量
            "obs": split_and_transform_obs(batch.obs_image, transform, device),
            "goal": transform_goal(batch.goal_image, transform, device),
            "viz_obs": viz_obs,
            "viz_goal": viz_goal,
            # actions 是绝对航点；objective 内会转换为增量并按 action_stats 归一化
            "actions": batch.actions.to(device, non_blocking=True),
            "distance": batch.distance.float().to(device, non_blocking=True),
            "action_mask": batch.action_mask.float().to(device, non_blocking=True),
            "goal_pos": batch.goal_pos,
            "dataset_index": batch.dataset_index,
            "metric_scale": batch.metric_scale,
        }

    # 训练步：计算距离损失 + 扩散损失
    def train_step(self, model, prepared, state: NoMaDState, config) -> StepResult:
        # train_losses 返回完整中间量，既包含 loss，也包含 noise_pred/noise/goal_mask 等可调试信息
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

    # 评估步：计算不同 mask 策略下的扩散损失
    def eval_step(self, model, prepared, state: NoMaDState, config) -> StepResult:
        # 评估不反传，因此只比较随机 mask、无 mask、全 goal mask 三种条件策略
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

    # 评估时优先使用 EMA 模型
    def model_for_eval(self, model, state: NoMaDState):
        # 未启用 EMA 时回退到即时模型；启用时统一用 averaged_model 做评估/可视化
        if state.ema_model is None:
            return unwrap_model(model)
        return state.ema_model.averaged_model

    # 更新 EMA 权重
    def after_optimizer_step(self, model, state: NoMaDState, config) -> None:
        if state.ema_model is not None:
            state.ema_model.step(unwrap_model(model))

    # 计算重指标（如行为评估）
    def heavy_metrics(self, model, prepared, state: NoMaDState, config, mode: str):
        logs = {}
        for metric_config in config.get("metrics", {}).get("heavy", []):
            metric_config = dict(metric_config)
            if not bool(metric_config.pop("enabled", True)):
                continue
            # heavy metric 通常会执行完整反向扩散采样，频率需要比轻量指标低
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

    # 可视化输出（动作分布、轨迹等）
    def visualize(self, *, model, prepared, result, state: NoMaDState, config, mode, project_folder, epoch, batch_idx, num_batches, recorder) -> None:
        # 图像日志未触发时不生成 Matplotlib 文件，避免训练热路径额外开销
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
                num_images_log=int(visualizer_config.get("num_images_log", config["visualization"].get("num_images_log", 8))),
                num_action_samples_log=int(
                    visualizer_config.get("num_action_samples_log", config["visualization"].get("num_action_samples_log", 30))
                ),
                action_stats=state.objective.action_stats,
            )

    # 保存 EMA 权重以便断点恢复
    def state_dict(self, state: NoMaDState):
        if state.ema_model is None:
            return {}
        # 只保存 averaged_model 的参数；即时模型由通用 checkpoint payload 保存
        return {"ema_model": state.ema_model.averaged_model.state_dict()}

    # NoMaD 的调度器按步更新
    def step_scheduler(self, scheduler, eval_summaries, config) -> None:
        if scheduler is not None:
            scheduler.step()
