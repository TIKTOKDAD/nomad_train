# ============================================================
# Supervised waypoint algorithm - GNM/ViNT shared trainer logic
# ============================================================
# 本文件把“监督式航点预测”抽象成一个算法：
# 1. 构建 GNM/ViNT 模型和 supervised_waypoint objective
# 2. 把 NavigationBatch 转成模型前向所需的 obs/goal/action/distance 张量
# 3. 计算距离回归损失、航点轨迹损失、轻量指标和可视化结果

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


# 监督航点算法：用于 GNM/ViNT 等基于回归的模型
class SupervisedWaypointAlgorithm(Algorithm):
    # 构建模型与额外组件
    def build_model(self, config):
        # build_model 返回 ModelBuild；监督模型没有额外运行时对象，extras 通常为空
        result = build_model(config)
        return result.model, result.extras

    # 构建损失目标
    def build_objective(self, config):
        return objective_registry.build(config["objective"]["name"], config["objective"])

    # 从最新检查点恢复模型与优化器状态
    def prepare_resume(self, model, optimizer, scheduler, config, device) -> ResumeState:
        # load_run 为空代表从头训练，不读取历史 checkpoint
        load_run = config["runtime"].get("load_run")
        if not load_run:
            return ResumeState(extra={})

        # 约定从 logs/<load_run>/latest.pth 恢复，兼容旧版权重命名
        load_project_folder = os.path.join("logs", load_run)
        print("Loading model from ", load_project_folder)
        latest_checkpoint = load_checkpoint(find_latest_checkpoint(load_project_folder), device)
        load_model_state(model, latest_checkpoint, strict=False, model_name=config["model"]["name"])

        # current_epoch 指向下一轮要训练的 epoch，因此 checkpoint epoch 需要 +1
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

    # 将 batch 转换为模型所需的输入格式
    def prepare_batch(self, batch, transform, device, mode: str, should_log_images: bool):
        # obs_image 是按通道拼接的多帧图像；最后 3 个通道代表当前观测帧
        obs_images = torch.split(batch.obs_image, 3, dim=1)
        # 可视化只保留 resize 后的当前观测和目标图像，不参与训练前向
        viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE) if should_log_images else None
        viz_goal_image = TF.resize(batch.goal_image, VISUALIZATION_IMAGE_SIZE) if should_log_images else None
        return {
            # split_and_transform_obs 会逐帧做 ImageNet Normalize，再重新按通道拼接
            "obs_image": split_and_transform_obs(batch.obs_image, transform, device),
            # 目标图像单独变换，保持 [B, 3, H, W]
            "goal_image": transform_goal(batch.goal_image, transform, device),
            # distance/action/action_mask 迁移到训练设备，non_blocking 配合 pin_memory 加速 H2D
            "dist_label": batch.distance.to(device, non_blocking=True),
            "action_label": batch.actions.to(device, non_blocking=True),
            "action_mask": batch.action_mask.to(device, non_blocking=True),
            # 以下字段留在 CPU 侧即可，主要服务日志和 Matplotlib 可视化
            "goal_pos": batch.goal_pos,
            "dataset_index": batch.dataset_index,
            "metric_scale": batch.metric_scale,
            "viz_obs_image": viz_obs_image,
            "viz_goal_image": viz_goal_image,
        }

    # 共享的前向与损失计算逻辑（训练/评估通用）
    def _step(self, model, prepared, state, config) -> StepResult:
        # 监督模型输出两个头：目标距离 dist_pred 和未来航点 action_pred
        dist_pred, action_pred = model(prepared["obs_image"], prepared["goal_image"])
        objective = state["objective"]
        # objective 内部根据 learn_angle 决定是否额外计算朝向相似度
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
            # 保存预测供 light_metrics/visualize 复用，避免重复前向
            extras={"dist_pred": dist_pred, "action_pred": action_pred},
        )

    # 训练步调用共享逻辑
    def train_step(self, model, prepared, state, config) -> StepResult:
        return self._step(model, prepared, state, config)

    # 评估步调用共享逻辑
    def eval_step(self, model, prepared, state, config) -> StepResult:
        return self._step(model, prepared, state, config)

    # 轻量指标（主要是动作误差类）
    def light_metrics(self, model, prepared, result, state, config, mode: str):
        del model, state
        # train/eval 可以配置不同轻量指标；未启用或为空时直接返回空日志
        entries = config.get("metrics", {}).get("train" if mode == "train" else "eval", [])
        logs = {}
        for metric_config in entries:
            metric_config = dict(metric_config)
            if not bool(metric_config.pop("enabled", True)):
                continue
            # name 是 registry key；log_name 允许同一指标用不同日志名输出
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

    # 可视化预测与目标轨迹
    def visualize(self, *, model, prepared, result, state, config, mode, project_folder, epoch, batch_idx, num_batches, recorder) -> None:
        # 没有开启图像日志时 prepared 中的 viz 图像为 None，直接跳过
        if prepared["viz_obs_image"] is None or prepared["viz_goal_image"] is None:
            return
        for visualizer_config in self.visualization_configs(config, mode, "supervised_waypoint"):
            visualizer_name = visualizer_config.pop("name")
            visualizer = visualizer_registry.build(visualizer_name, visualizer_config)
            # 可视化器只消费张量和元信息，不再触发模型前向
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
                dataset_metadata=config["data"].get("dataset_metadata", {}),
                use_latest=mode == "train",
            )

    # 选择用于调度器的主指标（默认总损失）
    def primary_metric(self, eval_summaries: Dict[str, Dict[str, float]]) -> float:
        values = [
            metrics["total_loss"]
            for metrics in eval_summaries.values()
            if "total_loss" in metrics and metrics["total_loss"] == metrics["total_loss"]
        ]
        return sum(values) / len(values) if values else float("nan")
