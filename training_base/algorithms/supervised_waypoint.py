# ============================================================
# Supervised waypoint algorithm - shared GNM/ViNT trainer logic
# ============================================================
# 本文件实现 GNM 和 ViNT 共用的监督式航点训练流程：
# 1. 统一准备 NavigationBatch 中的图像、距离标签和动作标签
# 2. 调用模型得到距离预测和航点预测，再交给 objective 计算损失
# 3. 按配置挂载轻量指标和可视化器，避免在模型类里写训练逻辑

from typing import Dict

import torch
import torchvision.transforms.functional as TF

from training_base.algorithms.base import Algorithm, StepResult
from training_base.core.checkpoint import (
    ResumeState,
    load_training_resume,
)
from training_base.data.batch import split_and_transform_obs, transform_goal
from training_base.data.data_utils import VISUALIZATION_IMAGE_SIZE
from training_base.models import build_model
from training_base.registry import metric_registry, objective_registry, visualizer_registry


# 监督航点算法基类：GNM/ViNT 只需要换模型 builder
class SupervisedWaypointAlgorithm(Algorithm):
    """Shared supervised waypoint recipe used by GNM and ViNT."""

    # 构建模型，返回 nn.Module 和可能的附加状态
    def build_model(self, config):
        result = build_model(config)
        return result.model, result.extras

    # 构建监督学习目标函数
    def build_objective(self, config):
        return objective_registry.build(config["objective"]["name"], config["objective"])

    # 使用通用 checkpoint 恢复逻辑
    def prepare_resume(self, model, optimizer, scheduler, config, device) -> ResumeState:
        return load_training_resume(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            device=device,
            model_name=config["model"]["name"],
        )

    # 将 NavigationBatch 转成模型/损失函数直接消费的 prepared 字典
    def prepare_batch(self, batch, transform, device, mode: str, should_log_images: bool, config=None):
        # obs_image 按通道拼了多帧，这里取最后一帧作为可视化观测图
        obs_images = torch.split(batch.obs_image, 3, dim=1)
        viz_size = tuple((config or {}).get("visualization", {}).get("image_size", VISUALIZATION_IMAGE_SIZE))
        viz_obs_image = TF.resize(obs_images[-1], viz_size) if should_log_images else None
        viz_goal_image = TF.resize(batch.goal_image, viz_size) if should_log_images else None
        return {
            # 训练图像先做 ImageNet normalize，再异步搬到目标设备
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

    # train/eval 共用的前向与损失计算
    def _step(self, model, prepared, state, config) -> StepResult:
        # 监督模型统一返回距离预测和未来航点预测
        dist_pred, action_pred = model(prepared["obs_image"], prepared["goal_image"])
        losses = state["objective"](
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

    # 训练阶段与评估阶段的计算路径相同，只由 Trainer 决定是否反传
    def train_step(self, model, prepared, state, config) -> StepResult:
        return self._step(model, prepared, state, config)

    def eval_step(self, model, prepared, state, config) -> StepResult:
        return self._step(model, prepared, state, config)

    # 计算配置中声明的轻量指标
    def light_metrics(self, model, prepared, result, state, config, mode: str):
        del model, state
        entries = config.get("metrics", {}).get("train" if mode == "train" else "eval", [])
        logs = {}
        for metric_config in entries:
            metric_config = dict(metric_config)
            if not bool(metric_config.pop("enabled", True)):
                continue
            # registry 中的 metric 函数只接收张量；log_name 只影响日志键名
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

    # 生成监督航点可视化图
    def visualize(self, *, model, prepared, result, state, config, mode, project_folder, epoch, batch_idx, num_batches, global_step, recorder) -> None:
        # 如果本 batch 没被调度到记录图片，prepare_batch 会把可视化图设为 None
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
                global_step=global_step,
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

    # 主指标：各评估集 total_loss 的平均值
    def primary_metric(self, eval_summaries: Dict[str, Dict[str, float]]) -> float:
        values = [
            metrics["total_loss"]
            for metrics in eval_summaries.values()
            if "total_loss" in metrics and metrics["total_loss"] == metrics["total_loss"]
        ]
        return sum(values) / len(values) if values else float("nan")
