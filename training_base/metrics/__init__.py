# ============================================================
# Metric exports - light and heavy navigation metrics
# ============================================================
# 本模块注册轻量动作指标，并导出 NoMaD 重指标。
# 轻量指标可在训练高频记录，NoMaD behavior 指标通常低频执行。

from training_base.metrics.action_metrics import flattened_waypoint_cosine, waypoint_cosine, waypoint_mse
from training_base.metrics.nomad_behavior import compute_nomad_behavior_metrics, model_output
from training_base.registry import metric_registry

# 注册常用指标函数
metric_registry.register("waypoint_mse")(waypoint_mse)
metric_registry.register("waypoint_cosine")(waypoint_cosine)
metric_registry.register("flattened_waypoint_cosine")(flattened_waypoint_cosine)

__all__ = [
    "compute_nomad_behavior_metrics",
    "flattened_waypoint_cosine",
    "model_output",
    "waypoint_cosine",
    "waypoint_mse",
]
