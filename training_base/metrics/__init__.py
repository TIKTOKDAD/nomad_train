from training_base.metrics.action_metrics import flattened_waypoint_cosine, waypoint_cosine, waypoint_mse
from training_base.metrics.nomad_behavior import compute_nomad_behavior_metrics, model_output
from training_base.registry import metric_registry

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
