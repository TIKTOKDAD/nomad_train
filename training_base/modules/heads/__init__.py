from training_base.modules.heads.distance_mlp import DistanceMLP
from training_base.modules.heads.waypoint import WaypointPredictionHead
from training_base.registry import module_registry


def build_waypoint_head(config):
    return WaypointPredictionHead(
        input_dim=config["input_dim"],
        len_traj_pred=config["len_traj_pred"],
        num_action_params=config["num_action_params"],
        learn_angle=bool(config["learn_angle"]),
    )


module_registry.register("distance_mlp")(DistanceMLP)
module_registry.register("waypoint_head")(build_waypoint_head)

__all__ = ["DistanceMLP", "WaypointPredictionHead", "build_waypoint_head"]
