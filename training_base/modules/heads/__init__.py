# ============================================================
# Head builders - waypoint and distance prediction heads
# ============================================================
# 本模块注册两类输出头：
# waypoint_head 由配置字典构建，distance_mlp 直接以 embedding_dim 构建。

from training_base.modules.heads.distance_mlp import DistanceMLP
from training_base.modules.heads.waypoint import WaypointPredictionHead
from training_base.registry import module_registry


# 构建航点预测头
def build_waypoint_head(config):
    # config 由 model builder 组装，包含 input_dim/len_traj_pred/action_dim 等派生字段
    return WaypointPredictionHead(
        input_dim=config["input_dim"],
        len_traj_pred=config["len_traj_pred"],
        num_action_params=config["num_action_params"],
        learn_angle=bool(config["learn_angle"]),
    )


# 注册预测头模块
module_registry.register("distance_mlp")(DistanceMLP)
module_registry.register("waypoint_head")(build_waypoint_head)

__all__ = ["DistanceMLP", "WaypointPredictionHead", "build_waypoint_head"]
