# ============================================================
# GNM algorithm registration - supervised waypoint variant
# ============================================================
# GNM 的训练逻辑完全复用 SupervisedWaypointAlgorithm；
# 本文件只负责把算法名称 gnm 注册到 algorithm_registry。

from training_base.algorithms.supervised_waypoint import SupervisedWaypointAlgorithm
from training_base.registry import algorithm_registry


# 注册 GNM 算法（复用监督航点算法实现）
@algorithm_registry.register("gnm")
class GNMAlgorithm(SupervisedWaypointAlgorithm):
    # name 只用于日志/进度条；训练逻辑继承 SupervisedWaypointAlgorithm
    name = "gnm"
