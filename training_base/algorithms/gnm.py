from training_base.algorithms.supervised_waypoint import SupervisedWaypointAlgorithm
from training_base.registry import algorithm_registry


# 注册 GNM 算法（复用监督航点算法实现）
@algorithm_registry.register("gnm")
class GNMAlgorithm(SupervisedWaypointAlgorithm):
    # name 只用于日志/进度条；训练逻辑继承 SupervisedWaypointAlgorithm
    name = "gnm"
