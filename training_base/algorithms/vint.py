from training_base.algorithms.supervised_waypoint import SupervisedWaypointAlgorithm
from training_base.registry import algorithm_registry


# 注册 ViNT 算法（复用监督航点算法实现）
@algorithm_registry.register("vint")
class ViNTAlgorithm(SupervisedWaypointAlgorithm):
    # ViNT 与 GNM 使用同一套监督航点训练流程，差异在模型 builder
    name = "vint"
