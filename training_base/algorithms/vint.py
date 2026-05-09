# ============================================================
# ViNT algorithm registration - supervised waypoint variant
# ============================================================
# ViNT 与 GNM 一样走监督航点算法；
# 本文件只注册名称，模型结构差异由 models/vint.py 决定。

from training_base.algorithms.supervised_waypoint import SupervisedWaypointAlgorithm
from training_base.registry import algorithm_registry


# 注册 ViNT 算法（复用监督航点算法实现）
@algorithm_registry.register("vint")
class ViNTAlgorithm(SupervisedWaypointAlgorithm):
    # ViNT 与 GNM 使用同一套监督航点训练流程，差异在模型 builder
    name = "vint"
