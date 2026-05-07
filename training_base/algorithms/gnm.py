from training_base.algorithms.supervised_waypoint import SupervisedWaypointAlgorithm
from training_base.registry import algorithm_registry


@algorithm_registry.register("gnm")
class GNMAlgorithm(SupervisedWaypointAlgorithm):
    name = "gnm"
