# ============================================================
# Visualizer exports - supervised and NoMaD image logging
# ============================================================
# 导入可视化器会触发 visualizer_registry 注册。
# supervised_waypoint 绘制预测/标签轨迹，nomad_action_distribution 绘制采样分布。
# 可视化模块导出入口

from training_base.visualizers.nomad_action_distribution import NoMaDActionDistributionVisualizer
from training_base.visualizers.supervised_waypoint import SupervisedWaypointVisualizer


__all__ = ["NoMaDActionDistributionVisualizer", "SupervisedWaypointVisualizer"]
