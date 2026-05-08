# ============================================================
# Objective exports - supervised and diffusion training losses
# ============================================================
# 导入本模块会触发 objective_registry 注册：
# supervised_waypoint 服务 GNM/ViNT，nomad_diffusion 服务 NoMaD。
# 目标函数模块导出入口
from training_base.objectives.nomad_diffusion import NoMaDDiffusionObjective
from training_base.objectives.supervised_waypoint import SupervisedWaypointObjective

__all__ = ["NoMaDDiffusionObjective", "SupervisedWaypointObjective"]
