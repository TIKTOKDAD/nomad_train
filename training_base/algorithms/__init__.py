# ============================================================
# Algorithm exports - built-in algorithm registrations
# ============================================================
# 导入本模块会触发 gnm/nomad/vint 算法类上的 registry 装饰器。
# Trainer 获取的是 Algorithm 子类实例，而不是直接依赖这些具体文件。
# 算法模块导出入口
from training_base.algorithms.gnm import GNMAlgorithm
from training_base.algorithms.nomad import NoMaDAlgorithm
from training_base.algorithms.vint import ViNTAlgorithm

__all__ = ["GNMAlgorithm", "NoMaDAlgorithm", "ViNTAlgorithm"]
