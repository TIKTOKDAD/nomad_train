# ============================================================
# Global registries - configurable training components
# ============================================================
# 本文件创建全局注册表实例：
# 1. 各子模块在 import 时通过装饰器注册自身
# 2. YAML 中的 name 字段会映射到对应 registry 的 key
# 3. register_builtins 负责导入内置模块，触发这些注册动作

# 全局注册表：用于按名称构建算法/模型/组件等
from training_base.core.registry import Registry


# 各子系统的注册表（字符串名 -> 构造函数/类）
algorithm_registry = Registry("algorithm")
model_registry = Registry("model")
objective_registry = Registry("objective")
metric_registry = Registry("metric")
visualizer_registry = Registry("visualizer")
callback_registry = Registry("callback")
log_sink_registry = Registry("log_sink")
dataset_registry = Registry("dataset")
data_module_registry = Registry("data_module")
module_registry = Registry("module")
loss_registry = Registry("loss")
optimizer_registry = Registry("optimizer")
scheduler_registry = Registry("scheduler")
noise_scheduler_registry = Registry("noise_scheduler")


# 导入内置模块触发注册（模块内通过装饰器注册）
def register_builtins() -> None:
    # noqa: F401 表示这些导入只为了触发副作用注册，而不是直接使用模块名
    from training_base import algorithms  # noqa: F401
    from training_base import callbacks  # noqa: F401
    from training_base import data  # noqa: F401
    from training_base import loggers  # noqa: F401
    from training_base import losses  # noqa: F401
    from training_base import metrics  # noqa: F401
    from training_base import models  # noqa: F401
    from training_base import modules  # noqa: F401
    from training_base import noise_schedulers  # noqa: F401
    from training_base import objectives  # noqa: F401
    from training_base import optimizers  # noqa: F401
    from training_base import schedulers  # noqa: F401
    from training_base import visualizers  # noqa: F401
