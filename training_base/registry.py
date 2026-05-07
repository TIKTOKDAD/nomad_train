from training_base.core.registry import Registry


algorithm_registry = Registry("algorithm")
model_registry = Registry("model")
objective_registry = Registry("objective")
metric_registry = Registry("metric")
visualizer_registry = Registry("visualizer")
callback_registry = Registry("callback")
log_sink_registry = Registry("log_sink")
module_registry = Registry("module")
loss_registry = Registry("loss")
optimizer_registry = Registry("optimizer")
scheduler_registry = Registry("scheduler")
noise_scheduler_registry = Registry("noise_scheduler")


def register_builtins() -> None:
    from training_base import algorithms  # noqa: F401
    from training_base import callbacks  # noqa: F401
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
