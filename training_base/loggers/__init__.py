# ============================================================
# Logger exports - recorder and built-in sink registrations
# ============================================================
# 导入 console/wandb 模块是为了触发 log_sink_registry 注册。
# Recorder 是 Trainer 直接使用的日志门面。
# 日志模块导出入口
from training_base.loggers.recorder import Recorder
from training_base.loggers import console as _console  # noqa: F401
from training_base.loggers import wandb as _wandb  # noqa: F401

__all__ = ["Recorder"]
