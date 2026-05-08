# ============================================================
# Callback exports - checkpoint/performance lifecycle hooks
# ============================================================
# 导入本模块会触发内置回调注册：
# checkpoint 负责保存训练状态，perf_monitor 负责记录吞吐和耗时。
# 回调模块导出入口
from training_base.callbacks.checkpoint import CheckpointCallback
from training_base.callbacks.manager import CallbackManager
from training_base.callbacks.perf_monitor import PerfMonitorCallback

__all__ = ["CheckpointCallback", "CallbackManager", "PerfMonitorCallback"]
