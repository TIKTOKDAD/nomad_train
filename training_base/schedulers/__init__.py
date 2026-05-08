# ============================================================
# Scheduler exports - registry-backed LR schedule builder
# ============================================================
# build_scheduler 根据 scheduler.name 构建学习率调度器，并可选套 warmup。
# 学习率调度器模块导出入口
from training_base.schedulers.builders import build_scheduler

__all__ = ["build_scheduler"]
