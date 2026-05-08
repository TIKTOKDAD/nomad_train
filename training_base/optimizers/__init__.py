# ============================================================
# Optimizer exports - registry-backed optimizer builder
# ============================================================
# build_optimizer 根据 optimizer.name 从 registry 构建 torch.optim 优化器。
# 优化器模块导出入口
from training_base.optimizers.builders import build_optimizer

__all__ = ["build_optimizer"]
