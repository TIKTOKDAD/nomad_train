# ============================================================
# Scheduler builders - learning rate schedule registry
# ============================================================
# 本文件把 scheduler.name 映射到 PyTorch 学习率调度器：
# 1. none/cosine/cyclic/plateau 四种基础策略
# 2. 可选 GradualWarmupScheduler 包装
# 3. Algorithm.step_scheduler 决定按 metric 还是普通 step 更新

import torch

from training_base.registry import scheduler_registry


# 不使用调度器
@scheduler_registry.register("none")
def build_none(optimizer, config):
    # 显式返回 None，Trainer/Algorithm 会跳过 scheduler.step
    return None


# CosineAnnealing 调度器
@scheduler_registry.register("cosine")
def build_cosine(optimizer, config):
    # T_max 使用总 epoch 数，按 epoch 调度
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config["epochs"]))


# CyclicLR 调度器
@scheduler_registry.register("cyclic")
def build_cyclic(optimizer, config):
    lr = float(config["lr"])
    # base_lr 设为 max_lr 的 1/10，cycle_momentum=False 兼容 Adam/AdamW
    return torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=lr / 10.0,
        max_lr=lr,
        step_size_up=int(config["cyclic_period"]) // 2,
        cycle_momentum=False,
    )


# ReduceLROnPlateau 调度器
@scheduler_registry.register("plateau")
def build_plateau(optimizer, config):
    # ReduceLROnPlateau 由 Algorithm.primary_metric 提供评估指标
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=float(config["plateau_factor"]),
        patience=int(config["plateau_patience"]),
        verbose=True,
    )


# 统一调度器构建入口（支持 warmup 包装）
def build_scheduler(optimizer, config):
    if config is None:
        return None
    name = config.get("name")
    if name is None:
        return None
    # 先构建基础 scheduler，再按需套 warmup
    scheduler = scheduler_registry.build(str(name).lower(), optimizer, config)
    warmup = config.get("warmup", {})
    if scheduler is not None and bool(warmup.get("enabled", False)):
        from warmup_scheduler import GradualWarmupScheduler

        # multiplier=1 表示 warmup 到基础 lr，不额外放大学习率
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=1,
            total_epoch=int(warmup["epochs"]),
            after_scheduler=scheduler,
        )
    return scheduler
