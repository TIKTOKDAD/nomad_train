import torch

from training_base.registry import scheduler_registry


@scheduler_registry.register("none")
def build_none(optimizer, config):
    return None


@scheduler_registry.register("cosine")
def build_cosine(optimizer, config):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config["epochs"]))


@scheduler_registry.register("cyclic")
def build_cyclic(optimizer, config):
    lr = float(config["lr"])
    return torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=lr / 10.0,
        max_lr=lr,
        step_size_up=int(config["cyclic_period"]) // 2,
        cycle_momentum=False,
    )


@scheduler_registry.register("plateau")
def build_plateau(optimizer, config):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=float(config["plateau_factor"]),
        patience=int(config["plateau_patience"]),
        verbose=True,
    )


def build_scheduler(optimizer, config):
    if config is None:
        return None
    name = config.get("name")
    if name is None:
        return None
    scheduler = scheduler_registry.build(str(name).lower(), optimizer, config)
    warmup = config.get("warmup", {})
    if scheduler is not None and bool(warmup.get("enabled", False)):
        from warmup_scheduler import GradualWarmupScheduler

        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=1,
            total_epoch=int(warmup["epochs"]),
            after_scheduler=scheduler,
        )
    return scheduler
