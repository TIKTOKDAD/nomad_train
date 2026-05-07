from training_base.noise_schedulers.ddpm import build_ddpm_scheduler
from training_base.registry import noise_scheduler_registry

noise_scheduler_registry.register("ddpm")(build_ddpm_scheduler)

__all__ = ["build_ddpm_scheduler"]
