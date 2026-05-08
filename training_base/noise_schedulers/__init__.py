from training_base.noise_schedulers.ddpm import build_ddpm_scheduler
from training_base.registry import noise_scheduler_registry

# ============================================================
# Noise scheduler exports - diffusion scheduler registration
# ============================================================
# ddpm scheduler 负责 NoMaD 动作扩散的加噪和反向采样时间表。

# 注册噪声调度器
noise_scheduler_registry.register("ddpm")(build_ddpm_scheduler)

__all__ = ["build_ddpm_scheduler"]
