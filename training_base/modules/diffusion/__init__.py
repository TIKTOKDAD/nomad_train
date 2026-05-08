from training_base.modules.diffusion.builders import build_conditional_unet1d
from training_base.registry import module_registry

# ============================================================
# Diffusion exports - action denoiser module registration
# ============================================================
# conditional_unet1d 是 NoMaD 动作扩散模型的核心去噪网络。

# 注册扩散模型构建函数
module_registry.register("conditional_unet1d")(build_conditional_unet1d)

__all__ = ["build_conditional_unet1d"]
