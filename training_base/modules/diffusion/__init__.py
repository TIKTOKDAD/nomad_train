from training_base.modules.diffusion.builders import build_conditional_unet1d
from training_base.registry import module_registry

module_registry.register("conditional_unet1d")(build_conditional_unet1d)

__all__ = ["build_conditional_unet1d"]
