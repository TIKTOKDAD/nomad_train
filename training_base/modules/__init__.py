# ============================================================
# Module package exports - reusable neural network components
# ============================================================
# 导入 diffusion/heads/vision 子包会触发 module_registry 注册。
# 上层 model builder 只通过 registry key 构建这些可复用模块。
# 复用网络模块入口：视觉编码器、预测头、扩散模型等
"""Reusable neural-network modules for navigation models."""

from training_base.modules import diffusion as _diffusion  # noqa: F401
from training_base.modules import heads as _heads  # noqa: F401
from training_base.modules import vision as _vision  # noqa: F401
