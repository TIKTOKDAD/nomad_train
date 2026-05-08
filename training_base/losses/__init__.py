# ============================================================
# Loss registry - primitive losses and configured lookup
# ============================================================
# 本文件集中注册损失函数：
# 1. objective 配置中的 losses.<key> 可以写字符串或 {name: ...}
# 2. get_configured_loss 负责读取并返回实际函数
# 3. reductions.py 提供动作 mask 的专用归约方式

from training_base.losses.primitives import cross_entropy, cosine_similarity, mse
from training_base.losses.reductions import action_reduce, masked_reduce
from training_base.registry import loss_registry


# 注册常用损失函数
loss_registry.register("mse")(mse)
loss_registry.register("cross_entropy")(cross_entropy)
loss_registry.register("cosine_similarity")(cosine_similarity)


def get_configured_loss(losses_config, key: str, default: str = "mse"):
    # 支持两种 YAML 写法：distance: mse 或 distance: {name: mse}
    entry = (losses_config or {}).get(key, default)
    if isinstance(entry, dict):
        entry = entry.get("name", default)
    # 返回函数对象，objective 调用时再传 pred/target/reduction
    return loss_registry.get(str(entry).lower())


__all__ = [
    "action_reduce",
    "masked_reduce",
    "mse",
    "cross_entropy",
    "cosine_similarity",
    "get_configured_loss",
]
