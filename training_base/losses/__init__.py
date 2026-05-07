from training_base.losses.primitives import cross_entropy, cosine_similarity, mse
from training_base.losses.reductions import action_reduce, masked_reduce
from training_base.registry import loss_registry


loss_registry.register("mse")(mse)
loss_registry.register("cross_entropy")(cross_entropy)
loss_registry.register("cosine_similarity")(cosine_similarity)


def get_configured_loss(losses_config, key: str, default: str = "mse"):
    entry = (losses_config or {}).get(key, default)
    if isinstance(entry, dict):
        entry = entry.get("name", default)
    return loss_registry.get(str(entry).lower())


__all__ = [
    "action_reduce",
    "masked_reduce",
    "mse",
    "cross_entropy",
    "cosine_similarity",
    "get_configured_loss",
]
