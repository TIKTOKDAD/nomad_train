import torch

from training_base.registry import optimizer_registry


@optimizer_registry.register("adam")
def build_adam(params, config):
    return torch.optim.Adam(params, lr=float(config["lr"]), betas=tuple(config.get("betas", (0.9, 0.98))))


@optimizer_registry.register("adamw")
def build_adamw(params, config):
    return torch.optim.AdamW(params, lr=float(config["lr"]), weight_decay=float(config.get("weight_decay", 0.0)))


@optimizer_registry.register("sgd")
def build_sgd(params, config):
    return torch.optim.SGD(params, lr=float(config["lr"]), momentum=float(config.get("momentum", 0.9)))


def build_optimizer(model, config):
    return optimizer_registry.build(config["name"], model.parameters(), config)
