# ============================================================
# Optimizer builders - registry based optimizer creation
# ============================================================
# 本文件把 optimizer.name 映射到 torch.optim 优化器：
# 1. 支持 adam/adamw/sgd
# 2. 只读取 optimizer 配置块，保持 Trainer 与具体优化器解耦

import torch

from training_base.registry import optimizer_registry


# 注册 Adam 优化器
@optimizer_registry.register("adam")
def build_adam(params, config):
    # betas 缺省沿用该项目常见设置 (0.9, 0.98)
    return torch.optim.Adam(params, lr=float(config["lr"]), betas=tuple(config.get("betas", (0.9, 0.98))))


# 注册 AdamW 优化器
@optimizer_registry.register("adamw")
def build_adamw(params, config):
    # weight_decay 缺省 0，避免用户未配置时悄悄改变正则化
    return torch.optim.AdamW(params, lr=float(config["lr"]), weight_decay=float(config.get("weight_decay", 0.0)))


# 注册 SGD 优化器
@optimizer_registry.register("sgd")
def build_sgd(params, config):
    # SGD 默认 momentum=0.9，适合需要显式对比 Adam 系列时使用
    return torch.optim.SGD(params, lr=float(config["lr"]), momentum=float(config.get("momentum", 0.9)))


# 统一优化器构建入口
def build_optimizer(model, config):
    # config["name"] 已在 normalize_config 中转小写
    return optimizer_registry.build(config["name"], model.parameters(), config)
