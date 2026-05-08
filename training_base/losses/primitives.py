# ============================================================
# Primitive losses - thin wrappers around torch.nn.functional
# ============================================================
# 本文件只封装最基础损失函数，方便注册表统一管理。
# objective 通过 get_configured_loss 拿到这些函数后再决定 reduction。

import torch
import torch.nn.functional as F


# 均方误差损失
def mse(pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    # 支持 reduction="none"，供 mask reduce 逐样本加权
    return F.mse_loss(pred, target, reduction=reduction)


# 交叉熵损失
def cross_entropy(pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    return F.cross_entropy(pred, target, reduction=reduction)


# 余弦相似度
def cosine_similarity(pred: torch.Tensor, target: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # 这里返回相似度而不是损失，主要用于日志指标
    return F.cosine_similarity(pred, target, dim=dim)
