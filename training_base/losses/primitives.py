import torch
import torch.nn.functional as F


def mse(pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    return F.mse_loss(pred, target, reduction=reduction)


def cross_entropy(pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    return F.cross_entropy(pred, target, reduction=reduction)


def cosine_similarity(pred: torch.Tensor, target: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.cosine_similarity(pred, target, dim=dim)
