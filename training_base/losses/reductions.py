import torch


def masked_reduce(unreduced: torch.Tensor, mask: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
    while unreduced.dim() > 1:
        unreduced = unreduced.mean(dim=-1)
    if unreduced.shape != mask.shape:
        raise ValueError(f"{unreduced.shape} != {mask.shape}")
    return (unreduced * mask).mean() / (mask.mean() + eps)


def action_reduce(unreduced: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    return masked_reduce(unreduced, action_mask)
