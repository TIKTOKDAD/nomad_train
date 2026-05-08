# ============================================================
# Loss reductions - mask aware averaging helpers
# ============================================================
# 本文件处理动作损失的有效样本归约：
# action_mask=1 的样本参与动作损失，action_mask=0 的样本只参与距离/其他任务。

import torch


# 带 mask 的均值归约
def masked_reduce(unreduced: torch.Tensor, mask: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
    # 若 unreduced 是 [B,T,D] 或 [B,T]，先沿最后维度逐层求均值，最后得到 [B]
    while unreduced.dim() > 1:
        unreduced = unreduced.mean(dim=-1)
    if unreduced.shape != mask.shape:
        raise ValueError(f"{unreduced.shape} != {mask.shape}")
    # eps 避免全 0 mask 时除零；全 0 时结果会接近 0
    return (unreduced * mask).mean() / (mask.mean() + eps)


# 动作损失的专用归约
def action_reduce(unreduced: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    return masked_reduce(unreduced, action_mask)
