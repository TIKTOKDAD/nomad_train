import torch
import torch.nn.functional as F

from training_base.losses import action_reduce


def waypoint_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return action_reduce(F.mse_loss(pred, target, reduction="none"), mask)


def waypoint_cosine(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return action_reduce(F.cosine_similarity(pred[:, :, :2], target[:, :, :2], dim=-1), mask)


def flattened_waypoint_cosine(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return action_reduce(
        F.cosine_similarity(
            torch.flatten(pred[:, :, :2], start_dim=1),
            torch.flatten(target[:, :, :2], start_dim=1),
            dim=-1,
        ),
        mask,
    )
