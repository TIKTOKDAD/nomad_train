from dataclasses import dataclass, replace
from typing import Any, Dict, Sequence

import torch
from torch.utils.data._utils.collate import default_collate


@dataclass
class NavigationBatch:
    """Named batch protocol for navigation algorithms."""

    obs_image: torch.Tensor
    goal_image: torch.Tensor
    actions: torch.Tensor
    distance: torch.Tensor
    goal_pos: torch.Tensor
    dataset_index: torch.Tensor
    action_mask: torch.Tensor
    metric_scale: torch.Tensor
    extras: Dict[str, Any]

    @property
    def batch_size(self) -> int:
        return int(self.obs_image.shape[0])

    def pin_memory(self):
        return replace(
            self,
            obs_image=self.obs_image.pin_memory(),
            goal_image=self.goal_image.pin_memory(),
            actions=self.actions.pin_memory(),
            distance=self.distance.pin_memory(),
            goal_pos=self.goal_pos.pin_memory(),
            dataset_index=self.dataset_index.pin_memory(),
            action_mask=self.action_mask.pin_memory(),
            metric_scale=self.metric_scale.pin_memory(),
        )


def as_navigation_batch(data: Sequence[torch.Tensor]) -> NavigationBatch:
    if isinstance(data, NavigationBatch):
        return data
    if len(data) not in {7, 8}:
        raise ValueError(f"Expected 7 or 8 tensors from NavigationDataset, got {len(data)}")
    metric_scale = data[7] if len(data) == 8 else torch.ones_like(data[5], dtype=torch.float32)
    return NavigationBatch(
        obs_image=data[0],
        goal_image=data[1],
        actions=data[2],
        distance=data[3],
        goal_pos=data[4],
        dataset_index=data[5],
        action_mask=data[6],
        metric_scale=metric_scale,
        extras={},
    )


def navigation_collate(samples):
    return as_navigation_batch(default_collate(samples))


def split_and_transform_obs(obs_image: torch.Tensor, transform, device: torch.device):
    obs_images = torch.split(obs_image, 3, dim=1)
    obs_images = [transform(obs).to(device, non_blocking=True) for obs in obs_images]
    return torch.cat(obs_images, dim=1)


def transform_goal(goal_image: torch.Tensor, transform, device: torch.device):
    return transform(goal_image).to(device, non_blocking=True)
