from typing import Optional, Tuple

import torch

from training_base.models.base import BaseModel


class ViNT(BaseModel):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
        encoder=None,
        head=None,
    ) -> None:
        super().__init__(context_size, len_traj_pred, learn_angle)
        if encoder is None or head is None:
            raise ValueError("ViNT requires prebuilt encoder and head modules.")
        self.encoder = encoder
        self.head = head

    def forward(
        self, obs_img: torch.Tensor, goal_img: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(obs_img, goal_img)
        return self.head(features)
