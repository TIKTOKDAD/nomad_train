import torch
import torch.nn as nn
import torch.nn.functional as F


class WaypointPredictionHead(nn.Module):
    """Predict distance and supervised waypoint trajectories from a fused feature."""

    def __init__(
        self,
        *,
        input_dim: int,
        len_traj_pred: int,
        num_action_params: int,
        learn_angle: bool,
    ) -> None:
        super().__init__()
        self.len_traj_pred = len_traj_pred
        self.num_action_params = num_action_params
        self.learn_angle = learn_angle
        self.dist_predictor = nn.Linear(input_dim, 1)
        self.action_predictor = nn.Linear(input_dim, self.len_traj_pred * self.num_action_params)

    def forward(self, features: torch.Tensor):
        dist_pred = self.dist_predictor(features)
        action_pred = self.action_predictor(features)
        action_pred = action_pred.reshape(
            (action_pred.shape[0], self.len_traj_pred, self.num_action_params)
        )
        action_pred[:, :, :2] = torch.cumsum(action_pred[:, :, :2], dim=1)
        if self.learn_angle:
            action_pred[:, :, 2:] = F.normalize(action_pred[:, :, 2:].clone(), dim=-1)
        return dist_pred, action_pred

