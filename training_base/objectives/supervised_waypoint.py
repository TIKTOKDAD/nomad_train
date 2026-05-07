from typing import Dict

import torch
import torch.nn.functional as F

from training_base.losses import action_reduce, get_configured_loss
from training_base.registry import objective_registry


@objective_registry.register("supervised_waypoint")
class SupervisedWaypointObjective:
    def __init__(self, config) -> None:
        self.alpha = float(config.get("alpha", 0.5))
        losses = config.get("losses", {})
        self.distance_loss = get_configured_loss(losses, "distance", "mse")
        self.action_loss = get_configured_loss(losses, "action", "mse")

    def __call__(
        self,
        *,
        dist_label: torch.Tensor,
        action_label: torch.Tensor,
        dist_pred: torch.Tensor,
        action_pred: torch.Tensor,
        learn_angle: bool,
        action_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        dist_loss = self.distance_loss(dist_pred.squeeze(-1), dist_label.float())
        if action_pred.shape != action_label.shape:
            raise ValueError(f"{action_pred.shape} != {action_label.shape}")

        action_loss = action_reduce(self.action_loss(action_pred, action_label, reduction="none"), action_mask)
        action_waypts_cos_sim = action_reduce(
            F.cosine_similarity(action_pred[:, :, :2], action_label[:, :, :2], dim=-1),
            action_mask,
        )
        multi_action_waypts_cos_sim = action_reduce(
            F.cosine_similarity(
                torch.flatten(action_pred[:, :, :2], start_dim=1),
                torch.flatten(action_label[:, :, :2], start_dim=1),
                dim=-1,
            ),
            action_mask,
        )
        results = {
            "dist_loss": dist_loss,
            "action_loss": action_loss,
            "action_waypts_cos_sim": action_waypts_cos_sim,
            "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim,
        }
        if learn_angle:
            results["action_orien_cos_sim"] = action_reduce(
                F.cosine_similarity(action_pred[:, :, 2:], action_label[:, :, 2:], dim=-1),
                action_mask,
            )
            results["multi_action_orien_cos_sim"] = action_reduce(
                F.cosine_similarity(
                    torch.flatten(action_pred[:, :, 2:], start_dim=1),
                    torch.flatten(action_label[:, :, 2:], start_dim=1),
                    dim=-1,
                ),
                action_mask,
            )
        results["total_loss"] = self.alpha * 1e-2 * dist_loss + (1 - self.alpha) * action_loss
        return results
