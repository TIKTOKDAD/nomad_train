import torch
import torch.nn as nn

from training_base.modules.vision.mobilenet import MobileNetEncoder


class GNMEncoder(nn.Module):
    """GNM visual encoder and fusion trunk."""

    output_dim = 32

    def __init__(
        self,
        *,
        context_size: int,
        obs_encoding_size: int,
        goal_encoding_size: int,
    ) -> None:
        super().__init__()
        self.context_size = context_size
        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = goal_encoding_size

        obs_mobilenet = MobileNetEncoder(num_images=1 + self.context_size)
        self.obs_mobilenet = obs_mobilenet.features
        self.compress_observation = nn.Sequential(
            nn.Linear(obs_mobilenet.last_channel, self.obs_encoding_size),
            nn.ReLU(),
        )

        goal_mobilenet = MobileNetEncoder(num_images=2 + self.context_size)
        self.goal_mobilenet = goal_mobilenet.features
        self.compress_goal = nn.Sequential(
            nn.Linear(goal_mobilenet.last_channel, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.goal_encoding_size),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(self.goal_encoding_size + self.obs_encoding_size, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim),
            nn.ReLU(),
        )

    @staticmethod
    def flatten_features(features: torch.Tensor) -> torch.Tensor:
        features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        return torch.flatten(features, 1)

    def forward(self, obs_img: torch.Tensor, goal_img: torch.Tensor) -> torch.Tensor:
        obs_encoding = self.obs_mobilenet(obs_img)
        obs_encoding = self.flatten_features(obs_encoding)
        obs_encoding = self.compress_observation(obs_encoding)

        obs_goal_input = torch.cat([obs_img, goal_img], dim=1)
        goal_encoding = self.goal_mobilenet(obs_goal_input)
        goal_encoding = self.flatten_features(goal_encoding)
        goal_encoding = self.compress_goal(goal_encoding)

        return self.fusion(torch.cat([obs_encoding, goal_encoding], dim=1))

