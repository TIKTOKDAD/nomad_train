from typing import Optional

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

from training_base.modules.transformer.decoder import MultiLayerDecoder


class ViNTEncoder(nn.Module):
    """ViNT EfficientNet plus transformer fusion trunk."""

    output_dim = 32

    def __init__(
        self,
        *,
        context_size: int,
        obs_encoder: str = "efficientnet-b0",
        obs_encoding_size: int = 512,
        late_fusion: Optional[bool] = False,
        mha_num_attention_heads: int = 2,
        mha_num_attention_layers: int = 2,
        mha_ff_dim_factor: int = 4,
    ) -> None:
        super().__init__()
        self.context_size = context_size
        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size
        self.late_fusion = bool(late_fusion)

        if obs_encoder.split("-")[0] != "efficientnet":
            raise NotImplementedError

        self.obs_encoder = EfficientNet.from_name(obs_encoder, in_channels=3)
        self.num_obs_features = self.obs_encoder._fc.in_features
        if self.late_fusion:
            self.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=3)
        else:
            self.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=6)
        self.num_goal_features = self.goal_encoder._fc.in_features

        self.compress_obs_enc = (
            nn.Linear(self.num_obs_features, self.obs_encoding_size)
            if self.num_obs_features != self.obs_encoding_size
            else nn.Identity()
        )
        self.compress_goal_enc = (
            nn.Linear(self.num_goal_features, self.goal_encoding_size)
            if self.num_goal_features != self.goal_encoding_size
            else nn.Identity()
        )

        self.decoder = MultiLayerDecoder(
            embed_dim=self.obs_encoding_size,
            seq_len=self.context_size + 2,
            output_layers=[256, 128, 64, self.output_dim],
            nhead=mha_num_attention_heads,
            num_layers=mha_num_attention_layers,
            ff_dim_factor=mha_ff_dim_factor,
        )

    def _encode_goal(self, obs_img: torch.Tensor, goal_img: torch.Tensor) -> torch.Tensor:
        if self.late_fusion:
            goal_encoding = self.goal_encoder.extract_features(goal_img)
        else:
            obsgoal_img = torch.cat([obs_img[:, 3 * self.context_size :, :, :], goal_img], dim=1)
            goal_encoding = self.goal_encoder.extract_features(obsgoal_img)
        goal_encoding = self.goal_encoder._avg_pooling(goal_encoding)
        if self.goal_encoder._global_params.include_top:
            goal_encoding = goal_encoding.flatten(start_dim=1)
            goal_encoding = self.goal_encoder._dropout(goal_encoding)
        goal_encoding = self.compress_goal_enc(goal_encoding)
        if len(goal_encoding.shape) == 2:
            goal_encoding = goal_encoding.unsqueeze(1)
        if goal_encoding.shape[2] != self.goal_encoding_size:
            raise ValueError(f"{goal_encoding.shape[2]} != {self.goal_encoding_size}")
        return goal_encoding

    def _encode_obs(self, obs_img: torch.Tensor) -> torch.Tensor:
        obs_frames = torch.split(obs_img, 3, dim=1)
        obs_frames = torch.concat(obs_frames, dim=0)

        obs_encoding = self.obs_encoder.extract_features(obs_frames)
        obs_encoding = self.obs_encoder._avg_pooling(obs_encoding)
        if self.obs_encoder._global_params.include_top:
            obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_encoding = self.obs_encoder._dropout(obs_encoding)
        obs_encoding = self.compress_obs_enc(obs_encoding)
        obs_encoding = obs_encoding.reshape((self.context_size + 1, -1, self.obs_encoding_size))
        return torch.transpose(obs_encoding, 0, 1)

    def forward(self, obs_img: torch.Tensor, goal_img: torch.Tensor) -> torch.Tensor:
        goal_encoding = self._encode_goal(obs_img, goal_img)
        obs_encoding = self._encode_obs(obs_img)
        tokens = torch.cat((obs_encoding, goal_encoding), dim=1)
        return self.decoder(tokens)

