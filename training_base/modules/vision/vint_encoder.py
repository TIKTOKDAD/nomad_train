# ============================================================
# ViNT encoder - EfficientNet features plus transformer decoder
# ============================================================
# 本文件实现 ViNT 监督模型的视觉主干：
# 1. EfficientNet 分别编码多帧观测和目标条件
# 2. late_fusion=False 时目标分支输入“当前观测 + 目标图”6 通道拼接
# 3. MultiLayerDecoder 用自注意力融合时序 token，输出 waypoint head 特征

from typing import Optional

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

from training_base.modules.transformer.decoder import MultiLayerDecoder


# ViNT 编码器：EfficientNet 编码 + Transformer 融合
class ViNTEncoder(nn.Module):
    """ViNT EfficientNet plus transformer fusion trunk."""

    output_dim = 32

    # 初始化编码器与融合解码器
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

        # 当前仅支持 EfficientNet 系列
        if obs_encoder.split("-")[0] != "efficientnet":
            raise NotImplementedError

        # 观测编码器每次处理单帧 RGB，之后再恢复成 [B, context+1, D] token 序列
        self.obs_encoder = EfficientNet.from_name(obs_encoder, in_channels=3)
        self.num_obs_features = self.obs_encoder._fc.in_features
        if self.late_fusion:
            # late_fusion=True 时目标分支只编码 goal 图，再在 token 层融合
            self.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=3)
        else:
            # 默认 ViNT 方式：目标分支输入当前观测帧和目标图的 6 通道拼接
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

        # decoder 输入 token 数 = context 历史帧 + 当前帧 + 目标 token
        self.decoder = MultiLayerDecoder(
            embed_dim=self.obs_encoding_size,
            seq_len=self.context_size + 2,
            output_layers=[256, 128, 64, self.output_dim],
            nhead=mha_num_attention_heads,
            num_layers=mha_num_attention_layers,
            ff_dim_factor=mha_ff_dim_factor,
        )

    # 编码目标（支持 late_fusion 方式）
    def _encode_goal(self, obs_img: torch.Tensor, goal_img: torch.Tensor) -> torch.Tensor:
        if self.late_fusion:
            # late_fusion: goal encoder 只看目标图，和观测 token 的交互交给 transformer
            goal_encoding = self.goal_encoder.extract_features(goal_img)
        else:
            # non-late-fusion: 使用当前观测帧辅助目标编码，保留原 ViNT 输入语义
            obsgoal_img = torch.cat([obs_img[:, 3 * self.context_size :, :, :], goal_img], dim=1)
            goal_encoding = self.goal_encoder.extract_features(obsgoal_img)
        goal_encoding = self.goal_encoder._avg_pooling(goal_encoding)
        if self.goal_encoder._global_params.include_top:
            goal_encoding = goal_encoding.flatten(start_dim=1)
            goal_encoding = self.goal_encoder._dropout(goal_encoding)
        goal_encoding = self.compress_goal_enc(goal_encoding)
        if len(goal_encoding.shape) == 2:
            # 目标编码转成单个 token：[B, 1, D]
            goal_encoding = goal_encoding.unsqueeze(1)
        if goal_encoding.shape[2] != self.goal_encoding_size:
            raise ValueError(f"{goal_encoding.shape[2]} != {self.goal_encoding_size}")
        return goal_encoding

    # 编码观测序列（多帧）
    def _encode_obs(self, obs_img: torch.Tensor) -> torch.Tensor:
        # obs_img 按通道拼接多帧；先切成若干 [B,3,H,W]，再在 batch 维合并提升编码效率
        obs_frames = torch.split(obs_img, 3, dim=1)
        obs_frames = torch.concat(obs_frames, dim=0)

        obs_encoding = self.obs_encoder.extract_features(obs_frames)
        obs_encoding = self.obs_encoder._avg_pooling(obs_encoding)
        if self.obs_encoder._global_params.include_top:
            obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_encoding = self.obs_encoder._dropout(obs_encoding)
        obs_encoding = self.compress_obs_enc(obs_encoding)
        # 从 [(context+1)*B, D] 还原到 [B, context+1, D]
        obs_encoding = obs_encoding.reshape((self.context_size + 1, -1, self.obs_encoding_size))
        return torch.transpose(obs_encoding, 0, 1)

    # 前向：拼接观测与目标 token 后输入 Transformer
    def forward(self, obs_img: torch.Tensor, goal_img: torch.Tensor) -> torch.Tensor:
        goal_encoding = self._encode_goal(obs_img, goal_img)
        obs_encoding = self._encode_obs(obs_img)
        # tokens: [B, context+2, D]，最后一个 token 是目标条件
        tokens = torch.cat((obs_encoding, goal_encoding), dim=1)
        return self.decoder(tokens)
