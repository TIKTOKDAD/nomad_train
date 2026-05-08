# ============================================================
# GNM model wrapper - encoder plus waypoint head
# ============================================================
# 本文件只负责把预构建的 GNM encoder 和 waypoint head 串起来：
# encoder: obs/goal 图像 -> 融合特征
# head: 融合特征 -> 距离预测 + 未来航点预测

from typing import Optional, Tuple

import torch

from training_base.models.base import BaseModel


# GNM 模型：编码器 + 预测头
class GNM(BaseModel):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
        encoder=None,
        head=None,
    ) -> None:
        super().__init__(context_size, len_traj_pred, learn_angle)
        # builder 必须显式注入 encoder/head，避免模型内部再读取全局配置
        if encoder is None or head is None:
            raise ValueError("GNM requires prebuilt encoder and head modules.")
        self.encoder = encoder
        self.head = head

    # 前向：提取特征并预测距离/动作
    def forward(
        self, obs_img: torch.Tensor, goal_img: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # obs_img: [B, 3*(context+1), H, W]；goal_img: [B, 3, H, W]
        features = self.encoder(obs_img, goal_img)
        # 返回 dist_pred [B,1] 与 action_pred [B,T,D]
        return self.head(features)
