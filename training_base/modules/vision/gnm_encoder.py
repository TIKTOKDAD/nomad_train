# ============================================================
# GNM encoder - MobileNet branches for observation and goal
# ============================================================
# 本文件实现 GNM 的视觉融合主干：
# 1. obs_mobilenet 编码历史观测序列
# 2. goal_mobilenet 编码“观测序列 + 目标图”的拼接输入
# 3. fusion MLP 将两个分支的向量融合为 waypoint head 的输入特征

import torch
import torch.nn as nn

from training_base.modules.vision.mobilenet import MobileNetEncoder


# GNM 视觉编码器：观测与目标特征融合
class GNMEncoder(nn.Module):
    """GNM visual encoder and fusion trunk."""

    output_dim = 32

    # 初始化观测/目标编码器与融合层
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

        # 观测分支输入通道数 = 3 * (context_size + 1)
        obs_mobilenet = MobileNetEncoder(num_images=1 + self.context_size)
        self.obs_mobilenet = obs_mobilenet.features
        # MobileNet 输出 last_channel 维，压缩到配置指定的 obs_encoding_size
        self.compress_observation = nn.Sequential(
            nn.Linear(obs_mobilenet.last_channel, self.obs_encoding_size),
            nn.ReLU(),
        )

        # 目标分支把观测序列和目标图一起拼接，因此输入图像数 = context + 当前 + goal
        goal_mobilenet = MobileNetEncoder(num_images=2 + self.context_size)
        self.goal_mobilenet = goal_mobilenet.features
        # 目标分支先扩到 1024 再压缩到 goal_encoding_size，保持原 GNM 结构
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

    # 全局平均池化并拉平成向量
    @staticmethod
    def flatten_features(features: torch.Tensor) -> torch.Tensor:
        features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        return torch.flatten(features, 1)

    # 前向：编码观测与目标后进行融合
    def forward(self, obs_img: torch.Tensor, goal_img: torch.Tensor) -> torch.Tensor:
        # 观测分支：只看历史/当前观测，得到环境上下文特征
        obs_encoding = self.obs_mobilenet(obs_img)
        obs_encoding = self.flatten_features(obs_encoding)
        obs_encoding = self.compress_observation(obs_encoding)

        # 目标分支：把目标图追加到通道维，让卷积编码器直接比较 obs 与 goal
        obs_goal_input = torch.cat([obs_img, goal_img], dim=1)
        goal_encoding = self.goal_mobilenet(obs_goal_input)
        goal_encoding = self.flatten_features(goal_encoding)
        goal_encoding = self.compress_goal(goal_encoding)

        # 两个分支拼接后输出固定 32 维特征，供 waypoint head 使用
        return self.fusion(torch.cat([obs_encoding, goal_encoding], dim=1))
