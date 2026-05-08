# ============================================================
# Distance MLP - NoMaD distance regression head
# ============================================================
# 本文件实现 NoMaD 的距离预测分支：
# 视觉编码器输出 obsgoal_cond 后，DistanceMLP 将其压缩为单个距离标量。

import torch.nn as nn


# 距离回归小 MLP
class DistanceMLP(nn.Module):
    """Small MLP distance regressor used by NoMaD-style models."""

    # 初始化多层感知机
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        # 逐步降维：D -> D/4 -> D/16 -> 1，保持结构轻量
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 4),
            nn.ReLU(),
            nn.Linear(embedding_dim // 4, embedding_dim // 16),
            nn.ReLU(),
            nn.Linear(embedding_dim // 16, 1),
        )

    # 前向：拉平后回归距离
    def forward(self, x):
        # 输入可能是 [B,1,D] 或 [B,D]，统一展平为 [-1,D]
        x = x.reshape((-1, self.embedding_dim))
        return self.network(x)
