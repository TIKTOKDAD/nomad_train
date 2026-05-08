# ============================================================
# Waypoint head - supervised distance and trajectory prediction
# ============================================================
# 本文件实现 GNM/ViNT 共用的预测头：
# 1. dist_predictor 输出观测到目标的离散距离回归值
# 2. action_predictor 输出未来 len_traj_pred 个航点
# 3. 位置分量用 cumsum 从增量形式恢复成绝对局部航点

import torch
import torch.nn as nn
import torch.nn.functional as F


# 航点预测头：输出距离与动作序列
class WaypointPredictionHead(nn.Module):
    """Predict distance and supervised waypoint trajectories from a fused feature."""

    # 初始化预测头参数与线性层
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
        # 距离头是单输出线性层：[B,input_dim] -> [B,1]
        self.dist_predictor = nn.Linear(input_dim, 1)
        # 动作头一次性输出整条轨迹，再 reshape 成 [B,T,D]
        self.action_predictor = nn.Linear(input_dim, self.len_traj_pred * self.num_action_params)

    # 前向：预测距离与动作，并进行累积与归一化
    def forward(self, features: torch.Tensor):
        dist_pred = self.dist_predictor(features)
        action_pred = self.action_predictor(features)
        # 展开为每个未来时间步的动作参数
        action_pred = action_pred.reshape(
            (action_pred.shape[0], self.len_traj_pred, self.num_action_params)
        )
        # 累积位移得到绝对轨迹
        # head 原始输出可看作逐步增量，cumsum 后与 Dataset 生成的绝对局部航点标签对齐
        action_pred[:, :, :2] = torch.cumsum(action_pred[:, :, :2], dim=1)
        # 若学习角度，归一化角度向量
        if self.learn_angle:
            # 方向以 cos/sin 向量表示，normalize 保证它落在单位圆附近
            action_pred[:, :, 2:] = F.normalize(action_pred[:, :, 2:].clone(), dim=-1)
        return dist_pred, action_pred
