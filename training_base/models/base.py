# ============================================================
# Base model - shared supervised navigation model contract
# ============================================================
# 本文件定义 GNM/ViNT 这类监督航点模型的基础属性：
# 1. context_size 控制观测历史帧数量
# 2. len_traj_pred 控制未来航点预测长度
# 3. learn_angle 决定动作输出维度是否包含方向表示

import torch
import torch.nn as nn

from typing import List, Dict, Optional, Tuple


# 基础模型：提供通用属性与接口
class BaseModel(nn.Module):
    # 初始化基础参数（上下文长度、预测长度、角度学习开关）
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
    ) -> None:
        """
        Base Model main class
        Args:
            context_size (int): how many previous observations to used for context
            len_traj_pred (int): how many waypoints to predict in the future
            learn_angle (bool): whether to predict the yaw of the robot
        """
        super(BaseModel, self).__init__()
        self.context_size = context_size
        self.learn_angle = learn_angle
        self.len_trajectory_pred = len_traj_pred
        if self.learn_angle:
            # 数据集侧会把 yaw 转成 cos/sin，因此网络 head 输出 4 维：(x, y, cos, sin)
            self.num_action_params = 4  # last two dims are the cos and sin of the angle
        else:
            # 不学习角度时只预测局部坐标系下的二维航点位置
            self.num_action_params = 2

    # 全局平均池化并拉平为向量
    def flatten(self, z: torch.Tensor) -> torch.Tensor:
        # 输入通常是 [B,C,H,W] 卷积特征；输出 [B,C]
        z = nn.functional.adaptive_avg_pool2d(z, (1, 1))
        z = torch.flatten(z, 1)
        return z

    # 子类需实现前向传播
    def forward(
        self, obs_img: torch.tensor, goal_img: torch.tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model
        Args:
            obs_img (torch.Tensor): batch of observations
            goal_img (torch.Tensor): batch of goals
        Returns:
            dist_pred (torch.Tensor): predicted distance to goal
            action_pred (torch.Tensor): predicted action
        """
        raise NotImplementedError
