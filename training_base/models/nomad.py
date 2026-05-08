# ============================================================
# NoMaD model wrapper - vision condition, diffusion, distance
# ============================================================
# 本文件把 NoMaD 的三个子模块组合成统一模型：
# 1. vision_encoder 生成带 goal mask 的条件向量
# 2. diffusion_model 预测动作扩散过程中的噪声
# 3. distance_predictor 回归观测到目标的离散距离

import torch.nn as nn


# NoMaD 模型：由视觉编码器、扩散模型与距离预测器组成
class NoMaD(nn.Module):
    """NoMaD model composed from explicit navigation modules."""

    # 初始化各子模块
    def __init__(self, vision_encoder, diffusion_model, distance_predictor) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.diffusion_model = diffusion_model
        self.distance_predictor = distance_predictor

    # 编码观测与目标图像（支持 goal mask）
    def encode_vision(self, obs_img, goal_img, goal_mask):
        # goal_mask=0 使用目标图；goal_mask=1 屏蔽目标图，用于探索/无目标条件
        return self.vision_encoder(obs_img, goal_img, input_goal_mask=goal_mask)

    # 扩散模型噪声预测
    def predict_noise(self, sample, timestep, global_cond):
        # sample 是当前 noisy action，global_cond 是视觉条件向量
        return self.diffusion_model(sample=sample, timestep=timestep, global_cond=global_cond)

    # 预测目标距离
    def predict_distance(self, obsgoal_cond):
        # 距离头只看视觉条件，不直接消费动作序列
        return self.distance_predictor(obsgoal_cond)
