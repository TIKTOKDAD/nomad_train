# ============================================================
# NoMaD model wrapper - vision condition, diffusion, distance
# ============================================================
# 本文件把 NoMaD 的三个子模块组合成统一模型：
# 1. vision_encoder 生成带 goal mask 的条件向量
# 2. diffusion_model 预测动作扩散过程中的噪声
# 3. distance_predictor 回归观测到目标的离散距离

import torch.nn as nn

from training_base.models import ModelBuild
from training_base.registry import model_registry, module_registry, noise_scheduler_registry


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


@model_registry.register("nomad")
def build_nomad(config) -> ModelBuild:
    model_config = config["model"]
    # vision_encoder 需要同时知道模型参数和数据参数（例如 context_size、image_size）
    vision_config = model_config["vision_encoder"]
    vision_encoder = module_registry.build(vision_config["name"], vision_config, config["data"])
    # diffusion 模块需要 global_cond_dim；未显式写时用视觉编码器输出维度兜底
    diffusion_config = dict(model_config["diffusion"])
    diffusion_config.setdefault("global_cond_dim", model_config["vision_encoder"]["encoding_size"])
    diffusion_model = module_registry.build(model_config["diffusion"]["name"], diffusion_config)
    # distance_predictor 是视觉条件上的距离头，独立于扩散动作头
    distance_predictor = module_registry.build(
        model_config["distance_predictor"]["name"],
        model_config["distance_predictor"]["embedding_dim"],
    )
    model = NoMaD(
        vision_encoder=vision_encoder,
        diffusion_model=diffusion_model,
        distance_predictor=distance_predictor,
    )
    # scheduler 不是 nn.Module，作为 extras 交给 NoMaDAlgorithm 管理
    scheduler = noise_scheduler_registry.build(model_config["diffusion_scheduler"]["name"], model_config["diffusion_scheduler"])
    return ModelBuild(model=model, extras={"noise_scheduler": scheduler})
