# ============================================================
# GNM model wrapper - encoder plus waypoint head
# ============================================================
# 本文件只负责把预构建的 GNM encoder 和 waypoint head 串起来：
# encoder: obs/goal 图像 -> 融合特征
# head: 融合特征 -> 距离预测 + 未来航点预测

from typing import Optional, Tuple

import torch

from training_base.models.base import BaseModel
from training_base.models import ModelBuild
from training_base.registry import model_registry, module_registry


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
            raise ValueError("GNM 需要预先构建好的 encoder 和 head 模块。")
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


@model_registry.register("gnm")
def build_gnm(config) -> ModelBuild:
    data = config["data"]
    model_config = config["model"]
    # encoder_config 从 model 顶层继承，再叠加 model.encoder，兼容旧配置和平铺配置
    encoder_config = dict(model_config)
    if "encoder" in model_config:
        encoder_config.update(model_config["encoder"])
        encoder_name = encoder_config.get("name", "gnm_encoder")
    else:
        encoder_name = "gnm_encoder"
    # GNM encoder 输出融合特征，head 再根据数据配置决定轨迹长度和动作维度
    encoder = module_registry.build(encoder_name, encoder_config, data)
    head = module_registry.build(
        model_config.get("head", {}).get("name", "waypoint_head"),
        {
            "input_dim": encoder.output_dim,
            "len_traj_pred": data["len_traj_pred"],
            "num_action_params": 4 if data["learn_angle"] else 2,
            "learn_angle": data["learn_angle"],
        },
    )
    # 外壳模型只保存 encoder/head，不在内部重复解析 config
    model = GNM(
        context_size=data["context_size"],
        len_traj_pred=data["len_traj_pred"],
        learn_angle=data["learn_angle"],
        encoder=encoder,
        head=head,
    )
    return ModelBuild(model=model, extras={})
