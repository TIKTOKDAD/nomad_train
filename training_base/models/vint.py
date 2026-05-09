# ============================================================
# ViNT model wrapper - encoder plus waypoint head
# ============================================================
# 本文件保持和 GNM 相同的监督模型外壳：
# 1. ViNTEncoder 负责 EfficientNet + Transformer 融合
# 2. WaypointPredictionHead 负责距离和轨迹输出
# 3. Trainer/Algorithm 可以用相同接口训练 GNM 与 ViNT

from typing import Optional, Tuple

import torch

from training_base.models.base import BaseModel
from training_base.models import ModelBuild
from training_base.registry import model_registry, module_registry


# ViNT 模型：编码器 + 预测头
class ViNT(BaseModel):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
        encoder=None,
        head=None,
    ) -> None:
        super().__init__(context_size, len_traj_pred, learn_angle)
        # encoder/head 由 models.__init__.build_vint 负责组装，这里只做完整性检查
        if encoder is None or head is None:
            raise ValueError("ViNT 需要预先构建好的 encoder 和 head 模块。")
        self.encoder = encoder
        self.head = head

    # 前向：提取特征并预测距离/动作
    def forward(
        self, obs_img: torch.Tensor, goal_img: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 编码器输出定长融合特征，head 再产生两个任务头输出
        features = self.encoder(obs_img, goal_img)
        return self.head(features)


@model_registry.register("vint")
def build_vint(config) -> ModelBuild:
    data = config["data"]
    model_config = config["model"]
    # encoder_config 兼容 model 顶层参数和 model.encoder 子配置两种写法
    encoder_config = dict(model_config)
    if "encoder" in model_config:
        encoder_config.update(model_config["encoder"])
        encoder_name = encoder_config.get("name", "vint_encoder")
    else:
        encoder_name = "vint_encoder"
    # ViNT encoder 输出 Transformer 融合特征；通用 waypoint_head 输出两个任务头
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
    # 返回 ModelBuild，保持和 NoMaD builder 的统一协议
    model = ViNT(
        context_size=data["context_size"],
        len_traj_pred=data["len_traj_pred"],
        learn_angle=data["learn_angle"],
        encoder=encoder,
        head=head,
    )
    return ModelBuild(model=model, extras={})
