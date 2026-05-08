# ============================================================
# Model builders - assemble GNM, ViNT, and NoMaD modules
# ============================================================
# 本文件是模型构建入口：
# 1. 根据 config["model"]["name"] 选择 GNM/ViNT/NoMaD
# 2. 通过 module_registry 构建 encoder/head/diffusion/distance_predictor
# 3. 返回 ModelBuild，其中 extras 保存训练时需要但不属于模型参数的对象

from dataclasses import dataclass
from typing import Any, Dict

from training_base.registry import model_registry, module_registry, noise_scheduler_registry


# 构建结果：返回模型实例与额外对象
@dataclass
class ModelBuild:
    # model 是最终 nn.Module，交给 Trainer 移动设备、包装 DDP、保存 checkpoint
    model: Any
    # extras 存放 noise_scheduler 等非 nn.Module 状态，算法 create_state 会接管
    extras: Dict[str, Any]


# 构建 GNM 模型
@model_registry.register("gnm")
def build_gnm(config) -> ModelBuild:
    from training_base.models.gnm import GNM

    data = config["data"]
    model_config = config["model"]
    # encoder_config 从 model_config 拷贝，允许 model.encoder 子块覆盖默认字段
    encoder_config = dict(model_config)
    if "encoder" in model_config:
        encoder_config.update(model_config["encoder"])
        encoder_name = encoder_config.get("name", "gnm_encoder")
    else:
        encoder_name = "gnm_encoder"
    # 组装编码器与预测头
    # encoder 输出融合特征，head 再拆成距离预测和航点预测两个头
    encoder = module_registry.build(encoder_name, encoder_config, data)
    head = module_registry.build(
        model_config.get("head", {}).get("name", "waypoint_head"),
        {
            # head 的输入维度来自 encoder.output_dim，避免配置重复写同一个数
            "input_dim": encoder.output_dim,
            "len_traj_pred": data["len_traj_pred"],
            "num_action_params": 4 if data["learn_angle"] else 2,
            "learn_angle": data["learn_angle"],
        },
    )
    model = GNM(
        context_size=data["context_size"],
        len_traj_pred=data["len_traj_pred"],
        learn_angle=data["learn_angle"],
        encoder=encoder,
        head=head,
    )
    return ModelBuild(model=model, extras={})


# 构建 ViNT 模型
@model_registry.register("vint")
def build_vint(config) -> ModelBuild:
    from training_base.models.vint import ViNT

    data = config["data"]
    model_config = config["model"]
    # ViNT 与 GNM 共享监督航点 head，但 encoder 默认切换为 vint_encoder
    encoder_config = dict(model_config)
    if "encoder" in model_config:
        encoder_config.update(model_config["encoder"])
        encoder_name = encoder_config.get("name", "vint_encoder")
    else:
        encoder_name = "vint_encoder"
    # 组装编码器与预测头
    encoder = module_registry.build(encoder_name, encoder_config, data)
    head = module_registry.build(
        model_config.get("head", {}).get("name", "waypoint_head"),
        {
            # learn_angle=True 时动作标签是 (x,y,cos,sin)，否则只有二维位移
            "input_dim": encoder.output_dim,
            "len_traj_pred": data["len_traj_pred"],
            "num_action_params": 4 if data["learn_angle"] else 2,
            "learn_angle": data["learn_angle"],
        },
    )
    model = ViNT(
        context_size=data["context_size"],
        len_traj_pred=data["len_traj_pred"],
        learn_angle=data["learn_angle"],
        encoder=encoder,
        head=head,
    )
    return ModelBuild(model=model, extras={})


# 构建 NoMaD 模型（视觉编码器 + 扩散模型 + 距离预测器）
@model_registry.register("nomad")
def build_nomad(config) -> ModelBuild:
    from training_base.models.nomad import NoMaD

    model_config = config["model"]
    vision_config = model_config["vision_encoder"]
    # 构建各子模块并注入配置
    # vision_encoder 输出 obsgoal condition，供扩散模型和距离预测器共用
    vision_encoder = module_registry.build(vision_config["name"], vision_config, config["data"])
    diffusion_config = dict(model_config["diffusion"])
    # ConditionalUnet1D 需要 global_cond_dim；默认等于视觉编码器输出维度
    diffusion_config.setdefault("global_cond_dim", model_config["vision_encoder"]["encoding_size"])
    diffusion_model = module_registry.build(model_config["diffusion"]["name"], diffusion_config)
    distance_predictor = module_registry.build(
        model_config["distance_predictor"]["name"],
        model_config["distance_predictor"]["embedding_dim"],
    )
    model = NoMaD(
        vision_encoder=vision_encoder,
        diffusion_model=diffusion_model,
        distance_predictor=distance_predictor,
    )
    # noise_scheduler 不是 nn.Module 参数，因此放入 extras 由 NoMaDAlgorithm 管理
    scheduler = noise_scheduler_registry.build(model_config["diffusion_scheduler"]["name"], model_config["diffusion_scheduler"])
    return ModelBuild(model=model, extras={"noise_scheduler": scheduler})


# 统一模型构建入口
def build_model(config) -> ModelBuild:
    return model_registry.build(config["model"]["name"], config)
