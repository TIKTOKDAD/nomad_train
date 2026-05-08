# ============================================================
# Vision module builders - registry adapters for visual encoders
# ============================================================
# 本文件把 YAML 中的视觉编码器配置转换为具体 nn.Module：
# 1. GNM 使用 MobileNet 双分支编码器
# 2. ViNT 使用 EfficientNet + Transformer decoder
# 3. NoMaD 使用 masked ViNT / 可选 ViT，并统一替换 GroupNorm

from training_base.registry import module_registry


# 构建 GNM 视觉编码器
def build_gnm_encoder(config, data_config):
    from training_base.modules.vision.gnm_encoder import GNMEncoder

    # data_config 提供 context_size；model config 提供编码维度
    return GNMEncoder(
        context_size=data_config["context_size"],
        obs_encoding_size=config["obs_encoding_size"],
        goal_encoding_size=config["goal_encoding_size"],
    )


# 构建 ViNT 视觉编码器
def build_vint_encoder(config, data_config):
    from training_base.modules.vision.vint_encoder import ViNTEncoder

    transformer = config["transformer"]
    # transformer 子配置拆成 ViNTEncoder 初始化参数，避免 encoder 自己直接读 YAML
    return ViNTEncoder(
        context_size=data_config["context_size"],
        obs_encoder=config.get("obs_encoder", "efficientnet-b0"),
        obs_encoding_size=config["obs_encoding_size"],
        late_fusion=bool(config.get("late_fusion", False)),
        mha_num_attention_heads=transformer["num_heads"],
        mha_num_attention_layers=transformer["num_layers"],
        mha_ff_dim_factor=transformer["ff_dim_factor"],
    )


# 构建 NoMaD 使用的 masked ViNT 编码器
def build_masked_vint(config, data_config):
    from training_base.modules.vision.masked_vint import NoMaD_ViNT, replace_bn_with_gn

    transformer = config["transformer"]
    # NoMaD 视觉编码器输出 encoding_size 维条件向量，供 diffusion/distance 共享
    encoder = NoMaD_ViNT(
        obs_encoder=config.get("obs_encoder", "efficientnet-b0"),
        obs_encoding_size=config["encoding_size"],
        context_size=data_config["context_size"],
        mha_num_attention_heads=transformer["num_heads"],
        mha_num_attention_layers=transformer["num_layers"],
        mha_ff_dim_factor=transformer["ff_dim_factor"],
    )
    # NoMaD 小 batch 训练下 GroupNorm 通常比 BatchNorm 更稳定
    return replace_bn_with_gn(encoder)


# 构建 ViT 编码器
def build_vit(config, data_config):
    from training_base.modules.vision.masked_vint import replace_bn_with_gn
    from training_base.modules.vision.vit import ViT

    transformer = config["transformer"]
    # ViT 需要 image_size/patch_size；image_size 来自 data 配置，patch_size 来自 model 配置
    encoder = ViT(
        obs_encoding_size=config["encoding_size"],
        context_size=data_config["context_size"],
        image_size=data_config["image_size"],
        patch_size=config["patch_size"],
        mha_num_attention_heads=transformer["num_heads"],
        mha_num_attention_layers=transformer["num_layers"],
    )
    return replace_bn_with_gn(encoder)


# 注册视觉编码器构建函数
# registry key 必须与 YAML model.vision_encoder.name 或 model.encoder.name 对齐
module_registry.register("gnm_encoder")(build_gnm_encoder)
module_registry.register("vint_encoder")(build_vint_encoder)
module_registry.register("masked_vint")(build_masked_vint)
module_registry.register("vit")(build_vit)

__all__ = ["build_gnm_encoder", "build_vint_encoder", "build_masked_vint", "build_vit"]
