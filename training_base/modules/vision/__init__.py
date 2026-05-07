from training_base.registry import module_registry


def build_gnm_encoder(config, data_config):
    from training_base.modules.vision.gnm_encoder import GNMEncoder

    return GNMEncoder(
        context_size=data_config["context_size"],
        obs_encoding_size=config["obs_encoding_size"],
        goal_encoding_size=config["goal_encoding_size"],
    )


def build_vint_encoder(config, data_config):
    from training_base.modules.vision.vint_encoder import ViNTEncoder

    transformer = config["transformer"]
    return ViNTEncoder(
        context_size=data_config["context_size"],
        obs_encoder=config.get("obs_encoder", "efficientnet-b0"),
        obs_encoding_size=config["obs_encoding_size"],
        late_fusion=bool(config.get("late_fusion", False)),
        mha_num_attention_heads=transformer["num_heads"],
        mha_num_attention_layers=transformer["num_layers"],
        mha_ff_dim_factor=transformer["ff_dim_factor"],
    )


def build_masked_vint(config, data_config):
    from training_base.modules.vision.masked_vint import NoMaD_ViNT, replace_bn_with_gn

    transformer = config["transformer"]
    encoder = NoMaD_ViNT(
        obs_encoder=config.get("obs_encoder", "efficientnet-b0"),
        obs_encoding_size=config["encoding_size"],
        context_size=data_config["context_size"],
        mha_num_attention_heads=transformer["num_heads"],
        mha_num_attention_layers=transformer["num_layers"],
        mha_ff_dim_factor=transformer["ff_dim_factor"],
    )
    return replace_bn_with_gn(encoder)


def build_vit(config, data_config):
    from training_base.modules.vision.masked_vint import replace_bn_with_gn
    from training_base.modules.vision.vit import ViT

    transformer = config["transformer"]
    encoder = ViT(
        obs_encoding_size=config["encoding_size"],
        context_size=data_config["context_size"],
        image_size=data_config["image_size"],
        patch_size=config["patch_size"],
        mha_num_attention_heads=transformer["num_heads"],
        mha_num_attention_layers=transformer["num_layers"],
    )
    return replace_bn_with_gn(encoder)


module_registry.register("gnm_encoder")(build_gnm_encoder)
module_registry.register("vint_encoder")(build_vint_encoder)
module_registry.register("masked_vint")(build_masked_vint)
module_registry.register("vit")(build_vit)

__all__ = ["build_gnm_encoder", "build_vint_encoder", "build_masked_vint", "build_vit"]
