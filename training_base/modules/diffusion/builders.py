def build_conditional_unet1d(config):
    from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

    return ConditionalUnet1D(
        input_dim=int(config.get("input_dim", 2)),
        global_cond_dim=int(config["global_cond_dim"]),
        down_dims=config["down_dims"],
        cond_predict_scale=bool(config.get("cond_predict_scale", False)),
    )
