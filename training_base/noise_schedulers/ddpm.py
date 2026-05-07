def build_ddpm_scheduler(config):
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    return DDPMScheduler(
        num_train_timesteps=int(config["num_train_timesteps"]),
        beta_schedule=config.get("beta_schedule", "squaredcos_cap_v2"),
        clip_sample=bool(config.get("clip_sample", True)),
        prediction_type=config.get("prediction_type", "epsilon"),
    )
