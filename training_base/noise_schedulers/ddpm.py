# ============================================================
# DDPM scheduler builder - diffusion timestep configuration
# ============================================================
# 本文件负责构建 diffusers 的 DDPMScheduler：
# NoMaD objective 用它加噪，NoMaD 行为指标和可视化用它反向采样。

# 构建 DDPM 噪声调度器
def build_ddpm_scheduler(config):
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    # num_train_timesteps 越大，扩散过程越细；这里从 YAML 明确读取
    return DDPMScheduler(
        num_train_timesteps=int(config["num_train_timesteps"]),
        beta_schedule=config.get("beta_schedule", "squaredcos_cap_v2"),
        clip_sample=bool(config.get("clip_sample", True)),
        prediction_type=config.get("prediction_type", "epsilon"),
    )
