# ============================================================
# Diffusion module builders - conditional action denoiser
# ============================================================
# 本文件构建 NoMaD 使用的 1D 条件 U-Net：
# 输入是 noisy action 序列，global_cond 是视觉编码器输出的条件向量。

# 构建 1D 条件 U-Net，用于动作扩散建模
def build_conditional_unet1d(config):
    from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

    # down_dims 控制 U-Net 每个尺度的通道数，global_cond_dim 必须和视觉编码器输出一致
    return ConditionalUnet1D(
        input_dim=int(config.get("input_dim", 2)),
        global_cond_dim=int(config["global_cond_dim"]),
        down_dims=config["down_dims"],
        cond_predict_scale=bool(config.get("cond_predict_scale", False)),
    )
