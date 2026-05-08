# ============================================================
# Log key formatter - W&B dashboard namespace layout
# ============================================================
# 本文件只负责“日志名字怎么显示”，不负责计算指标、不负责上传 W&B：
# 1. 把算法返回的原始 key 映射到 train/eval/media/system 的分区
# 2. 让 Trainer、Visualizer 共享同一套命名规则，避免各处手写字符串
# 3. 保持算法层只关心指标语义，不关心 W&B 面板排版


# loss 类指标统一进入 */loss/* 面板，便于训练和评估对齐查看
LOSS_KEYS = {
    "total_loss": "total",
    "dist_loss": "distance",
    "action_loss": "action",
    "diffusion_loss": "diffusion",
    "diffusion_eval_loss_random_masking": "diffusion_random_mask",
    "diffusion_eval_loss_no_masking": "diffusion_no_mask",
    "diffusion_eval_loss_goal_masking": "diffusion_goal_mask",
}

# GNM/ViNT 的直接动作回归指标进入 */action/* 面板
ACTION_KEYS = {
    "action_waypts_cos_sim": "waypts_cos_sim",
    "multi_action_waypts_cos_sim": "multi_waypts_cos_sim",
    "action_orien_cos_sim": "orien_cos_sim",
    "multi_action_orien_cos_sim": "multi_orien_cos_sim",
}

# NoMaD 的采样行为指标进入 */behavior/{uc,gc}/* 面板
# uc 表示 unconditioned，gc 表示 goal-conditioned
BEHAVIOR_KEYS = {
    "uc_action_loss": ("uc", "action_loss"),
    "uc_action_waypts_cos_sim": ("uc", "waypts_cos_sim"),
    "uc_multi_action_waypts_cos_sim": ("uc", "multi_waypts_cos_sim"),
    "gc_dist_loss": ("gc", "distance_loss"),
    "gc_action_loss": ("gc", "action_loss"),
    "gc_action_waypts_cos_sim": ("gc", "waypts_cos_sim"),
    "gc_multi_action_waypts_cos_sim": ("gc", "multi_waypts_cos_sim"),
}


def metric_prefix(mode: str) -> str:
    # mode=train 时写入训练面板；其他 mode 是数据集名，如 huron_test
    return "train" if mode == "train" else f"eval/{mode}"


def format_metric_key(key: str, mode: str, *, kind: str = "metric") -> str:
    # 将算法原始 key 转成最终 W&B chart key
    prefix = metric_prefix(mode)
    if kind == "behavior" or key in BEHAVIOR_KEYS:
        # behavior 指标必须按 uc/gc 拆开，避免和普通 action 指标混在一起
        branch, name = BEHAVIOR_KEYS.get(key, ("misc", key))
        return f"{prefix}/behavior/{branch}/{name}"
    if key in LOSS_KEYS:
        return f"{prefix}/loss/{LOSS_KEYS[key]}"
    if key in ACTION_KEYS:
        return f"{prefix}/action/{ACTION_KEYS[key]}"
    return f"{prefix}/metrics/{key}"


def format_metric_logs(logs: dict, mode: str, *, kind: str = "metric") -> dict:
    # 批量格式化一个日志字典，Trainer 只调用这里，不在热路径散落命名规则
    return {format_metric_key(key, mode, kind=kind): value for key, value in logs.items()}


def media_key(mode: str, name: str) -> str:
    # 图片日志单独进入 media 分区；eval 图片必须带数据集名
    if mode == "train":
        return f"media/train/{name}"
    return f"media/eval/{mode}/{name}"
