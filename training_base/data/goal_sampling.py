# ============================================================
# Goal sampling config - negative goal sampling adapter
# ============================================================
# 本文件把新版 goal_sampling YAML 配置转换成采样函数能直接消费的结构：
# 1. negative.enabled 控制是否允许跨轨迹负样本
# 2. negative.policy 当前只支持 offset_zero，即随机偏移为 0 时采负样本
# 3. negative.distance_label 控制负样本距离标签写成 max_dist_cat 还是 -1

from dataclasses import dataclass
from typing import Any, Dict

from training_base.data.labeling import sample_goal


# 负样本采样配置：不可变 dataclass，避免训练中被意外修改
@dataclass(frozen=True)
class NegativeGoalSamplingConfig:
    # enabled=False 时，offset=0 会被改成最近正样本，完全禁用负采样
    enabled: bool = True
    # 采样策略；当前仅支持与旧实现兼容的 offset_zero
    policy: str = "offset_zero"
    # 负样本的距离标签语义：最大距离桶或显式 -1
    distance_label: str = "max_dist_cat"


# 归一化 goal_sampling 配置，并校验当前实现支持的取值
def normalize_goal_sampling_config(config: Dict[str, Any] | None) -> NegativeGoalSamplingConfig:
    # 兼容两种写法：data.goal_sampling.negative 或直接把 negative 字段平铺到 goal_sampling
    config = config or {}
    negative = config.get("negative", config)
    if not isinstance(negative, dict):
        negative = {}
    # 统一小写，避免 YAML 中大小写差异影响判断
    policy = str(negative.get("policy", "offset_zero")).lower()
    distance_label = str(negative.get("distance_label", "max_dist_cat")).lower()
    if policy not in {"offset_zero"}:
        raise ValueError(f"data.goal_sampling.negative.policy 当前只支持 offset_zero，实际为 {policy!r}")
    if distance_label not in {"max_dist_cat", "minus_one"}:
        raise ValueError(
            "data.goal_sampling.negative.distance_label 必须是 max_dist_cat 或 minus_one，"
            f"实际为 {distance_label!r}"
        )
    return NegativeGoalSamplingConfig(
        enabled=bool(negative.get("enabled", True)),
        policy=policy,
        distance_label=distance_label,
    )


# 对外采样入口：把配置对象拆成 sample_goal 所需的参数
def sample_navigation_goal(
    trajectory_name,
    curr_time,
    max_goal_dist,
    waypoint_spacing,
    goals_index,
    *,
    config: NegativeGoalSamplingConfig,
    rng=None,
):
    # sample_goal 位于 labeling.py，负责具体随机偏移和负样本选择
    return sample_goal(
        trajectory_name,
        curr_time,
        max_goal_dist,
        waypoint_spacing,
        goals_index,
        rng=rng,
        negative_enabled=config.enabled,
        negative_policy=config.policy,
    )


# 根据采样结果生成距离标签
def distance_label_for_goal(goal_is_negative: bool, *, distance: int, max_dist_cat: int, config: NegativeGoalSamplingConfig) -> int:
    # 正样本直接使用观测和目标之间的离散步数
    if not goal_is_negative:
        return int(distance)
    # 负样本可以选择 -1 作为不可达类，也可以映射到最大距离桶兼容旧训练脚本
    if config.distance_label == "minus_one":
        return -1
    return int(max_dist_cat)
