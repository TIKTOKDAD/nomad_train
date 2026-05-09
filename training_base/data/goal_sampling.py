from dataclasses import dataclass
from typing import Any, Dict

from training_base.data.labeling import sample_goal


@dataclass(frozen=True)
class NegativeGoalSamplingConfig:
    enabled: bool = True
    policy: str = "offset_zero"
    distance_label: str = "max_dist_cat"


def normalize_goal_sampling_config(config: Dict[str, Any] | None) -> NegativeGoalSamplingConfig:
    config = config or {}
    negative = config.get("negative", config)
    if not isinstance(negative, dict):
        negative = {}
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


def distance_label_for_goal(goal_is_negative: bool, *, distance: int, max_dist_cat: int, config: NegativeGoalSamplingConfig) -> int:
    if not goal_is_negative:
        return int(distance)
    if config.distance_label == "minus_one":
        return -1
    return int(max_dist_cat)
