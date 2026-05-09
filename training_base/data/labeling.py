# ============================================================
# Navigation labeling - deterministic sampling and action targets
# ============================================================

from contextlib import contextmanager
import contextvars
from dataclasses import dataclass
import hashlib

import numpy as np

from training_base.data.data_utils import to_local_coords


@dataclass(frozen=True)
class SampleContext:
    seed: int
    epoch: int
    index: int


_SAMPLE_CONTEXT = contextvars.ContextVar("training_base_sample_context", default=None)


@contextmanager
def sample_context(seed: int, epoch: int, index: int):
    token = _SAMPLE_CONTEXT.set(SampleContext(int(seed), int(epoch), int(index)))
    try:
        yield
    finally:
        _SAMPLE_CONTEXT.reset(token)


def _context_rng():
    context = _SAMPLE_CONTEXT.get()
    if context is None:
        return np.random
    payload = f"{context.seed}:{context.epoch}:{context.index}".encode("utf-8")
    seed = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
    return np.random.default_rng(seed)


def _randint(rng, high: int) -> int:
    if hasattr(rng, "integers"):
        return int(rng.integers(0, high))
    return int(rng.randint(0, high))


def sample_negative(goals_index, rng=None):
    rng = rng or _context_rng()
    return goals_index[_randint(rng, len(goals_index))]


def sample_goal(trajectory_name, curr_time, max_goal_dist, waypoint_spacing, goals_index, rng=None):
    rng = rng or _context_rng()
    goal_offset = _randint(rng, max_goal_dist + 1)
    if goal_offset == 0:
        neg_trajectory_name, goal_time = sample_negative(goals_index, rng=rng)
        return neg_trajectory_name, goal_time, True
    goal_time = curr_time + int(goal_offset * waypoint_spacing)
    return trajectory_name, goal_time, False


def temporal_context(trajectory_name, curr_time, context_size, waypoint_spacing):
    context_times = list(
        range(
            curr_time + -context_size * waypoint_spacing,
            curr_time + 1,
            waypoint_spacing,
        )
    )
    return [(trajectory_name, time) for time in context_times]


def context_entries(context_type, trajectory_name, curr_time, context_size, waypoint_spacing):
    if context_type != "temporal":
        raise ValueError(f"data.context_type 当前只支持 temporal，实际为 {context_type!r}")
    return temporal_context(trajectory_name, curr_time, context_size, waypoint_spacing)


def _require_shape(value, expected_shape, *, field_name: str, dataset_name: str, trajectory_name: str, curr_time: int):
    if value.shape != expected_shape:
        raise ValueError(
            f"{dataset_name}:{trajectory_name}@{curr_time} 的 {field_name} 形状错误: "
            f"{value.shape} != {expected_shape}"
        )


def compute_navigation_actions(
    *,
    traj_data,
    curr_time: int,
    goal_time: int,
    len_traj_pred: int,
    waypoint_spacing: int,
    learn_angle: bool,
    normalize: bool,
    metric_waypoint_spacing: float,
    num_action_params: int,
    dataset_name: str,
    trajectory_name: str,
):
    start_index = curr_time
    end_index = curr_time + len_traj_pred * waypoint_spacing + 1

    yaw = traj_data["yaw"][start_index:end_index:waypoint_spacing]
    positions = traj_data["position"][start_index:end_index:waypoint_spacing]
    goal_pos = traj_data["position"][min(goal_time, len(traj_data["position"]) - 1)]

    if len(yaw.shape) == 2:
        yaw = yaw.squeeze(1)

    if yaw.shape[0] == 0 or positions.shape[0] == 0:
        raise ValueError(f"{dataset_name}:{trajectory_name}@{curr_time} 的轨迹片段为空")

    if yaw.shape != (len_traj_pred + 1,):
        const_len = len_traj_pred + 1 - yaw.shape[0]
        yaw = np.concatenate([yaw, np.repeat(yaw[-1], const_len)])
        positions = np.concatenate([positions, np.repeat(positions[-1][None], const_len, axis=0)], axis=0)

    _require_shape(
        yaw,
        (len_traj_pred + 1,),
        field_name="yaw",
        dataset_name=dataset_name,
        trajectory_name=trajectory_name,
        curr_time=curr_time,
    )
    _require_shape(
        positions,
        (len_traj_pred + 1, 2),
        field_name="positions",
        dataset_name=dataset_name,
        trajectory_name=trajectory_name,
        curr_time=curr_time,
    )

    waypoints = to_local_coords(positions, positions[0], yaw[0])
    goal_pos = to_local_coords(goal_pos, positions[0], yaw[0])
    _require_shape(
        waypoints,
        (len_traj_pred + 1, 2),
        field_name="waypoints",
        dataset_name=dataset_name,
        trajectory_name=trajectory_name,
        curr_time=curr_time,
    )

    if learn_angle:
        yaw = yaw[1:] - yaw[0]
        actions = np.concatenate([waypoints[1:], yaw[:, None]], axis=-1)
    else:
        actions = waypoints[1:]

    if normalize:
        scale = metric_waypoint_spacing * waypoint_spacing
        actions[:, :2] /= scale
        goal_pos /= scale

    _require_shape(
        actions,
        (len_traj_pred, num_action_params),
        field_name="actions",
        dataset_name=dataset_name,
        trajectory_name=trajectory_name,
        curr_time=curr_time,
    )

    return actions, goal_pos
