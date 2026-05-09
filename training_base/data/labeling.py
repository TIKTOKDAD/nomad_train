# ============================================================
# Navigation labeling - deterministic sampling and action targets
# ============================================================
# 本文件承接 NavigationDataset 中最核心的“标签语义”：
# 1. 用 sample_context 让目标采样在多 worker 下仍可复现
# 2. 根据当前观测采样正目标或跨轨迹负目标
# 3. 将全局轨迹裁剪成机器人局部坐标系下的动作标签

from contextlib import contextmanager
import contextvars
from dataclasses import dataclass
import hashlib

import numpy as np

from training_base.data.data_utils import to_local_coords


# 单个样本的随机上下文：seed + epoch + index 唯一决定采样随机性
@dataclass(frozen=True)
class SampleContext:
    seed: int
    epoch: int
    index: int


# ContextVar 保证同一进程内并发/嵌套调用时上下文不会互相污染
_SAMPLE_CONTEXT = contextvars.ContextVar("training_base_sample_context", default=None)


# 临时写入样本上下文，供 sample_goal/sample_negative 获取确定性随机数
@contextmanager
def sample_context(seed: int, epoch: int, index: int):
    token = _SAMPLE_CONTEXT.set(SampleContext(int(seed), int(epoch), int(index)))
    try:
        yield
    finally:
        _SAMPLE_CONTEXT.reset(token)


# 根据当前样本上下文构造随机数生成器
def _context_rng():
    context = _SAMPLE_CONTEXT.get()
    if context is None:
        # 兼容没有 EpochAwareDataset 包装的旧调用路径
        return np.random
    # 使用 blake2b 混合 seed/epoch/index，避免简单相加导致不同组合碰撞
    payload = f"{context.seed}:{context.epoch}:{context.index}".encode("utf-8")
    seed = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
    return np.random.default_rng(seed)


# 兼容 numpy.RandomState 与 numpy.Generator 的 randint 接口差异
def _randint(rng, high: int) -> int:
    if hasattr(rng, "integers"):
        return int(rng.integers(0, high))
    return int(rng.randint(0, high))


# 从全局 goals_index 中采样一个跨轨迹目标，作为不可达负样本
def sample_negative(goals_index, rng=None):
    rng = rng or _context_rng()
    return goals_index[_randint(rng, len(goals_index))]


# 采样一个目标图像位置，并返回它是否为负样本
def sample_goal(
    trajectory_name,
    curr_time,
    max_goal_dist,
    waypoint_spacing,
    goals_index,
    rng=None,
    *,
    negative_enabled: bool = True,
    negative_policy: str = "offset_zero",
):
    rng = rng or _context_rng()
    if str(negative_policy).lower() != "offset_zero":
        raise ValueError(f"negative_policy 当前只支持 offset_zero，实际为 {negative_policy!r}")
    # offset 单位是 waypoint_spacing；0 被保留给负样本采样策略
    goal_offset = _randint(rng, max_goal_dist + 1)
    if goal_offset == 0 and negative_enabled:
        neg_trajectory_name, goal_time = sample_negative(goals_index, rng=rng)
        return neg_trajectory_name, goal_time, True
    if goal_offset == 0:
        # 禁用负样本时，把 0 偏移推进到最近正目标，避免目标等于当前观测
        goal_offset = 1 if max_goal_dist > 0 else 0
    goal_time = curr_time + int(goal_offset * waypoint_spacing)
    return trajectory_name, goal_time, False


# 构造 temporal 上下文帧列表：从最早历史帧到当前帧
def temporal_context(trajectory_name, curr_time, context_size, waypoint_spacing):
    context_times = list(
        range(
            curr_time + -context_size * waypoint_spacing,
            curr_time + 1,
            waypoint_spacing,
        )
    )
    return [(trajectory_name, time) for time in context_times]


# 根据配置选择上下文策略；当前只实现 temporal
def context_entries(context_type, trajectory_name, curr_time, context_size, waypoint_spacing):
    if context_type != "temporal":
        raise ValueError(f"data.context_type 当前只支持 temporal，实际为 {context_type!r}")
    return temporal_context(trajectory_name, curr_time, context_size, waypoint_spacing)


# 统一的形状校验，错误信息带上数据集/轨迹/时间，方便定位坏数据
def _require_shape(value, expected_shape, *, field_name: str, dataset_name: str, trajectory_name: str, curr_time: int):
    if value.shape != expected_shape:
        raise ValueError(
            f"{dataset_name}:{trajectory_name}@{curr_time} 的 {field_name} 形状错误: "
            f"{value.shape} != {expected_shape}"
        )


# 计算单个样本的动作标签和目标位置
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
    # 多取一个点：第 0 个点是当前观测，其余 len_traj_pred 个点是未来动作标签
    start_index = curr_time
    end_index = curr_time + len_traj_pred * waypoint_spacing + 1

    # 按 waypoint_spacing 从轨迹中抽取位置和朝向
    yaw = traj_data["yaw"][start_index:end_index:waypoint_spacing]
    positions = traj_data["position"][start_index:end_index:waypoint_spacing]
    # 目标位置可能来自负样本或远目标，越界时夹到轨迹最后一帧
    goal_pos = traj_data["position"][min(goal_time, len(traj_data["position"]) - 1)]

    if len(yaw.shape) == 2:
        # 某些数据集把 yaw 存成 [T,1]，这里统一成 [T]
        yaw = yaw.squeeze(1)

    if yaw.shape[0] == 0 or positions.shape[0] == 0:
        raise ValueError(f"{dataset_name}:{trajectory_name}@{curr_time} 的轨迹片段为空")

    if yaw.shape != (len_traj_pred + 1,):
        # 轨迹末端偶尔不够长时，用最后一个状态补齐，保持 batch 张量形状固定
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

    # 以当前机器人位姿为原点，把未来路径和目标位置转到局部坐标系
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
        # 角度标签使用相对当前朝向的 yaw，后续 Dataset 会转成 cos/sin 表示
        yaw = yaw[1:] - yaw[0]
        actions = np.concatenate([waypoints[1:], yaw[:, None]], axis=-1)
    else:
        # 不学习角度时只保留未来航点的局部 x/y
        actions = waypoints[1:]

    if normalize:
        # 用米制 waypoint 间距归一化，使不同数据集/采样间隔落到相近尺度
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
