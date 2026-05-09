# ============================================================
# Navigation indexing - split files to sample/goal indices
# ============================================================
# 本文件只负责索引文件命名、加载和构建，不读取图像、不计算标签。

import os
import pickle
from typing import Callable, List, Tuple

import tqdm


def resolved_distance_bounds(min_dist_cat: int, max_dist_cat: int, waypoint_spacing: int) -> Tuple[int, int]:
    # 距离类别按 waypoint_spacing 离散化，索引文件名使用实际可达的首尾桶
    distance_categories = list(range(min_dist_cat, max_dist_cat + 1, waypoint_spacing))
    if len(distance_categories) == 0:
        raise ValueError(
            f"无效的距离范围: min={min_dist_cat}, max={max_dist_cat}, spacing={waypoint_spacing}"
        )
    return distance_categories[0], distance_categories[-1]


def get_dataset_index_path(
    data_split_folder: str,
    min_dist_cat: int,
    max_dist_cat: int,
    waypoint_spacing: int,
    context_type: str,
    context_size: int,
    end_slack: int,
) -> str:
    # 参数变化会改变样本集合，因此全部写入文件名，避免误复用旧索引
    min_dist_cat, max_dist_cat = resolved_distance_bounds(
        min_dist_cat,
        max_dist_cat,
        waypoint_spacing,
    )
    return os.path.join(
        data_split_folder,
        f"dataset_dist_{min_dist_cat}_to_{max_dist_cat}_context_{context_type}_n{context_size}_slack_{end_slack}.pkl",
    )


def load_expected_lmdb_count(index_path: str) -> int:
    # LMDB 缓存按 goals_index 存每个可作为目标的图像，因此期望数量等于 goals_index 长度
    with open(index_path, "rb") as f:
        _, goals_index = pickle.load(f)
    return len(goals_index)


def build_navigation_index(
    *,
    traj_names: List[str],
    get_trajectory: Callable[[str], dict],
    context_size: int,
    waypoint_spacing: int,
    len_traj_pred: int,
    end_slack: int,
    max_dist_cat: int,
    use_tqdm: bool = False,
):
    samples_index = []
    goals_index = []

    for traj_name in tqdm.tqdm(traj_names, disable=not use_tqdm, dynamic_ncols=True):
        traj_data = get_trajectory(traj_name)
        traj_len = len(traj_data["position"])

        # goals_index 覆盖所有时间步，用于正样本目标和跨轨迹负样本目标
        for goal_time in range(0, traj_len):
            goals_index.append((traj_name, goal_time))

        # begin_time 保证历史上下文帧完整；end_time 保证未来动作标签长度完整
        begin_time = context_size * waypoint_spacing
        end_time = traj_len - end_slack - len_traj_pred * waypoint_spacing

        for curr_time in range(begin_time, end_time):
            # 当前观测能采到的最远目标受配置上限和轨迹剩余长度共同约束
            max_goal_distance = min(max_dist_cat * waypoint_spacing, traj_len - curr_time - 1)
            samples_index.append((traj_name, curr_time, max_goal_distance))

    return samples_index, goals_index


def load_or_build_navigation_index(
    *,
    index_path: str,
    build_fn: Callable[[], Tuple[list, list]],
) -> Tuple[list, list]:
    if os.path.exists(index_path):
        try:
            # 索引文件是纯采样元数据，直接 pickle 读取即可
            with open(index_path, "rb") as f:
                return pickle.load(f)
        except (OSError, EOFError, pickle.PickleError, ValueError, AttributeError) as exc:
            raise RuntimeError(
                "加载数据集索引失败。请检查数据划分和参数后，删除或重建索引文件: "
                f"{index_path}"
            ) from exc

    # 不存在时由调用方提供的 build_fn 构建，并写回磁盘供后续复用
    index_to_data, goals_index = build_fn()
    with open(index_path, "wb") as f:
        pickle.dump((index_to_data, goals_index), f)
    return index_to_data, goals_index
