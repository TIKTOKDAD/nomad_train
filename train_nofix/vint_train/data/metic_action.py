# 这个文件主要是用来分析每个数据集并且得到 data_config.yaml里面的参数是多少



import os
import pickle
import numpy as np


def yaw_rotmat(yaw: float) -> np.ndarray:
    return np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw),  np.cos(yaw)],
    ], dtype=np.float32)


def to_local_coords(positions: np.ndarray, curr_pos: np.ndarray, curr_yaw: float) -> np.ndarray:
    rot = yaw_rotmat(curr_yaw)
    return (positions - curr_pos) @ rot


def collect_one_dataset_stats(dataset_root: str, len_traj_pred: int = 8, stride: int = 1):
    all_spacings = []
    all_deltas = []

    traj_dirs = sorted([
        d for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d))
    ])

    for traj_name in traj_dirs:
        pkl_path = os.path.join(dataset_root, traj_name, "traj_data.pkl")
        if not os.path.exists(pkl_path):
            continue

        with open(pkl_path, "rb") as f:
            traj = pickle.load(f)

        pos = np.asarray(traj["position"], dtype=np.float32)
        yaw = np.asarray(traj["yaw"], dtype=np.float32)

        if len(pos) < 2:
            continue

        spacings = np.linalg.norm(pos[1:] - pos[:-1], axis=1)
        spacings = spacings[np.isfinite(spacings)]
        spacings = spacings[spacings > 1e-6]
        if len(spacings) > 0:
            all_spacings.append(spacings)

        max_start = len(pos) - (len_traj_pred * stride) - 1
        if max_start <= 0:
            continue

        for t in range(max_start):
            idx = [t + k * stride for k in range(len_traj_pred + 1)]
            future_pos = pos[idx]
            local_future = to_local_coords(future_pos, future_pos[0], yaw[t])
            deltas = local_future[1:] - local_future[:-1]
            all_deltas.append(deltas)

    if not all_spacings:
        return None

    metric_waypoint_spacing = float(np.mean(np.concatenate(all_spacings, axis=0)))
    deltas = np.concatenate(all_deltas, axis=0) if all_deltas else None

    return {
        "metric_waypoint_spacing": metric_waypoint_spacing,
        "deltas": deltas,
    }


def collect_multi_dataset_stats(all_datasets_root: str, len_traj_pred: int = 8, stride: int = 1):
    dataset_names = sorted([
        d for d in os.listdir(all_datasets_root)
        if os.path.isdir(os.path.join(all_datasets_root, d))
    ])

    per_dataset = {}
    global_deltas = []

    for name in dataset_names:
        ds_root = os.path.join(all_datasets_root, name)
        stats = collect_one_dataset_stats(ds_root, len_traj_pred=len_traj_pred, stride=stride)
        if stats is None:
            print(f"[SKIP] {name}: 没找到有效 traj_data.pkl")
            continue

        per_dataset[name] = {
            "metric_waypoint_spacing": stats["metric_waypoint_spacing"]
        }

        if stats["deltas"] is not None:
            deltas_norm = stats["deltas"] / stats["metric_waypoint_spacing"]
            global_deltas.append(deltas_norm)

        print(f"[OK] {name}: spacing = {stats['metric_waypoint_spacing']:.6f}")

    if not global_deltas:
        raise RuntimeError("没有统计到任何有效动作增量。")

    global_deltas = np.concatenate(global_deltas, axis=0)
    action_min = np.percentile(global_deltas, 1, axis=0).tolist()
    action_max = np.percentile(global_deltas, 99, axis=0).tolist()

    return {
        "action_stats": {
            "min": [float(action_min[0]), float(action_min[1])],
            "max": [float(action_max[0]), float(action_max[1])],
        },
        "datasets": per_dataset,
    }


if __name__ == "__main__":
    all_root = "/root/data1/visualnav-transformer_4_60/datasets/"
    stats = collect_multi_dataset_stats(all_root, len_traj_pred=8, stride=1)

    print("\n===== data_config.yaml 建议内容 =====")
    print("action_stats:")
    print(f"  min: {stats['action_stats']['min']}")
    print(f"  max: {stats['action_stats']['max']}")
    print()

    for name, v in stats["datasets"].items():
        print(f"{name}:")
        print(f"  metric_waypoint_spacing: {v['metric_waypoint_spacing']}")
        print()