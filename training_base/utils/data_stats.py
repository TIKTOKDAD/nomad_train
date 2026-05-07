"""Collect navigation dataset statistics for data_config.yaml.

This is an offline maintenance utility, not part of the training data path.
It estimates per-dataset waypoint spacing and global action delta bounds.
"""

import argparse
import os
import pickle
from typing import Dict, Optional

import numpy as np


def yaw_rotmat(yaw: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)],
        ],
        dtype=np.float32,
    )


def to_local_coords(positions: np.ndarray, curr_pos: np.ndarray, curr_yaw: float) -> np.ndarray:
    return (positions - curr_pos) @ yaw_rotmat(curr_yaw)


def collect_one_dataset_stats(dataset_root: str, len_traj_pred: int = 8, stride: int = 1) -> Optional[Dict[str, np.ndarray]]:
    all_spacings = []
    all_deltas = []
    traj_dirs = sorted(
        name
        for name in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, name))
    )

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
            local_future = to_local_coords(future_pos, future_pos[0], float(yaw[t]))
            all_deltas.append(local_future[1:] - local_future[:-1])

    if not all_spacings:
        return None

    deltas = np.concatenate(all_deltas, axis=0) if all_deltas else None
    return {
        "metric_waypoint_spacing": float(np.mean(np.concatenate(all_spacings, axis=0))),
        "deltas": deltas,
    }


def collect_multi_dataset_stats(all_datasets_root: str, len_traj_pred: int = 8, stride: int = 1) -> Dict[str, Dict]:
    dataset_names = sorted(
        name
        for name in os.listdir(all_datasets_root)
        if os.path.isdir(os.path.join(all_datasets_root, name))
    )
    per_dataset = {}
    global_deltas = []

    for name in dataset_names:
        ds_root = os.path.join(all_datasets_root, name)
        stats = collect_one_dataset_stats(ds_root, len_traj_pred=len_traj_pred, stride=stride)
        if stats is None:
            print(f"[SKIP] {name}: no valid traj_data.pkl files")
            continue

        spacing = float(stats["metric_waypoint_spacing"])
        per_dataset[name] = {"metric_waypoint_spacing": spacing}
        if stats["deltas"] is not None:
            global_deltas.append(stats["deltas"] / spacing)
        print(f"[OK] {name}: spacing = {spacing:.6f}")

    if not global_deltas:
        raise RuntimeError("No valid action deltas were collected.")

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets_root", help="Directory containing dataset subdirectories.")
    parser.add_argument("--len-traj-pred", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    stats = collect_multi_dataset_stats(
        args.datasets_root,
        len_traj_pred=args.len_traj_pred,
        stride=args.stride,
    )
    print("\n# Suggested data_config.yaml entries")
    print("action_stats:")
    print(f"  min: {stats['action_stats']['min']}")
    print(f"  max: {stats['action_stats']['max']}")
    print()
    for name, values in stats["datasets"].items():
        print(f"{name}:")
        print(f"  metric_waypoint_spacing: {values['metric_waypoint_spacing']}")
        print()


if __name__ == "__main__":
    main()

