#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
按照轨迹点数量重新切分 data_splits：

规则：
1. 满足要求的轨迹 -> train/traj_names.txt
2. 不满足要求的轨迹 -> test/traj_names.txt
3. 读取失败的轨迹默认也放入 test，避免进入训练集

判断标准：
- 读取每个轨迹目录下的 traj_data.pkl
- 使用 data["position"] 的长度作为轨迹点数量
- num_points >= threshold 视为满足要求
"""

import os
import csv
import pickle
import shutil
import argparse
from collections import defaultdict


def remove_files_in_dir(dir_path: str):
    """
    清空目录内容，但保留目录本身。
    """
    if not os.path.exists(dir_path):
        return

    for f in os.listdir(dir_path):
        file_path = os.path.join(dir_path, f)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"[WARN] 删除失败: {file_path}, 原因: {e}")


def get_num_points_from_pkl(pkl_path: str):
    """
    从 traj_data.pkl 中读取轨迹点数量。
    默认使用 data["position"] 的第一维长度。
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if "position" not in data:
        raise KeyError("traj_data.pkl 中不存在 key: position")

    position = data["position"]

    if hasattr(position, "shape"):
        return int(position.shape[0])

    return len(position)


def find_valid_trajectory_dirs(data_dir: str):
    """
    查找一个数据集目录下所有包含 traj_data.pkl 的轨迹目录。
    """
    traj_dirs = []

    for name in sorted(os.listdir(data_dir)):
        traj_dir = os.path.join(data_dir, name)

        if not os.path.isdir(traj_dir):
            continue

        pkl_path = os.path.join(traj_dir, "traj_data.pkl")

        if os.path.isfile(pkl_path):
            traj_dirs.append((name, traj_dir, pkl_path))

    return traj_dirs


def write_traj_names(output_dir: str, names):
    """
    写入 traj_names.txt。
    """
    os.makedirs(output_dir, exist_ok=True)

    txt_path = os.path.join(output_dir, "traj_names.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for name in names:
            f.write(name + "\n")

    return txt_path


def split_single_dataset_by_requirement(
    data_dir: str,
    dataset_name: str,
    threshold: int,
    data_splits_dir: str,
    clean_old: bool = True,
):
    """
    单个数据集切分：
    - 满足要求 -> train
    - 不满足要求 -> test
    """
    print(f"\n--- 处理数据集: {dataset_name} ---")
    print(f"数据目录: {data_dir}")
    print(f"判断阈值: 轨迹点数 >= {threshold} 进入 train，否则进入 test")

    traj_items = find_valid_trajectory_dirs(data_dir)

    if len(traj_items) == 0:
        print(f"[跳过] {dataset_name}: 未找到包含 traj_data.pkl 的轨迹目录")
        return None

    train_names = []
    test_names = []
    records = []

    for traj_name, traj_dir, pkl_path in traj_items:
        try:
            num_points = get_num_points_from_pkl(pkl_path)

            if num_points >= threshold:
                split = "train"
                train_names.append(traj_name)
            else:
                split = "test"
                test_names.append(traj_name)

            records.append({
                "dataset": dataset_name,
                "trajectory": traj_name,
                "num_points": num_points,
                "threshold": threshold,
                "split": split,
                "status": "ok",
                "error": "",
                "pkl_path": pkl_path,
            })

        except Exception as e:
            # 读取失败的轨迹默认放入 test，避免进入训练集
            split = "test"
            test_names.append(traj_name)

            records.append({
                "dataset": dataset_name,
                "trajectory": traj_name,
                "num_points": -1,
                "threshold": threshold,
                "split": split,
                "status": "error",
                "error": str(e),
                "pkl_path": pkl_path,
            })

            print(f"[WARN] 读取失败，已放入 test: {traj_name}, 原因: {e}")

    train_dir = os.path.join(data_splits_dir, dataset_name, "train")
    test_dir = os.path.join(data_splits_dir, dataset_name, "test")

    if clean_old:
        for d in [train_dir, test_dir]:
            if os.path.exists(d):
                print(f"清空旧目录: {d}")
                remove_files_in_dir(d)

    train_txt = write_traj_names(train_dir, train_names)
    test_txt = write_traj_names(test_dir, test_names)

    total = len(traj_items)
    train_count = len(train_names)
    test_count = len(test_names)

    print(f"[完成] {dataset_name}")
    print(f"  总轨迹数: {total}")
    print(f"  train 满足要求数量: {train_count}")
    print(f"  test 不满足要求/读取失败数量: {test_count}")
    print(f"  train 文件: {train_txt}")
    print(f"  test 文件: {test_txt}")

    return {
        "dataset": dataset_name,
        "total": total,
        "train": train_count,
        "test": test_count,
        "records": records,
    }


def write_detail_csv(csv_path: str, all_records):
    """
    写入详细切分记录。
    """
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)

    fieldnames = [
        "dataset",
        "trajectory",
        "num_points",
        "threshold",
        "split",
        "status",
        "error",
        "pkl_path",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in all_records:
            writer.writerow(row)


def write_summary_csv(csv_path: str, summaries, threshold: int):
    """
    写入每个数据集的汇总。
    """
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)

    fieldnames = [
        "dataset",
        "total_trajectories",
        "train_trajectories",
        "test_trajectories",
        "threshold",
        "train_ratio_percent",
        "test_ratio_percent",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for item in summaries:
            total = item["total"]
            train_count = item["train"]
            test_count = item["test"]

            if total > 0:
                train_ratio = train_count / total * 100
                test_ratio = test_count / total * 100
            else:
                train_ratio = 0.0
                test_ratio = 0.0

            writer.writerow({
                "dataset": item["dataset"],
                "total_trajectories": total,
                "train_trajectories": train_count,
                "test_trajectories": test_count,
                "threshold": threshold,
                "train_ratio_percent": f"{train_ratio:.2f}",
                "test_ratio_percent": f"{test_ratio:.2f}",
            })


def main():
    parser = argparse.ArgumentParser(
        description="按照轨迹点数量重新切分 data_splits：满足要求进入 train，不满足要求进入 test。"
    )

    parser.add_argument(
        "--data-dir",
        "-i",
        required=True,
        help="数据目录。单数据集模式下指向具体数据集；批量模式下指向 datasets 根目录。"
    )

    parser.add_argument(
        "--dataset-name",
        "-d",
        default=None,
        help="单数据集模式下的数据集名称。批量模式不需要。"
    )

    parser.add_argument(
        "--batch",
        "-b",
        action="store_true",
        help="批量模式：自动扫描 data-dir 下的所有子目录，每个子目录视为一个数据集。"
    )

    parser.add_argument(
        "--threshold",
        "-t",
        type=int,
        default=20,
        help="轨迹点数量阈值。num_points >= threshold 进入 train，否则进入 test。默认 20。"
    )

    parser.add_argument(
        "--data-splits-dir",
        "-o",
        default="/root/data1/visualnav-transformer_4_60/data_splits",
        help="data_splits 输出目录。"
    )

    parser.add_argument(
        "--detail-csv",
        default="split_by_requirement_detail.csv",
        help="详细切分记录 CSV 输出路径。"
    )

    parser.add_argument(
        "--summary-csv",
        default="split_by_requirement_summary.csv",
        help="数据集切分汇总 CSV 输出路径。"
    )

    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="不清空旧的 train/test 目录。默认会清空旧文件。"
    )

    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    data_splits_dir = os.path.abspath(args.data_splits_dir)
    clean_old = not args.no_clean

    print("=" * 80)
    print("按轨迹点数量重新切分 data_splits")
    print("=" * 80)
    print(f"data_dir: {data_dir}")
    print(f"data_splits_dir: {data_splits_dir}")
    print(f"threshold: {args.threshold}")
    print(f"规则: num_points >= {args.threshold} -> train, num_points < {args.threshold} -> test")
    print("=" * 80)

    summaries = []
    all_records = []

    if args.batch:
        sub_dirs = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ])

        if len(sub_dirs) == 0:
            print(f"[ERROR] 批量模式下没有找到任何子数据集目录: {data_dir}")
            return

        print(f"批量模式：发现 {len(sub_dirs)} 个子目录")
        print(sub_dirs)

        for dataset_name in sub_dirs:
            dataset_path = os.path.join(data_dir, dataset_name)

            result = split_single_dataset_by_requirement(
                data_dir=dataset_path,
                dataset_name=dataset_name,
                threshold=args.threshold,
                data_splits_dir=data_splits_dir,
                clean_old=clean_old,
            )

            if result is not None:
                summaries.append(result)
                all_records.extend(result["records"])

    else:
        if not args.dataset_name:
            print("[ERROR] 单数据集模式必须指定 --dataset-name")
            print("例如：")
            print("python data_split_by_requirement.py --data-dir /path/to/recon --dataset-name recon")
            return

        result = split_single_dataset_by_requirement(
            data_dir=data_dir,
            dataset_name=args.dataset_name,
            threshold=args.threshold,
            data_splits_dir=data_splits_dir,
            clean_old=clean_old,
        )

        if result is not None:
            summaries.append(result)
            all_records.extend(result["records"])

    write_detail_csv(args.detail_csv, all_records)
    write_summary_csv(args.summary_csv, summaries, args.threshold)

    print("\n" + "=" * 80)
    print("全部切分完成")
    print("=" * 80)

    total_dataset = len(summaries)
    total_traj = sum(x["total"] for x in summaries)
    total_train = sum(x["train"] for x in summaries)
    total_test = sum(x["test"] for x in summaries)

    print(f"成功处理数据集数量: {total_dataset}")
    print(f"总轨迹数量: {total_traj}")
    print(f"train 满足要求数量: {total_train}")
    print(f"test 不满足要求/读取失败数量: {total_test}")
    print(f"详细记录 CSV: {args.detail_csv}")
    print(f"汇总记录 CSV: {args.summary_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()