# ============================================================
# LMDB cache management - build, validate, and open image cache
# ============================================================
# 本文件只处理图像缓存生命周期，不关心样本采样或训练标签。

import json
import os
import shutil
from typing import List, Tuple

import lmdb
import tqdm

from training_base.data.data_utils import get_data_path
from training_base.data.indexing import get_dataset_index_path, load_expected_lmdb_count


LMDB_CACHE_VERSION = 1


def get_lmdb_cache_paths(data_split_folder: str, dataset_name: str) -> Tuple[str, str]:
    cache_path = os.path.join(data_split_folder, f"dataset_{dataset_name}.lmdb")
    complete_path = f"{cache_path}.complete.json"
    return cache_path, complete_path


def path_exists(path: str) -> bool:
    return os.path.isdir(path) or os.path.isfile(path)


def remove_path(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


def validate_lmdb_cache(
    cache_path: str,
    complete_path: str,
    dataset_name: str,
    expected_num_images: int,
) -> Tuple[bool, List[str]]:
    errors = []
    if not path_exists(cache_path):
        errors.append(f"缺少 LMDB 缓存: {cache_path}")
    if not os.path.exists(complete_path):
        errors.append(f"缺少完成标记: {complete_path}")
        return False, errors

    try:
        with open(complete_path, "r", encoding="utf-8") as f:
            marker = json.load(f)
    except Exception as exc:
        errors.append(f"完成标记无效 {complete_path}: {exc}")
        return False, errors

    if int(marker.get("version", -1)) != LMDB_CACHE_VERSION:
        errors.append(
            f"完成标记版本不匹配 {complete_path}: "
            f"{marker.get('version')} != {LMDB_CACHE_VERSION}"
        )
    if marker.get("dataset_name") != dataset_name:
        errors.append(
            f"完成标记数据集不匹配 {complete_path}: "
            f"{marker.get('dataset_name')} != {dataset_name}"
        )
    if int(marker.get("num_cached_images", -1)) != int(expected_num_images):
        errors.append(
            f"缓存图像数量不匹配 {complete_path}: "
            f"{marker.get('num_cached_images')} != {expected_num_images}"
        )
    return len(errors) == 0, errors


def check_lmdb_cache_ready(
    data_split_folder: str,
    dataset_name: str,
    min_dist_cat: int,
    max_dist_cat: int,
    waypoint_spacing: int,
    context_type: str,
    context_size: int,
    end_slack: int,
) -> Tuple[bool, List[str]]:
    index_path = get_dataset_index_path(
        data_split_folder,
        min_dist_cat,
        max_dist_cat,
        waypoint_spacing,
        context_type,
        context_size,
        end_slack,
    )
    cache_path, complete_path = get_lmdb_cache_paths(data_split_folder, dataset_name)
    if not os.path.exists(index_path):
        return False, [f"缺少数据集索引: {index_path}"]

    try:
        expected_num_images = load_expected_lmdb_count(index_path)
    except Exception as exc:
        return False, [f"加载数据集索引失败 {index_path}: {exc}"]

    return validate_lmdb_cache(cache_path, complete_path, dataset_name, expected_num_images)


def build_or_open_lmdb_cache(
    *,
    data_folder: str,
    data_split_folder: str,
    dataset_name: str,
    goals_index,
    lmdb_lock: bool,
    lmdb_readahead: bool,
    lmdb_meminit: bool,
    lmdb_max_readers: int,
    lmdb_map_size: int,
    lmdb_cache_mode: str,
    rebuild_incomplete_lmdb: bool,
    use_tqdm: bool = True,
) -> lmdb.Environment:
    cache_filename, complete_filename = get_lmdb_cache_paths(
        data_split_folder,
        dataset_name,
    )
    expected_num_images = len(goals_index)
    cache_ready, cache_errors = validate_lmdb_cache(
        cache_filename,
        complete_filename,
        dataset_name,
        expected_num_images,
    )

    should_build = lmdb_cache_mode == "build" or (
        lmdb_cache_mode == "auto" and not cache_ready
    )
    if lmdb_cache_mode == "read" and not cache_ready:
        joined_errors = "\n  - ".join(cache_errors)
        raise RuntimeError(
            "LMDB 缓存缺失或不完整，但 lmdb_cache_mode='read'。\n"
            f"数据集: {dataset_name}\n"
            f"划分目录: {data_split_folder}\n"
            f"问题:\n  - {joined_errors}\n"
            "请运行: python -m training_base.cli -c <config> --build-lmdb-only"
        )

    if should_build and not cache_ready:
        if path_exists(cache_filename) or os.path.exists(complete_filename):
            if not rebuild_incomplete_lmdb:
                joined_errors = "\n  - ".join(cache_errors)
                raise RuntimeError(
                    "发现不完整或未经校验的 LMDB 缓存。\n"
                    f"数据集: {dataset_name}\n"
                    f"划分目录: {data_split_folder}\n"
                    f"问题:\n  - {joined_errors}\n"
                    "如需安全重建，请重新运行时加 --rebuild-incomplete-lmdb，"
                    "或设置 rebuild_incomplete_lmdb: True。"
                )
            remove_path(cache_filename)
            if os.path.exists(complete_filename):
                os.remove(complete_filename)

        tmp_cache_filename = f"{cache_filename}.tmp.{os.getpid()}"
        remove_path(tmp_cache_filename)
        tqdm_iterator = tqdm.tqdm(
            goals_index,
            disable=not use_tqdm,
            dynamic_ncols=True,
            desc=f"正在为 {dataset_name} 构建 LMDB 缓存",
        )
        num_cached_images = 0
        with lmdb.open(tmp_cache_filename, map_size=int(lmdb_map_size)) as image_cache:
            with image_cache.begin(write=True) as txn:
                for traj_name, time in tqdm_iterator:
                    image_path = get_data_path(data_folder, traj_name, time)
                    with open(image_path, "rb") as f:
                        txn.put(image_path.encode(), f.read())
                    num_cached_images += 1
            image_cache.sync()

        if num_cached_images != expected_num_images:
            remove_path(tmp_cache_filename)
            raise RuntimeError(
                f"{dataset_name} 的 LMDB 构建数量不匹配: "
                f"{num_cached_images} != {expected_num_images}"
            )

        os.rename(tmp_cache_filename, cache_filename)
        marker = {
            "version": LMDB_CACHE_VERSION,
            "dataset_name": dataset_name,
            "num_cached_images": num_cached_images,
            "cache_path": cache_filename,
        }
        with open(complete_filename, "w", encoding="utf-8") as f:
            json.dump(marker, f, ensure_ascii=False, indent=2)

    elif should_build and cache_ready:
        print(f"{dataset_name} 的 LMDB 缓存已完整: {cache_filename}")

    return lmdb.open(
        cache_filename,
        readonly=True,
        lock=lmdb_lock,
        readahead=lmdb_readahead,
        meminit=lmdb_meminit,
        max_readers=lmdb_max_readers,
    )
