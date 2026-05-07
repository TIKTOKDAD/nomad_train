# ============================================================
# Navigation dataset - visual navigation data loader
# ============================================================
# 本文件实现了用于训练GNM、ViNT、NoMaD模型的数据集类
# 主要功能：
# 1. 加载机器人轨迹数据（图像、位置、航向）
# 2. 采样观测-目标对（包括负样本挖掘）
# 3. 计算局部坐标系下的动作标签
# 4. 使用LMDB缓存加速图像加载

import numpy as np
import os
import yaml
from typing import Any, Dict, List, Optional, Tuple
import tqdm
import io
import json
import lmdb  # Lightning Memory-Mapped Database - 高效的键值存储
import shutil

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

import pickle
import sys
import numpy.core as np_core

sys.modules.setdefault("numpy._core", np_core)
if hasattr(np_core, "multiarray"):
    sys.modules.setdefault("numpy._core.multiarray", np_core.multiarray)

# 导入数据处理工具函数
from training_base.data.data_utils import (
    img_path_to_data,  # 加载并预处理图像
    calculate_sin_cos,  # 将角度转换为sin/cos表示
    get_data_path,  # 获取数据文件路径
    to_local_coords,  # 转换到局部坐标系
)


LMDB_CACHE_VERSION = 1


def _resolved_distance_bounds(min_dist_cat: int, max_dist_cat: int, waypoint_spacing: int) -> Tuple[int, int]:
    distance_categories = list(range(min_dist_cat, max_dist_cat + 1, waypoint_spacing))
    if len(distance_categories) == 0:
        raise ValueError(
            f"Invalid distance range: min={min_dist_cat}, max={max_dist_cat}, spacing={waypoint_spacing}"
        )
    return distance_categories[0], distance_categories[-1]


def get_lmdb_cache_paths(data_split_folder: str, dataset_name: str) -> Tuple[str, str]:
    cache_path = os.path.join(data_split_folder, f"dataset_{dataset_name}.lmdb")
    complete_path = f"{cache_path}.complete.json"
    return cache_path, complete_path


def get_dataset_index_path(
    data_split_folder: str,
    min_dist_cat: int,
    max_dist_cat: int,
    waypoint_spacing: int,
    context_type: str,
    context_size: int,
    end_slack: int,
) -> str:
    min_dist_cat, max_dist_cat = _resolved_distance_bounds(
        min_dist_cat,
        max_dist_cat,
        waypoint_spacing,
    )
    return os.path.join(
        data_split_folder,
        f"dataset_dist_{min_dist_cat}_to_{max_dist_cat}_context_{context_type}_n{context_size}_slack_{end_slack}.pkl",
    )


def _path_exists(path: str) -> bool:
    return os.path.isdir(path) or os.path.isfile(path)


def _remove_path(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


def _load_expected_lmdb_count(index_path: str) -> int:
    with open(index_path, "rb") as f:
        _, goals_index = pickle.load(f)
    return len(goals_index)


def _validate_lmdb_cache(
    cache_path: str,
    complete_path: str,
    dataset_name: str,
    expected_num_images: int,
) -> Tuple[bool, List[str]]:
    errors = []
    if not _path_exists(cache_path):
        errors.append(f"missing LMDB cache: {cache_path}")
    if not os.path.exists(complete_path):
        errors.append(f"missing completion marker: {complete_path}")
        return False, errors

    try:
        with open(complete_path, "r", encoding="utf-8") as f:
            marker = json.load(f)
    except Exception as exc:
        errors.append(f"invalid completion marker {complete_path}: {exc}")
        return False, errors

    if int(marker.get("version", -1)) != LMDB_CACHE_VERSION:
        errors.append(
            f"marker version mismatch for {complete_path}: "
            f"{marker.get('version')} != {LMDB_CACHE_VERSION}"
        )
    if marker.get("dataset_name") != dataset_name:
        errors.append(
            f"marker dataset mismatch for {complete_path}: "
            f"{marker.get('dataset_name')} != {dataset_name}"
        )
    if int(marker.get("num_cached_images", -1)) != int(expected_num_images):
        errors.append(
            f"cached image count mismatch for {complete_path}: "
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
        return False, [f"missing dataset index: {index_path}"]

    try:
        expected_num_images = _load_expected_lmdb_count(index_path)
    except Exception as exc:
        return False, [f"failed to load dataset index {index_path}: {exc}"]

    return _validate_lmdb_cache(cache_path, complete_path, dataset_name, expected_num_images)


class NavigationDataset(Dataset):
    def __init__(
            self,
            data_folder: str,  # 包含所有图像数据的目录
            # 结构: data_folder/trajectory_name/0.jpg, 1.jpg, ..., traj_data.pkl
            data_split_folder: str,  # 包含traj_names.txt的数据划分目录
            dataset_name: str,  # 数据集名称（用于查找data_config.yaml中的配置）
            image_size: Tuple[int, int],  # 图像尺寸 (宽度, 高度)
            waypoint_spacing: int,  # 航点间隔（帧数）
            min_dist_cat: int,  # 最小距离类别（航点步）
            max_dist_cat: int,  # 最大距离类别（航点步）
            min_action_distance: int,  # 动作预测最小距离（航点步）
            max_action_distance: int,  # 动作预测最大距离（航点步）
            negative_mining: bool,  # 是否启用负样本相关逻辑
            len_traj_pred: int,  # 预测轨迹长度（航点数量）
            learn_angle: bool,  # 是否学习航向角
            context_size: int,  # 历史上下文帧数
            context_type: str = "temporal",  # 上下文类型：temporal/randomized/randomized_temporal
            end_slack: int = 0,  # 轨迹末端忽略步数
            goals_per_obs: int = 1,  # 每个观测采样目标数
            normalize: bool = True,  # 是否按米制间隔归一化动作
            obs_type: str = "image",  # 观测数据类型（目前仅支持image）
            goal_type: str = "image",  # 目标数据类型（目前仅支持image）
            lmdb_lock: bool = False,  # 多 worker 只读训练建议关闭锁，减少 LMDB 读锁竞争
            lmdb_readahead: bool = False,  # 随机访问图像时关闭预读，避免无效磁盘/页缓存压力
            lmdb_meminit: bool = False,  # 只读场景无需初始化内存页，可减少 LMDB 打开开销
            lmdb_max_readers: int = 512,  # 允许更多 DataLoader worker 同时读取 LMDB
            lmdb_cache_mode: str = "auto",  # auto=缺失时构建；read=只读已完成缓存；build=只构建/补齐缓存
            rebuild_incomplete_lmdb: bool = False,  # True 时删除未完成/不匹配的 LMDB 后重建
    ):
        """
        Navigation dataset class - used to train visual navigation models

        核心功能：
        1. 从机器人轨迹中采样观测-目标对
        2. 计算局部坐标系下的动作序列
        3. 支持负样本挖掘（提升模型鲁棒性）
        4. 使用LMDB缓存加速图像加载

        参数:
            data_folder (str): 包含所有图像数据的目录
                结构: data_folder/trajectory_name/0.jpg, 1.jpg, ..., traj_data.pkl

            data_split_folder (str): 包含traj_names.txt的目录
                traj_names.txt列出了该数据划分中的所有轨迹名称（每行一个）

            dataset_name (str): 数据集名称
                例如: recon, go_stanford, scand, tartandrive等
                用于从data_config.yaml中查找数据集特定参数

            image_size (Tuple[int, int]): 图像尺寸 (宽度, 高度)
                例如: (85, 64) for ViNT, (96, 96) for NoMaD

            waypoint_spacing (int): 航点间隔（帧数）
                例如: 1表示每帧都是一个航点，2表示每隔一帧
                用于控制轨迹的时间分辨率

            min_dist_cat (int): 最小距离类别（航点步）
                目标必须距离观测至少这么远

            max_dist_cat (int): 最大距离类别（航点步）
                目标最多距离观测这么远

            min_action_distance (int): 动作预测的最小距离（航点步）
                太近的目标不适合预测动作序列

            max_action_distance (int): 动作预测的最大距离（航点步）
                太远的目标动作预测不准确

            negative_mining (bool): 负样本相关开关
                来自ViNG论文 (Shah et al. 2020)
                当前实现中该开关主要影响距离类别集合（是否附加-1标签位）；
                目标采样在goal_offset=0时仍会走跨轨迹采样路径

            len_traj_pred (int): 预测的轨迹长度（航点数量）
                例如: 5 for GNM/ViNT, 8 for NoMaD

            learn_angle (bool): 是否学习机器人的航向角
                True: 预测(x, y, yaw)，输出维度为3或4（sin/cos编码）
                False: 仅预测(x, y)，输出维度为2

            context_size (int): 作为上下文的历史观测数量
                例如: 5表示使用过去5帧作为上下文
                0表示不使用历史上下文（仅当前帧）

            context_type (str): 上下文类型
                "temporal": 时序上下文（使用最近的N帧）
                "randomized": 随机采样上下文
                "randomized_temporal": 随机时序上下文

            end_slack (int): 轨迹末端忽略的时间步数
                因为很多轨迹以碰撞结束，需要裁剪末端

            goals_per_obs (int): 每个观测采样的目标数量
                增加此值可以扩大数据集规模

            normalize (bool): 是否归一化距离或动作
                True: 将动作归一化到[-1, 1]范围

            obs_type (str): 观测数据类型（目前仅支持"image"）

            goal_type (str): 目标数据类型（目前仅支持"image"）
        """
        # ========== 基本参数设置 ==========
        self.data_folder = data_folder
        self.data_split_folder = data_split_folder
        self.dataset_name = dataset_name

        # 读取轨迹名称列表
        traj_names_file = os.path.join(data_split_folder, "traj_names.txt")
        with open(traj_names_file, "r") as f:
            file_lines = f.read()
            self.traj_names = file_lines.split("\n")
        # 移除空字符串
        if "" in self.traj_names:
            self.traj_names.remove("")

        # ========== 距离和航点配置 ==========
        self.image_size = image_size
        self.waypoint_spacing = waypoint_spacing
        # 生成距离类别列表：[min_dist_cat, min_dist_cat+spacing, ..., max_dist_cat]
        self.distance_categories = list(
            range(min_dist_cat, max_dist_cat + 1, self.waypoint_spacing)
        )
        self.min_dist_cat = self.distance_categories[0]
        self.max_dist_cat = self.distance_categories[-1]

        # ========== 负样本挖掘 ==========
        self.negative_mining = negative_mining
        if self.negative_mining:
            # 在距离类别中添加-1作为不可达类别标记（仅用于类别集合定义）
            self.distance_categories.append(-1)

        # ========== 轨迹预测参数 ==========
        self.len_traj_pred = len_traj_pred  # 预测的航点数量
        self.learn_angle = learn_angle  # 是否学习航向角

        # ========== 动作距离范围 ==========
        self.min_action_distance = min_action_distance
        self.max_action_distance = max_action_distance

        # ========== 上下文配置 ==========
        self.context_size = context_size
        assert context_type in {
            "temporal",
            "randomized",
            "randomized_temporal",
        }, "context_type must be one of temporal, randomized, randomized_temporal"
        self.context_type = context_type

        # ========== 其他配置 ==========
        self.end_slack = end_slack  # 轨迹末端裁剪
        self.goals_per_obs = goals_per_obs  # 每个观测的目标数
        self.normalize = normalize  # 是否归一化
        self.obs_type = obs_type  # 观测类型
        self.goal_type = goal_type  # 目标类型
        self.lmdb_lock = lmdb_lock
        self.lmdb_readahead = lmdb_readahead
        self.lmdb_meminit = lmdb_meminit
        self.lmdb_max_readers = lmdb_max_readers
        self.lmdb_cache_mode = str(lmdb_cache_mode).lower()
        if self.lmdb_cache_mode not in {"auto", "read", "build"}:
            raise ValueError("lmdb_cache_mode must be one of: auto, read, build")
        self.rebuild_incomplete_lmdb = rebuild_incomplete_lmdb

        # ========== 加载数据集配置 ==========
        # 从data_config.yaml加载数据集特定参数（如metric_waypoint_spacing）
        with open(
                os.path.join(os.path.dirname(__file__), "data_config.yaml"), "r"
        ,encoding="utf-8") as f:
            all_data_config = yaml.safe_load(f)
        assert (
                self.dataset_name in all_data_config
        ), f"Dataset {self.dataset_name} not found in data_config.yaml"

        # 获取数据集索引（用于多数据集训练时的标识）
        dataset_names = list(all_data_config.keys())
        dataset_names.sort()

        self.dataset_index = dataset_names.index(self.dataset_name)
        self.data_config = all_data_config[self.dataset_name]
        self.metric_scale = float(self.data_config.get("metric_waypoint_spacing", 1.0)) * float(self.waypoint_spacing)

        # ========== 初始化缓存 ==========
        self.trajectory_cache = {}  # 轨迹数据缓存（内存）
        self._load_index()  # 加载或构建数据索引
        self._build_caches()  # 根据 lmdb_cache_mode 构建或只读打开 LMDB 图像缓存

        # ========== 动作参数维度 ==========
        if self.learn_angle:
            self.num_action_params = 3  # (x, y, yaw) 或 (x, y, sin, cos)
        else:
            self.num_action_params = 2  # (x, y)

    def __getstate__(self):
        """
        序列化对象状态（用于pickle）
        移除LMDB缓存引用，因为LMDB对象不能被pickle
        """
        state = self.__dict__.copy()
        state["_image_cache"] = None
        return state

    def __setstate__(self, state):
        """
        反序列化对象状态（用于pickle）
        重新打开LMDB缓存
        """
        self.__dict__ = state
        self._build_caches()

    def _build_caches(self, use_tqdm: bool = True):
        """
        使用LMDB构建/打开图像缓存以加速加载

        LMDB (Lightning Memory-Mapped Database) 优势：
        - 内存映射：直接映射到内存，避免频繁的磁盘I/O
        - 快速读取：比直接读取文件快10-100倍
        - 零拷贝：数据直接从磁盘映射到内存

        参数:
            use_tqdm (bool): 是否显示进度条
        """
        cache_filename, complete_filename = get_lmdb_cache_paths(
            self.data_split_folder,
            self.dataset_name,
        )
        expected_num_images = len(self.goals_index)
        cache_ready, cache_errors = _validate_lmdb_cache(
            cache_filename,
            complete_filename,
            self.dataset_name,
            expected_num_images,
        )

        should_build = self.lmdb_cache_mode == "build" or (
            self.lmdb_cache_mode == "auto" and not cache_ready
        )
        if self.lmdb_cache_mode == "read" and not cache_ready:
            joined_errors = "\n  - ".join(cache_errors)
            raise RuntimeError(
                "LMDB cache is missing or incomplete, but lmdb_cache_mode='read'.\n"
                f"Dataset: {self.dataset_name}\n"
                f"Split folder: {self.data_split_folder}\n"
                f"Problems:\n  - {joined_errors}\n"
                "Please run: python train.py -c <config> --build-lmdb-only"
            )

        if should_build and not cache_ready:
            if _path_exists(cache_filename) or os.path.exists(complete_filename):
                if not self.rebuild_incomplete_lmdb:
                    joined_errors = "\n  - ".join(cache_errors)
                    raise RuntimeError(
                        "Found an incomplete or unverified LMDB cache.\n"
                        f"Dataset: {self.dataset_name}\n"
                        f"Split folder: {self.data_split_folder}\n"
                        f"Problems:\n  - {joined_errors}\n"
                        "To rebuild it safely, rerun with --rebuild-incomplete-lmdb "
                        "or set rebuild_incomplete_lmdb: True."
                    )
                _remove_path(cache_filename)
                if os.path.exists(complete_filename):
                    os.remove(complete_filename)

            tmp_cache_filename = f"{cache_filename}.tmp.{os.getpid()}"
            _remove_path(tmp_cache_filename)
            tqdm_iterator = tqdm.tqdm(
                self.goals_index,
                disable=not use_tqdm,
                dynamic_ncols=True,
                desc=f"Building LMDB cache for {self.dataset_name}"
            )
            # 写入阶段使用临时 LMDB，全部成功后再原子重命名，避免中断后留下“看似存在但不完整”的缓存。
            num_cached_images = 0
            with lmdb.open(tmp_cache_filename, map_size=2 ** 40) as image_cache:
                with image_cache.begin(write=True) as txn:
                    for traj_name, time in tqdm_iterator:
                        image_path = get_data_path(self.data_folder, traj_name, time)
                        # 读取图像文件并存储到LMDB
                        with open(image_path, "rb") as f:
                            txn.put(image_path.encode(), f.read())
                        num_cached_images += 1
                image_cache.sync()

            if num_cached_images != expected_num_images:
                _remove_path(tmp_cache_filename)
                raise RuntimeError(
                    f"LMDB build count mismatch for {self.dataset_name}: "
                    f"{num_cached_images} != {expected_num_images}"
                )

            os.rename(tmp_cache_filename, cache_filename)
            marker = {
                "version": LMDB_CACHE_VERSION,
                "dataset_name": self.dataset_name,
                "num_cached_images": num_cached_images,
                "cache_path": cache_filename,
            }
            with open(complete_filename, "w", encoding="utf-8") as f:
                json.dump(marker, f, ensure_ascii=False, indent=2)

        elif should_build and cache_ready:
            print(f"LMDB cache already complete for {self.dataset_name}: {cache_filename}")

        # 以只读模式重新打开缓存文件；这些参数只影响读取性能，不改变读取到的数据内容。
        self._image_cache: lmdb.Environment = lmdb.open(
            cache_filename,
            readonly=True,
            lock=self.lmdb_lock,
            readahead=self.lmdb_readahead,
            meminit=self.lmdb_meminit,
            max_readers=self.lmdb_max_readers,
        )

    def _build_index(self, use_tqdm: bool = False):
        """
        构建数据索引

        创建两个索引：
        1. samples_index: 所有有效的(轨迹名, 当前时间, 最大目标距离)元组
           - 用于训练时采样观测-目标对
        2. goals_index: 所有可能的(轨迹名, 时间)元组
           - 用于负样本挖掘

        返回:
            samples_index: 训练样本索引列表
            goals_index: 目标候选索引列表
        """
        samples_index = []
        goals_index = []

        for traj_name in tqdm.tqdm(self.traj_names, disable=not use_tqdm, dynamic_ncols=True):
            traj_data = self._get_trajectory(traj_name)
            traj_len = len(traj_data["position"])

            # 将轨迹中的每个时间步添加到goals_index
            # 这些是潜在的目标候选
            for goal_time in range(0, traj_len):
                goals_index.append((traj_name, goal_time))

            # 计算有效的观测时间范围
            # begin_time: 必须有足够的历史上下文
            begin_time = self.context_size * self.waypoint_spacing
            # end_time: 必须有足够的未来轨迹用于预测和目标采样
            end_time = traj_len - self.end_slack - self.len_traj_pred * self.waypoint_spacing

            for curr_time in range(begin_time, end_time):
                # 计算从当前时间可以采样的最大目标距离
                # 不能超过轨迹末端
                max_goal_distance = min(self.max_dist_cat * self.waypoint_spacing, traj_len - curr_time - 1)
                samples_index.append((traj_name, curr_time, max_goal_distance))

        return samples_index, goals_index

    def _sample_goal(self, trajectory_name, curr_time, max_goal_dist):
        """
        从同一轨迹的未来采样一个目标

        采样策略：
        - 如果goal_offset=0：采样负样本（来自其他轨迹）
        - 否则：采样同一轨迹未来的某个时间点

        参数:
            trajectory_name: 当前轨迹名称
            curr_time: 当前时间步
            max_goal_dist: 最大目标距离（帧数）

        返回:
            (trajectory_name, goal_time, goal_is_negative)
            - trajectory_name: 目标所在的轨迹名称
            - goal_time: 目标时间步
            - goal_is_negative: 是否为负样本
        """
        goal_offset = np.random.randint(0, max_goal_dist + 1)
        if goal_offset == 0:
            # 采样负样本（来自可能不同的轨迹）
            trajectory_name, goal_time = self._sample_negative()
            return trajectory_name, goal_time, True
        else:
            # 采样同一轨迹的未来目标
            goal_time = curr_time + int(goal_offset * self.waypoint_spacing)
            return trajectory_name, goal_time, False

    def _sample_negative(self):
        """
        从（可能）不同的轨迹中采样一个目标

        负样本挖掘的作用：
        - 训练模型识别不可达的目标
        - 提升模型的鲁棒性
        - 避免模型对所有目标都输出"前进"

        返回:
            (trajectory_name, goal_time): 随机采样的目标
        """
        return self.goals_index[np.random.randint(0, len(self.goals_index))]

    def _load_index(self) -> None:
        """
        生成或加载数据索引

        索引包含数据集中每个观测的元组：
        (obs_traj_name, goal_traj_name, obs_time, goal_time)

        索引文件命名包含关键参数，确保参数变化时重新构建：
        - 距离范围: dist_{min}_to_{max}
        - 上下文类型和大小: context_{type}_n{size}
        - 末端裁剪: slack_{end_slack}

        如果索引文件已存在，直接加载以节省时间
        否则，调用_build_index()构建新索引
        """
        index_to_data_path = get_dataset_index_path(
            self.data_split_folder,
            self.min_dist_cat,
            self.max_dist_cat,
            self.waypoint_spacing,
            self.context_type,
            self.context_size,
            self.end_slack,
        )
        if os.path.exists(index_to_data_path):
            try:
                # 尝试加载已存在的索引（节省时间）
                with open(index_to_data_path, "rb") as f:
                    self.index_to_data, self.goals_index = pickle.load(f)
                return
            except (OSError, EOFError, pickle.PickleError, ValueError, AttributeError) as exc:
                raise RuntimeError(
                    "Failed to load dataset index. Remove or rebuild the index file after "
                    f"checking the dataset split and parameters: {index_to_data_path}"
                ) from exc

        # 如果索引文件不存在，创建它
        if not os.path.exists(index_to_data_path):
            self.index_to_data, self.goals_index = self._build_index()
            with open(index_to_data_path, "wb") as f:
                pickle.dump((self.index_to_data, self.goals_index), f)

    def _load_image(self, trajectory_name, time):
        """
        从LMDB缓存加载图像

        使用LMDB的优势：
        - 比直接读取文件快10-100倍
        - 内存映射，零拷贝
        - 适合频繁随机访问

        参数:
            trajectory_name: 轨迹名称
            time: 时间步

        返回:
            预处理后的图像张量
        """
        image_path = get_data_path(self.data_folder, trajectory_name, time)

        try:
            with self._image_cache.begin() as txn:
                image_buffer = txn.get(image_path.encode())
            if image_buffer is None:
                raise KeyError(image_path)
            image_bytes = io.BytesIO(bytes(image_buffer))
            return img_path_to_data(image_bytes, self.image_size)
        except Exception as exc:
            raise RuntimeError(
                f"Unable to read image from LMDB cache for dataset '{self.dataset_name}': {image_path}"
            ) from exc

    def _compute_actions(self, traj_data, curr_time, goal_time):
        """
        计算从当前位置到目标的动作序列（局部坐标系）

        核心步骤：
        1. 提取位置和航向序列
        2. 转换到机器人局部坐标系（以当前位置为原点，当前朝向为x轴）
        3. 计算相对航点位置
        4. 可选：计算相对航向角
        5. 归一化动作

        参数:
            traj_data: 轨迹数据字典（包含position和yaw）
            curr_time: 当前时间步
            goal_time: 目标时间步

        返回:
            actions: 动作序列 [len_traj_pred, num_action_params]
            goal_pos: 目标在局部坐标系中的位置 [2]
        """
        # 提取动作序列的起止索引
        start_index = curr_time
        end_index = curr_time + self.len_traj_pred * self.waypoint_spacing + 1

        # 提取航向和位置序列
        yaw = traj_data["yaw"][start_index:end_index:self.waypoint_spacing]
        positions = traj_data["position"][start_index:end_index:self.waypoint_spacing]
        goal_pos = traj_data["position"][min(goal_time, len(traj_data["position"]) - 1)]

        # 处理航向维度
        if len(yaw.shape) == 2:
            yaw = yaw.squeeze(1)

        # 如果序列长度不足，用最后一个值填充
        if yaw.shape != (self.len_traj_pred + 1,):
            const_len = self.len_traj_pred + 1 - yaw.shape[0]
            yaw = np.concatenate([yaw, np.repeat(yaw[-1], const_len)])
            positions = np.concatenate([positions, np.repeat(positions[-1][None], const_len, axis=0)], axis=0)

        # 验证形状
        assert yaw.shape == (self.len_traj_pred + 1,), f"{yaw.shape} and {(self.len_traj_pred + 1,)} should be equal"
        assert positions.shape == (self.len_traj_pred + 1,
                                   2), f"{positions.shape} and {(self.len_traj_pred + 1, 2)} should be equal"

        # 转换到局部坐标系（以当前位置和朝向为参考）
        # 这使得动作与机器人的绝对位置和朝向无关
        waypoints = to_local_coords(positions, positions[0], yaw[0])
        goal_pos = to_local_coords(goal_pos, positions[0], yaw[0])

        assert waypoints.shape == (self.len_traj_pred + 1,
                                   2), f"{waypoints.shape} and {(self.len_traj_pred + 1, 2)} should be equal"

        # 构建动作序列
        if self.learn_angle:
            # 计算相对航向角（相对于当前朝向）
            yaw = yaw[1:] - yaw[0]
            # 拼接位置和航向：[x, y, yaw]
            actions = np.concatenate([waypoints[1:], yaw[:, None]], axis=-1)
        else:
            # 仅使用位置：[x, y]
            actions = waypoints[1:]

        # 归一化动作（除以实际的米制距离）
        if self.normalize:
            # metric_waypoint_spacing: 数据集中航点之间的实际距离（米）
            actions[:, :2] /= self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing
            goal_pos /= self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing

        assert actions.shape == (self.len_traj_pred,
                                 self.num_action_params), f"{actions.shape} and {(self.len_traj_pred, self.num_action_params)} should be equal"

        return actions, goal_pos

    def _get_trajectory(self, trajectory_name):
        """
        获取轨迹数据（带缓存）

        轨迹数据包含：
        - position: [T, 2] 机器人的xy坐标
        - yaw: [T] 机器人的航向角

        使用内存缓存避免重复加载同一轨迹

        参数:
            trajectory_name: 轨迹名称

        返回:
            traj_data: 轨迹数据字典
        """
        if trajectory_name in self.trajectory_cache:
            return self.trajectory_cache[trajectory_name]
        else:
            # 从磁盘加载轨迹数据
            with open(os.path.join(self.data_folder, trajectory_name, "traj_data.pkl"), "rb") as f:
                traj_data = pickle.load(f)
            # 缓存到内存
            self.trajectory_cache[trajectory_name] = traj_data
            return traj_data

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.index_to_data)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        """
        获取第i个数据样本

        数据流程：
        1. 从索引获取观测和目标的轨迹名称、时间
        2. 加载上下文图像序列（历史观测）
        3. 加载当前观测图像
        4. 加载目标图像
        5. 计算动作标签（局部坐标系下的航点序列）
        6. 计算距离标签
        7. 确定动作掩码（是否应该学习这个动作）

        参数:
            i (int): 数据点索引

        返回:
            包含以下张量的元组：
            - obs_image (torch.Tensor): [3*(context_size+1), H, W] 观测图像序列
                包含context_size帧历史 + 1帧当前观测
            - goal_image (torch.Tensor): [3, H, W] 目标图像
            - actions_torch (torch.Tensor): [len_traj_pred, num_action_params] 动作标签
                如果learn_angle=True: [len_traj_pred, 4] (x, y, sin(yaw), cos(yaw))
                如果learn_angle=False: [len_traj_pred, 2] (x, y)
            - distance (torch.Tensor): [1] 从观测到目标的距离标签（航点数）
            - goal_pos (torch.Tensor): [2] 目标在局部坐标系中的位置
            - dataset_index (torch.Tensor): [1] 数据集索引（用于多数据集训练时的可视化）
            - action_mask (torch.Tensor): [1] 动作掩码（1=应该学习，0=不学习）
        """
        # 从索引获取当前观测信息  index_to_data 实际就是  samples_index.append((traj_name, curr_time, max_goal_distance))
        # f_curr  其实就是 traj_name
        f_curr, curr_time, max_goal_dist = self.index_to_data[i]
        # 采样目标（可能是正样本或负样本）
        # f_goal 其实也是  traj_name
        f_goal, goal_time, goal_is_negative = self._sample_goal(f_curr, curr_time, max_goal_dist)

        # ========== 加载图像 ==========
        context = []
        if self.context_type == "temporal":
            # 时序上下文：采样最近context_size帧历史并包含当前帧（共context_size+1帧）
            # 例如：context_size=5, curr_time=10, waypoint_spacing=1
            # 则采样时间: [5, 6, 7, 8, 9, 10]
            context_times = list(
                range(
                    curr_time + -self.context_size * self.waypoint_spacing,
                    curr_time + 1,
                    self.waypoint_spacing,
                )
            )
            # 其实就是 context =（ traj_name，采样时间: [5, 6, 7, 8, 9, 10] ）
            context = [(f_curr, t) for t in context_times]
        else:
            raise ValueError(f"Invalid context type {self.context_type}")

        # 拼接所有上下文图像（包括当前观测）
        # 形状: [3*(context_size+1), H, W]
        obs_image = torch.cat([
            self._load_image(f, t) for f, t in context
        ])

        # 加载目标图像
        # 形状: [3, H, W]
        goal_image = self._load_image(f_goal, goal_time)

        # ========== 加载轨迹数据 加载轨迹名的数据==========
        curr_traj_data = self._get_trajectory(f_curr)
        curr_traj_len = len(curr_traj_data["position"])
        assert curr_time < curr_traj_len, f"{curr_time} and {curr_traj_len}"

        goal_traj_data = self._get_trajectory(f_goal)
        goal_traj_len = len(goal_traj_data["position"])
        assert goal_time < goal_traj_len, f"{goal_time} an {goal_traj_len}"

        # ========== 计算动作标签 ==========
        # actions: [len_traj_pred, num_action_params] 局部坐标系下的航点序列

        # goal_pos: [2] 目标在局部坐标系中的位置  这个目标位置其实是在一个数组里面随机产生的
        actions, goal_pos = self._compute_actions(curr_traj_data, curr_time, goal_time)

        # ========== 计算距离标签 ==========
        if goal_is_negative:
            # 负样本：在当前实现中将距离标签置为max_dist_cat
            # （等效为最远可达桶，而不是-1）
            distance = self.max_dist_cat
        else:
            # 正样本：计算实际距离（航点数）
            distance = (goal_time - curr_time) // self.waypoint_spacing
            assert (
                               goal_time - curr_time) % self.waypoint_spacing == 0, f"{goal_time} and {curr_time} should be separated by an integer multiple of {self.waypoint_spacing}"

        # ========== 处理动作标签 ==========
        actions_torch = torch.as_tensor(actions, dtype=torch.float32)
        if self.learn_angle:
            # 将航向角转换为sin/cos表示（避免角度不连续性）
            # [len_traj_pred, 3] -> [len_traj_pred, 4]
            # (x, y, yaw) -> (x, y, sin(yaw), cos(yaw))
            actions_torch = calculate_sin_cos(actions_torch)

        # ========== 计算动作掩码 ==========
        # 只有满足以下条件才学习动作：
        # 1. 距离在有效范围内 [min_action_distance, max_action_distance]
        # 2. 不是负样本
        action_mask = (
                (distance < self.max_action_distance) and
                (distance > self.min_action_distance) and
                (not goal_is_negative)
        )

        return (
            torch.as_tensor(obs_image, dtype=torch.float32),
            torch.as_tensor(goal_image, dtype=torch.float32),
            actions_torch,
            torch.as_tensor(distance, dtype=torch.int64),
            torch.as_tensor(goal_pos, dtype=torch.float32),
            torch.as_tensor(self.dataset_index, dtype=torch.int64),
            torch.as_tensor(action_mask, dtype=torch.float32),
            torch.as_tensor(self.metric_scale, dtype=torch.float32),
        )


#
