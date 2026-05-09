# ============================================================
# Trajectory store - cached legacy pickle reader
# ============================================================
# 本文件读取每条轨迹目录下的 traj_data.pkl：
# 1. 兼容历史 pickle 中可能引用的 numpy._core 模块路径
# 2. 以 trajectory_name 为 key 做进程内缓存，避免重复反序列化
# 3. 返回的数据通常包含 position、yaw 等标签计算所需数组

import os
import pickle
import sys

import numpy.core as np_core


# 兼容不同 numpy 版本生成的 pickle 模块路径
sys.modules.setdefault("numpy._core", np_core)
if hasattr(np_core, "multiarray"):
    sys.modules.setdefault("numpy._core.multiarray", np_core.multiarray)


# 轨迹数据读取器：按需加载并缓存 traj_data.pkl
class PickleTrajectoryStore:
    """Cached reader for legacy traj_data.pkl files."""

    # 初始化数据根目录和内存缓存
    def __init__(self, data_folder: str) -> None:
        self.data_folder = data_folder
        self.cache = {}

    # 获取单条轨迹的元数据
    def get(self, trajectory_name: str):
        # 第一次访问时从磁盘读取，后续访问直接复用缓存对象
        if trajectory_name not in self.cache:
            path = os.path.join(self.data_folder, trajectory_name, "traj_data.pkl")
            with open(path, "rb") as f:
                self.cache[trajectory_name] = pickle.load(f)
        return self.cache[trajectory_name]
