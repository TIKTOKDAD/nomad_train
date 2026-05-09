# ============================================================
# Navigation dataset - visual navigation data loader
# ============================================================
# 本文件实现了用于训练GNM、ViNT、NoMaD模型的数据集类
# 主要功能：
# 1. 加载机器人轨迹数据（图像、位置、航向）
# 2. 采样观测-目标对（包括负样本挖掘）
# 3. 计算局部坐标系下的动作标签
# 4. 使用LMDB缓存加速图像加载

import os
from typing import Tuple

import torch
from torch.utils.data import Dataset

# 导入数据处理工具函数
from training_base.data.data_utils import (
    calculate_sin_cos,  # 将角度转换为sin/cos表示
)
from training_base.data.action_labels import NavigationActionLabelBuilder
from training_base.data.goal_sampling import (
    distance_label_for_goal,
    normalize_goal_sampling_config,
    sample_navigation_goal,
)
from training_base.data.image_store import LmdbImageStore
from training_base.data.navigation_spec import LmdbCacheConfig, NavigationDatasetSpec
from training_base.data.indexing import (
    build_navigation_index,
    get_dataset_index_path,
    load_or_build_navigation_index,
)
from training_base.data.labeling import context_entries
from training_base.data.trajectory_store import PickleTrajectoryStore
from training_base.registry import dataset_registry


# 核心数据集类：负责采样、标签构造与图像缓存读取
@dataset_registry.register("navigation")
class NavigationDataset(Dataset):
    def __init__(
            self,
            *,
            spec: NavigationDatasetSpec,
            cache_config: LmdbCacheConfig,
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
                "temporal": 时序上下文（使用最近的N帧）；当前实现只支持该值

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
        # DataModule 已经负责把 YAML/runtime 翻译成 spec；Dataset 只消费明确的构造契约。
        self.data_folder = spec.data_folder
        self.data_split_folder = spec.data_split_folder
        self.dataset_name = spec.dataset_name

        # 读取轨迹名称列表（每行一个轨迹目录名）
        traj_names_file = os.path.join(self.data_split_folder, "traj_names.txt")
        with open(traj_names_file, "r", encoding="utf-8") as f:
            file_lines = f.read()
            self.traj_names = file_lines.split("\n")
        # 移除末尾空行造成的空字符串
        if "" in self.traj_names:
            self.traj_names.remove("")

        # ========== 距离和航点配置 ==========
        # 保存图像尺寸与航点间隔
        self.image_size = spec.image_size
        self.waypoint_spacing = spec.waypoint_spacing
        # 生成距离类别列表：[min_dist_cat, min_dist_cat+spacing, ..., max_dist_cat]
        self.distance_categories = list(
            range(spec.distance.min_dist_cat, spec.distance.max_dist_cat + 1, self.waypoint_spacing)
        )
        # 便于后续快速访问最小/最大距离桶
        self.min_dist_cat = self.distance_categories[0]
        self.max_dist_cat = self.distance_categories[-1]

        # ========== 负样本挖掘 ==========
        # negative_mining 控制是否加入“不可达”类别
        self.negative_mining = spec.negative_mining
        if self.negative_mining:
            # 在距离类别中添加-1作为不可达类别标记（仅用于类别集合定义）
            self.distance_categories.append(-1)

        # ========== 轨迹预测参数 ==========
        # 预测未来轨迹长度与角度学习开关
        self.len_traj_pred = spec.len_traj_pred  # 预测的航点数量
        self.learn_angle = spec.learn_angle  # 是否学习航向角

        # ========== 动作距离范围 ==========
        # 控制训练动作标签的有效距离区间
        self.min_action_distance = spec.action.min_dist_cat
        self.max_action_distance = spec.action.max_dist_cat

        # ========== 上下文配置 ==========
        # context_size 为历史帧数量，context_type 控制采样策略
        self.context_size = spec.context.context_size
        context_type = spec.context.context_type
        if context_type != "temporal":
            raise ValueError(f"data.context_type 当前只支持 temporal，实际为 {context_type!r}")
        self.context_type = context_type

        # ========== 其他配置 ==========
        # 轨迹末端裁剪、目标数量与归一化配置
        self.end_slack = spec.end_slack  # 轨迹末端裁剪
        self.goals_per_obs = spec.goals_per_obs  # 每个观测的目标数
        self.normalize = spec.normalize  # 是否归一化
        self.obs_type = spec.obs_type  # 观测类型
        self.goal_type = spec.goal_type  # 目标类型
        if self.obs_type != "image":
            raise ValueError(f"data.obs_type 当前只支持 image，实际为 {self.obs_type!r}")
        if self.goal_type != "image":
            raise ValueError(f"data.goal_type 当前只支持 image，实际为 {self.goal_type!r}")
        # LMDB 读取参数与缓存策略
        self.lmdb_lock = cache_config.lock
        self.lmdb_readahead = cache_config.readahead
        self.lmdb_meminit = cache_config.meminit
        self.lmdb_max_readers = cache_config.max_readers
        self.lmdb_map_size = int(cache_config.map_size)
        self.lmdb_cache_mode = str(cache_config.cache_mode).lower()
        if self.lmdb_cache_mode not in {"auto", "read", "build"}:
            raise ValueError("lmdb_cache_mode 必须是 auto、read 或 build 之一")
        self.rebuild_incomplete_lmdb = cache_config.rebuild_incomplete
        self.image_aspect_ratio = float(spec.image_aspect_ratio)
        self.goal_sampling_config = normalize_goal_sampling_config(spec.goal_sampling)

        # ========== 数据集元信息 ==========
        self.dataset_index = int(spec.metadata.dataset_index)
        self.data_config = dict(spec.metadata.data_config)
        self.metric_scale = float(spec.metadata.metric_scale)

        # ========== 初始化缓存 ==========
        # trajectory_cache 用于重复访问时避免磁盘 IO
        self.trajectory_store = PickleTrajectoryStore(self.data_folder)
        self.trajectory_cache = self.trajectory_store.cache  # 兼容旧测试/调试入口
        self._load_index()  # 加载或构建数据索引
        self._build_caches()  # 根据 lmdb_cache_mode 构建或只读打开 LMDB 图像缓存

        # ========== 动作参数维度 ==========
        # learn_angle 决定动作维度是否包含角度信息
        if self.learn_angle:
            # labeling 阶段先生成 (x, y, yaw)，__getitem__ 返回前再转成 (x, y, cos, sin)
            self.num_action_params = 3
        else:
            self.num_action_params = 2  # (x, y)
        self.action_label_builder = NavigationActionLabelBuilder(
            len_traj_pred=self.len_traj_pred,
            waypoint_spacing=self.waypoint_spacing,
            learn_angle=self.learn_angle,
            normalize=self.normalize,
            metric_waypoint_spacing=float(self.data_config["metric_waypoint_spacing"]),
            num_action_params=self.num_action_params,
            dataset_name=self.dataset_name,
        )

    def __getstate__(self):
        """
        序列化对象状态（用于pickle）
        移除LMDB缓存引用，因为LMDB对象不能被pickle
        """
        state = self.__dict__.copy()
        # LMDB 环境对象不可序列化，序列化前置空
        state["_image_cache"] = None
        state["image_store"] = None
        return state

    def __setstate__(self, state):
        """
        反序列化对象状态（用于pickle）
        重新打开LMDB缓存
        """
        self.__dict__ = state
        # 反序列化后重建 LMDB 缓存连接
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
        self.image_store = LmdbImageStore(
            data_folder=self.data_folder,
            data_split_folder=self.data_split_folder,
            dataset_name=self.dataset_name,
            goals_index=self.goals_index,
            image_size=self.image_size,
            image_aspect_ratio=self.image_aspect_ratio,
            lmdb_lock=self.lmdb_lock,
            lmdb_readahead=self.lmdb_readahead,
            lmdb_meminit=self.lmdb_meminit,
            lmdb_max_readers=self.lmdb_max_readers,
            lmdb_map_size=self.lmdb_map_size,
            lmdb_cache_mode=self.lmdb_cache_mode,
            rebuild_incomplete_lmdb=self.rebuild_incomplete_lmdb,
            use_tqdm=use_tqdm,
        )
        self._image_cache = self.image_store.env

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
        return build_navigation_index(
            traj_names=self.traj_names,
            get_trajectory=self._get_trajectory,
            context_size=self.context_size,
            waypoint_spacing=self.waypoint_spacing,
            len_traj_pred=self.len_traj_pred,
            end_slack=self.end_slack,
            max_dist_cat=self.max_dist_cat,
            use_tqdm=use_tqdm,
        )

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
        return sample_navigation_goal(
            trajectory_name,
            curr_time,
            max_goal_dist,
            self.waypoint_spacing,
            self.goals_index,
            config=self.goal_sampling_config,
        )

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
        self.index_to_data, self.goals_index = load_or_build_navigation_index(
            index_path=index_to_data_path,
            build_fn=self._build_index,
        )

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
        return self.image_store.load(trajectory_name, time)

    def _compute_actions(self, traj_data, curr_time, goal_time, trajectory_name):
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
        return self.action_label_builder.build(
            traj_data=traj_data,
            curr_time=curr_time,
            goal_time=goal_time,
            trajectory_name=trajectory_name,
        )

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
        return self.trajectory_store.get(trajectory_name)

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
                如果learn_angle=True: [len_traj_pred, 4] (x, y, cos(yaw), sin(yaw))
                如果learn_angle=False: [len_traj_pred, 2] (x, y)
            - distance (torch.Tensor): [1] 从观测到目标的距离标签（航点数）
            - goal_pos (torch.Tensor): [2] 目标在局部坐标系中的位置
            - dataset_index (torch.Tensor): [1] 数据集索引（用于多数据集训练时的可视化）
            - action_mask (torch.Tensor): [1] 动作掩码（1=应该学习，0=不学习）
        """
        # ========== 1. 读取索引并采样目标 ==========
        # index_to_data 中每条记录来自 _build_index:
        #   (obs_traj_name, curr_time, max_goal_distance)
        # obs_traj_name 表示当前观测所在轨迹，curr_time 表示当前观测帧
        f_curr, curr_time, max_goal_dist = self.index_to_data[i]
        # 根据当前轨迹和最大可达距离采样目标：
        # 正样本来自同一轨迹未来帧，负样本来自全局 goals_index 随机目标
        f_goal, goal_time, goal_is_negative = self._sample_goal(f_curr, curr_time, max_goal_dist)

        # ========== 2. 加载上下文观测图像 ==========
        # context 列表保存 (traj_name, time) 的采样对
        context = context_entries(
            self.context_type,
            f_curr,
            curr_time,
            self.context_size,
            self.waypoint_spacing,
        )

        # 拼接所有上下文图像（包括当前观测）
        # 形状: [3*(context_size+1), H, W]
        # 逐帧读取并在通道维拼接
        obs_image = torch.cat([
            self._load_image(f, t) for f, t in context
        ])

        # 加载目标图像
        # 形状: [3, H, W]
        # 目标图像单独读取，后续模型会与观测序列一起编码
        goal_image = self._load_image(f_goal, goal_time)

        # ========== 3. 加载轨迹元数据 ==========
        # 获取当前轨迹与目标轨迹（可能相同也可能不同）
        curr_traj_data = self._get_trajectory(f_curr)
        curr_traj_len = len(curr_traj_data["position"])
        if curr_time >= curr_traj_len:
            raise ValueError(
                f"{self.dataset_name}:{f_curr} 当前时间 {curr_time} 必须小于当前轨迹长度 {curr_traj_len}"
            )

        goal_traj_data = self._get_trajectory(f_goal)
        goal_traj_len = len(goal_traj_data["position"])
        if goal_time >= goal_traj_len:
            raise ValueError(
                f"{self.dataset_name}:{f_goal} 目标时间 {goal_time} 必须小于目标轨迹长度 {goal_traj_len}"
            )

        # ========== 计算动作标签 ==========
        # actions: [len_traj_pred, num_action_params] 局部坐标系下的航点序列

        # goal_pos: [2] 目标在当前机器人局部坐标系中的位置
        # 注意：动作与 goal_pos 仅依赖当前轨迹与目标时间
        actions, goal_pos = self._compute_actions(curr_traj_data, curr_time, goal_time, f_curr)

        # ========== 计算距离标签 ==========
        if goal_is_negative:
            distance = distance_label_for_goal(
                True,
                distance=self.max_dist_cat,
                max_dist_cat=self.max_dist_cat,
                config=self.goal_sampling_config,
            )
        else:
            # 正样本：计算实际距离（航点数）
            positive_distance = (goal_time - curr_time) // self.waypoint_spacing
            if (goal_time - curr_time) % self.waypoint_spacing != 0:
                raise ValueError(
                    f"{self.dataset_name}:{f_curr}->{f_goal} 的 goal_time={goal_time} 与 "
                    f"curr_time={curr_time} 间隔必须是 waypoint_spacing={self.waypoint_spacing} 的整数倍"
                )
            distance = distance_label_for_goal(
                False,
                distance=positive_distance,
                max_dist_cat=self.max_dist_cat,
                config=self.goal_sampling_config,
            )

        # ========== 处理动作标签 ==========
        # numpy -> torch，并按需转换角度表示
        actions_torch = torch.as_tensor(actions, dtype=torch.float32)
        if self.learn_angle:
            # 将航向角转换为sin/cos表示（避免角度不连续性）
            # [len_traj_pred, 3] -> [len_traj_pred, 4]
            # (x, y, yaw) -> (x, y, cos(yaw), sin(yaw))
            actions_torch = calculate_sin_cos(actions_torch)

        # ========== 计算动作掩码 ==========
        # 只有满足以下条件才学习动作：
        # 1. 距离在有效范围内 [min_action_distance, max_action_distance]
        # 2. 不是负样本
        # 只在有效距离范围内学习动作（负样本不回归动作）
        action_mask = (
                (distance < self.max_action_distance) and
                (distance > self.min_action_distance) and
                (not goal_is_negative)
        )

        # 统一转换为 torch.Tensor 并返回
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
