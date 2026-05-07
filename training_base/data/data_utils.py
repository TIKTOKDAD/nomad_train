# ============================================================
# 数据处理工具函数模块
# ============================================================
# 本文件提供训练数据处理的核心工具函数
# 主要功能：
# 1. 图像加载和预处理（裁剪、缩放、归一化）
# 2. 坐标系转换（全局坐标 -> 局部坐标）
# 3. 路径点处理（计算增量、角度表示转换）
# 4. 数据路径管理
#
# 关键概念：
# - 局部坐标系：以机器人当前位置为原点的坐标系
# - 增量表示：相邻路径点之间的位置变化
# - sin/cos角度表示：避免角度不连续问题
# - 纵横比裁剪：保持图像比例一致性

import numpy as np
import os
from PIL import Image
from typing import Any, Iterable, Tuple

import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import io
from typing import Union

# ========== 全局常量 ==========

VISUALIZATION_IMAGE_SIZE = (160, 120)
"""可视化图像尺寸 (宽, 高)
用途：生成用于wandb日志和调试的可视化图像
说明：较小的尺寸节省存储和带宽
"""

IMAGE_ASPECT_RATIO = 4 / 3
"""图像纵横比（宽/高）
用途：训练时所有图像都会中心裁剪到4:3的纵横比
原因：
  - 保持输入尺寸一致性
  - 避免图像变形
  - 匹配大多数相机的原生比例
"""


def get_data_path(data_folder: str, f: str, time: int, data_type: str = "image"):
    """
    生成数据文件路径
    
    参数:
        data_folder: 数据根目录
        f: 轨迹文件夹名称
        time: 时间步索引
        data_type: 数据类型（默认"image"）
    
    返回:
        完整的数据文件路径
    
    路径格式:
        {data_folder}/{f}/{time}.{ext}
        例如: data/traj_001/42.jpg
    
    设计说明:
        - 支持扩展到其他数据类型（深度图、语义分割等）
        - 使用字典映射数据类型到文件扩展名
    """
    data_ext = {
        "image": ".jpg",
        # 可以在这里添加更多数据类型
        # "depth": ".png",
        # "semantic": ".npy",
    }
    return os.path.join(data_folder, f, f"{str(time)}{data_ext[data_type]}")


def yaw_rotmat(yaw: float) -> np.ndarray:
    """
    生成绕Z轴旋转的旋转矩阵
    
    参数:
        yaw: 偏航角（弧度），绕Z轴的旋转角度
    
    返回:
        3x3旋转矩阵
    
    数学原理:
        绕Z轴旋转的旋转矩阵：
        [cos(θ)  -sin(θ)  0]
        [sin(θ)   cos(θ)  0]
        [  0        0     1]
    
    使用场景:
        - 将全局坐标转换为机器人局部坐标
        - 考虑机器人的朝向
    """
    return np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
    )


def to_local_coords(
    positions: np.ndarray, curr_pos: np.ndarray, curr_yaw: float
) -> np.ndarray:
    """
    将位置从全局坐标系转换到机器人局部坐标系
    
    参数:
        positions: 要转换的位置 [N, 2] 或 [N, 3]
        curr_pos: 机器人当前位置 [2] 或 [3]
        curr_yaw: 机器人当前偏航角（弧度）
    
    返回:
        局部坐标系中的位置 [N, 2] 或 [N, 3]
    
    转换步骤:
        1. 平移：positions - curr_pos（将原点移到机器人位置）
        2. 旋转：乘以旋转矩阵（对齐坐标轴到机器人朝向）
    
    坐标系定义:
        - 全局坐标系：固定的世界坐标系
        - 局部坐标系：以机器人为原点，x轴指向前方
    
    使用场景:
        - 将目标位置转换为相对于机器人的位置
        - 训练时使用局部坐标，使模型学习相对导航
        - 提高泛化能力（不依赖全局位置）
    """
    # 生成旋转矩阵
    rotmat = yaw_rotmat(curr_yaw)
    
    # 根据位置维度选择合适的旋转矩阵
    if positions.shape[-1] == 2:
        # 2D位置 (x, y)
        rotmat = rotmat[:2, :2]
    elif positions.shape[-1] == 3:
        # 3D位置 (x, y, z)
        pass
    else:
        raise ValueError("位置维度必须是2或3")

    # 应用平移和旋转
    return (positions - curr_pos).dot(rotmat)


def calculate_deltas(waypoints: torch.Tensor) -> torch.Tensor:
    """
    计算路径点之间的增量
    
    参数:
        waypoints: 路径点张量 [T, D]
                  T = 时间步数
                  D = 维度（2表示(x,y)，3表示(x,y,θ)）
    
    返回:
        增量张量 [T, D']
        - 如果D=2: 返回 [T, 2]，表示(Δx, Δy)
        - 如果D=3: 返回 [T, 4]，表示(Δx, Δy, cos(Δθ), sin(Δθ))
    
    工作流程:
        1. 在路径点前添加原点(0,0)或(0,0,0)
        2. 计算相邻点之间的差值
        3. 如果包含角度，转换为sin/cos表示
    
    为什么使用增量:
        - 增量表示更稳定，避免绝对位置的累积误差
        - 更容易学习，因为增量通常较小且分布更均匀
        - 便于归一化
    
    使用场景:
        - 训练时将绝对路径点转换为增量
        - 推理时累积增量得到绝对路径
    """
    num_params = waypoints.shape[1]
    
    # 创建原点作为第一个"前一个"路径点
    origin = torch.zeros(1, num_params)
    
    # 拼接原点和路径点（除了最后一个）
    prev_waypoints = torch.concat((origin, waypoints[:-1]), axis=0)
    
    # 计算增量
    deltas = waypoints - prev_waypoints
    
    # 如果包含角度（3维），转换为sin/cos表示
    if num_params > 2:
        return calculate_sin_cos(deltas)
    
    return deltas


def calculate_sin_cos(waypoints: torch.Tensor) -> torch.Tensor:
    """
    将角度表示转换为sin/cos表示
    
    参数:
        waypoints: 路径点张量 [T, 3]，格式为(x, y, θ)
    
    返回:
        转换后的张量 [T, 4]，格式为(x, y, cos(θ), sin(θ))
    
    为什么使用sin/cos表示:
        1. 避免角度不连续问题：
           - 弧度表示：-π和π是同一个角度，但数值差异大
           - sin/cos表示：连续且平滑
        
        2. 更容易学习：
           - 神经网络难以学习周期性函数
           - sin/cos是连续的实数，更适合回归
        
        3. 归一化友好：
           - sin和cos的范围都是[-1, 1]
           - 便于与位置坐标一起归一化
    
    数学关系:
        θ = atan2(sin(θ), cos(θ))
        可以从sin/cos恢复原始角度
    
    使用场景:
        - 训练时将角度转换为sin/cos
        - 推理时从sin/cos恢复角度
    """
    assert waypoints.shape[1] == 3, "输入必须是3维 (x, y, θ)"
    
    # 创建角度表示张量
    angle_repr = torch.zeros_like(waypoints[:, :2])
    angle_repr[:, 0] = torch.cos(waypoints[:, 2])  # cos(θ)
    angle_repr[:, 1] = torch.sin(waypoints[:, 2])  # sin(θ)
    
    # 拼接位置和角度表示
    return torch.concat((waypoints[:, :2], angle_repr), axis=1)


def transform_images(
    img: Image.Image, 
    transform: transforms, 
    image_resize_size: Tuple[int, int], 
    aspect_ratio: float = IMAGE_ASPECT_RATIO
):
    """
    转换图像用于训练和可视化
    
    参数:
        img: PIL图像
        transform: torchvision变换（归一化等）
        image_resize_size: 训练图像尺寸 (宽, 高)
        aspect_ratio: 目标纵横比（默认4:3）
    
    返回:
        (viz_img, transf_img): 
        - viz_img: 可视化图像张量 [3, 120, 160]
        - transf_img: 训练图像张量 [3, H, W]
    
    处理流程:
        1. 中心裁剪到目标纵横比
        2. 生成可视化版本（小尺寸）
        3. 生成训练版本（目标尺寸 + 归一化）
    
    为什么需要中心裁剪:
        - 不同相机可能有不同的纵横比
        - 保持一致的纵横比避免图像变形
        - 中心裁剪保留最重要的内容
    
    使用场景:
        - 数据加载时预处理图像
        - 同时生成训练和可视化版本
    """
    w, h = img.size
    
    # 中心裁剪到目标纵横比
    if w > h:
        # 宽度大于高度，裁剪宽度
        img = TF.center_crop(img, (h, int(h * aspect_ratio)))
    else:
        # 高度大于宽度，裁剪高度
        img = TF.center_crop(img, (int(w / aspect_ratio), w))
    
    # 生成可视化版本（小尺寸，用于wandb）
    viz_img = img.resize(VISUALIZATION_IMAGE_SIZE)
    viz_img = TF.to_tensor(viz_img)
    
    # 生成训练版本（目标尺寸 + 归一化）
    img = img.resize(image_resize_size)
    transf_img = transform(img)
    
    return viz_img, transf_img


def resize_and_aspect_crop(
    img: Image.Image, 
    image_resize_size: Tuple[int, int], 
    aspect_ratio: float = IMAGE_ASPECT_RATIO
):
    """
    裁剪并调整图像大小（不进行归一化）
    
    参数:
        img: PIL图像
        image_resize_size: 目标尺寸 (宽, 高)
        aspect_ratio: 目标纵横比（默认4:3）
    
    返回:
        调整后的图像张量 [3, H, W]，范围[0, 1]
    
    与transform_images的区别:
        - 不应用归一化变换
        - 只返回一个版本
        - 用于不需要ImageNet归一化的场景
    
    使用场景:
        - 加载拓扑地图图像
        - 推理时的图像预处理
        - 不需要训练时归一化的情况
    """
    w, h = img.size
    
    # 中心裁剪到目标纵横比
    if w > h:
        img = TF.center_crop(img, (h, int(h * aspect_ratio)))
    else:
        img = TF.center_crop(img, (int(w / aspect_ratio), w))
    
    # 调整大小并转换为张量
    img = img.resize(image_resize_size)
    resize_img = TF.to_tensor(img)
    
    return resize_img


def img_path_to_data(path: Union[str, io.BytesIO], image_resize_size: Tuple[int, int]) -> torch.Tensor:
    """
    从路径加载图像并转换为张量
    
    参数:
        path: 图像文件路径或字节流
        image_resize_size: 目标尺寸 (宽, 高)
    
    返回:
        图像张量 [3, H, W]，范围[0, 1]
    
    功能:
        1. 加载图像
        2. 裁剪到标准纵横比
        3. 调整到目标尺寸
        4. 转换为张量
    
    使用场景:
        - 数据集加载器中的图像加载
        - 从LMDB数据库加载图像
        - 推理时加载单张图像
    
    设计说明:
        - 支持文件路径和字节流（用于LMDB）
        - 不进行归一化，保持原始像素值范围[0, 1]
    """
    return resize_and_aspect_crop(Image.open(path), image_resize_size)    

