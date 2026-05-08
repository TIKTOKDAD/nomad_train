# ============================================================
# Navigation batch - DataLoader batch protocol
# ============================================================
# 本文件把 NavigationDataset 返回的 tuple 转成具名结构：
# 1. 算法侧可以用 batch.obs_image / batch.actions 访问字段，减少下标错误
# 2. collate_fn 保持兼容 Dataset 旧返回格式（7 个或 8 个张量）
# 3. pin_memory 支持 DataLoader 在 CUDA 训练时加速 CPU->GPU 拷贝

from dataclasses import dataclass, replace
from typing import Any, Dict, Sequence

import torch
from torch.utils.data._utils.collate import default_collate


# 标准化的导航批次结构，方便算法侧统一访问
@dataclass
class NavigationBatch:
    """Named batch protocol for navigation algorithms."""

    # obs_image: [B, 3*(context_size+1), H, W]，多帧观测按通道拼接
    obs_image: torch.Tensor
    # goal_image: [B, 3, H, W]，目标图像
    goal_image: torch.Tensor
    # actions: [B, len_traj_pred, action_dim]，局部坐标系下的航点标签
    actions: torch.Tensor
    # distance: [B]，当前观测到目标的离散距离标签
    distance: torch.Tensor
    # goal_pos: [B, 2]，目标点在当前机器人局部坐标系中的位置
    goal_pos: torch.Tensor
    # dataset_index: [B]，多数据集训练时标识样本来源
    dataset_index: torch.Tensor
    # action_mask: [B]，控制哪些样本参与动作损失
    action_mask: torch.Tensor
    # metric_scale: [B]，把归一化航点还原为米制轨迹的缩放系数
    metric_scale: torch.Tensor
    # extras: 为未来扩展保留的非标准字段
    extras: Dict[str, Any]

    # 批次大小
    @property
    def batch_size(self) -> int:
        return int(self.obs_image.shape[0])

    # 将张量固定到页锁定内存，提升数据传输速度
    def pin_memory(self):
        # replace 会创建新的 dataclass 实例，不修改原 batch 对象
        return replace(
            self,
            obs_image=self.obs_image.pin_memory(),
            goal_image=self.goal_image.pin_memory(),
            actions=self.actions.pin_memory(),
            distance=self.distance.pin_memory(),
            goal_pos=self.goal_pos.pin_memory(),
            dataset_index=self.dataset_index.pin_memory(),
            action_mask=self.action_mask.pin_memory(),
            metric_scale=self.metric_scale.pin_memory(),
        )


# 将默认 collate 的结果转换为 NavigationBatch
def as_navigation_batch(data: Sequence[torch.Tensor]) -> NavigationBatch:
    # 若上游已经返回 NavigationBatch，则直接透传
    if isinstance(data, NavigationBatch):
        return data
    if len(data) not in {7, 8}:
        raise ValueError(f"NavigationDataset 应返回 7 或 8 个张量，实际得到 {len(data)} 个")
    # 旧 Dataset 版本没有 metric_scale，第 8 个字段缺失时用 1.0 兜底
    metric_scale = data[7] if len(data) == 8 else torch.ones_like(data[5], dtype=torch.float32)
    return NavigationBatch(
        obs_image=data[0],
        goal_image=data[1],
        actions=data[2],
        distance=data[3],
        goal_pos=data[4],
        dataset_index=data[5],
        action_mask=data[6],
        metric_scale=metric_scale,
        extras={},
    )


# DataLoader 的 collate_fn：输出 NavigationBatch
def navigation_collate(samples):
    # default_collate 先把样本 tuple 堆叠成 batch tuple，再转成具名 batch
    return as_navigation_batch(default_collate(samples))


# 将拼接后的观测图像切分并应用变换
def split_and_transform_obs(obs_image: torch.Tensor, transform, device: torch.device):
    # 每 3 个通道是一帧 RGB 图像，逐帧 normalize 后再拼回通道维
    obs_images = torch.split(obs_image, 3, dim=1)
    obs_images = [transform(obs).to(device, non_blocking=True) for obs in obs_images]
    return torch.cat(obs_images, dim=1)


# 对目标图像执行同样的预处理并迁移到设备
def transform_goal(goal_image: torch.Tensor, transform, device: torch.device):
    return transform(goal_image).to(device, non_blocking=True)
