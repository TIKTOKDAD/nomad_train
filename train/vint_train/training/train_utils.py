# ============================================================
# 训练工具函数模块
# ============================================================
# 本文件包含 ViNT / GNM / NoMaD 的训练、评估、日志与可视化相关工具函数。

import wandb  # Weights & Biases 实验跟踪
import os
import numpy as np
import yaml
from typing import List, Optional, Dict
from prettytable import PrettyTable
import tqdm
import itertools

# 可视化工具
from vint_train.visualizing.action_utils import visualize_traj_pred, plot_trajs_and_points
from vint_train.visualizing.distance_utils import visualize_dist_pred
from vint_train.visualizing.visualize_utils import to_numpy, from_numpy
from vint_train.training.logger import Logger
from vint_train.data.data_utils import VISUALIZATION_IMAGE_SIZE

# 扩散模型组件
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel  # 指数滑动平均模型

# PyTorch 相关库
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

# ========== 加载数据配置 ==========
# 加载动作统计信息（用于归一化）
with open(os.path.join(os.path.dirname(__file__), "../data/data_config.yaml"), "r",encoding="utf-8") as f:
    data_config = yaml.safe_load(f)

# ========== 动作统计表 ==========
# ACTION_STATS 包含每个数据集动作的 min/max，用于映射到 [-1, 1]
ACTION_STATS = {}
for key in data_config['action_stats']:
    ACTION_STATS[key] = np.array(data_config['action_stats'][key])

# ========== ViNT / GNM 训练工具函数 ==========

def _compute_losses(
    dist_label: torch.Tensor,  # 真实距离标签，形状 [B]
    action_label: torch.Tensor,  # 真实动作/轨迹标签，形状 [B, T, D]
    dist_pred: torch.Tensor,  # 距离预测值，形状 [B, 1]
    action_pred: torch.Tensor,  # 动作/轨迹预测值，形状 [B, T, D]
    alpha: float,  # 距离损失与动作损失的融合权重
    learn_angle: bool,  # 是否计算朝向分量相似度指标
    action_mask: torch.Tensor = None,  # 动作有效样本掩码，形状 [B]
):
    """
    计算 ViNT/GNM 在单个 batch 上的距离损失、动作损失及相似度指标。

    参数：
    - `dist_label` (`[B]`)：样本到目标的真实距离。
    - `action_label` (`[B, T, D]`)：真实轨迹/动作序列。
    - `dist_pred` (`[B, 1]`)：模型预测的距离。
    - `action_pred` (`[B, T, D]`)：模型预测的轨迹/动作序列。
    - `alpha`：距离损失与动作损失的融合系数。
    - `learn_angle`：若为 True，额外计算朝向分量（如 yaw/sin-cos）的相似度指标。
    - `action_mask` (`[B]`)：动作有效性掩码。无效样本会被削弱或忽略。

    返回：
    - `dist_loss`：距离回归 MSE。
    - `action_loss`：动作回归 MSE（应用 action_mask）。
    - `action_waypts_cos_sim`：逐时间步的平面方向余弦相似度（x,y）。
    - `multi_action_waypts_cos_sim`：整段轨迹展平后的平面方向相似度。
    - `action_orien_cos_sim` / `multi_action_orien_cos_sim`：朝向分量相似度（仅 learn_angle=True 时）。
    - `total_loss`：用于反向传播的总损失。

    实现细节：
    - 距离损失单独计算并在总损失中乘以 `1e-2` 缩放（保持与原实现一致）。
    - 动作相关损失/相似度都通过 `action_reduce` 做“按样本归约 + mask 加权平均”。
    - 使用断言保证预测与标签形状一致，尽早发现数据错配问题。
    """
    dist_loss = F.mse_loss(dist_pred.squeeze(-1), dist_label.float())

    def action_reduce(unreduced_loss: torch.Tensor):  # 未归约损失，形状 [B, ...]
        # 将 [B, ...] 的损失逐步归约成 [B]，再做 mask 加权平均。
        # 分母加 1e-2 用于避免有效样本极少时的数值不稳定。
        while unreduced_loss.dim() > 1:
            unreduced_loss = unreduced_loss.mean(dim=-1)
        assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
        return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)

    assert action_pred.shape == action_label.shape, f"{action_pred.shape} != {action_label.shape}"
    action_loss = action_reduce(F.mse_loss(action_pred, action_label, reduction="none"))

    # 路径点方向相似度（仅看 x,y，忽略幅值尺度差异）
    action_waypts_cos_similairity = action_reduce(
        F.cosine_similarity(action_pred[:, :, :2], action_label[:, :, :2], dim=-1)
    )
    multi_action_waypts_cos_sim = action_reduce(
        F.cosine_similarity(
            torch.flatten(action_pred[:, :, :2], start_dim=1),
            torch.flatten(action_label[:, :, :2], start_dim=1),
            dim=-1,
        )
    )

    results = {
        "dist_loss": dist_loss,
        "action_loss": action_loss,
        "action_waypts_cos_sim": action_waypts_cos_similairity,
        "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim,
    }

    if learn_angle:
        # 当动作维度包含朝向分量时，额外统计朝向方向一致性
        action_orien_cos_sim = action_reduce(
            F.cosine_similarity(action_pred[:, :, 2:], action_label[:, :, 2:], dim=-1)
        )
        multi_action_orien_cos_sim = action_reduce(
            F.cosine_similarity(
                torch.flatten(action_pred[:, :, 2:], start_dim=1),
                torch.flatten(action_label[:, :, 2:], start_dim=1),
                dim=-1,
            )
        )
        results["action_orien_cos_sim"] = action_orien_cos_sim
        results["multi_action_orien_cos_sim"] = multi_action_orien_cos_sim

    # 与历史实现保持一致：distance loss 量级缩放到与 action loss 相近
    total_loss = alpha * 1e-2 * dist_loss + (1 - alpha) * action_loss
    results["total_loss"] = total_loss
    return results
def _log_data(
    i,  # 当前 batch 索引
    epoch,  # 当前 epoch 编号
    num_batches,  # 当前阶段总 batch 数
    normalized,  # 可视化轨迹是否按归一化坐标解释
    project_folder,  # 可视化输出根目录
    num_images_log,  # 每次可视化记录的样本数
    loggers,  # 指标 logger 字典
    obs_image,  # 观测图像（用于可视化）
    goal_image,  # 目标图像（用于可视化）
    action_pred,  # 预测动作/轨迹
    action_label,  # 真实动作/轨迹
    dist_pred,  # 预测距离
    dist_label,  # 真实距离
    goal_pos,  # 目标位置点
    dataset_index,  # 数据集索引/来源标识
    use_wandb,  # 是否启用 wandb 日志
    mode,  # 日志模式名称（train / eval_xxx）
    use_latest,  # True 用最新值，False 用平均值
    wandb_log_freq=1,  # wandb 标量日志频率
    print_log_freq=1,  # 控制台打印频率
    image_log_freq=1,  # 图像可视化日志频率
    wandb_increment_step=True,  # 记录时是否推进 wandb step
):
    """
    统一处理训练/评估过程中的标量日志与可视化日志。

    参数：
    - `i` / `epoch` / `num_batches`：当前训练进度信息。
    - `loggers`：`Logger` 字典，保存每个指标的滑动窗口/平均值。
    - `use_latest`：True 表示记录当前 batch 最新值；False 表示记录累计平均值。
    - `wandb_*_freq` / `print_log_freq` / `image_log_freq`：不同输出通道的记录频率。
    - `wandb_increment_step`：是否推进 wandb step（评估阶段常置 False）。
    - 其余参数用于图像可视化（观测图、目标图、轨迹、距离等）。

    行为说明：
    1. 汇总 logger 指标并按频率打印到控制台；
    2. 按频率写入 wandb 标量；
    3. 按频率生成距离与轨迹可视化图，支持本地保存与 wandb 上报。
    """
    data_log = {}
    for key, logger in loggers.items():
        if use_latest:
            data_log[logger.full_name()] = logger.latest()
            if i % print_log_freq == 0 and print_log_freq != 0:
                print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")
        else:
            data_log[logger.full_name()] = logger.average()
            if i % print_log_freq == 0 and print_log_freq != 0:
                print(f"(epoch {epoch}) {logger.full_name()} {logger.average()}")

    if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
        wandb.log(data_log, commit=wandb_increment_step)

    if image_log_freq != 0 and i % image_log_freq == 0:
        # 距离预测可视化
        visualize_dist_pred(
            to_numpy(obs_image),
            to_numpy(goal_image),
            to_numpy(dist_pred),
            to_numpy(dist_label),
            mode,
            project_folder,
            epoch,
            num_images_log,
            use_wandb=use_wandb,
        )
        # 轨迹预测可视化
        visualize_traj_pred(
            to_numpy(obs_image),
            to_numpy(goal_image),
            to_numpy(dataset_index),
            to_numpy(goal_pos),
            to_numpy(action_pred),
            to_numpy(action_label),
            mode,
            normalized,
            project_folder,
            epoch,
            num_images_log,
            use_wandb=use_wandb,
        )


def train(
    model: nn.Module,  # ViNT/GNM 模型
    optimizer: Adam,  # 优化器
    dataloader: DataLoader,  # 训练数据加载器
    transform: transforms,  # 图像预处理变换
    device: torch.device,  # 训练设备
    project_folder: str,  # 项目输出目录（日志/可视化）
    normalized: bool,  # 可视化时动作是否按归一化空间解释
    epoch: int,  # 当前 epoch 编号
    alpha: float = 0.5,  # 距离损失与动作损失融合权重
    learn_angle: bool = True,  # 是否统计朝向分量相似度
    print_log_freq: int = 100,  # 控制台打印频率（按 batch）
    wandb_log_freq: int = 10,  # wandb 标量日志频率
    image_log_freq: int = 1000,  # 图像可视化频率
    num_images_log: int = 8,  # 每次可视化样本数量
    use_wandb: bool = True,  # 是否启用 wandb
    use_tqdm: bool = True,  # 是否显示 tqdm 进度条
):
    """
    训练 ViNT/GNM 一个 epoch（标准监督学习循环）。

    参数：
    - `model`：待训练模型，前向返回 `(dist_pred, action_pred)`。
    - `optimizer`：优化器（如 Adam）。
    - `dataloader`：训练数据加载器，单个 batch 返回：
      `(obs_image, goal_image, action_label, dist_label, goal_pos, dataset_index, action_mask)`。
    - `transform`：图像预处理（归一化、resize 等）。
    - `device`：训练设备（CPU/GPU）。
    - `project_folder`：可视化输出目录。
    - `normalized`：动作是否在归一化坐标系下可视化。
    - `alpha` / `learn_angle`：损失构成相关开关。
    - `*_log_freq` / `num_images_log`：日志与可视化频率控制。

    训练流程：
    1. `model.train()` 切换训练模式；
    2. 初始化各项 logger；
    3. 遍历 batch：
       a) 处理多帧观测与目标图像；
       b) 前向得到距离/动作预测；
       c) 调用 `_compute_losses` 计算所有指标；
       d) 反向传播 + 参数更新；
       e) 记录日志和可视化。
    """
    model.train()

    dist_loss_logger = Logger("dist_loss", "train", window_size=print_log_freq)
    action_loss_logger = Logger("action_loss", "train", window_size=print_log_freq)
    action_waypts_cos_sim_logger = Logger("action_waypts_cos_sim", "train", window_size=print_log_freq)
    multi_action_waypts_cos_sim_logger = Logger("multi_action_waypts_cos_sim", "train", window_size=print_log_freq)
    total_loss_logger = Logger("total_loss", "train", window_size=print_log_freq)
    loggers = {
        "dist_loss": dist_loss_logger,
        "action_loss": action_loss_logger,
        "action_waypts_cos_sim": action_waypts_cos_sim_logger,
        "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim_logger,
        "total_loss": total_loss_logger,
    }

    if learn_angle:
        # 仅在模型学习角度分量时才记录该类指标
        action_orien_cos_sim_logger = Logger("action_orien_cos_sim", "train", window_size=print_log_freq)
        multi_action_orien_cos_sim_logger = Logger("multi_action_orien_cos_sim", "train", window_size=print_log_freq)
        loggers["action_orien_cos_sim"] = action_orien_cos_sim_logger
        loggers["multi_action_orien_cos_sim"] = multi_action_orien_cos_sim_logger

    num_batches = len(dataloader)
    tqdm_iter = tqdm.tqdm(
        dataloader,
        disable=not use_tqdm,
        dynamic_ncols=True,
        desc=f"Training epoch {epoch}",
    )

    for i, data in enumerate(tqdm_iter):
        (
            obs_image,
            goal_image,
            action_label,
            dist_label,
            goal_pos,
            dataset_index,
            action_mask,
        ) = data

        # 观测输入通常是“多帧 RGB 在通道维拼接”：
        # 例如 [B, 3*K, H, W] -> 拆成 K 帧 [B, 3, H, W] 再做 transform。
        obs_images = torch.split(obs_image, 3, dim=1)
        # 可视化时只看最后一帧，避免多帧叠加导致图像不可读。
        viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE)
        obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
        obs_image = torch.cat(obs_images, dim=1)

        viz_goal_image = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE)
        goal_image = transform(goal_image).to(device)

        dist_label = dist_label.to(device)
        action_label = action_label.to(device)
        action_mask = action_mask.to(device)

        # 标准优化步骤：清空梯度 -> 前向 -> 反向 -> step
        optimizer.zero_grad()
        dist_pred, action_pred = model(obs_image, goal_image)

        losses = _compute_losses(
            dist_label=dist_label,
            action_label=action_label,
            dist_pred=dist_pred,
            action_pred=action_pred,
            alpha=alpha,
            learn_angle=learn_angle,
            action_mask=action_mask,
        )

        losses["total_loss"].backward()
        optimizer.step()

        for key, value in losses.items():
            if key in loggers:
                loggers[key].log_data(value.item())

        _log_data(
            i=i,
            epoch=epoch,
            num_batches=num_batches,
            normalized=normalized,
            project_folder=project_folder,
            num_images_log=num_images_log,
            loggers=loggers,
            obs_image=viz_obs_image,
            goal_image=viz_goal_image,
            action_pred=action_pred,
            action_label=action_label,
            dist_pred=dist_pred,
            dist_label=dist_label,
            goal_pos=goal_pos,
            dataset_index=dataset_index,
            wandb_log_freq=wandb_log_freq,
            print_log_freq=print_log_freq,
            image_log_freq=image_log_freq,
            use_wandb=use_wandb,
            mode="train",
            use_latest=True,
        )


def evaluate(
    eval_type: str,  # 评估类型标识（用于日志命名）
    model: nn.Module,  # 待评估模型
    dataloader: DataLoader,  # 评估数据加载器
    transform: transforms,  # 图像预处理变换
    device: torch.device,  # 评估设备
    project_folder: str,  # 项目输出目录（日志/可视化）
    normalized: bool,  # 可视化时动作是否按归一化空间解释
    epoch: int = 0,  # 当前 epoch 编号
    alpha: float = 0.5,  # 距离损失与动作损失融合权重
    learn_angle: bool = True,  # 是否统计朝向分量相似度
    num_images_log: int = 8,  # 每次可视化样本数量
    use_wandb: bool = True,  # 是否启用 wandb
    eval_fraction: float = 1.0,  # 评估数据比例（0~1）
    use_tqdm: bool = True,  # 是否显示 tqdm 进度条
):
    """
    在指定数据集上评估 ViNT/GNM（无梯度）。

    与训练阶段差异：
    - 使用 `model.eval()` 关闭 dropout 等训练态行为；
    - 使用 `torch.no_grad()` 降低显存并提升速度；
    - 支持 `eval_fraction` 仅评估部分 batch；
    - 结果以平均指标为主，避免单个 batch 波动影响判断。

    返回：
    - `(avg_dist_loss, avg_action_loss, avg_total_loss)`。
    """
    model.eval()

    dist_loss_logger = Logger("dist_loss", eval_type)
    action_loss_logger = Logger("action_loss", eval_type)
    action_waypts_cos_sim_logger = Logger("action_waypts_cos_sim", eval_type)
    multi_action_waypts_cos_sim_logger = Logger("multi_action_waypts_cos_sim", eval_type)
    total_loss_logger = Logger("total_loss", eval_type)
    loggers = {
        "dist_loss": dist_loss_logger,
        "action_loss": action_loss_logger,
        "action_waypts_cos_sim": action_waypts_cos_sim_logger,
        "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim_logger,
        "total_loss": total_loss_logger,
    }

    if learn_angle:
        # 评估阶段同样可统计方向相关指标，便于与训练对齐
        loggers["action_orien_cos_sim"] = Logger("action_orien_cos_sim", eval_type)
        loggers["multi_action_orien_cos_sim"] = Logger("multi_action_orien_cos_sim", eval_type)

    num_batches = max(int(len(dataloader) * eval_fraction), 1)

    # 保存最后一个 batch 的可视化内容，循环结束后统一记录（避免每步都画图开销过大）
    viz_obs_image = None
    with torch.no_grad():
        tqdm_iter = tqdm.tqdm(
            itertools.islice(dataloader, num_batches),
            total=num_batches,
            disable=not use_tqdm,
            dynamic_ncols=True,
            desc=f"Evaluating {eval_type} for epoch {epoch}",
        )
        for i, data in enumerate(tqdm_iter):
            (
                obs_image,
                goal_image,
                action_label,
                dist_label,
                goal_pos,
                dataset_index,
                action_mask,
            ) = data

            obs_images = torch.split(obs_image, 3, dim=1)
            viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE)
            obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
            obs_image = torch.cat(obs_images, dim=1)

            viz_goal_image = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE)
            goal_image = transform(goal_image).to(device)

            dist_label = dist_label.to(device)
            action_label = action_label.to(device)
            action_mask = action_mask.to(device)

            dist_pred, action_pred = model(obs_image, goal_image)

            losses = _compute_losses(
                dist_label=dist_label,
                action_label=action_label,
                dist_pred=dist_pred,
                action_pred=action_pred,
                alpha=alpha,
                learn_angle=learn_angle,
                action_mask=action_mask,
            )

            for key, value in losses.items():
                if key in loggers:
                    loggers[key].log_data(value.item())

    _log_data(
        i=i,
        epoch=epoch,
        num_batches=num_batches,
        normalized=normalized,
        project_folder=project_folder,
        num_images_log=num_images_log,
        loggers=loggers,
        obs_image=viz_obs_image,
        goal_image=viz_goal_image,
        action_pred=action_pred,
        action_label=action_label,
        goal_pos=goal_pos,
        dist_pred=dist_pred,
        dist_label=dist_label,
        dataset_index=dataset_index,
        use_wandb=use_wandb,
        mode=eval_type,
        use_latest=False,
        wandb_increment_step=False,
    )

    return dist_loss_logger.average(), action_loss_logger.average(), total_loss_logger.average()


# ========== NoMaD 训练工具函数 ==========
def _compute_losses_nomad(
    ema_model,  # EMA 平滑后的 NoMaD 模型
    noise_scheduler,  # DDPM 噪声调度器
    batch_obs_images,  # 观测图像批次
    batch_goal_images,  # 目标图像批次
    batch_dist_label: torch.Tensor,  # 距离真值，形状 [B]
    batch_action_label: torch.Tensor,  # 动作真值，形状 [B, T, D]
    device: torch.device,  # 计算设备
    action_mask: torch.Tensor,  # 动作有效掩码，形状 [B]
):
    """
    计算 NoMaD 在 UC/GC 两种推理模式下的动作与距离指标。

    参数：
    - `ema_model`：EMA 平滑后的模型权重（评估更稳定）。
    - `noise_scheduler`：DDPM 调度器，控制去噪步序。
    - `batch_obs_images` / `batch_goal_images`：输入图像批次。
    - `batch_dist_label`：真实距离 `[B]`。
    - `batch_action_label`：真实动作轨迹 `[B, T, D]`。
    - `action_mask`：样本有效掩码 `[B]`。

    返回：
    - UC/GC 动作 MSE；
    - UC/GC 逐步与整轨迹余弦相似度；
    - GC 距离 MSE。

    说明：
    - 该函数主要用于训练/评估阶段的“解释型指标统计”，
      与主训练损失（扩散噪声预测）互补。
    """
    pred_horizon = batch_action_label.shape[1]
    action_dim = batch_action_label.shape[2]

    # 一次调用同时得到两种控制模式的动作：
    # UC（探索）和 GC（目标条件导航），并返回 GC 距离预测。
    model_output_dict = model_output(
        ema_model,
        noise_scheduler,
        batch_obs_images,
        batch_goal_images,
        pred_horizon,
        action_dim,
        num_samples=1,
        device=device,
    )
    uc_actions = model_output_dict["uc_actions"]
    gc_actions = model_output_dict["gc_actions"]
    gc_distance = model_output_dict["gc_distance"]

    # GC 分支具备目标条件，可直接评估距离回归质量
    gc_dist_loss = F.mse_loss(gc_distance, batch_dist_label.unsqueeze(-1))

    def action_reduce(unreduced_loss: torch.Tensor):  # 未归约损失，形状 [B, ...]
        # 先把非 batch 维归约，再用 action_mask 过滤无效样本
        while unreduced_loss.dim() > 1:
            unreduced_loss = unreduced_loss.mean(dim=-1)
        assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
        return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)

    assert uc_actions.shape == batch_action_label.shape, f"{uc_actions.shape} != {batch_action_label.shape}"
    assert gc_actions.shape == batch_action_label.shape, f"{gc_actions.shape} != {batch_action_label.shape}"

    uc_action_loss = action_reduce(F.mse_loss(uc_actions, batch_action_label, reduction="none"))
    gc_action_loss = action_reduce(F.mse_loss(gc_actions, batch_action_label, reduction="none"))

    # 逐时间步余弦相似度（只看 x,y）
    uc_action_waypts_cos_similairity = action_reduce(
        F.cosine_similarity(uc_actions[:, :, :2], batch_action_label[:, :, :2], dim=-1)
    )
    gc_action_waypts_cos_similairity = action_reduce(
        F.cosine_similarity(gc_actions[:, :, :2], batch_action_label[:, :, :2], dim=-1)
    )

    # 全轨迹展平后的整体余弦相似度
    uc_multi_action_waypts_cos_sim = action_reduce(
        F.cosine_similarity(
            torch.flatten(uc_actions[:, :, :2], start_dim=1),
            torch.flatten(batch_action_label[:, :, :2], start_dim=1),
            dim=-1,
        )
    )
    gc_multi_action_waypts_cos_sim = action_reduce(
        F.cosine_similarity(
            torch.flatten(gc_actions[:, :, :2], start_dim=1),
            torch.flatten(batch_action_label[:, :, :2], start_dim=1),
            dim=-1,
        )
    )

    return {
        "uc_action_loss": uc_action_loss,
        "uc_action_waypts_cos_sim": uc_action_waypts_cos_similairity,
        "uc_multi_action_waypts_cos_sim": uc_multi_action_waypts_cos_sim,
        "gc_dist_loss": gc_dist_loss,
        "gc_action_loss": gc_action_loss,
        "gc_action_waypts_cos_sim": gc_action_waypts_cos_similairity,
        "gc_multi_action_waypts_cos_sim": gc_multi_action_waypts_cos_sim,
    }


def train_nomad(
    model: nn.Module,  # NoMaD 主模型
    ema_model: EMAModel,  # 指数滑动平均模型容器
    optimizer: Adam,  # 优化器
    dataloader: DataLoader,  # 训练数据加载器
    transform: transforms,  # 图像预处理变换
    device: torch.device,  # 训练设备
    noise_scheduler: DDPMScheduler,  # DDPM 噪声调度器
    goal_mask_prob: float,  # 目标被 mask 的概率（控制探索/导航比例）
    project_folder: str,  # 项目输出目录（日志/可视化）
    epoch: int,  # 当前 epoch 编号
    alpha: float = 1e-4,  # 距离辅助损失权重
    print_log_freq: int = 100,  # 控制台打印频率
    wandb_log_freq: int = 10,  # wandb 标量日志频率
    image_log_freq: int = 1000,  # 图像可视化频率
    num_images_log: int = 8,  # 每次可视化样本数量
    use_wandb: bool = True,  # 是否启用 wandb
):
    """
    训练 NoMaD 模型一个 epoch（扩散主任务 + 距离辅助任务）。

    该函数是 NoMaD 的核心训练循环，单个 batch 的输入通常为：
    - `obs_image`: [B, 3*K, H, W]，K 帧观测图像在通道维拼接；
    - `goal_image`: [B, 3, H, W]，目标图像；
    - `actions`: [B, T, 2]，真实绝对轨迹；
    - `distance`: [B]，观测到目标的距离标签；
    - `goal_pos`: [B, 2]，目标点位置（主要用于可视化）；
    - `dataset_idx`: [B]，样本来源数据集索引；
    - `action_mask`: [B]，动作有效样本掩码。

    训练目标分两部分：
    1. 扩散主损失 `diffusion_loss`：
       随机采样时间步 `t`，给真实动作加噪，训练网络预测噪声残差。
    2. 距离辅助损失 `dist_loss`：
       仅在目标未被 mask 的样本上计算距离回归误差。

    最终损失：
    - `loss = alpha * dist_loss + (1 - alpha) * diffusion_loss`
      其中 `alpha` 通常较小，使训练重点落在扩散动作建模上。

    额外机制：
    - 通过 `goal_mask_prob` 混合“探索（无目标）”与“导航（有目标）”训练样本；
    - 每次参数更新后同步更新 EMA 权重，后续评估/采样优先使用 EMA 模型。
    """
    # 防御式裁剪：避免配置文件传入非法概率（<0 或 >1）
    goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    model.train()
    # 当前 epoch 的 batch 总数（不是 batch_size）
    num_batches = len(dataloader)

    # 初始化日志器：
    # UC = Unconditioned（目标被 mask，偏探索）
    # GC = Goal-conditioned（目标可见，偏导航）
    uc_action_loss_logger = Logger("uc_action_loss", "train", window_size=print_log_freq)
    uc_action_waypts_cos_sim_logger = Logger("uc_action_waypts_cos_sim", "train", window_size=print_log_freq)
    uc_multi_action_waypts_cos_sim_logger = Logger("uc_multi_action_waypts_cos_sim", "train", window_size=print_log_freq)
    gc_dist_loss_logger = Logger("gc_dist_loss", "train", window_size=print_log_freq)
    gc_action_loss_logger = Logger("gc_action_loss", "train", window_size=print_log_freq)
    gc_action_waypts_cos_sim_logger = Logger("gc_action_waypts_cos_sim", "train", window_size=print_log_freq)
    gc_multi_action_waypts_cos_sim_logger = Logger("gc_multi_action_waypts_cos_sim", "train", window_size=print_log_freq)
    loggers = {
        "uc_action_loss": uc_action_loss_logger,
        "uc_action_waypts_cos_sim": uc_action_waypts_cos_sim_logger,
        "uc_multi_action_waypts_cos_sim": uc_multi_action_waypts_cos_sim_logger,
        "gc_dist_loss": gc_dist_loss_logger,
        "gc_action_loss": gc_action_loss_logger,
        "gc_action_waypts_cos_sim": gc_action_waypts_cos_sim_logger,
        "gc_multi_action_waypts_cos_sim": gc_multi_action_waypts_cos_sim_logger,
    }

    # 逐 batch 训练（使用 tqdm 展示进度）
    with tqdm.tqdm(dataloader, desc="Train Batch", leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            # 解包当前 batch 数据
            (
                obs_image,
                goal_image,
                actions,
                distance,
                goal_pos,
                dataset_idx,
                action_mask,
            ) = data

            # ===== 1) 图像预处理 =====
            # 多帧观测输入：按 RGB 三通道拆帧，再逐帧 transform，最后拼接回模型输入
            obs_images = torch.split(obs_image, 3, dim=1)

            # 可视化只使用最后一帧观测，避免多帧叠加难以阅读
            batch_viz_obs_images = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            batch_viz_goal_images = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])

            # 训练输入保留多帧上下文信息
            batch_obs_images = [transform(obs) for obs in obs_images]
            batch_obs_images = torch.cat(batch_obs_images, dim=1).to(device)
            # 目标图像
            batch_goal_images = transform(goal_image).to(device)
            action_mask = action_mask.to(device)

            B = actions.shape[0]

            # ===== 2) 条件编码（探索/导航混合） =====
            # goal_mask=1：屏蔽目标（探索模式）
            # goal_mask=0：使用目标（导航模式）
            # 单个 batch 内混合两类样本，可提升策略泛化能力
            goal_mask = (torch.rand((B,)) < goal_mask_prob).long().to(device)
            obsgoal_cond = model(
                "vision_encoder",
                obs_img=batch_obs_images,
                goal_img=batch_goal_images,
                input_goal_mask=goal_mask,
            )

            # 距离标签转 float 并迁移到训练设备  其实这个是时间步数
            distance = distance.float().to(device)

            # ===== 3) 轨迹标签处理 =====
            # 在“增量动作空间”学习扩散过程，通常比直接建模绝对坐标更稳定 变成相对坐标 x，y
            deltas = get_delta(actions)
            # 进行归一化 将相对坐标。
            ndeltas = normalize_data(deltas, ACTION_STATS)
            #
            naction = from_numpy(ndeltas).to(device)
            assert naction.shape[-1] == 2, "action dim must be 2"

            # ===== 4) 距离辅助损失（仅 goal 可见样本） =====
            # 对探索样本（goal_mask=1）不计算距离监督，避免引入无意义梯度
            dist_pred = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
            dist_loss = nn.functional.mse_loss(dist_pred.squeeze(-1), distance)
            # 使用 (1-goal_mask) 进行样本级过滤，并做有效样本数归一化
            dist_loss = (dist_loss * (1 - goal_mask.float())).mean() / (1e-2 + (1 - goal_mask.float()).mean())

            # ===== 5) 扩散主损失 =====
            # 采样高斯噪声与扩散时间步，对真实动作加噪后让网络预测噪声残差
            noise = torch.randn(naction.shape, device=device)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,), device=device).long()
            noisy_action = noise_scheduler.add_noise(naction, noise, timesteps)
            noise_pred = model("noise_pred_net", sample=noisy_action, timestep=timesteps, global_cond=obsgoal_cond)

            def action_reduce(unreduced_loss: torch.Tensor):  # 未归约损失，形状 [B, ...]
                # 将非 batch 维度归约，再按 action_mask 过滤无效样本
                while unreduced_loss.dim() > 1:
                    unreduced_loss = unreduced_loss.mean(dim=-1)
                assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
                return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)

            # 扩散主损失：预测噪声与真实噪声的 MSE（掩码加权后）
            diffusion_loss = action_reduce(F.mse_loss(noise_pred, noise, reduction="none"))
            # 总损失：以扩散主损失为主，距离损失为辅助
            loss = alpha * dist_loss + (1 - alpha) * diffusion_loss

            # ===== 6) 反向传播与参数更新 =====
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 训练后更新 EMA 参数（评估与可视化更稳定）
            ema_model.step(model)

            # ===== 7) 基础日志 =====
            loss_cpu = loss.item()
            tepoch.set_postfix(loss=loss_cpu)
            wandb.log({"total_loss": loss_cpu})
            wandb.log({"dist_loss": dist_loss.item()})
            wandb.log({"diffusion_loss": diffusion_loss.item()})

            if i % print_log_freq == 0:
                # ===== 8) 详细指标日志（较重） =====
                # 该分支会触发额外采样计算，能更直观看到 UC/GC 的行为质量
                losses = _compute_losses_nomad(
                    ema_model.averaged_model,
                    noise_scheduler,
                    batch_obs_images,
                    batch_goal_images,
                    distance.to(device),
                    actions.to(device),
                    device,
                    action_mask.to(device),
                )

                for key, value in losses.items():
                    if key in loggers:
                        loggers[key].log_data(value.item())

                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)

            if image_log_freq != 0 and i % image_log_freq == 0:
                # ===== 9) 可视化日志 =====
                # 定期导出动作采样分布图，用于检查：
                # - 采样是否发散
                # - 导航轨迹是否明显朝向目标
                # - 模型不确定性是否异常增大
                visualize_diffusion_action_distribution(
                    ema_model.averaged_model,
                    noise_scheduler,
                    batch_obs_images,
                    batch_goal_images,
                    batch_viz_obs_images,
                    batch_viz_goal_images,
                    actions,
                    distance,
                    goal_pos,
                    device,
                    "train",
                    project_folder,
                    epoch,
                    num_images_log,
                    30,
                    use_wandb,
                )


def evaluate_nomad(
    eval_type: str,  # 评估类型标识（用于日志命名）
    ema_model: EMAModel,  # EMA 模型容器（函数内会取 averaged_model）
    dataloader: DataLoader,  # 评估数据加载器
    transform: transforms,  # 图像预处理变换
    device: torch.device,  # 评估设备
    noise_scheduler: DDPMScheduler,  # DDPM 噪声调度器
    goal_mask_prob: float,  # 随机目标 mask 概率
    project_folder: str,  # 项目输出目录（日志/可视化）
    epoch: int,  # 当前 epoch 编号
    print_log_freq: int = 100,  # 控制台打印频率
    wandb_log_freq: int = 10,  # wandb 标量日志频率
    image_log_freq: int = 1000,  # 图像可视化频率
    num_images_log: int = 8,  # 每次可视化样本数量
    eval_fraction: float = 0.25,  # 评估数据比例（0~1）
    use_wandb: bool = True,  # 是否启用 wandb
):
    """
    评估 NoMaD（使用 EMA 平滑权重）。

    评估包含两类信号：
    - 扩散噪声预测误差（随机 mask / 全不 mask / 全 mask 三种条件）。
    - UC/GC 轨迹与距离指标（按日志频率调用 `_compute_losses_nomad`）。

    与训练阶段不同：
    - 仅前向，不更新参数；
    - 默认使用 `ema_model.averaged_model`；
    - 通过 `eval_fraction` 控制评估开销。
    """
    # 与训练保持一致，保证 mask 概率稳定
    goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    ema_model = ema_model.averaged_model
    ema_model.eval()

    num_batches = len(dataloader)

    uc_action_loss_logger = Logger("uc_action_loss", eval_type, window_size=print_log_freq)
    uc_action_waypts_cos_sim_logger = Logger("uc_action_waypts_cos_sim", eval_type, window_size=print_log_freq)
    uc_multi_action_waypts_cos_sim_logger = Logger("uc_multi_action_waypts_cos_sim", eval_type, window_size=print_log_freq)
    gc_dist_loss_logger = Logger("gc_dist_loss", eval_type, window_size=print_log_freq)
    gc_action_loss_logger = Logger("gc_action_loss", eval_type, window_size=print_log_freq)
    gc_action_waypts_cos_sim_logger = Logger("gc_action_waypts_cos_sim", eval_type, window_size=print_log_freq)
    gc_multi_action_waypts_cos_sim_logger = Logger("gc_multi_action_waypts_cos_sim", eval_type, window_size=print_log_freq)
    loggers = {
        "uc_action_loss": uc_action_loss_logger,
        "uc_action_waypts_cos_sim": uc_action_waypts_cos_sim_logger,
        "uc_multi_action_waypts_cos_sim": uc_multi_action_waypts_cos_sim_logger,
        "gc_dist_loss": gc_dist_loss_logger,
        "gc_action_loss": gc_action_loss_logger,
        "gc_action_waypts_cos_sim": gc_action_waypts_cos_sim_logger,
        "gc_multi_action_waypts_cos_sim": gc_multi_action_waypts_cos_sim_logger,
    }
    # 只评估前若干 batch，可在大数据集上快速做 sanity check
    num_batches = max(int(num_batches * eval_fraction), 1)

    with tqdm.tqdm(
        itertools.islice(dataloader, num_batches),
        total=num_batches,
        dynamic_ncols=True,
        desc=f"Evaluating {eval_type} for epoch {epoch}",
        leave=False,
    ) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_image,
                goal_image,
                actions,
                distance,
                goal_pos,
                dataset_idx,
                action_mask,
            ) = data

            # 观测处理方式与训练一致，防止 train/eval 输入分布不一致
            obs_images = torch.split(obs_image, 3, dim=1)
            batch_viz_obs_images = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            batch_viz_goal_images = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
            batch_obs_images = [transform(obs) for obs in obs_images]
            batch_obs_images = torch.cat(batch_obs_images, dim=1).to(device)
            batch_goal_images = transform(goal_image).to(device)
            action_mask = action_mask.to(device)

            B = actions.shape[0]

            # 三种掩码设置：
            # rand_goal_mask: 与训练分布一致的随机掩码；
            # goal_mask: 全探索；
            # no_mask: 全目标条件。
            rand_goal_mask = (torch.rand((B,)) < goal_mask_prob).long().to(device)
            goal_mask = torch.ones_like(rand_goal_mask).long().to(device)
            no_mask = torch.zeros_like(rand_goal_mask).long().to(device)

            rand_mask_cond = ema_model(
                "vision_encoder",
                obs_img=batch_obs_images,
                goal_img=batch_goal_images,
                input_goal_mask=rand_goal_mask,
            )
            obsgoal_cond = ema_model(
                "vision_encoder",
                obs_img=batch_obs_images,
                goal_img=batch_goal_images,
                input_goal_mask=no_mask,
            )
            obsgoal_cond = obsgoal_cond.flatten(start_dim=1)
            goal_mask_cond = ema_model(
                "vision_encoder",
                obs_img=batch_obs_images,
                goal_img=batch_goal_images,
                input_goal_mask=goal_mask,
            )

            distance = distance.to(device)

            deltas = get_delta(actions)
            ndeltas = normalize_data(deltas, ACTION_STATS)
            naction = from_numpy(ndeltas).to(device)
            assert naction.shape[-1] == 2, "action dim must be 2"

            # 在同一批输入上比较三种条件下的噪声预测误差
            noise = torch.randn(naction.shape, device=device)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,), device=device).long()
            noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)

            rand_mask_noise_pred = ema_model(
                "noise_pred_net",
                sample=noisy_actions,
                timestep=timesteps,
                global_cond=rand_mask_cond,
            )
            rand_mask_loss = nn.functional.mse_loss(rand_mask_noise_pred, noise)

            no_mask_noise_pred = ema_model(
                "noise_pred_net",
                sample=noisy_actions,
                timestep=timesteps,
                global_cond=obsgoal_cond,
            )
            no_mask_loss = nn.functional.mse_loss(no_mask_noise_pred, noise)

            goal_mask_noise_pred = ema_model(
                "noise_pred_net",
                sample=noisy_actions,
                timestep=timesteps,
                global_cond=goal_mask_cond,
            )
            goal_mask_loss = nn.functional.mse_loss(goal_mask_noise_pred, noise)

            loss_cpu = rand_mask_loss.item()
            tepoch.set_postfix(loss=loss_cpu)

            wandb.log({"diffusion_eval_loss (random masking)": rand_mask_loss})
            wandb.log({"diffusion_eval_loss (no masking)": no_mask_loss})
            wandb.log({"diffusion_eval_loss (goal masking)": goal_mask_loss})

            if i % print_log_freq == 0 and print_log_freq != 0:
                losses = _compute_losses_nomad(
                    ema_model,
                    noise_scheduler,
                    batch_obs_images,
                    batch_goal_images,
                    distance.to(device),
                    actions.to(device),
                    device,
                    action_mask.to(device),
                )

                for key, value in losses.items():
                    if key in loggers:
                        loggers[key].log_data(value.item())

                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)

            if image_log_freq != 0 and i % image_log_freq == 0:
                visualize_diffusion_action_distribution(
                    ema_model,
                    noise_scheduler,
                    batch_obs_images,
                    batch_goal_images,
                    batch_viz_obs_images,
                    batch_viz_goal_images,
                    actions,
                    distance,
                    goal_pos,
                    device,
                    eval_type,
                    project_folder,
                    epoch,
                    num_images_log,
                    30,
                    use_wandb,
                )


# ========== 数据归一化工具函数 ==========
def get_data_stats(
    data,  # 原始数据数组（最后一维为动作维度）
):
    """
    统计输入数据在每个动作维度上的最小值和最大值。

    参数：
    - `data`：任意形状数组，最后一维必须是动作维度。

    返回：
    - `{"min": ..., "max": ...}`，用于 `normalize_data/unnormalize_data`。
    """
    data = data.reshape(-1, data.shape[-1])
    stats = {
        "min": np.min(data, axis=0),
        "max": np.max(data, axis=0),
    }
    return stats


def normalize_data(
    data,  # 待归一化数据
    stats,  # 归一化统计量字典，包含 min/max
):
    """
    将数据按统计量归一化到 [-1, 1]。

    公式：
    1. [min, max] -> [0, 1]
    2. [0, 1] -> [-1, 1]
    """
    # 映射到 [0,1]
    ndata = (data - stats["min"]) / (stats["max"] - stats["min"])
    # 再线性映射到 [-1,1]
    ndata = ndata * 2 - 1
    return ndata


def unnormalize_data(
    ndata,  # 已归一化数据（通常范围 [-1, 1]）
    stats,  # 归一化统计量字典，包含 min/max
):
    """
    将 [-1, 1] 的归一化数据还原到原始范围。

    公式是 `normalize_data` 的逆变换。
    """
    # [-1,1] -> [0,1]
    ndata = (ndata + 1) / 2
    # [0,1] -> [min,max]
    data = ndata * (stats["max"] - stats["min"]) + stats["min"]
    return data


def get_delta(
    actions,  # 绝对动作序列，形状 [B, T, D]
):
    """
    将绝对位置动作转换为增量动作（相邻时刻做差）。

    这样更适配扩散模型学习“局部变化”而非绝对坐标。
    """
    # 在起点前补零，保证 delta 序列长度与原动作序列一致
    ex_actions = np.concatenate([np.zeros((actions.shape[0], 1, actions.shape[-1])), actions], axis=1)
    delta = ex_actions[:, 1:] - ex_actions[:, :-1]
    return delta


def get_action(
    diffusion_output,  # 扩散模型输出的归一化增量动作，形状 [B, T, 2]（或可 reshape 到该形状）
    action_stats=ACTION_STATS,  # 动作统计量（用于反归一化）
):
    """
    将扩散模型输出（归一化增量）还原为绝对动作轨迹。

    步骤：
    1. reshape 到 [B, T, 2]；
    2. 反归一化；
    3. 累积求和得到绝对轨迹。
    """
    # 先保留设备，后续 numpy 计算完成再转回 torch
    device = diffusion_output.device
    ndeltas = diffusion_output.reshape(diffusion_output.shape[0], -1, 2)
    ndeltas = to_numpy(ndeltas)
    ndeltas = unnormalize_data(ndeltas, action_stats)
    actions = np.cumsum(ndeltas, axis=1)
    return from_numpy(actions).to(device)

def model_output(
    model: nn.Module,  # NoMaD 模型（支持 vision_encoder/noise_pred_net/dist_pred_net）
    noise_scheduler: DDPMScheduler,  # DDPM 噪声调度器
    batch_obs_images: torch.Tensor,  # 观测图像批次
    batch_goal_images: torch.Tensor,  # 目标图像批次
    pred_horizon: int,  # 预测轨迹长度 T
    action_dim: int,  # 动作维度 D（常见为 2）
    num_samples: int,  # 每个样本采样的轨迹条数
    device: torch.device,  # 推理设备
):
    """
    使用 NoMaD 扩散模型生成动作与距离预测。

    参数:
        model: NoMaD 模型（支持 "vision_encoder" / "noise_pred_net" / "dist_pred_net" 三个子调用）。
        noise_scheduler: DDPM 噪声调度器。
        batch_obs_images: 观测图像批次。
        batch_goal_images: 目标图像批次。
        pred_horizon: 预测轨迹长度 T。
        action_dim: 动作维度（通常为 2）。
        num_samples: 每个输入采样多少条动作轨迹。
        device: 推理设备。

    返回:
        dict:
        - uc_actions: 无条件动作（探索模式）[B*num_samples, T, 2]
        - gc_actions: 目标条件动作（导航模式）[B*num_samples, T, 2]
        - gc_distance: 目标条件距离预测 [B*num_samples, 1]

    说明：
    - 两个分支都从高斯噪声出发做完整 DDPM 反向过程。
    - 差异只在条件向量：UC 不看目标、GC 使用目标。
    """
    # ========== 无条件分支（目标被 mask）==========
    # 目标 mask 为 1 时，视觉编码器学习“仅根据观测历史进行动作生成”。
    goal_mask = torch.ones((batch_goal_images.shape[0],), dtype=torch.long, device=device)
    obs_cond = model(
        "vision_encoder",
        obs_img=batch_obs_images,
        goal_img=batch_goal_images,
        input_goal_mask=goal_mask,
    )
    # repeat_interleave: 为同一观测采样多条候选轨迹
    obs_cond = obs_cond.repeat_interleave(num_samples, dim=0)

    diffusion_output = torch.randn(
        (len(obs_cond), pred_horizon, action_dim),
        device=device,
    )
    # 逐步去噪：从高斯噪声动作开始，经过多个时间步恢复轨迹
    for k in noise_scheduler.timesteps[:]:
        noise_pred = model(
            "noise_pred_net",
            sample=diffusion_output,
            timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
            global_cond=obs_cond,
        )
        diffusion_output = noise_scheduler.step(
            model_output=noise_pred,
            timestep=k,
            sample=diffusion_output,
        ).prev_sample

    uc_actions = get_action(diffusion_output, ACTION_STATS)

    # ========== 目标条件分支（目标可见）==========
    # 目标 mask 为 0 时，编码器显式融合目标图像条件。
    no_mask = torch.zeros((batch_goal_images.shape[0],), dtype=torch.long, device=device)
    obsgoal_cond = model(
        "vision_encoder",
        obs_img=batch_obs_images,
        goal_img=batch_goal_images,
        input_goal_mask=no_mask,
    )
    obsgoal_cond = obsgoal_cond.repeat_interleave(num_samples, dim=0)

    diffusion_output = torch.randn(
        (len(obsgoal_cond), pred_horizon, action_dim),
        device=device,
    )
    # GC 分支重复同样去噪流程，但条件向量包含目标信息，理论上更“有方向性”
    for k in noise_scheduler.timesteps[:]:
        noise_pred = model(
            "noise_pred_net",
            sample=diffusion_output,
            timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
            global_cond=obsgoal_cond,
        )
        diffusion_output = noise_scheduler.step(
            model_output=noise_pred,
            timestep=k,
            sample=diffusion_output,
        ).prev_sample

    gc_actions = get_action(diffusion_output, ACTION_STATS)
    # 距离头输入展平特征，用于回归“当前观测到目标”的距离估计
    gc_distance = model("dist_pred_net", obsgoal_cond=obsgoal_cond.flatten(start_dim=1))

    return {
        "uc_actions": uc_actions,
        "gc_actions": gc_actions,
        "gc_distance": gc_distance,
    }
def visualize_diffusion_action_distribution(
    ema_model: nn.Module,  # EMA 平滑后的 NoMaD 模型
    noise_scheduler: DDPMScheduler,  # DDPM 噪声调度器
    batch_obs_images: torch.Tensor,  # 模型输入观测图像批次
    batch_goal_images: torch.Tensor,  # 模型输入目标图像批次
    batch_viz_obs_images: torch.Tensor,  # 可视化用观测图像批次（通常未归一化）
    batch_viz_goal_images: torch.Tensor,  # 可视化用目标图像批次（通常未归一化）
    batch_action_label: torch.Tensor,  # 真实动作轨迹标签
    batch_distance_labels: torch.Tensor,  # 真实距离标签
    batch_goal_pos: torch.Tensor,  # 目标点坐标
    device: torch.device,  # 推理设备
    eval_type: str,  # 日志模式（train/eval_xxx）
    project_folder: str,  # 项目输出目录
    epoch: int,  # 当前 epoch 编号
    num_images_log: int,  # 本次可视化样本数
    num_samples: int = 30,  # 每个样本采样的动作轨迹数量
    use_wandb: bool = True,  # 是否上传可视化到 wandb
):
    """
    可视化 NoMaD 扩散模型的动作采样分布。

    图中会对比：
    - 无条件动作（红色，探索模式）
    - 目标条件动作（绿色，导航模式）
    - 真实动作（洋红色）

    该可视化用于快速判断：
    - 采样分布是否发散；
    - GC 轨迹是否明显朝向目标点；
    - 模型不确定性是否过高（轨迹散布过宽）。

    输出结构：
    - 左图：轨迹分布（红=UC，绿=GC，洋红=GT）；
    - 中图：观测图像；
    - 右图：目标图像 + 距离标签与预测统计。
    """
    visualize_path = os.path.join(
        project_folder,
        "visualize",
        eval_type,
        f"epoch{epoch}",
        "action_sampling_prediction",
    )
    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path)

    max_batch_size = batch_obs_images.shape[0]

    num_images_log = min(
        num_images_log,
        batch_obs_images.shape[0],
        batch_goal_images.shape[0],
        batch_action_label.shape[0],
        batch_goal_pos.shape[0],
    )
    batch_obs_images = batch_obs_images[:num_images_log]
    batch_goal_images = batch_goal_images[:num_images_log]
    batch_action_label = batch_action_label[:num_images_log]
    batch_goal_pos = batch_goal_pos[:num_images_log]

    wandb_list = []

    pred_horizon = batch_action_label.shape[1]
    action_dim = batch_action_label.shape[2]

    # 分批采样，避免一次性显存占用过大。
    batch_obs_images_list = torch.split(batch_obs_images, max_batch_size, dim=0)
    batch_goal_images_list = torch.split(batch_goal_images, max_batch_size, dim=0)

    uc_actions_list = []
    gc_actions_list = []
    gc_distances_list = []

    for obs, goal in zip(batch_obs_images_list, batch_goal_images_list):
        model_output_dict = model_output(
            ema_model,
            noise_scheduler,
            obs,
            goal,
            pred_horizon,
            action_dim,
            num_samples,
            device,
        )
        uc_actions_list.append(to_numpy(model_output_dict["uc_actions"]))
        gc_actions_list.append(to_numpy(model_output_dict["gc_actions"]))
        gc_distances_list.append(to_numpy(model_output_dict["gc_distance"]))

    uc_actions_list = np.concatenate(uc_actions_list, axis=0)
    gc_actions_list = np.concatenate(gc_actions_list, axis=0)
    gc_distances_list = np.concatenate(gc_distances_list, axis=0)

    # 按观测样本拆分，便于逐样本绘图
    uc_actions_list = np.split(uc_actions_list, num_images_log, axis=0)
    gc_actions_list = np.split(gc_actions_list, num_images_log, axis=0)
    gc_distances_list = np.split(gc_distances_list, num_images_log, axis=0)

    gc_distances_avg = [np.mean(dist) for dist in gc_distances_list]
    gc_distances_std = [np.std(dist) for dist in gc_distances_list]

    assert len(uc_actions_list) == len(gc_actions_list) == num_images_log

    np_distance_labels = to_numpy(batch_distance_labels)

    for i in range(num_images_log):
        # 每个样本绘制 1x3 子图：轨迹、观测、目标
        fig, ax = plt.subplots(1, 3)
        uc_actions = uc_actions_list[i]
        gc_actions = gc_actions_list[i]
        action_label = to_numpy(batch_action_label[i])

        traj_list = np.concatenate([uc_actions, gc_actions, action_label[None]], axis=0)
        traj_colors = ["red"] * len(uc_actions) + ["green"] * len(gc_actions) + ["magenta"]
        traj_alphas = [0.1] * (len(uc_actions) + len(gc_actions)) + [1.0]

        # 起点固定为机器人坐标原点，目标点来自数据集标签
        point_list = [np.array([0, 0]), to_numpy(batch_goal_pos[i])]
        point_colors = ["green", "red"]
        point_alphas = [1.0, 1.0]

        plot_trajs_and_points(
            ax[0],
            traj_list,
            point_list,
            traj_colors,
            point_colors,
            traj_labels=None,
            point_labels=None,
            quiver_freq=0,
            traj_alphas=traj_alphas,
            point_alphas=point_alphas,
        )

        obs_image = to_numpy(batch_viz_obs_images[i])
        goal_image = to_numpy(batch_viz_goal_images[i])
        # matplotlib 需要 HWC，张量一般是 CHW
        obs_image = np.moveaxis(obs_image, 0, -1)
        goal_image = np.moveaxis(goal_image, 0, -1)
        ax[1].imshow(obs_image)
        ax[2].imshow(goal_image)

        ax[0].set_title("diffusion action predictions")
        ax[1].set_title("observation")
        ax[2].set_title(
            f"goal: label={np_distance_labels[i]} gc_dist={gc_distances_avg[i]:.2f}±{gc_distances_std[i]:.2f}"
        )

        fig.set_size_inches(18.5, 10.5)

        save_path = os.path.join(visualize_path, f"sample_{i}.png")
        plt.savefig(save_path)
        wandb_list.append(wandb.Image(save_path))
        plt.close(fig)

    if len(wandb_list) > 0 and use_wandb:
        wandb.log({f"{eval_type}_action_samples": wandb_list}, commit=False)






