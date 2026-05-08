# 导入实验跟踪和工具库
import wandb  # Weights & Biases实验跟踪
import os
import numpy as np
from typing import List, Optional, Dict
from prettytable import PrettyTable  # 用于美化表格输出

# 导入训练和评估工具函数
from vint_train.training.train_utils import train, evaluate  # 标准训练评估（GNM/ViNT）
from vint_train.training.train_utils import train_nomad, evaluate_nomad  # NoMaD专用训练评估

# 导入PyTorch相关库
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms

# 导入扩散模型相关组件
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler  # DDPM噪声调度器
from diffusers.training_utils import EMAModel  # 指数移动平均模型（用于稳定训练）

def train_eval_loop(
    train_model: bool,  # 是否进行训练（False 时仅评估）
    model: nn.Module,  # 待训练/评估模型
    optimizer: Adam,  # 优化器
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],  # 学习率调度器（可选）
    dataloader: DataLoader,  # 训练数据加载器
    test_dataloaders: Dict[str, DataLoader],  # 各测试集的数据加载器
    transform: transforms,  # 图像变换
    epochs: int,  # 本轮运行的 epoch 数
    device: torch.device,  # 运行设备（CPU/GPU）
    project_folder: str,  # 日志与检查点目录
    normalized: bool,  # 是否对动作空间归一化
    wandb_log_freq: int = 10,  # wandb 标量日志频率
    print_log_freq: int = 100,  # 控制台打印频率
    image_log_freq: int = 1000,  # wandb 图像日志频率
    num_images_log: int = 8,  # 每次记录的图像数量
    current_epoch: int = 0,  # 起始 epoch（用于断点续训）
    alpha: float = 0.5,  # 距离损失与动作损失权重
    learn_angle: bool = True,  # 是否学习角度
    use_wandb: bool = True,  # 是否启用 wandb 记录
    eval_fraction: float = 0.25,  # 评估时采样比例
):
    """
    训练和评估模型多个epoch（用于ViNT或GNM模型）

    参数:
        train_model: 是否训练模型（False则仅评估）
        model: 要训练的模型
        optimizer: 优化器
        scheduler: 学习率调度器（可选）
        dataloader: 训练数据集的数据加载器
        test_dataloaders: 测试数据集的数据加载器字典
        transform: 应用于图像的变换
        epochs: 训练的epoch数量
        device: 训练设备（CPU或GPU）
        project_folder: 保存检查点和日志的文件夹
        normalized: 是否归一化动作空间
        wandb_log_freq: 记录到wandb的频率
        print_log_freq: 打印到控制台的频率
        image_log_freq: 记录图像到wandb的频率
        num_images_log: 记录到wandb的图像数量
        current_epoch: 开始训练的epoch（用于恢复训练）
        alpha: 距离损失和动作损失之间的权衡系数（0-1之间，具体缩放见train_utils）
        learn_angle: 是否学习角度
        use_wandb: 是否记录到wandb
        eval_fraction: 用于评估的训练数据比例
    """
    # 确保alpha在有效范围内
    assert 0 <= alpha <= 1
    # 最新检查点的保存路径
    latest_path = os.path.join(project_folder, f"latest.pth")

    # ========== 主训练循环 ==========
    for epoch in range(current_epoch, current_epoch + epochs):
        # 训练阶段
        if train_model:
            print(
            f"Start ViNT Training Epoch {epoch}/{current_epoch + epochs - 1}"
            )
            train(
                model=model,
                optimizer=optimizer,
                dataloader=dataloader,
                transform=transform,
                device=device,
                project_folder=project_folder,
                normalized=normalized,
                epoch=epoch,
                alpha=alpha,
                learn_angle=learn_angle,
                print_log_freq=print_log_freq,
                wandb_log_freq=wandb_log_freq,
                image_log_freq=image_log_freq,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
            )

        # ========== 评估阶段 ==========
        avg_total_test_loss = []  # 存储所有测试集的平均损失
        # 遍历所有测试数据集
        for dataset_type in test_dataloaders:
            print(
                f"Start {dataset_type} ViNT Testing Epoch {epoch}/{current_epoch + epochs - 1}"
            )
            loader = test_dataloaders[dataset_type]

            # 在当前测试集上评估模型
            test_dist_loss, test_action_loss, total_eval_loss = evaluate(
                eval_type=dataset_type,
                model=model,
                dataloader=loader,
                transform=transform,
                device=device,
                project_folder=project_folder,
                normalized=normalized,
                epoch=epoch,
                alpha=alpha,
                learn_angle=learn_angle,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
                eval_fraction=eval_fraction,
            )

            avg_total_test_loss.append(total_eval_loss)

        # ========== 保存检查点 ==========
        checkpoint = {
            "epoch": epoch,
            "model": model,
            "optimizer": optimizer,
            "avg_total_test_loss": np.mean(avg_total_test_loss),  # 所有测试集的平均损失
            "scheduler": scheduler
        }
        # 记录平均评估损失（空提交，等待后续数据）
        wandb.log({}, commit=False)

        # ========== 学习率调度 ==========
        if scheduler is not None:
            # 根据调度器类型调用不同的step方法
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # ReduceLROnPlateau需要传入指标值
                scheduler.step(np.mean(avg_total_test_loss))
            else:
                # 其他调度器直接调用step
                scheduler.step()
        
        # 记录平均测试损失和当前学习率
        wandb.log({
            "avg_total_test_loss": np.mean(avg_total_test_loss),
            "lr": optimizer.param_groups[0]["lr"],
        }, commit=False)

        # 保存检查点文件
        numbered_path = os.path.join(project_folder, f"{epoch}.pth")
        torch.save(checkpoint, latest_path)  # 保存最新检查点
        torch.save(checkpoint, numbered_path)  # 保存带编号的检查点（每个epoch一个）

    # 刷新最后一组评估日志
    wandb.log({})
    print()

def train_eval_loop_nomad(
    train_model: bool,  # 是否进行训练（False 时仅评估）
    model: nn.Module,  # NoMaD 模型
    optimizer: Adam,  # 优化器
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,  # 学习率调度器
    noise_scheduler: DDPMScheduler,  # 扩散噪声调度器
    train_loader: DataLoader,  # 训练数据加载器
    test_dataloaders: Dict[str, DataLoader],  # 各测试集的数据加载器
    transform: transforms,  # 图像变换
    goal_mask_prob: float,  # 目标 token mask 概率
    epochs: int,  # 本轮运行的 epoch 数
    device: torch.device,  # 运行设备（CPU/GPU）
    project_folder: str,  # 日志与检查点目录
    print_log_freq: int = 100,  # 控制台打印频率
    wandb_log_freq: int = 10,  # wandb 标量日志频率
    image_log_freq: int = 1000,  # wandb 图像日志频率
    num_images_log: int = 8,  # 每次记录的图像数量
    current_epoch: int = 0,  # 起始 epoch（用于断点续训）
    alpha: float = 1e-4,  # 距离损失与扩散损失权重
    use_wandb: bool = True,  # 是否启用 wandb 记录
    eval_fraction: float = 0.25,  # 评估时采样比例
    eval_freq: int = 1,  # 每隔多少个 epoch 评估一次
):
    """
    训练和评估NoMaD模型多个epoch（基于扩散策略的导航模型）

    参数:
        train_model: 是否训练模型
        model: 要训练的模型
        optimizer: 优化器
        lr_scheduler: 学习率调度器
        noise_scheduler: 噪声调度器（用于扩散过程）
        train_loader: 训练数据集的数据加载器
        test_dataloaders: 测试数据集的数据加载器字典
        transform: 应用于图像的变换
        goal_mask_prob: 训练期间掩码目标token的概率（用于目标条件训练）
        epochs: 训练的epoch数量
        device: 训练设备（CPU或GPU）
        project_folder: 保存检查点和日志的文件夹
        print_log_freq: 打印到控制台的频率
        wandb_log_freq: 记录到wandb的频率
        image_log_freq: 记录图像到wandb的频率
        num_images_log: 记录到wandb的图像数量
        current_epoch: 开始训练的epoch
        alpha: 距离损失和扩散损失之间的权衡系数
        use_wandb: 是否记录到wandb
        eval_fraction: 用于评估的训练数据比例
        eval_freq: 评估频率（每隔多少个epoch评估一次）
    """
    latest_path = os.path.join(project_folder, f"latest.pth")
    # 创建EMA模型（指数移动平均，用于稳定训练和提升性能）
    ema_model = EMAModel(model=model, power=0.75)
    
    # ========== NoMaD主训练循环 ==========
    for epoch in range(current_epoch, current_epoch + epochs):
        # 训练阶段
        if train_model:
            print(
            f"Start ViNT DP Training Epoch {epoch}/{current_epoch + epochs - 1}"
            )
            # 使用NoMaD专用训练函数（基于扩散策略）
            train_nomad(
                model=model,
                ema_model=ema_model,  # EMA模型用于稳定训练
                optimizer=optimizer,
                dataloader=train_loader,
                transform=transform,
                device=device,
                noise_scheduler=noise_scheduler,  # 扩散过程的噪声调度器
                goal_mask_prob=goal_mask_prob,  # 目标掩码概率
                project_folder=project_folder,
                epoch=epoch,
                print_log_freq=print_log_freq,
                wandb_log_freq=wandb_log_freq,
                image_log_freq=image_log_freq,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
                alpha=alpha,
            )
            # 更新学习率
            lr_scheduler.step()

        # ========== 保存EMA模型 ==========
        # 保存带编号的EMA模型
        numbered_path = os.path.join(project_folder, f"ema_{epoch}.pth")
        torch.save(ema_model.averaged_model.state_dict(), numbered_path)
        # 保存最新的EMA模型
        numbered_path = os.path.join(project_folder, f"ema_latest.pth")
        print(f"Saved EMA model to {numbered_path}")

        # ========== 保存标准模型 ==========
        numbered_path = os.path.join(project_folder, f"{epoch}.pth")
        torch.save(model.state_dict(), numbered_path)
        torch.save(model.state_dict(), latest_path)
        print(f"Saved model to {numbered_path}")

        # ========== 保存优化器状态 ==========
        numbered_path = os.path.join(project_folder, f"optimizer_{epoch}.pth")
        latest_optimizer_path = os.path.join(project_folder, f"optimizer_latest.pth")
        torch.save(optimizer.state_dict(), latest_optimizer_path)

        # ========== 保存调度器状态 ==========
        numbered_path = os.path.join(project_folder, f"scheduler_{epoch}.pth")
        latest_scheduler_path = os.path.join(project_folder, f"scheduler_latest.pth")
        torch.save(lr_scheduler.state_dict(), latest_scheduler_path)

        # ========== 定期评估 ==========
        # 根据eval_freq决定是否在当前epoch进行评估
        if (epoch + 1) % eval_freq == 0: 
            for dataset_type in test_dataloaders:
                print(
                    f"Start {dataset_type} ViNT DP Testing Epoch {epoch}/{current_epoch + epochs - 1}"
                )
                loader = test_dataloaders[dataset_type]
                # 使用EMA模型进行评估（通常比标准模型更稳定）
                evaluate_nomad(
                    eval_type=dataset_type,
                    ema_model=ema_model,
                    dataloader=loader,
                    transform=transform,
                    device=device,
                    noise_scheduler=noise_scheduler,
                    goal_mask_prob=goal_mask_prob,
                    project_folder=project_folder,
                    epoch=epoch,
                    print_log_freq=print_log_freq,
                    num_images_log=num_images_log,
                    wandb_log_freq=wandb_log_freq,
                    use_wandb=use_wandb,
                    eval_fraction=eval_fraction,
                )
        
        # 记录当前学习率
        wandb.log({
            "lr": optimizer.param_groups[0]["lr"],
        }, commit=False)

        # 更新学习率调度器
        if lr_scheduler is not None:
            lr_scheduler.step()

        # 记录平均评估损失（空提交）
        wandb.log({}, commit=False)

        # 再次记录学习率（确保记录）
        wandb.log({
            "lr": optimizer.param_groups[0]["lr"],
        }, commit=False)

    # 刷新最后一组评估日志
    wandb.log({})
    print()

def load_model(
    model,  # 需要加载权重的模型
    model_type,  # 模型类型（如 "nomad"）
    checkpoint: dict,  # 检查点字典
) -> None:
    """
    从检查点加载模型
    
    参数:
        model: 要加载权重的模型
        model_type: 模型类型（"nomad"或其他）
        checkpoint: 检查点字典
    """
    if model_type == "nomad":
        # NoMaD模型直接使用检查点作为状态字典
        state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    else:
        # 其他模型（GNM/ViNT）从检查点中提取模型
        loaded_model = checkpoint["model"]
        try:
            # 尝试从DataParallel包装的模型中提取状态字典
            state_dict = loaded_model.module.state_dict()
            model.load_state_dict(state_dict, strict=False)
        except AttributeError as e:
            # 如果不是DataParallel模型，直接提取状态字典
            state_dict = loaded_model.state_dict()
            model.load_state_dict(state_dict, strict=False)


def load_ema_model(
    ema_model,  # EMA 模型对象
    state_dict: dict,  # EMA 状态字典
) -> None:
    """
    从状态字典加载EMA模型
    
    参数:
        ema_model: EMA模型对象
        state_dict: 状态字典
    """
    ema_model.load_state_dict(state_dict)


def count_parameters(
    model,  # 待统计参数的模型
):
    """
    统计并打印模型的可训练参数数量
    
    参数:
        model: 要统计的模型
        
    返回:
        total_params: 总参数数量
    """
    # 创建美化的表格用于显示参数信息
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    # 遍历所有命名参数
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue  # 跳过不需要梯度的参数
        params = parameter.numel()  # 获取参数数量
        table.add_row([name, params])
        total_params += params
    # print(table)  # 可选：打印详细的参数表格
    # 打印总参数数量（以百万为单位）
    print(f"Total Trainable Params: {total_params/1e6:.2f}M")
    return total_params
