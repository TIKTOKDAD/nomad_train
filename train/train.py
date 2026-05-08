# 导入标准库
import os
import wandb  # Weights & Biases 实验跟踪工具
import argparse
import numpy as np
import yaml
import time
import pdb

# 导入PyTorch相关库
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam, AdamW
from torchvision import transforms
import torch.backends.cudnn as cudnn
from warmup_scheduler import GradualWarmupScheduler  # 学习率预热调度器

# 导入扩散模型相关组件
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler

"""
导入模型定义
支持三种模型架构：GNM、ViNT、NoMaD
"""
from vint_train.models.gnm.gnm import GNM  # General Navigation Model
from vint_train.models.vint.vint import ViNT  # Visual Navigation Transformer
from vint_train.models.vint.vit import ViT  # Vision Transformer
from vint_train.models.nomad.nomad import NoMaD, DenseNetwork  # NoMaD模型和密集网络
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn  # NoMaD-ViNT变体
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D  # 条件UNet1D用于扩散策略

# 导入数据集和训练循环
from vint_train.data.vint_dataset import ViNT_Dataset
from vint_train.training.train_eval_loop import (
    train_eval_loop,  # 标准训练评估循环（用于GNM和ViNT）
    train_eval_loop_nomad,  # NoMaD专用训练评估循环
    load_model,  # 模型加载函数
)


def main(config):
    """
    主训练函数
    
    参数:
        config: 配置字典，包含所有训练超参数和设置
    """
    # 验证距离和动作的分类范围配置是否合理
    assert config["distance"]["min_dist_cat"] < config["distance"]["max_dist_cat"]
    assert config["action"]["min_dist_cat"] < config["action"]["max_dist_cat"]

    # ========== GPU设备配置 ==========
    if torch.cuda.is_available():
        # 设置CUDA设备顺序为PCI总线ID
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # 如果未指定GPU ID，默认使用GPU 0
        if "gpu_ids" not in config:
            config["gpu_ids"] = [0]
        elif type(config["gpu_ids"]) == int:
            config["gpu_ids"] = [config["gpu_ids"]]
        # 设置可见的CUDA设备
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(x) for x in config["gpu_ids"]]
        )
        print("Using cuda devices:", os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        print("Using cpu")

    # 获取第一个GPU ID并创建设备对象
    first_gpu_id = config["gpu_ids"][0]
    device = torch.device(
        f"cuda:{first_gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    # ========== 随机种子设置（确保可复现性） ==========
    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        cudnn.deterministic = True  # 使用确定性算法

    # 启用cudnn自动调优（当输入尺寸固定时可以提升性能）
    cudnn.benchmark = False
    
    # ========== 图像预处理变换 ==========
    # 使用ImageNet统计量进行归一化（与EfficientNet常用输入规范保持一致）
    transform = ([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform = transforms.Compose(transform)

    # ========== 数据加载 ==========
    train_dataset = []  # 存储所有训练数据集
    test_dataloaders = {}  # 存储所有测试数据加载器

    # 设置默认配置值
    if "context_type" not in config:
        config["context_type"] = "temporal"  # 上下文类型：时序上下文

    if "clip_goals" not in config:
        config["clip_goals"] = False  # 是否裁剪目标（当前训练入口未直接使用该项）

    # 遍历配置中的所有数据集
    for dataset_name in config["datasets"]:
        data_config = config["datasets"][dataset_name]
        # 为每个数据集设置默认参数
        if "negative_mining" not in data_config:
            data_config["negative_mining"] = True  # 负样本相关配置项（采样逻辑见ViNT_Dataset实现）
        if "goals_per_obs" not in data_config:
            data_config["goals_per_obs"] = 1  # 每个观测对应的目标数量
        if "end_slack" not in data_config:
            data_config["end_slack"] = 0  # 轨迹末端松弛量：忽略/裁掉末尾若干时间步，避免学习到末端不稳定数据
        if "waypoint_spacing" not in data_config:
            data_config["waypoint_spacing"] = 1  # 路径点间隔

        # 为训练集和测试集分别创建数据集对象
        for data_split_type in ["train", "test"]:
            if data_split_type in data_config:
                    # 创建ViNT数据集实例
                    dataset = ViNT_Dataset(
                        data_folder=data_config["data_folder"],  # 数据文件夹路径
                        data_split_folder=data_config[data_split_type],  # 数据划分文件夹
                        dataset_name=dataset_name,  # 数据集名称
                        image_size=config["image_size"],  # 图像尺寸
                        waypoint_spacing=data_config["waypoint_spacing"],  # 路径点间隔
                        min_dist_cat=config["distance"]["min_dist_cat"],  # 最小距离类别（航点步）
                        max_dist_cat=config["distance"]["max_dist_cat"],  # 最大距离类别（航点步）
                        min_action_distance=config["action"]["min_dist_cat"],  # 最小动作距离
                        max_action_distance=config["action"]["max_dist_cat"],  # 最大动作距离
                        negative_mining=data_config["negative_mining"],  # 负样本挖掘
                        len_traj_pred=config["len_traj_pred"],  # 预测轨迹长度
                        learn_angle=config["learn_angle"],  # 是否学习角度
                        context_size=config["context_size"],  # 上下文大小
                        context_type=config["context_type"],  # 上下文类型
                        end_slack=data_config["end_slack"],  # 末端松弛
                        goals_per_obs=data_config["goals_per_obs"],  # 每个观测的目标数
                        normalize=config["normalize"],  # 是否归一化
                        goal_type=config["goal_type"],  # 目标类型
                    )
                    # ==================== 数据集注册逻辑 ====================
                    # 将当前构建好的 dataset 放入与 split 对应的容器中：
                    # 1) train split -> 放入 train_dataset 列表，后续通过 ConcatDataset 合并。
                    # 2) 非 train split（当前循环中主要是 test）-> 放入 test_dataloaders 字典。
                    if data_split_type == "train":
                        # 训练集使用 append 追加，保留多个数据源的子数据集。
                        # 这样后面可以把 [ds1_train, ds2_train, ...] 合并为一个总训练集。
                        train_dataset.append(dataset)
                    else:
                        # 为评估/测试集构造唯一键，避免不同数据源之间相互覆盖。
                        # 键格式："{dataset_name}_{data_split_type}"，例如 "recon_test"。
                        dataset_type = f"{dataset_name}_{data_split_type}"

                        # 如果该键还不存在，先初始化占位。
                        # 注意：下一行会直接写入 dataset，因此这里主要是增强可读性与显式性。
                        if dataset_type not in test_dataloaders:
                            test_dataloaders[dataset_type] = {}

                        # 注册该 split 对应的数据集对象。
                        # 最终 test_dataloaders 形态示例：
                        # {
                        #   "datasetA_test": <ViNT_Dataset>,
                        #   "datasetB_test": <ViNT_Dataset>,
                        # }
                        # 后续会遍历该字典，把每个 dataset 包装成 DataLoader。
                        test_dataloaders[dataset_type] = dataset

    # ========== 合并数据集并创建数据加载器 ==========
    # 训练：1个合并后的train_loader
    # 测试：多个按数据集分开的test_loader
    # 将来自不同数据集的训练子集合并
    train_dataset = ConcatDataset(train_dataset)

    # 创建训练数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],  # 批次大小
        shuffle=True,  # 打乱数据
        num_workers=config["num_workers"],  # 数据加载的工作进程数
        drop_last=False,  # 不丢弃最后一个不完整的批次
        persistent_workers=True,  # 保持工作进程活跃（提高效率）
    )

    # 设置评估批次大小（如果未指定，使用训练批次大小）
    if "eval_batch_size" not in config:
        config["eval_batch_size"] = config["batch_size"]

    # 为每个测试数据集创建数据加载器
    for dataset_type, dataset in test_dataloaders.items():
        test_dataloaders[dataset_type] = DataLoader(
            dataset,
            batch_size=config["eval_batch_size"],
            shuffle=True,
            num_workers=0,  # 测试时不使用多进程
            drop_last=False,
        )

    # ========== 创建模型 ==========
    if config["model_type"] == "gnm":
        # General Navigation Model（通用导航模型）
        model = GNM(
            config["context_size"],  # 上下文大小
            config["len_traj_pred"],  # 预测轨迹长度
            config["learn_angle"],  # 是否学习角度
            config["obs_encoding_size"],  # 观测编码维度
            config["goal_encoding_size"],  # 目标编码维度
        )
    elif config["model_type"] == "vint":
        # Visual Navigation Transformer（视觉导航Transformer）
        model = ViNT(
            context_size=config["context_size"],  # 上下文大小
            len_traj_pred=config["len_traj_pred"],  # 预测轨迹长度
            learn_angle=config["learn_angle"],  # 是否学习角度
            obs_encoder=config["obs_encoder"],  # 观测编码器类型
            obs_encoding_size=config["obs_encoding_size"],  # 观测编码维度
            late_fusion=config["late_fusion"],  # 是否使用后期融合
            mha_num_attention_heads=config["mha_num_attention_heads"],  # 多头注意力头数
            mha_num_attention_layers=config["mha_num_attention_layers"],  # 注意力层数
            mha_ff_dim_factor=config["mha_ff_dim_factor"],  # 前馈网络维度因子
        )
    elif config["model_type"] == "nomad":
        # NoMaD模型（基于扩散策略的导航模型）
        # 根据配置选择视觉编码器
        if config["vision_encoder"] == "nomad_vint":
            # NoMaD-ViNT视觉编码器
            vision_encoder = NoMaD_ViNT(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
            )
            # 将批归一化替换为组归一化（更适合小批次训练）
            vision_encoder = replace_bn_with_gn(vision_encoder)
        elif config["vision_encoder"] == "vib": 
            # ViB视觉编码器
            vision_encoder = ViB(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        elif config["vision_encoder"] == "vit": 
            # Vision Transformer编码器
            vision_encoder = ViT(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                image_size=config["image_size"],
                patch_size=config["patch_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        else: 
            raise ValueError(f"Vision encoder {config['vision_encoder']} not supported")
        
        # 噪声预测网络（用于扩散策略）
        noise_pred_net = ConditionalUnet1D(
                input_dim=2,  # 输入维度（x, y坐标）
                global_cond_dim=config["encoding_size"],  # 全局条件维度
                down_dims=config["down_dims"],  # 下采样维度
                cond_predict_scale=config["cond_predict_scale"],  # 条件预测缩放
            )
        # 距离预测网络
        dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])
        
        # 组装NoMaD模型
        model = NoMaD(
            vision_encoder=vision_encoder,
            noise_pred_net=noise_pred_net,
            dist_pred_net=dist_pred_network,
        )

        # 创建DDPM噪声调度器（用于扩散过程）
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=config["num_diffusion_iters"],  # 扩散迭代次数
            beta_schedule='squaredcos_cap_v2',  # beta调度策略
            clip_sample=True,  # 裁剪样本
            prediction_type='epsilon'  # 预测类型：噪声
        )
    else:
        raise ValueError(f"Model {config['model']} not supported")

    # ========== 梯度裁剪设置 ==========
    if config["clipping"]:
        print("Clipping gradients to", config["max_norm"])
        # 为所有需要梯度的参数注册梯度裁剪钩子
        for p in model.parameters():
            if not p.requires_grad:
                continue
            # 将梯度裁剪到[-max_norm, max_norm]范围内
            p.register_hook(
                lambda grad: torch.clamp(
                    grad, -1 * config["max_norm"], config["max_norm"]
                )
            )

    # ========== 优化器配置 ==========
    lr = float(config["lr"])  # 学习率
    config["optimizer"] = config["optimizer"].lower()
    if config["optimizer"] == "adam":
        # Adam优化器（自适应矩估计）
        optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
    elif config["optimizer"] == "adamw":
        # AdamW优化器（带权重衰减的Adam）
        optimizer = AdamW(model.parameters(), lr=lr)
    elif config["optimizer"] == "sgd":
        # 随机梯度下降优化器
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Optimizer {config['optimizer']} not supported")

    # ========== 学习率调度器配置 ==========
    scheduler = None
    if config["scheduler"] is not None:
        config["scheduler"] = config["scheduler"].lower()
        if config["scheduler"] == "cosine":
            # 余弦退火调度器（学习率按余弦曲线衰减）
            print("Using cosine annealing with T_max", config["epochs"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config["epochs"]
            )
        elif config["scheduler"] == "cyclic":
            # 循环学习率调度器（学习率在最小值和最大值之间循环）
            print("Using cyclic LR with cycle", config["cyclic_period"])
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=lr / 10.,  # 基础学习率
                max_lr=lr,  # 最大学习率
                step_size_up=config["cyclic_period"] // 2,  # 上升步数
                cycle_momentum=False,  # 不循环动量
            )
        elif config["scheduler"] == "plateau":
            # 平台调度器（当指标停止改善时降低学习率）
            print("Using ReduceLROnPlateau")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=config["plateau_factor"],  # 学习率衰减因子
                patience=config["plateau_patience"],  # 容忍的epoch数
                verbose=True,
            )
        else:
            raise ValueError(f"Scheduler {config['scheduler']} not supported")

        # 如果启用预热，在调度器前添加预热阶段
        if config["warmup"]:
            print("Using warmup scheduler")
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=1,  # 预热结束时的学习率倍数
                total_epoch=config["warmup_epochs"],  # 预热的epoch数
                after_scheduler=scheduler,  # 预热后使用的调度器
            )

    # ========== 模型加载和多GPU设置 ==========
    current_epoch = 0  # 当前训练轮次
    if "load_run" in config:
        # 从之前的训练运行中加载模型
        load_project_folder = os.path.join("logs", config["load_run"])
        print("Loading model from ", load_project_folder)
        latest_path = os.path.join(load_project_folder, "latest.pth")
        latest_checkpoint = torch.load(latest_path)
        load_model(model, config["model_type"], latest_checkpoint)
        # 如果检查点中包含epoch信息，从下一个epoch继续训练
        if "epoch" in latest_checkpoint:
            current_epoch = latest_checkpoint["epoch"] + 1

    # 多GPU并行训练设置
    if len(config["gpu_ids"]) > 1:
        model = nn.DataParallel(model, device_ids=config["gpu_ids"])
    model = model.to(device)

    # 在数据并行之后加载优化器和调度器状态（如果有）
    if "load_run" in config:
        if "optimizer" in latest_checkpoint:
            optimizer.load_state_dict(latest_checkpoint["optimizer"].state_dict())
        if scheduler is not None and "scheduler" in latest_checkpoint:
            scheduler.load_state_dict(latest_checkpoint["scheduler"].state_dict())

    # ========== 开始训练 ==========
    if config["model_type"] == "vint" or config["model_type"] == "gnm": 
        # 使用标准训练评估循环（适用于ViNT和GNM模型）
        train_eval_loop(
            train_model=config["train"],  # 是否训练模型（或仅评估）
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader=train_loader,
            test_dataloaders=test_dataloaders,
            transform=transform,
            epochs=config["epochs"],  # 训练轮次
            device=device,
            project_folder=config["project_folder"],  # 项目保存文件夹
            normalized=config["normalize"],  # 是否归一化
            print_log_freq=config["print_log_freq"],  # 打印日志频率
            image_log_freq=config["image_log_freq"],  # 图像日志频率
            num_images_log=config["num_images_log"],  # 记录的图像数量
            current_epoch=current_epoch,
            learn_angle=config["learn_angle"],  # 是否学习角度
            alpha=config["alpha"],  # 损失函数权重
            use_wandb=config["use_wandb"],  # 是否使用wandb记录
            eval_fraction=config["eval_fraction"],  # 评估数据比例
        )
    else:
        # 使用NoMaD专用训练评估循环（基于扩散策略）
        train_eval_loop_nomad(
            train_model=config["train"],
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            noise_scheduler=noise_scheduler,  # 扩散噪声调度器
            train_loader=train_loader,
            test_dataloaders=test_dataloaders,
            transform=transform,
            goal_mask_prob=config["goal_mask_prob"],  # 目标掩码概率
            epochs=config["epochs"],
            device=device,
            project_folder=config["project_folder"],
            print_log_freq=config["print_log_freq"],
            wandb_log_freq=config["wandb_log_freq"],  # wandb日志频率
            image_log_freq=config["image_log_freq"],
            num_images_log=config["num_images_log"],
            current_epoch=current_epoch,
            alpha=float(config["alpha"]),
            use_wandb=config["use_wandb"],
            eval_fraction=config["eval_fraction"],
            eval_freq=config["eval_freq"],  # 评估频率
        )

    print("FINISHED TRAINING")


if __name__ == "__main__":
    os.environ["WANDB_API_KEY"] = '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'
    # 设置多进程启动方法为spawn（适用于CUDA）
    torch.multiprocessing.set_start_method("spawn")

    # ========== 命令行参数解析 ==========
    parser = argparse.ArgumentParser(description="Visual Navigation Transformer")

    # 项目设置
    parser.add_argument(
        "--config",
        "-c",
        default="config/nomad_retrain.yaml",
        type=str,
        help="Path to the config file in train_config folder",
    )
    args = parser.parse_args()

    # ========== 配置文件加载 ==========
    # 加载默认配置
    with open("config/defaults.yaml", "r",encoding="utf-8") as f:
        default_config = yaml.safe_load(f)

    config = default_config

    # 加载用户指定的配置并覆盖默认配置
    with open(args.config, "r",encoding="utf-8") as f:
        user_config = yaml.safe_load(f)

    config.update(user_config)

    # ========== 创建项目文件夹 ==========
    # 为运行名称添加时间戳
    config["run_name"] += "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
    config["project_folder"] = os.path.join(
        "logs", config["project_name"], config["run_name"]
    )
    # 创建项目文件夹（如果已存在会报错，避免覆盖旧项目）
    os.makedirs(
        config[
            "project_folder"
        ],
    )

    # ========== Weights & Biases初始化 ==========
    if config["use_wandb"]:
        wandb.login()
        wandb.init(
            project=config["project_name"],
            settings=wandb.Settings(start_method="fork"),
            entity="1948627929-university-of-central-florida", # TODO: 修改为你的wandb实体名称
        )
        wandb.save(args.config, policy="now")  # 保存配置文件
        wandb.run.name = config["run_name"]
        # 将训练配置更新到wandb
        if wandb.run:
            wandb.config.update(config)

    print(config)
    main(config)
