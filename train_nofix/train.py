
import os
import wandb
import argparse
import numpy as np
import yaml
import time
import pdb
import inspect


import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam, AdamW
from torchvision import transforms
import torch.backends.cudnn as cudnn
from warmup_scheduler import GradualWarmupScheduler


from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler

"""
Model imports.
Supported architectures: GNM, ViNT, and NoMaD.
"""
from vint_train.models.gnm.gnm import GNM  # General Navigation Model
from vint_train.models.vint.vint import ViNT  # Visual Navigation Transformer
from vint_train.models.vint.vit import ViT  # Vision Transformer
from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D


from vint_train.data.vint_dataset import ViNT_Dataset, check_lmdb_cache_ready
from vint_train.training.train_eval_loop import (
    train_eval_loop,
    train_eval_loop_nomad,
    load_model,
)


def _env_int(name, default=0):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _is_torchrun():
    return _env_int("WORLD_SIZE", 1) > 1


def _init_distributed(config):
    world_size = _env_int("WORLD_SIZE", 1)
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)
    distributed = bool(config.get("distributed", True)) and world_size > 1
    if distributed and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() and os.name != "nt" else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    return distributed, rank, local_rank, world_size


def _is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def _barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _broadcast_string(value, src=0):
    if not (dist.is_available() and dist.is_initialized()):
        return value
    obj = [value if dist.get_rank() == src else None]
    dist.broadcast_object_list(obj, src=src)
    return obj[0]


def _loader_kwargs(num_workers, pin_memory, prefetch_factor, persistent_workers):
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


def _ddp_kwargs(config, local_rank, device):
    """根据当前 PyTorch 版本安全组装 DDP 参数，老版本不支持的参数会自动跳过。"""
    kwargs = {
        "device_ids": [local_rank] if device.type == "cuda" else None,
        "output_device": local_rank if device.type == "cuda" else None,
        "find_unused_parameters": bool(config.get("find_unused_parameters", False)),
    }
    optional_kwargs = {
        "static_graph": bool(config.get("ddp_static_graph", False)),
        "gradient_as_bucket_view": bool(config.get("ddp_gradient_as_bucket_view", False)),
        "broadcast_buffers": bool(config.get("ddp_broadcast_buffers", True)),
    }
    supported = inspect.signature(DDP.__init__).parameters
    for key, value in optional_kwargs.items():
        if key in supported:
            kwargs[key] = value
    return kwargs


def _normalize_gpu_ids(config):
    gpu_ids = config.get("gpu_ids", [0])
    if isinstance(gpu_ids, int):
        gpu_ids = [gpu_ids]
    config["gpu_ids"] = [int(gpu_id) for gpu_id in gpu_ids]
    return config["gpu_ids"]


def _configure_cuda_visibility(config):
    gpu_ids = _normalize_gpu_ids(config)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if not _is_torchrun():
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in gpu_ids)
    return gpu_ids


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


def _configured_dataset_splits(config):
    context_type = config.get("context_type", "temporal")
    for dataset_name, data_config in config["datasets"].items():
        waypoint_spacing = data_config.get("waypoint_spacing", 1)
        end_slack = data_config.get("end_slack", 0)
        for data_split_type in ["train", "test"]:
            if data_split_type not in data_config:
                continue
            yield {
                "dataset_name": dataset_name,
                "split": data_split_type,
                "data_split_folder": data_config[data_split_type],
                "waypoint_spacing": waypoint_spacing,
                "end_slack": end_slack,
                "context_type": context_type,
                "context_size": config["context_size"],
                "min_dist_cat": config["distance"]["min_dist_cat"],
                "max_dist_cat": config["distance"]["max_dist_cat"],
            }


def _format_lmdb_cache_errors(errors):
    lines = []
    for item in errors:
        header = f"{item['dataset_name']}:{item['split']} ({item['data_split_folder']})"
        lines.append(f"- {header}")
        for problem in item["problems"]:
            lines.append(f"  * {problem}")
    return "\n".join(lines)


def _check_lmdb_caches_ready(config):
    errors = []
    for split_config in _configured_dataset_splits(config):
        ready, problems = check_lmdb_cache_ready(
            data_split_folder=split_config["data_split_folder"],
            dataset_name=split_config["dataset_name"],
            min_dist_cat=split_config["min_dist_cat"],
            max_dist_cat=split_config["max_dist_cat"],
            waypoint_spacing=split_config["waypoint_spacing"],
            context_type=split_config["context_type"],
            context_size=split_config["context_size"],
            end_slack=split_config["end_slack"],
        )
        if not ready:
            errors.append({
                "dataset_name": split_config["dataset_name"],
                "split": split_config["split"],
                "data_split_folder": split_config["data_split_folder"],
                "problems": problems,
            })
    return errors


def _require_lmdb_ready_before_ddp(config):
    if not bool(config.get("require_lmdb_ready_for_ddp", True)):
        return
    errors = _check_lmdb_caches_ready(config)
    if errors:
        raise RuntimeError(
            "DDP training requires all LMDB caches to be prebuilt and complete.\n"
            "Run this first with a single process:\n"
            "  python train.py -c config/nomad_retrain.yaml --build-lmdb-only\n"
            "If a previous build was interrupted, add:\n"
            "  --rebuild-incomplete-lmdb\n\n"
            f"LMDB problems:\n{_format_lmdb_cache_errors(errors)}"
        )


def main(config):
    """
    Build datasets, model, optimizer, and run the configured train/eval loop.

    Args:
        config: Training configuration dictionary.
    """

    assert config["distance"]["min_dist_cat"] < config["distance"]["max_dist_cat"]
    assert config["action"]["min_dist_cat"] < config["action"]["max_dist_cat"]

    build_lmdb_only = bool(config.get("build_lmdb_only", False))
    if build_lmdb_only and _is_torchrun():
        raise RuntimeError(
            "build_lmdb_only must run as a single process. "
            "Use: python train.py -c config/nomad_retrain.yaml --build-lmdb-only"
        )
    if build_lmdb_only:
        config["distributed"] = False
        config["require_ddp_for_multigpu"] = False
        config["gpu_ids"] = [int(config.get("build_lmdb_gpu_id", 0))]
        config["use_wandb"] = False

    if _is_torchrun() and bool(config.get("distributed", True)):
        # DDP/NCCL 初始化前先检查缓存，避免 rank 等待 rank0 构建 LMDB 导致 10 分钟超时。
        _require_lmdb_ready_before_ddp(config)

    gpu_ids = _configure_cuda_visibility(config)
    if torch.cuda.is_available() and _is_torchrun():
        torch.cuda.set_device(_env_int("LOCAL_RANK", 0))
    distributed, rank, local_rank, world_size = _init_distributed(config)
    is_main_process = _is_main_process()
    config["distributed_active"] = distributed
    config["rank"] = rank
    config["local_rank"] = local_rank
    config["world_size"] = world_size
    # 多卡训练必须通过 torchrun 激活 DDP；否则会退回 DataParallel，效率和主卡负载都更差。
    if (
        torch.cuda.is_available()
        and len(gpu_ids) > 1
        and not distributed
        and bool(config.get("require_ddp_for_multigpu", True))
    ):
        raise RuntimeError(
            "Multi-GPU training requires torchrun/DDP. "
            "Launch from the train directory with: "
            "CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py "
            "-c config/nomad_retrain.yaml"
        )

    if torch.cuda.is_available():
        if distributed:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cuda", 0)
        if is_main_process:
            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", ",".join(str(x) for x in gpu_ids))
            print("Using cuda devices:", visible_devices)
    else:
        device = torch.device("cpu")
        if is_main_process:
            print("Using cpu")

    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S") if is_main_process else ""
    timestamp = _broadcast_string(timestamp)
    config["run_name"] += "_" + timestamp
    config["project_folder"] = os.path.join(
        "logs", config["project_name"], config["run_name"]
    )
    if is_main_process:
        os.makedirs(config["project_folder"], exist_ok=True)
    _barrier()

    if config.get("use_wandb", False) and is_main_process:
        wandb.login()
        wandb.init(
            project=config["project_name"],
            settings=wandb.Settings(start_method="thread"),
            entity=config.get("wandb_entity", "1948627929-university-of-central-florida"),
        )
        if config.get("config_path"):
            wandb.save(config["config_path"], policy="now")
        wandb.run.name = config["run_name"]
        if wandb.run:
            wandb.config.update(config)
    else:
        config["use_wandb"] = False


    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config["seed"])

    deterministic = bool(config.get("deterministic", False))
    cudnn.deterministic = deterministic
    cudnn.benchmark = bool(config.get("cudnn_benchmark", not deterministic))
    if is_main_process:
        print(config)
    


    transform = ([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform = transforms.Compose(transform)


    train_dataset = []
    test_dataloaders = {}
    distributed_eval = bool(config.get("distributed_eval", False))


    if "context_type" not in config:
        config["context_type"] = "temporal"

    if "clip_goals" not in config:
        config["clip_goals"] = False

    lmdb_cache_mode = str(config.get("lmdb_cache_mode", "auto")).lower()
    if build_lmdb_only:
        lmdb_cache_mode = "build"
    elif distributed and bool(config.get("require_lmdb_ready_for_ddp", True)):
        lmdb_cache_mode = "read"

    if distributed and not is_main_process:
        _barrier()


    for dataset_name in config["datasets"]:
        data_config = config["datasets"][dataset_name]

        if "negative_mining" not in data_config:
            data_config["negative_mining"] = True
        if "goals_per_obs" not in data_config:
            data_config["goals_per_obs"] = 1
        if "end_slack" not in data_config:
            data_config["end_slack"] = 0
        if "waypoint_spacing" not in data_config:
            data_config["waypoint_spacing"] = 1


        for data_split_type in ["train", "test"]:
            if distributed and not is_main_process and data_split_type != "train" and not distributed_eval:
                continue
            if data_split_type in data_config:

                    dataset = ViNT_Dataset(
                        data_folder=data_config["data_folder"],
                        data_split_folder=data_config[data_split_type],
                        dataset_name=dataset_name,
                        image_size=config["image_size"],
                        waypoint_spacing=data_config["waypoint_spacing"],
                        min_dist_cat=config["distance"]["min_dist_cat"],
                        max_dist_cat=config["distance"]["max_dist_cat"],
                        min_action_distance=config["action"]["min_dist_cat"],
                        max_action_distance=config["action"]["max_dist_cat"],
                        negative_mining=data_config["negative_mining"],
                        len_traj_pred=config["len_traj_pred"],
                        learn_angle=config["learn_angle"],
                        context_size=config["context_size"],
                        context_type=config["context_type"],
                        end_slack=data_config["end_slack"],
                        goals_per_obs=data_config["goals_per_obs"],
                        normalize=config["normalize"],
                        goal_type=config["goal_type"],
                        lmdb_lock=bool(config.get("lmdb_lock", False)),
                        lmdb_readahead=bool(config.get("lmdb_readahead", False)),
                        lmdb_meminit=bool(config.get("lmdb_meminit", False)),
                        lmdb_max_readers=int(config.get("lmdb_max_readers", 512)),
                        lmdb_cache_mode=lmdb_cache_mode,
                        rebuild_incomplete_lmdb=bool(config.get("rebuild_incomplete_lmdb", False)),
                    )




                    if data_split_type == "train":


                        train_dataset.append(dataset)
                    else:


                        dataset_type = f"{dataset_name}_{data_split_type}"



                        if dataset_type not in test_dataloaders:
                            test_dataloaders[dataset_type] = {}



                        # {
                        #   "datasetA_test": <ViNT_Dataset>,
                        #   "datasetB_test": <ViNT_Dataset>,
                        # }

                        test_dataloaders[dataset_type] = dataset

    if distributed and is_main_process:
        _barrier()



    if build_lmdb_only:
        if is_main_process:
            print("FINISHED LMDB CACHE BUILD")
        _cleanup_distributed()
        return



    train_dataset = ConcatDataset(train_dataset)

    train_sampler = None
    configured_global_batch_size = config.get("global_batch_size")
    global_batch_size = int(
        config["batch_size"] if configured_global_batch_size is None else configured_global_batch_size
    )
    # Keep batch_size as the effective/global batch size for logging and legacy config consumers.
    config["global_batch_size"] = global_batch_size
    config["batch_size"] = global_batch_size

    train_batch_size = global_batch_size
    if distributed:
        if train_batch_size % world_size != 0:
            raise ValueError(
                f"Global batch_size={train_batch_size} must be divisible by world_size={world_size} "
                "to preserve the original effective batch size under DDP."
            )
        train_batch_size = train_batch_size // world_size
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )
    config["per_device_batch_size"] = train_batch_size

    pin_memory = bool(config.get("pin_memory", torch.cuda.is_available()))
    prefetch_factor = int(config.get("prefetch_factor", 2))
    base_num_workers = int(config.get("num_workers", 0))
    configured_workers_per_rank = config.get("num_workers_per_rank")
    # 默认把 num_workers 当作“全局 worker 预算”，DDP 下按 world_size 拆到每个 rank。
    if configured_workers_per_rank is not None:
        train_num_workers = int(configured_workers_per_rank)
    elif distributed and bool(config.get("num_workers_is_global", True)):
        train_num_workers = base_num_workers // world_size
        if base_num_workers > 0:
            train_num_workers = max(1, train_num_workers)
    else:
        train_num_workers = base_num_workers
    config["num_workers_per_rank"] = train_num_workers

    persistent_workers = bool(config.get("persistent_workers", train_num_workers > 0))


    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=False,
        worker_init_fn=_seed_worker,
        **_loader_kwargs(
            train_num_workers,
            pin_memory,
            prefetch_factor,
            persistent_workers,
        ),
    )


    if "eval_batch_size" not in config:
        config["eval_batch_size"] = config["batch_size"]


    base_test_num_workers = int(config.get("test_num_workers", config["num_workers"]))
    if distributed and distributed_eval and bool(config.get("test_num_workers_is_global", True)):
        test_num_workers = base_test_num_workers // world_size
        if base_test_num_workers > 0:
            test_num_workers = max(1, test_num_workers)
    else:
        test_num_workers = base_test_num_workers
    if is_main_process:
        # 训练启动时打印真实生效的吞吐相关配置，方便确认多卡拆分是否符合预期。
        print(
            "DataLoader config: "
            f"global_batch_size={global_batch_size}, "
            f"per_device_batch_size={train_batch_size}, "
            f"eval_batch_size={config['eval_batch_size']}, "
            f"train_num_workers_per_rank={train_num_workers}, "
            f"test_num_workers={test_num_workers}, "
            f"distributed_eval={distributed_eval}, "
            f"pin_memory={pin_memory}"
    )
    for dataset_type, dataset in test_dataloaders.items():
        if distributed and distributed_eval:
            # 分布式评估只改变评估阶段的数据分片；训练数据和训练 loss 不受影响。
            dataset = Subset(dataset, list(range(rank, len(dataset), world_size)))
        test_dataloaders[dataset_type] = DataLoader(
            dataset,
            batch_size=config["eval_batch_size"],
            shuffle=False,
            drop_last=False,
            worker_init_fn=_seed_worker,
            **_loader_kwargs(
                test_num_workers,
                pin_memory,
                prefetch_factor,
                bool(config.get("test_persistent_workers", test_num_workers > 0)),
            ),
        )


    if config["model_type"] == "gnm":

        model = GNM(
            config["context_size"],
            config["len_traj_pred"],
            config["learn_angle"],
            config["obs_encoding_size"],
            config["goal_encoding_size"],
        )
    elif config["model_type"] == "vint":

        model = ViNT(
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],
            obs_encoding_size=config["obs_encoding_size"],
            late_fusion=config["late_fusion"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )
    elif config["model_type"] == "nomad":


        if config["vision_encoder"] == "nomad_vint":

            vision_encoder = NoMaD_ViNT(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
            )

            vision_encoder = replace_bn_with_gn(vision_encoder)
        elif config["vision_encoder"] == "vib": 

            vision_encoder = ViB(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        elif config["vision_encoder"] == "vit": 

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
        

        noise_pred_net = ConditionalUnet1D(
                input_dim=2,
                global_cond_dim=config["encoding_size"],
                down_dims=config["down_dims"],
                cond_predict_scale=config["cond_predict_scale"],
            )

        dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])
        

        model = NoMaD(
            vision_encoder=vision_encoder,
            noise_pred_net=noise_pred_net,
            dist_pred_net=dist_pred_network,
        )


        noise_scheduler = DDPMScheduler(
            num_train_timesteps=config["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
    else:
        raise ValueError(f"Model {config['model']} not supported")


    if config["clipping"]:
        print("Clipping gradients to", config["max_norm"])

        for p in model.parameters():
            if not p.requires_grad:
                continue

            p.register_hook(
                lambda grad: torch.clamp(
                    grad, -1 * config["max_norm"], config["max_norm"]
                )
            )


    lr = float(config["lr"])
    config["optimizer"] = config["optimizer"].lower()
    if config["optimizer"] == "adam":

        optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
    elif config["optimizer"] == "adamw":

        optimizer = AdamW(model.parameters(), lr=lr)
    elif config["optimizer"] == "sgd":

        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Optimizer {config['optimizer']} not supported")


    scheduler = None
    if config["scheduler"] is not None:
        config["scheduler"] = config["scheduler"].lower()
        if config["scheduler"] == "cosine":

            print("Using cosine annealing with T_max", config["epochs"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config["epochs"]
            )
        elif config["scheduler"] == "cyclic":

            print("Using cyclic LR with cycle", config["cyclic_period"])
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=lr / 10.,
                max_lr=lr,
                step_size_up=config["cyclic_period"] // 2,
                cycle_momentum=False,
            )
        elif config["scheduler"] == "plateau":

            print("Using ReduceLROnPlateau")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=config["plateau_factor"],
                patience=config["plateau_patience"],
                verbose=True,
            )
        else:
            raise ValueError(f"Scheduler {config['scheduler']} not supported")


        if config["warmup"]:
            print("Using warmup scheduler")
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=1,
                total_epoch=config["warmup_epochs"],
                after_scheduler=scheduler,
            )


    current_epoch = 0
    load_project_folder = None
    latest_checkpoint = None
    resume_ema_state = None
    if "load_run" in config:

        load_project_folder = os.path.join("logs", config["load_run"])
        print("Loading model from ", load_project_folder)
        latest_path = os.path.join(load_project_folder, "latest.pth")
        latest_checkpoint = torch.load(latest_path, map_location=device)
        load_model(model, config["model_type"], latest_checkpoint)

        if isinstance(latest_checkpoint, dict) and "epoch" in latest_checkpoint:
            current_epoch = latest_checkpoint["epoch"] + 1


        if config["model_type"] == "nomad" and current_epoch == 0:
            epoch_ids = []
            for filename in os.listdir(load_project_folder):
                stem, ext = os.path.splitext(filename)
                if ext == ".pth" and stem.isdigit():
                    epoch_ids.append(int(stem))
            if len(epoch_ids) > 0:
                current_epoch = max(epoch_ids) + 1


        if isinstance(latest_checkpoint, dict) and "ema_model" in latest_checkpoint:
            resume_ema_state = latest_checkpoint["ema_model"]
        elif config["model_type"] == "nomad":
            ema_latest_path = os.path.join(load_project_folder, "ema_latest.pth")
            if os.path.exists(ema_latest_path):
                resume_ema_state = torch.load(ema_latest_path, map_location=device)

    model = model.to(device)
    if distributed:
        model = DDP(model, **_ddp_kwargs(config, local_rank, device))
    elif len(config["gpu_ids"]) > 1:
        # Fallback for single-process launches. For efficient multi-GPU training, use torchrun/DDP.
        model = nn.DataParallel(model, device_ids=list(range(len(config["gpu_ids"]))))


    if "load_run" in config:
        optimizer_state = None
        scheduler_state = None

        if isinstance(latest_checkpoint, dict):
            optimizer_state = latest_checkpoint.get("optimizer", None)
            scheduler_state = latest_checkpoint.get("scheduler", None)


        if config["model_type"] == "nomad":
            if optimizer_state is None:
                optimizer_latest_path = os.path.join(load_project_folder, "optimizer_latest.pth")
                if os.path.exists(optimizer_latest_path):
                    optimizer_state = torch.load(optimizer_latest_path, map_location=device)
            if scheduler is not None and scheduler_state is None:
                scheduler_latest_path = os.path.join(load_project_folder, "scheduler_latest.pth")
                if os.path.exists(scheduler_latest_path):
                    scheduler_state = torch.load(scheduler_latest_path, map_location=device)

        if optimizer_state is not None:
            if isinstance(optimizer_state, dict):
                optimizer.load_state_dict(optimizer_state)
            else:
                optimizer.load_state_dict(optimizer_state.state_dict())

        if scheduler is not None and scheduler_state is not None:
            if isinstance(scheduler_state, dict):
                scheduler.load_state_dict(scheduler_state)
            else:
                scheduler.load_state_dict(scheduler_state.state_dict())

        print(f"Resuming training from epoch {current_epoch}")


    if config["model_type"] == "vint" or config["model_type"] == "gnm": 

        train_eval_loop(
            train_model=config["train"],
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader=train_loader,
            test_dataloaders=test_dataloaders,
            transform=transform,
            epochs=config["epochs"],
            device=device,
            project_folder=config["project_folder"],
            normalized=config["normalize"],
            print_log_freq=config["print_log_freq"],
            wandb_log_freq=config["wandb_log_freq"],
            image_log_freq=config["image_log_freq"],
            num_images_log=config["num_images_log"],
            current_epoch=current_epoch,
            learn_angle=config["learn_angle"],
            alpha=config["alpha"],
            use_wandb=config["use_wandb"],
            eval_fraction=config["eval_fraction"],
            train_sampler=train_sampler,
            distributed=distributed,
            is_main_process=is_main_process,
            amp_enabled=bool(config.get("amp", False)),
            amp_dtype=config.get("amp_dtype", "fp16"),
            use_grad_scaler=bool(config.get("use_grad_scaler", True)),
            log_by_global_step=bool(config.get("log_by_global_step", True)),
            log_first_step=bool(config.get("log_first_step", False)),
            image_log_start_step=int(config.get("image_log_start_step", 0)),
            perf_log_freq=int(config.get("perf_log_freq", 0)),
            distributed_eval=distributed_eval,
            save_latest_every_epoch=bool(config.get("save_latest_every_epoch", True)),
            checkpoint_freq=int(config.get("checkpoint_freq", 1)),
        )
    else:

        train_eval_loop_nomad(
            train_model=config["train"],
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            noise_scheduler=noise_scheduler,
            train_loader=train_loader,
            test_dataloaders=test_dataloaders,
            transform=transform,
            goal_mask_prob=config["goal_mask_prob"],
            epochs=config["epochs"],
            device=device,
            project_folder=config["project_folder"],
            print_log_freq=config["print_log_freq"],
            wandb_log_freq=config["wandb_log_freq"],
            image_log_freq=config["image_log_freq"],
            num_images_log=config["num_images_log"],
            current_epoch=current_epoch,
            alpha=float(config["alpha"]),
            use_wandb=config["use_wandb"],
            eval_fraction=config["eval_fraction"],
            eval_freq=config["eval_freq"],
            ema_state_dict=resume_ema_state,
            train_sampler=train_sampler,
            distributed=distributed,
            is_main_process=is_main_process,
            heavy_metric_log_freq=int(config.get("heavy_metric_log_freq", config["print_log_freq"])),
            heavy_metric_start_step=int(config.get("heavy_metric_start_step", 0)),
            num_action_samples_log=int(config.get("num_action_samples_log", 30)),
            amp_enabled=bool(config.get("amp", False)),
            amp_dtype=config.get("amp_dtype", "fp16"),
            use_grad_scaler=bool(config.get("use_grad_scaler", True)),
            log_by_global_step=bool(config.get("log_by_global_step", True)),
            log_first_step=bool(config.get("log_first_step", False)),
            image_log_start_step=int(config.get("image_log_start_step", 0)),
            perf_log_freq=int(config.get("perf_log_freq", 0)),
            distributed_eval=distributed_eval,
            save_latest_every_epoch=bool(config.get("save_latest_every_epoch", True)),
            checkpoint_freq=int(config.get("checkpoint_freq", 1)),
            ema_checkpoint_freq=int(config.get("ema_checkpoint_freq", 1)),
            optimizer_checkpoint_freq=int(config.get("optimizer_checkpoint_freq", 1)),
            scheduler_checkpoint_freq=int(config.get("scheduler_checkpoint_freq", 1)),
        )

    if is_main_process:
        print("FINISHED TRAINING")
    _cleanup_distributed()


if __name__ == "__main__":

    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass


    parser = argparse.ArgumentParser(description="Visual Navigation Transformer")


    parser.add_argument(
        "--config",
        "-c",
        default="config/nomad_retrain.yaml",
        type=str,
        help="Path to the config file in train_config folder",
    )
    parser.add_argument(
        "--build-lmdb-only",
        action="store_true",
        help="Build all configured LMDB caches with one process, then exit before training.",
    )
    parser.add_argument(
        "--rebuild-incomplete-lmdb",
        action="store_true",
        help="Remove incomplete/unverified LMDB caches and rebuild them during --build-lmdb-only.",
    )
    args = parser.parse_args()



    with open("config/defaults.yaml", "r",encoding="utf-8") as f:
        default_config = yaml.safe_load(f)

    config = default_config


    with open(args.config, "r",encoding="utf-8") as f:
        user_config = yaml.safe_load(f)

    config.update(user_config)
    config["config_path"] = args.config
    if args.build_lmdb_only:
        config["build_lmdb_only"] = True
    if args.rebuild_incomplete_lmdb:
        config["rebuild_incomplete_lmdb"] = True

    main(config)
