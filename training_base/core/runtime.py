import inspect
import os
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass
class RuntimeContext:
    device: torch.device
    distributed: bool
    rank: int
    local_rank: int
    world_size: int
    is_main_process: bool
    gpu_ids: List[int]


def env_int(name: str, default: int = 0) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def is_torchrun() -> bool:
    return env_int("WORLD_SIZE", 1) > 1


def normalize_gpu_ids(config) -> List[int]:
    gpu_ids = config.get("gpu_ids", [0])
    if isinstance(gpu_ids, int):
        gpu_ids = [gpu_ids]
    config["gpu_ids"] = [int(gpu_id) for gpu_id in gpu_ids]
    return config["gpu_ids"]


def configure_cuda_visibility(config) -> List[int]:
    gpu_ids = normalize_gpu_ids(config)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if not is_torchrun():
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in gpu_ids)
    return gpu_ids


def init_distributed(config):
    world_size = env_int("WORLD_SIZE", 1)
    rank = env_int("RANK", 0)
    local_rank = env_int("LOCAL_RANK", 0)
    distributed = bool(config.get("distributed", True)) and world_size > 1
    if distributed and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() and os.name != "nt" else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    return distributed, rank, local_rank, world_size


def is_main_process() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def broadcast_string(value: str, src: int = 0) -> str:
    if not (dist.is_available() and dist.is_initialized()):
        return value
    obj = [value if dist.get_rank() == src else None]
    dist.broadcast_object_list(obj, src=src)
    return obj[0]


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


def setup_seed(config) -> None:
    if "seed" not in config:
        return
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])


def setup_cudnn(config) -> None:
    deterministic = bool(config.get("deterministic", False))
    cudnn.deterministic = deterministic
    cudnn.benchmark = bool(config.get("cudnn_benchmark", not deterministic))


def ddp_kwargs(config, local_rank: int, device: torch.device):
    """Build DDP kwargs while skipping arguments unsupported by old PyTorch."""
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


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def setup_runtime(config) -> RuntimeContext:
    gpu_ids = configure_cuda_visibility(config)
    if torch.cuda.is_available() and is_torchrun():
        torch.cuda.set_device(env_int("LOCAL_RANK", 0))

    distributed, rank, local_rank, world_size = init_distributed(config)
    main = is_main_process()
    config["distributed_active"] = distributed
    config["rank"] = rank
    config["local_rank"] = local_rank
    config["world_size"] = world_size

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
        if main:
            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", ",".join(str(x) for x in gpu_ids))
            print("Using cuda devices:", visible_devices)
    else:
        device = torch.device("cpu")
        if main:
            print("Using cpu")

    return RuntimeContext(
        device=device,
        distributed=distributed,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        is_main_process=main,
        gpu_ids=gpu_ids,
    )


def wrap_distributed_model(model: nn.Module, config, context: RuntimeContext) -> nn.Module:
    if context.distributed:
        return DDP(model, **ddp_kwargs(config, context.local_rank, context.device))
    if torch.cuda.is_available() and len(config.get("gpu_ids", [0])) > 1:
        raise RuntimeError(
            "Multi-GPU training requires torchrun/DDP. Single-process multi-GPU fallback is not supported "
            "by the native training_base runtime."
        )
    return model
