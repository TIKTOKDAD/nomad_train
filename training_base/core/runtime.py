# ============================================================
# Runtime utilities - device, DDP, seeds, and CUDA policy
# ============================================================
# 本文件集中处理训练启动时的运行时环境：
# 1. 解析 torchrun 环境变量，初始化分布式进程组
# 2. 设置 CUDA_VISIBLE_DEVICES、当前 device、cuDNN 策略和随机种子
# 3. 封装 DDP 参数，禁止回退到单进程多 GPU DataParallel
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


# 运行时上下文：统一保存设备与分布式信息
@dataclass
class RuntimeContext:
    # 当前进程使用的 torch.device，例如 cuda:0 或 cpu
    device: torch.device
    # distributed=True 表示当前已经处于 torch.distributed 多进程训练
    distributed: bool
    # 全局 rank 和本机 rank，分别用于通信与设备选择
    rank: int
    local_rank: int
    # world_size 是参与训练的进程总数
    world_size: int
    # 主进程负责日志、保存、进度条和可视化
    is_main_process: bool
    # YAML 中声明的可见 GPU 列表，非 torchrun 启动时会写入 CUDA_VISIBLE_DEVICES
    gpu_ids: List[int]


# 从环境变量读取整数值
def env_int(name: str, default: int = 0) -> int:
    try:
        # torchrun 会设置 WORLD_SIZE/RANK/LOCAL_RANK 等环境变量
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# 判断是否通过 torchrun 启动
def is_torchrun() -> bool:
    return env_int("WORLD_SIZE", 1) > 1


# 规范化 GPU ID 列表并写回配置
def normalize_gpu_ids(config) -> List[int]:
    gpu_ids = config.get("gpu_ids", [0])
    if isinstance(gpu_ids, int):
        gpu_ids = [gpu_ids]
    config["gpu_ids"] = [int(gpu_id) for gpu_id in gpu_ids]
    return config["gpu_ids"]


# 配置 CUDA 可见设备（非 torchrun 时设置 CUDA_VISIBLE_DEVICES）
def configure_cuda_visibility(config) -> List[int]:
    gpu_ids = normalize_gpu_ids(config)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if not is_torchrun():
        # torchrun 场景一般由外层 CUDA_VISIBLE_DEVICES 控制，避免在子进程中重复覆盖
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in gpu_ids)
    return gpu_ids


# 初始化分布式进程组
def init_distributed(config):
    world_size = env_int("WORLD_SIZE", 1)
    rank = env_int("RANK", 0)
    local_rank = env_int("LOCAL_RANK", 0)
    # distributed 配置为 False 时，即使环境变量存在也不主动初始化
    distributed = bool(config.get("distributed", True)) and world_size > 1
    if distributed and not dist.is_initialized():
        # Windows/NCCL 不可用时使用 gloo；Linux CUDA 多卡优先 nccl
        backend = "nccl" if torch.cuda.is_available() and os.name != "nt" else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    return distributed, rank, local_rank, world_size


# 判断当前是否主进程
def is_main_process() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


# 分布式同步屏障
def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


# 关闭分布式进程组
def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# 广播字符串到所有进程
def broadcast_string(value: str, src: int = 0) -> str:
    if not (dist.is_available() and dist.is_initialized()):
        return value
    # object_list 允许直接广播 Python 字符串，用于同步 run_name 时间戳
    obj = [value if dist.get_rank() == src else None]
    dist.broadcast_object_list(obj, src=src)
    return obj[0]


# DataLoader worker 随机种子初始化
def seed_worker(worker_id: int) -> None:
    # torch DataLoader 会为每个 worker 派生 initial_seed，这里同步到 numpy
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


# 设置全局随机种子
def setup_seed(config) -> None:
    if "seed" not in config:
        return
    # 同时设置 numpy、CPU torch 和 CUDA torch，尽量提升实验可复现性
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])


# 配置 cuDNN 的确定性与性能选项
def setup_cudnn(config) -> None:
    deterministic = bool(config.get("deterministic", False))
    cudnn.deterministic = deterministic
    cudnn.benchmark = bool(config.get("cudnn_benchmark", not deterministic))


# 构建 DDP 参数，兼容旧版本 PyTorch
def ddp_kwargs(config, local_rank: int, device: torch.device):
    """Build DDP kwargs while skipping arguments unsupported by old PyTorch."""
    # CUDA DDP 需要绑定本进程对应的 local_rank；CPU/gloo 不传 device_ids
    kwargs = {
        "device_ids": [local_rank] if device.type == "cuda" else None,
        "output_device": local_rank if device.type == "cuda" else None,
        "find_unused_parameters": bool(config.get("find_unused_parameters", False)),
    }
    # 这些参数不是所有 PyTorch 版本都支持，下面用 inspect 过滤
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


# 解除 DDP 包装
def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


# 运行时初始化：设备选择、分布式、可见 GPU 及环境检查
def setup_runtime(config) -> RuntimeContext:
    gpu_ids = configure_cuda_visibility(config)
    if torch.cuda.is_available() and is_torchrun():
        # torchrun 每个进程只绑定自己的 local_rank，避免多个 rank 抢同一张卡
        torch.cuda.set_device(env_int("LOCAL_RANK", 0))

    distributed, rank, local_rank, world_size = init_distributed(config)
    main = is_main_process()
    config["distributed_active"] = distributed
    config["rank"] = rank
    config["local_rank"] = local_rank
    config["world_size"] = world_size

    # 多 GPU 且未启用 DDP 时给出明确提示
    # training_base 明确不走 DataParallel，避免单进程多卡性能和稳定性问题
    if (
        torch.cuda.is_available()
        and len(gpu_ids) > 1
        and not distributed
        and bool(config.get("require_ddp_for_multigpu", True))
    ):
        raise RuntimeError(
            "多 GPU 训练必须使用 torchrun/DDP。"
            "请从训练目录启动: "
            "CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py "
            "-c config/nomad_retrain.yaml"
        )

    # 选择训练设备（GPU/CPU）并打印可见设备
    if torch.cuda.is_available():
        if distributed:
            # DDP 每个进程只使用一张本地 GPU
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            # 单进程只使用 CUDA_VISIBLE_DEVICES 后的第 0 张可见卡
            device = torch.device("cuda", 0)
        if main:
            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", ",".join(str(x) for x in gpu_ids))
            print("使用 CUDA 设备:", visible_devices)
    else:
        device = torch.device("cpu")
        if main:
            print("使用 CPU")

    return RuntimeContext(
        device=device,
        distributed=distributed,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        is_main_process=main,
        gpu_ids=gpu_ids,
    )


# 按运行时配置包装模型（DDP 或保持单机）
def wrap_distributed_model(model: nn.Module, config, context: RuntimeContext) -> nn.Module:
    if context.distributed:
        # 只有已经初始化分布式上下文时才包 DDP
        return DDP(model, **ddp_kwargs(config, context.local_rank, context.device))
    if torch.cuda.is_available() and len(config.get("gpu_ids", [0])) > 1:
        # 防止用户误以为 gpu_ids=[0,1] 会自动 DataParallel
        raise RuntimeError(
            "多 GPU 训练必须使用 torchrun/DDP。training_base 原生运行时不支持单进程多 GPU 回退。"
        )
    return model
