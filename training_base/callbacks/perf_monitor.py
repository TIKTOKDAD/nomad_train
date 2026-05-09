# ============================================================
# Performance monitor callback - throughput and timing metrics
# ============================================================
# 本文件负责 system/runtime 类日志，不参与模型训练逻辑：
# 1. log_runtime_config 在训练启动时记录 DDP/DataLoader/AMP 的静态配置
# 2. log_perf 在训练热路径中记录 data/compute/step 耗时和吞吐
# 3. log_system_gpu 独立记录 CUDA 显存状态，频率由 logging.system.gpu 控制

import torch

from training_base.loggers.schedule import build_logging_schedules
from training_base.registry import callback_registry


@callback_registry.register("perf_monitor")
class PerfMonitorCallback:
    # 保存配置和运行上下文；是否主进程写日志由 callback 内部判断
    def __init__(self, config, context) -> None:
        self.config = config
        self.context = context

    # 记录一次性运行配置：这些值通常不随 step 变化，但对复现实验和排查吞吐很关键
    def log_runtime_config(self, *, recorder, config, global_step) -> None:
        if not self.context.is_main_process:
            return
        schedules = build_logging_schedules(config.get("logging", {}))
        runtime = config["runtime"]
        data = {
            "runtime/dataloader/train_num_workers_configured": runtime.get("num_workers"),
            "runtime/dataloader/train_num_workers_per_rank": runtime.get("num_workers_per_rank"),
            "runtime/dataloader/test_num_workers_configured": runtime.get("test_num_workers", runtime.get("num_workers")),
            "runtime/dataloader/test_num_workers_per_rank": runtime.get("test_num_workers_per_rank"),
            "runtime/dataloader/prefetch_factor": runtime.get("prefetch_factor"),
            "runtime/dataloader/persistent_workers": float(bool(runtime.get("persistent_workers", False))),
            "runtime/dataloader/pin_memory": float(bool(runtime.get("pin_memory", False))),
            "runtime/amp/enabled": float(bool(runtime.get("amp", False))),
        }
        if schedules.system_ddp_log_once:
            data.update(
                {
                    "system/ddp/world_size": self.context.world_size,
                    "system/ddp/global_batch_size": runtime.get("global_batch_size", runtime.get("batch_size")),
                    "system/ddp/per_device_batch_size": runtime.get("per_device_batch_size", runtime.get("batch_size")),
                }
            )
        recorder.log_metrics({key: value for key, value in data.items() if value is not None}, step=global_step, commit=False)

    # 记录训练热路径性能：频率由 Trainer 的 logging schedule 控制
    def log_perf(self, *, recorder, mode, epoch, batch_idx, batch_size, data_time, compute_time, step_time, device, global_step) -> None:
        # step_time 可能非常小，max 避免除零导致无穷吞吐
        samples_per_sec = batch_size / max(step_time, 1e-12)
        data = {
            f"runtime/{mode}/time/data_time": data_time,
            f"runtime/{mode}/time/compute_time": compute_time,
            f"runtime/{mode}/time/step_time": step_time,
            f"runtime/{mode}/throughput/samples_per_sec": samples_per_sec,
            f"runtime/{mode}/progress/epoch": epoch,
            f"runtime/{mode}/progress/batch": batch_idx,
        }
        recorder.log_metrics(data, step=global_step, commit=False)

    # 记录 GPU 系统状态：和 runtime/perf 解耦，便于只看吞吐或只看显存
    def log_system_gpu(self, *, recorder, device, global_step) -> None:
        if not self.context.is_main_process or device.type != "cuda":
            return
        # device.index 可能为空，回退到当前 CUDA 设备，保证 key 稳定
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        gpu_prefix = f"system/gpu/{device_index}"
        data = {
            f"{gpu_prefix}/memory_allocated_mb": torch.cuda.memory_allocated(device) / (1024 ** 2),
            f"{gpu_prefix}/memory_reserved_mb": torch.cuda.memory_reserved(device) / (1024 ** 2),
            f"{gpu_prefix}/max_memory_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
        }
        recorder.log_metrics(data, step=global_step, commit=False)
