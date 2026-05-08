# ============================================================
# Performance monitor callback - throughput and timing metrics
# ============================================================
# 本文件记录训练热路径性能：
# 1. data_time 近似 DataLoader 等待时间
# 2. compute_time 近似 batch 准备 + 前向/反向/优化时间
# 3. samples_per_sec 和显存指标用于观察多 GPU/worker 配置是否有效

import torch

from training_base.registry import callback_registry


# 注册性能监控回调
@callback_registry.register("perf_monitor")
class PerfMonitorCallback:
    # 保存配置与上下文
    def __init__(self, config, context) -> None:
        self.config = config
        self.context = context

    # 记录性能指标（吞吐、耗时、显存）
    def log_perf(self, *, recorder, mode, epoch, batch_idx, batch_size, data_time, compute_time, step_time, device) -> None:
        # step_time 可能极小，max 避免除零
        samples_per_sec = batch_size / max(step_time, 1e-12)
        data = {
            f"perf/{mode}_data_time": data_time,
            f"perf/{mode}_compute_time": compute_time,
            f"perf/{mode}_step_time": step_time,
            f"perf/{mode}_samples_per_sec": samples_per_sec,
            f"perf/{mode}_epoch": epoch,
            f"perf/{mode}_batch": batch_idx,
        }
        if device.type == "cuda":
            # memory_allocated 是当前分配显存，不含缓存池 reserved memory
            data[f"perf/{mode}_gpu_mem_allocated_mb"] = torch.cuda.memory_allocated(device) / (1024 ** 2)
        recorder.log_metrics(data, commit=False)
