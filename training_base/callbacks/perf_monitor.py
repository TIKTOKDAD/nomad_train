import torch

from training_base.registry import callback_registry


@callback_registry.register("perf_monitor")
class PerfMonitorCallback:
    def __init__(self, config, context) -> None:
        self.config = config
        self.context = context

    def log_perf(self, *, recorder, mode, epoch, batch_idx, batch_size, data_time, compute_time, step_time, device) -> None:
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
            data[f"perf/{mode}_gpu_mem_allocated_mb"] = torch.cuda.memory_allocated(device) / (1024 ** 2)
        recorder.log_metrics(data, commit=False)
