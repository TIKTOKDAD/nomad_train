# ============================================================
# Optimizer monitor callback - learning rate and training health
# ============================================================
# 本文件只负责记录优化器和模型健康状态，不参与训练数学：
# 1. log_optimizer_step 记录 step 级 lr、grad_norm、GradScaler scale 和可选 param_norm
# 2. log_epoch_optimizer 在调度器更新后记录 epoch 级学习率
# 3. 这些日志属于 runtime/optim 与 runtime/model 分区，不放进算法或可视化模块

from training_base.registry import callback_registry


@callback_registry.register("optim_monitor")
class OptimMonitorCallback:
    # 保存上下文；是否写日志由 callback 内部按主进程判断
    def __init__(self, config, context) -> None:
        self.config = config
        self.context = context

    # 从 optimizer param_groups 中抽取学习率；多组学习率时保留 group 明细
    def _learning_rate_logs(self, optimizer) -> dict:
        if optimizer is None or not optimizer.param_groups:
            return {}
        logs = {"runtime/optim/lr": optimizer.param_groups[0].get("lr")}
        if len(optimizer.param_groups) > 1:
            for index, group in enumerate(optimizer.param_groups):
                logs[f"runtime/optim/lr_group_{index}"] = group.get("lr")
        return {key: value for key, value in logs.items() if value is not None}

    # 记录一次 optimizer step 的健康指标，频率由 Trainer 根据 logging.optim_log_freq 控制
    def log_optimizer_step(self, *, recorder, optimizer, stats, global_step) -> None:
        if not self.context.is_main_process:
            return
        data = self._learning_rate_logs(optimizer)
        if stats is not None:
            if stats.grad_norm is not None:
                data["runtime/optim/grad_norm"] = stats.grad_norm
            if stats.grad_scale is not None:
                data["runtime/optim/grad_scale"] = stats.grad_scale
            if stats.param_norm is not None:
                data["runtime/model/param_norm"] = stats.param_norm
        if data:
            recorder.log_metrics(data, step=global_step, commit=False)

    # epoch 结束时调度器可能已经更新学习率，因此额外记录一次更新后的 lr
    def log_epoch_optimizer(self, *, recorder, optimizer, global_step) -> None:
        if not self.context.is_main_process:
            return
        data = self._learning_rate_logs(optimizer)
        if data:
            recorder.log_metrics(data, step=global_step, commit=False)
