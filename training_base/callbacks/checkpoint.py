# ============================================================
# Checkpoint callback - epoch-end model and optimizer snapshots
# ============================================================
# 本文件在每个 epoch 结束时保存训练状态：
# 1. latest.pth 保存通用 checkpoint payload
# 2. 按频率保存历史 epoch checkpoint
# 3. NoMaD 若启用 EMA，还会保存 ema/optimizer/scheduler 的独立 latest 和历史文件

import os

import torch

from training_base.core.checkpoint import save_checkpoint
from training_base.core.native_utils import should_save_epoch
from training_base.registry import callback_registry


# 注册检查点保存回调
@callback_registry.register("checkpoint")
class CheckpointCallback:
    # 保存配置与运行上下文
    def __init__(self, config, context) -> None:
        self.config = config
        self.context = context

    # epoch 结束时保存模型与相关状态
    def on_epoch_end(
        self,
        *,
        epoch,
        global_step,
        model,
        optimizer,
        scheduler,
        algorithm,
        state,
        config,
        eval_summaries,
    ) -> None:
        # 仅主进程执行保存
        if not self.context.is_main_process:
            return
        project_folder = config["runtime"]["project_folder"]
        algorithm_state = algorithm.state_dict(state)
        # 保存 latest.pth
        if bool(self.config.get("save_latest_every_epoch", True)):
            # latest.pth 用于常规断点恢复，包含模型、优化器、调度器、算法状态和配置
            save_checkpoint(
                os.path.join(project_folder, "latest.pth"),
                epoch=epoch,
                global_step=global_step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                algorithm_state=algorithm_state,
                callback_state={},
                config=config,
                eval_summaries=eval_summaries,
            )
        checkpoint_freq = int(self.config.get("checkpoint_freq", 1))
        # 按频率保存 epoch 检查点
        if should_save_epoch(epoch, checkpoint_freq):
            # 数字命名 checkpoint 用于保留历史快照和兼容旧恢复逻辑
            save_checkpoint(
                os.path.join(project_folder, f"{epoch}.pth"),
                epoch=epoch,
                global_step=global_step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                algorithm_state=algorithm_state,
                callback_state={},
                config=config,
                eval_summaries=eval_summaries,
            )

        # 若不存在 EMA 权重则直接返回
        if "ema_model" not in algorithm_state:
            return

        save_latest = bool(self.config.get("save_latest_every_epoch", True))
        ema_checkpoint_freq = int(self.config.get("ema_checkpoint_freq", checkpoint_freq))
        optimizer_checkpoint_freq = int(self.config.get("optimizer_checkpoint_freq", checkpoint_freq))
        scheduler_checkpoint_freq = int(self.config.get("scheduler_checkpoint_freq", checkpoint_freq))

        # 保存 EMA 与优化器/调度器最新状态
        if save_latest:
            # 独立 latest 文件兼容历史 NoMaD 脚本，也方便只加载 EMA 权重
            torch.save(algorithm_state["ema_model"], os.path.join(project_folder, "ema_latest.pth"))
            if optimizer is not None:
                torch.save(optimizer.state_dict(), os.path.join(project_folder, "optimizer_latest.pth"))
            if scheduler is not None:
                torch.save(scheduler.state_dict(), os.path.join(project_folder, "scheduler_latest.pth"))

        # 分别按频率保存 EMA/优化器/调度器历史状态
        if should_save_epoch(epoch, ema_checkpoint_freq):
            torch.save(algorithm_state["ema_model"], os.path.join(project_folder, f"ema_{epoch}.pth"))
        if optimizer is not None and should_save_epoch(epoch, optimizer_checkpoint_freq):
            torch.save(optimizer.state_dict(), os.path.join(project_folder, f"optimizer_{epoch}.pth"))
        if scheduler is not None and should_save_epoch(epoch, scheduler_checkpoint_freq):
            torch.save(scheduler.state_dict(), os.path.join(project_folder, f"scheduler_{epoch}.pth"))
