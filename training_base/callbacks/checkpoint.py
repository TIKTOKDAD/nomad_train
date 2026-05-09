# ============================================================
# Checkpoint callback - epoch-end model and optimizer snapshots
# ============================================================
# 本文件实现训练结束一个 epoch 后的快照保存策略：
# 1. latest.pth 保存完整训练状态，支持精确恢复
# 2. 按 checkpoint_freq 额外保存 epoch 编号 checkpoint
# 3. 对 NoMaD EMA/optimizer/scheduler 兼容旧版单独文件保存

import os

from training_base.core.checkpoint import atomic_torch_save, save_checkpoint
from training_base.core.native_utils import should_save_epoch
from training_base.registry import callback_registry


# 检查点回调：优先级较高，确保其他回调状态先更新再保存
@callback_registry.register("checkpoint")
class CheckpointCallback:
    # 数值越大越靠后执行；保存 checkpoint 应尽量放在 epoch_end 的最后
    on_epoch_end_priority = 1000

    # 保存回调自身配置和运行时上下文
    def __init__(self, config, context) -> None:
        self.config = config
        self.context = context

    # 本回调本身没有额外可恢复状态
    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state) -> None:
        return None

    # epoch 结束时保存完整 checkpoint 和可选的独立 EMA/优化器快照
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
        callback_state=None,
        callback_manager=None,
        grad_scaler=None,
    ) -> None:
        # 只允许主进程写文件，避免 DDP 多 rank 同时覆盖同一路径
        if not self.context.is_main_process:
            return

        project_folder = config["runtime"]["project_folder"]
        # 算法状态主要用于 NoMaD EMA；普通监督算法通常返回空 dict
        algorithm_state = algorithm.state_dict(state)
        if callback_manager is not None:
            # 保存所有其他回调状态，避免把正在执行的 checkpoint callback 递归写入
            callback_state = callback_manager.state_dict_for(self)
        else:
            callback_state = callback_state or {}

        if bool(self.config.get("save_latest_every_epoch", True)):
            # latest.pth 始终代表最近一个完整 epoch，可直接 resume
            save_checkpoint(
                os.path.join(project_folder, "latest.pth"),
                epoch=epoch,
                global_step=global_step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                algorithm_state=algorithm_state,
                callback_state=callback_state,
                config=config,
                eval_summaries=eval_summaries,
                grad_scaler=grad_scaler,
            )

        checkpoint_freq = int(self.config.get("checkpoint_freq", 1))
        if should_save_epoch(epoch, checkpoint_freq):
            # 编号 checkpoint 用于回滚到历史 epoch 或做离线评估
            save_checkpoint(
                os.path.join(project_folder, f"{epoch}.pth"),
                epoch=epoch,
                global_step=global_step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                algorithm_state=algorithm_state,
                callback_state=callback_state,
                config=config,
                eval_summaries=eval_summaries,
                grad_scaler=grad_scaler,
            )

        if "ema_model" not in algorithm_state:
            # 只有 NoMaD 等带 EMA 的算法才需要下面的兼容性单独文件
            return

        save_latest = bool(self.config.get("save_latest_every_epoch", True))
        ema_checkpoint_freq = int(self.config.get("ema_checkpoint_freq", checkpoint_freq))
        optimizer_checkpoint_freq = int(self.config.get("optimizer_checkpoint_freq", checkpoint_freq))
        scheduler_checkpoint_freq = int(self.config.get("scheduler_checkpoint_freq", checkpoint_freq))

        if save_latest:
            # 旧版 NoMaD 恢复逻辑会查找这些 latest 文件，因此保留独立保存
            atomic_torch_save(algorithm_state["ema_model"], os.path.join(project_folder, "ema_latest.pth"), backup_existing=True)
            if optimizer is not None:
                atomic_torch_save(optimizer.state_dict(), os.path.join(project_folder, "optimizer_latest.pth"), backup_existing=True)
            if scheduler is not None:
                atomic_torch_save(scheduler.state_dict(), os.path.join(project_folder, "scheduler_latest.pth"), backup_existing=True)

        if should_save_epoch(epoch, ema_checkpoint_freq):
            # 独立 EMA checkpoint 只保存 EMA 模型权重，不含训练全状态
            atomic_torch_save(algorithm_state["ema_model"], os.path.join(project_folder, f"ema_{epoch}.pth"))
        if optimizer is not None and should_save_epoch(epoch, optimizer_checkpoint_freq):
            atomic_torch_save(optimizer.state_dict(), os.path.join(project_folder, f"optimizer_{epoch}.pth"))
        if scheduler is not None and should_save_epoch(epoch, scheduler_checkpoint_freq):
            atomic_torch_save(scheduler.state_dict(), os.path.join(project_folder, f"scheduler_{epoch}.pth"))
