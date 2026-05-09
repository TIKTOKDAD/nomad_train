# ============================================================
# Checkpoint callback - epoch-end model and optimizer snapshots
# ============================================================

import os

from training_base.core.checkpoint import atomic_torch_save, save_checkpoint
from training_base.core.native_utils import should_save_epoch
from training_base.registry import callback_registry


@callback_registry.register("checkpoint")
class CheckpointCallback:
    on_epoch_end_priority = 1000

    def __init__(self, config, context) -> None:
        self.config = config
        self.context = context

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state) -> None:
        return None

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
    ) -> None:
        if not self.context.is_main_process:
            return

        project_folder = config["runtime"]["project_folder"]
        algorithm_state = algorithm.state_dict(state)
        if callback_manager is not None:
            callback_state = callback_manager.state_dict_for(self)
        else:
            callback_state = callback_state or {}

        if bool(self.config.get("save_latest_every_epoch", True)):
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
            )

        checkpoint_freq = int(self.config.get("checkpoint_freq", 1))
        if should_save_epoch(epoch, checkpoint_freq):
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
            )

        if "ema_model" not in algorithm_state:
            return

        save_latest = bool(self.config.get("save_latest_every_epoch", True))
        ema_checkpoint_freq = int(self.config.get("ema_checkpoint_freq", checkpoint_freq))
        optimizer_checkpoint_freq = int(self.config.get("optimizer_checkpoint_freq", checkpoint_freq))
        scheduler_checkpoint_freq = int(self.config.get("scheduler_checkpoint_freq", checkpoint_freq))

        if save_latest:
            atomic_torch_save(algorithm_state["ema_model"], os.path.join(project_folder, "ema_latest.pth"), backup_existing=True)
            if optimizer is not None:
                atomic_torch_save(optimizer.state_dict(), os.path.join(project_folder, "optimizer_latest.pth"), backup_existing=True)
            if scheduler is not None:
                atomic_torch_save(scheduler.state_dict(), os.path.join(project_folder, "scheduler_latest.pth"), backup_existing=True)

        if should_save_epoch(epoch, ema_checkpoint_freq):
            atomic_torch_save(algorithm_state["ema_model"], os.path.join(project_folder, f"ema_{epoch}.pth"))
        if optimizer is not None and should_save_epoch(epoch, optimizer_checkpoint_freq):
            atomic_torch_save(optimizer.state_dict(), os.path.join(project_folder, f"optimizer_{epoch}.pth"))
        if scheduler is not None and should_save_epoch(epoch, scheduler_checkpoint_freq):
            atomic_torch_save(scheduler.state_dict(), os.path.join(project_folder, f"scheduler_{epoch}.pth"))
