# ============================================================
# WandB sink - optional experiment tracking backend
# ============================================================
# This module owns W&B-specific behavior only: run initialization,
# artifact upload, metric/image logging, and image object wrapping.

import os

from training_base.registry import log_sink_registry


def _iter_config_artifact_paths(paths):
    if isinstance(paths, dict):
        return [path for path in paths.values() if path]
    if isinstance(paths, (list, tuple)):
        return [path for path in paths if path]
    return []


def _save_config_artifact(wandb, path) -> None:
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        return
    rel_path = os.path.relpath(abs_path, os.getcwd())
    if rel_path == os.pardir or rel_path.startswith(os.pardir + os.sep):
        print(f"W&B config artifact save skipped outside cwd: {abs_path}")
        return
    try:
        wandb.save(rel_path, policy="now")
    except Exception as exc:
        print(f"W&B config artifact save skipped: {exc}")


@log_sink_registry.register("wandb")
class WandBSink:
    def __init__(self, config, context) -> None:
        self.enabled = bool(config.get("enabled", True)) and context.is_main_process
        self.run = None
        if not self.enabled:
            return

        import wandb

        wandb.login()
        wandb.init(
            project=config["project"],
            settings=wandb.Settings(start_method="thread"),
            entity=config.get("entity"),
        )
        for artifact_path in _iter_config_artifact_paths(config.get("config_artifact_paths", {})):
            _save_config_artifact(wandb, artifact_path)
        if config.get("run_name"):
            wandb.run.name = config["run_name"]
        if wandb.run and config.get("full_config") is not None:
            wandb.config.update(config["full_config"], allow_val_change=True)
        self.run = wandb

    def log_metrics(self, data, *, step=None, commit=True) -> None:
        if self.run is not None and data:
            self.run.log(data, step=step, commit=commit)

    def log_images(self, data, *, step=None, commit=False) -> None:
        if self.run is not None and data:
            self.run.log(data, step=step, commit=commit)

    def image(self, path):
        if self.run is None:
            return path
        return self.run.Image(path)

    def close(self) -> None:
        if self.run is not None:
            self.run.log({}, commit=True)
