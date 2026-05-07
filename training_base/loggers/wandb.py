from training_base.registry import log_sink_registry


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
        if config.get("config_path"):
            wandb.save(config["config_path"], policy="now")
        if config.get("run_name"):
            wandb.run.name = config["run_name"]
        if wandb.run and config.get("full_config") is not None:
            wandb.config.update(config["full_config"])
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
