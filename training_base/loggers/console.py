from training_base.registry import log_sink_registry


@log_sink_registry.register("console")
class ConsoleSink:
    def __init__(self, config, context) -> None:
        self.enabled = bool(config.get("enabled", True)) and context.is_main_process

    def log_metrics(self, data, *, step=None, commit=True) -> None:
        return None

    def log_images(self, data, *, step=None, commit=False) -> None:
        return None

    def close(self) -> None:
        return None
