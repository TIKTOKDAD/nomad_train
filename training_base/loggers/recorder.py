from training_base.loggers.metric_store import MetricStore
from training_base.registry import log_sink_registry


class Recorder:
    def __init__(self, config, context) -> None:
        self.config = config
        self.context = context
        self.sinks = []
        for sink_config in config["logging"].get("sinks", []):
            sink_config = dict(sink_config)
            name = sink_config.pop("name")
            self.sinks.append(log_sink_registry.build(name, sink_config, context))

    def metric_store(self) -> MetricStore:
        return MetricStore(window_size=int(self.config["logging"].get("window_size", 10)))

    def log_metrics(self, data, *, step=None, commit=True) -> None:
        for sink in self.sinks:
            sink.log_metrics(data, step=step, commit=commit)

    def log_images(self, data, *, step=None, commit=False) -> None:
        for sink in self.sinks:
            sink.log_images(data, step=step, commit=commit)

    def image(self, path):
        for sink in self.sinks:
            if hasattr(sink, "image"):
                return sink.image(path)
        return path

    def close(self) -> None:
        for sink in self.sinks:
            sink.close()
