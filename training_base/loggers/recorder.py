# ============================================================
# Recorder - fan-out metrics and images to configured sinks
# ============================================================
# 本文件是日志系统的统一门面：
# 1. 根据 logging.sinks 构建 console/wandb 等输出端
# 2. Trainer 只调用 Recorder，不关心每个 sink 的实现细节
# 3. image() 负责把本地图片路径包装成 sink 需要的对象

from training_base.loggers.metric_store import MetricStore
from training_base.registry import log_sink_registry


# 日志记录器：统一向多个 sink 写入指标与图像
class Recorder:
    # 按配置构建日志 sink 列表
    def __init__(self, config, context) -> None:
        self.config = config
        self.context = context
        self.sinks = []
        for sink_config in config["logging"].get("sinks", []):
            sink_config = dict(sink_config)
            # name 是 log_sink_registry 的 key，其余字段透传给 sink 构造函数
            name = sink_config.pop("name")
            self.sinks.append(log_sink_registry.build(name, sink_config, context))

    # 创建一个新的指标缓存器
    def metric_store(self) -> MetricStore:
        return MetricStore(window_size=int(self.config["logging"].get("window_size", 10)))

    # 记录数值指标
    def log_metrics(self, data, *, step=None, commit=True) -> None:
        for sink in self.sinks:
            sink.log_metrics(data, step=step, commit=commit)

    # 记录图像数据
    def log_images(self, data, *, step=None, commit=False) -> None:
        for sink in self.sinks:
            sink.log_images(data, step=step, commit=commit)

    def log_status(self, message: str) -> None:
        for sink in self.sinks:
            method = getattr(sink, "log_status", None)
            if method is not None:
                method(message)

    # 生成可被 sink 接受的图像对象
    def image(self, path):
        # 优先使用第一个支持 image() 的 sink，例如 WandB Image；没有则直接返回路径
        for sink in self.sinks:
            if hasattr(sink, "image"):
                return sink.image(path)
        return path

    # 关闭所有 sink
    def close(self) -> None:
        for sink in self.sinks:
            sink.close()
