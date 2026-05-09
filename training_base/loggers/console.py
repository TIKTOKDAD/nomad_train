# ============================================================
# Console sink - lightweight placeholder log backend
# ============================================================
# 本文件保留 console sink 的统一接口：
# 目前实际打印由 Trainer._print_store 负责，ConsoleSink 主要用于配置占位和未来扩展。

from training_base.registry import log_sink_registry


# 注册控制台日志输出（当前为占位实现）
@log_sink_registry.register("console")
class ConsoleSink:
    # 仅主进程启用
    def __init__(self, config, context) -> None:
        self.enabled = bool(config.get("enabled", True)) and context.is_main_process

    # 控制台输出（占位）
    def log_metrics(self, data, *, step=None, commit=True) -> None:
        return None

    # 控制台不处理图像日志
    def log_images(self, data, *, step=None, commit=False) -> None:
        return None

    def log_status(self, message: str) -> None:
        if self.enabled:
            print(message)

    # 关闭占位
    def close(self) -> None:
        return None
