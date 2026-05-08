# ============================================================
# Callback manager - lifecycle hook dispatcher
# ============================================================
# 本文件负责管理回调列表：
# 1. 根据 callbacks YAML 列表构建回调实例
# 2. Trainer 在关键节点调用 call("hook", ...)
# 3. 只有实现了对应方法的回调才会收到该 hook

from training_base.registry import callback_registry


# 回调管理器：统一调度回调生命周期
class CallbackManager:
    # 按配置构建回调实例
    def __init__(self, config, context) -> None:
        self.callbacks = []
        for callback_config in config.get("callbacks", []):
            callback_config = dict(callback_config)
            # name 是 callback_registry key，其余字段作为该回调的私有配置
            name = callback_config.pop("name")
            self.callbacks.append(callback_registry.build(name, callback_config, context))

    # 调用指定 hook（如果回调实现了该方法）
    def call(self, hook: str, **kwargs) -> None:
        for callback in self.callbacks:
            # getattr(..., None) 让不同回调只实现自己关心的 hook
            method = getattr(callback, hook, None)
            if method is not None:
                method(**kwargs)

    # 统一关闭回调
    def close(self) -> None:
        self.call("close")
