from training_base.registry import callback_registry


class CallbackManager:
    def __init__(self, config, context) -> None:
        self.callbacks = []
        for callback_config in config.get("callbacks", []):
            callback_config = dict(callback_config)
            name = callback_config.pop("name")
            self.callbacks.append(callback_registry.build(name, callback_config, context))

    def call(self, hook: str, **kwargs) -> None:
        for callback in self.callbacks:
            method = getattr(callback, hook, None)
            if method is not None:
                method(**kwargs)

    def close(self) -> None:
        self.call("close")
