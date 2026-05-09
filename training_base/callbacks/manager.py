# ============================================================
# Callback manager - lifecycle hook dispatcher and state holder
# ============================================================

from training_base.registry import callback_registry


class CallbackManager:
    """Build callbacks from config, dispatch hooks, and persist callback state."""

    def __init__(self, config, context) -> None:
        self._entries = []
        for callback_config in config.get("callbacks", []):
            callback_config = dict(callback_config)
            name = callback_config.pop("name")
            callback = callback_registry.build(name, callback_config, context)
            self._entries.append((name, callback))
        self.callbacks = [callback for _, callback in self._entries]

    def call(self, hook: str, **kwargs) -> None:
        entries = self._entries
        if hook == "on_epoch_end":
            entries = sorted(self._entries, key=lambda item: getattr(item[1], "on_epoch_end_priority", 0))
        for _, callback in entries:
            method = getattr(callback, hook, None)
            if method is not None:
                method(**kwargs)

    def state_dict(self, *, exclude=None) -> dict:
        exclude = set(exclude or [])
        states = []
        for name, callback in self._entries:
            if callback in exclude:
                continue
            method = getattr(callback, "state_dict", None)
            state = method() if method is not None else {}
            states.append({"name": name, "state": state or {}})
        return {"callbacks": states}

    def state_dict_for(self, current_callback=None) -> dict:
        return self.state_dict(exclude={current_callback} if current_callback is not None else None)

    def load_state_dict(self, state) -> None:
        if not state:
            return
        if isinstance(state, dict) and "callbacks" in state:
            pending = list(state.get("callbacks") or [])
            used = set()
            for name, callback in self._entries:
                for index, saved_entry in enumerate(pending):
                    if index in used:
                        continue
                    if isinstance(saved_entry, dict):
                        saved_name = saved_entry.get("name")
                        if saved_name is not None and saved_name != name:
                            continue
                        saved_state = saved_entry.get("state", {})
                    else:
                        saved_state = saved_entry
                    self._load_callback_state(callback, saved_state)
                    used.add(index)
                    break
            return
        if isinstance(state, dict):
            for name, callback in self._entries:
                if name in state:
                    self._load_callback_state(callback, state[name])

    @staticmethod
    def _load_callback_state(callback, state) -> None:
        method = getattr(callback, "load_state_dict", None)
        if method is not None:
            method(state or {})

    def close(self) -> None:
        self.call("close")
