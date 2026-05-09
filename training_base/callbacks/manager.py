# ============================================================
# Callback manager - lifecycle hook dispatcher and state holder
# ============================================================
# 本文件是回调系统的统一调度器：
# 1. 根据 config.callbacks 构建所有回调实例
# 2. Trainer 在关键生命周期点通过 call(hook, **kwargs) 广播事件
# 3. checkpoint 保存时汇总各回调自己的 state_dict，恢复时再分发回去

from training_base.registry import callback_registry


# 回调管理器：负责构建、调用、保存和恢复回调状态
class CallbackManager:
    """Build callbacks from config, dispatch hooks, and persist callback state."""

    # 按配置顺序实例化回调
    def __init__(self, config, context) -> None:
        self._entries = []
        for callback_config in config.get("callbacks", []):
            callback_config = dict(callback_config)
            name = callback_config.pop("name")
            # callback_registry 让新增回调只需要注册，不需要改 Trainer
            callback = callback_registry.build(name, callback_config, context)
            self._entries.append((name, callback))
        self.callbacks = [callback for _, callback in self._entries]

    # 调用所有实现了指定 hook 的回调
    def call(self, hook: str, **kwargs) -> None:
        entries = self._entries
        if hook == "on_epoch_end":
            # epoch_end 需要先让普通回调更新状态，最后再由 checkpoint 保存状态
            entries = sorted(self._entries, key=lambda item: getattr(item[1], "on_epoch_end_priority", 0))
        for _, callback in entries:
            method = getattr(callback, hook, None)
            if method is not None:
                method(**kwargs)

    # 汇总所有回调状态，用于写入训练 checkpoint
    def state_dict(self, *, exclude=None) -> dict:
        exclude = set(exclude or [])
        states = []
        for name, callback in self._entries:
            if callback in exclude:
                continue
            method = getattr(callback, "state_dict", None)
            # 没有状态的回调保存空 dict，恢复时仍能按名称对齐
            state = method() if method is not None else {}
            states.append({"name": name, "state": state or {}})
        return {"callbacks": states}

    # 当前回调主动保存 checkpoint 时，可排除自己，避免递归依赖正在生成的状态
    def state_dict_for(self, current_callback=None) -> dict:
        return self.state_dict(exclude={current_callback} if current_callback is not None else None)

    # 从 checkpoint 恢复回调状态
    def load_state_dict(self, state) -> None:
        if not state:
            return
        if isinstance(state, dict) and "callbacks" in state:
            # 新格式：按列表保存 name/state，支持同名回调顺序匹配
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
            # 旧格式：直接按 callback name 取状态
            for name, callback in self._entries:
                if name in state:
                    self._load_callback_state(callback, state[name])

    # 对单个回调调用 load_state_dict（若存在）
    @staticmethod
    def _load_callback_state(callback, state) -> None:
        method = getattr(callback, "load_state_dict", None)
        if method is not None:
            method(state or {})

    # 关闭回调资源，例如 W&B sink 或文件句柄
    def close(self) -> None:
        self.call("close")
