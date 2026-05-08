# ============================================================
# Registry core - name based configurable component lookup
# ============================================================
# 本文件提供最小化注册表实现：
# 1. 各模块用 @registry.register("name") 把类或函数挂到字符串 key
# 2. 配置文件只需要写 name，即可通过 registry.build 实例化组件
# 3. 报错时列出可用 key，方便排查配置拼写错误

from typing import Callable, Dict, Iterable, Optional


# 简单注册表：用于按名称查找可配置组件
class Registry:
    """Small explicit registry used by configurable training items."""

    # name 用于报错提示，标识注册表类型
    def __init__(self, name: str) -> None:
        self.name = name
        self._items: Dict[str, Callable] = {}

    # 装饰器：将对象注册到表中
    def register(self, name: Optional[str] = None):
        def decorator(obj: Callable):
            # 未显式传 name 时使用对象自身 __name__，但配置化组件通常显式传小写 key
            key = name or obj.__name__
            if key in self._items:
                raise KeyError(f"{self.name} 注册表中已经包含 '{key}'")
            self._items[key] = obj
            return obj

        return decorator

    # 根据名称获取对象，不存在则抛出详细错误
    def get(self, name: str) -> Callable:
        if name not in self._items:
            available = ", ".join(sorted(self._items)) or "<空>"
            raise KeyError(f"未知 {self.name} '{name}'。可用项: {available}")
        return self._items[name]

    # 直接构建对象实例
    def build(self, name: str, *args, **kwargs):
        # build 只负责查找并调用，具体参数结构由注册对象自己决定
        return self.get(name)(*args, **kwargs)

    # 返回已注册的名称列表
    def names(self) -> Iterable[str]:
        return sorted(self._items)
