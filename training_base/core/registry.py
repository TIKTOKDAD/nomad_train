from typing import Callable, Dict, Iterable, Optional


class Registry:
    """Small explicit registry used by configurable training items."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._items: Dict[str, Callable] = {}

    def register(self, name: Optional[str] = None):
        def decorator(obj: Callable):
            key = name or obj.__name__
            if key in self._items:
                raise KeyError(f"{self.name} registry already contains '{key}'")
            self._items[key] = obj
            return obj

        return decorator

    def get(self, name: str) -> Callable:
        if name not in self._items:
            available = ", ".join(sorted(self._items)) or "<empty>"
            raise KeyError(f"Unknown {self.name} '{name}'. Available: {available}")
        return self._items[name]

    def build(self, name: str, *args, **kwargs):
        return self.get(name)(*args, **kwargs)

    def names(self) -> Iterable[str]:
        return sorted(self._items)
