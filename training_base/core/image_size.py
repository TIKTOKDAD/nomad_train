# ============================================================
# Image size helpers - config width/height to library shapes
# ============================================================
# YAML 中的 image_size 统一使用 [width, height]，但 torchvision.resize
# 需要 [height, width]。本文件集中处理这个转换，避免各算法手写反转。

from typing import Sequence, Tuple


def as_width_height(value: Sequence[int], field_name: str = "image_size") -> Tuple[int, int]:
    """Validate and return an image size stored as (width, height)."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field_name} 必须是 [width, height]")
    width = int(value[0])
    height = int(value[1])
    if width <= 0:
        raise ValueError(f"{field_name}[0] 必须是正整数，实际为 {value[0]!r}")
    if height <= 0:
        raise ValueError(f"{field_name}[1] 必须是正整数，实际为 {value[1]!r}")
    return width, height


def as_torch_resize_size(value: Sequence[int], field_name: str = "image_size") -> Tuple[int, int]:
    """Return torchvision.resize size order: (height, width)."""
    width, height = as_width_height(value, field_name)
    return height, width
