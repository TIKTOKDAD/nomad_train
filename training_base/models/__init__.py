# ============================================================
# Model registry entrypoint - build contract and registrations
# ============================================================
# 本文件只定义模型构建返回协议、导入内置模型以触发注册，并暴露统一 build_model 入口。
# 具体模型的装配逻辑放在各自的 models/<model>.py 中，避免 __init__.py 继续膨胀。

from dataclasses import dataclass
from typing import Any, Dict

from training_base.registry import model_registry


@dataclass
class ModelBuild:
    # model 是最终 nn.Module，交给 Trainer 移动设备、包 DDP、保存 checkpoint
    model: Any
    # extras 存放 noise_scheduler 等非 nn.Module 状态，算法 create_state 会接管
    extras: Dict[str, Any]


from training_base.models import gnm as _gnm  # noqa: F401,E402
from training_base.models import nomad as _nomad  # noqa: F401,E402
from training_base.models import vint as _vint  # noqa: F401,E402


# 统一模型构建入口：通过 registry 根据 config["model"]["name"] 分派
def build_model(config) -> ModelBuild:
    return model_registry.build(config["model"]["name"], config)
