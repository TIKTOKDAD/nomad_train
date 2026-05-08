# ============================================================
# Algorithm base - training/evaluation protocol
# ============================================================
# 本文件定义所有算法类必须遵守的“协议层”：
# 1. StepResult 统一描述一次 train/eval step 的输出
# 2. Algorithm 规定模型、目标函数、优化器、batch 准备、指标、可视化等钩子
# 3. Trainer 只依赖这些接口，因此 GNM/ViNT/NoMaD 可以复用同一套训练循环

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

from training_base.optimizers import build_optimizer
from training_base.schedulers import build_scheduler
from training_base.core.checkpoint import ResumeState
from training_base.core.native_utils import unwrap_model


# 训练/评估一步的统一返回结构
@dataclass
class StepResult:
    # loss: 训练时用于 backward 的标量张量；评估阶段允许为 None
    loss: Optional[torch.Tensor]
    # logs: 需要写入 MetricStore/Recorder 的损失和轻量指标
    logs: Dict[str, Any] = field(default_factory=dict)
    # batch_size: 性能监控使用，避免从不同 prepared 结构中猜测 batch 大小
    batch_size: int = 0
    # extras: 算法私有中间结果，如预测轨迹、扩散噪声、goal mask 等
    extras: Dict[str, Any] = field(default_factory=dict)


# 算法基类：定义训练所需的最小接口
class Algorithm:
    # name 用于日志、进度条、输出文件命名，子类应覆盖
    name: str
    # 评估时默认只可视化最后一个 batch，避免评估阶段生成过多图片
    visualize_eval_last: bool = True

    # 构建模型
    def build_model(self, config):
        raise NotImplementedError

    # 构建损失目标/目标函数
    def build_objective(self, config):
        raise NotImplementedError

    # 构建优化器与学习率调度器
    def configure_optimizers(self, model, config):
        # optimizer 只接收模型参数和 optimizer 配置块
        optimizer = build_optimizer(model, config["optimizer"])
        # scheduler 允许缺省；缺省时显式走 name=none，统一下游处理
        scheduler_config = dict(config.get("scheduler") or {"name": "none"})
        # 部分调度器需要 epochs/lr，若用户配置未写则从 runtime/optimizer 推导
        scheduler_config.setdefault("epochs", config["runtime"]["epochs"])
        scheduler_config.setdefault("lr", config["optimizer"]["lr"])
        scheduler = build_scheduler(optimizer, scheduler_config)
        return optimizer, scheduler

    # 创建算法级状态（可携带额外对象）
    def create_state(self, model, model_extras, objective, config, device, resume_state: ResumeState):
        return {"model_extras": model_extras, "objective": objective}

    # 断点恢复准备
    def prepare_resume(self, model, optimizer, scheduler, config, device) -> ResumeState:
        return ResumeState()

    # 数据批次预处理（张量迁移、归一化等）
    def prepare_batch(self, batch, transform, device, mode: str, should_log_images: bool):
        raise NotImplementedError

    # 单步训练逻辑
    def train_step(self, model, prepared, state, config) -> StepResult:
        raise NotImplementedError

    # 单步评估逻辑
    def eval_step(self, model, prepared, state, config) -> StepResult:
        raise NotImplementedError

    # 获取用于评估的模型（可替换为 EMA 等）
    def model_for_eval(self, model, state):
        return unwrap_model(model)

    # 重计算指标（默认空）
    def heavy_metrics(self, model, prepared, state, config, mode: str) -> Dict[str, Any]:
        return {}

    # 轻量指标（默认空）
    def light_metrics(self, model, prepared, result: StepResult, state, config, mode: str) -> Dict[str, Any]:
        return {}

    # 可视化钩子（默认空）
    def visualize(self, **kwargs) -> None:
        return None

    # 读取可视化配置并过滤启用项
    def visualization_configs(self, config, mode: str, default_name: str):
        # visualization.train / visualization.eval 可以是列表，也可以是单个 dict
        section = config.get("visualization", {})
        key = "train" if mode == "train" else "eval"
        # 没配置时回退到算法默认可视化器，保证老配置仍可运行
        entries = section.get(key, [{"name": default_name}])
        if isinstance(entries, dict):
            entries = [entries]
        return [dict(entry) for entry in entries if bool(entry.get("enabled", True))]

    # 优化器更新后的回调
    def after_optimizer_step(self, model, state, config) -> None:
        return None

    # 序列化算法自身状态
    def state_dict(self, state) -> Dict[str, Any]:
        return {}

    # 选择用于调度器/早停的主指标
    def primary_metric(self, eval_summaries: Dict[str, Dict[str, float]]) -> float:
        # 对所有评估数据集的 total_loss 求平均；过滤 NaN，避免调度器收到无效值
        values = [
            metrics["total_loss"]
            for metrics in eval_summaries.values()
            if "total_loss" in metrics and metrics["total_loss"] == metrics["total_loss"]
        ]
        return sum(values) / len(values) if values else float("nan")

    # 更新学习率调度器
    def step_scheduler(self, scheduler, eval_summaries, config) -> None:
        if scheduler is None:
            return
        metric = self.primary_metric(eval_summaries)
        # Plateau 调度器需要显式指标；其他调度器按 epoch/step 自增
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if metric == metric:
                scheduler.step(metric)
        else:
            scheduler.step()
