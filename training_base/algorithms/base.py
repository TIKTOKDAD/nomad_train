from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

from training_base.optimizers import build_optimizer
from training_base.schedulers import build_scheduler
from training_base.core.checkpoint import ResumeState
from training_base.core.native_utils import unwrap_model


@dataclass
class StepResult:
    loss: Optional[torch.Tensor]
    logs: Dict[str, Any] = field(default_factory=dict)
    batch_size: int = 0
    extras: Dict[str, Any] = field(default_factory=dict)


class Algorithm:
    name: str
    visualize_eval_last: bool = True

    def build_model(self, config):
        raise NotImplementedError

    def build_objective(self, config):
        raise NotImplementedError

    def configure_optimizers(self, model, config):
        optimizer = build_optimizer(model, config["optimizer"])
        scheduler_config = dict(config.get("scheduler") or {"name": "none"})
        scheduler_config.setdefault("epochs", config["runtime"]["epochs"])
        scheduler_config.setdefault("lr", config["optimizer"]["lr"])
        scheduler = build_scheduler(optimizer, scheduler_config)
        return optimizer, scheduler

    def create_state(self, model, model_extras, objective, config, device, resume_state: ResumeState):
        return {"model_extras": model_extras, "objective": objective}

    def prepare_resume(self, model, optimizer, scheduler, config, device) -> ResumeState:
        return ResumeState()

    def prepare_batch(self, batch, transform, device, mode: str, should_log_images: bool):
        raise NotImplementedError

    def train_step(self, model, prepared, state, config) -> StepResult:
        raise NotImplementedError

    def eval_step(self, model, prepared, state, config) -> StepResult:
        raise NotImplementedError

    def model_for_eval(self, model, state):
        return unwrap_model(model)

    def heavy_metrics(self, model, prepared, state, config, mode: str) -> Dict[str, Any]:
        return {}

    def light_metrics(self, model, prepared, result: StepResult, state, config, mode: str) -> Dict[str, Any]:
        return {}

    def visualize(self, **kwargs) -> None:
        return None

    def visualization_configs(self, config, mode: str, default_name: str):
        section = config.get("visualization", {})
        key = "train" if mode == "train" else "eval"
        entries = section.get(key, [{"name": default_name}])
        if isinstance(entries, dict):
            entries = [entries]
        return [dict(entry) for entry in entries if bool(entry.get("enabled", True))]

    def after_optimizer_step(self, model, state, config) -> None:
        return None

    def state_dict(self, state) -> Dict[str, Any]:
        return {}

    def primary_metric(self, eval_summaries: Dict[str, Dict[str, float]]) -> float:
        values = [
            metrics["total_loss"]
            for metrics in eval_summaries.values()
            if "total_loss" in metrics and metrics["total_loss"] == metrics["total_loss"]
        ]
        return sum(values) / len(values) if values else float("nan")

    def step_scheduler(self, scheduler, eval_summaries, config) -> None:
        if scheduler is None:
            return
        metric = self.primary_metric(eval_summaries)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if metric == metric:
                scheduler.step(metric)
        else:
            scheduler.step()
