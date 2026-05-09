# ============================================================
# Logging schedule - frequency rules for W&B/console panels
# ============================================================
# 本文件只负责“什么时候记录日志”，不负责计算指标、不负责上传 W&B：
# 1. 将 logging 配置解析成面向 train/media/runtime/eval 的调度对象
# 2. 复用 core.native_utils.should_log_event 保持频率语义一致
# 3. 让 Trainer 不直接依赖 YAML 字段路径，降低训练循环和配置结构的耦合

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from training_base.core.native_utils import should_log_event


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _nested_get(config: Mapping[str, Any], path: Sequence[str]) -> Optional[Any]:
    current: Any = config
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _read_int(config: Mapping[str, Any], path: Sequence[str], legacy_key: Optional[str], default: int) -> int:
    value = _nested_get(config, path)
    if value is None and legacy_key:
        value = config.get(legacy_key)
    if value is None:
        value = default
    return int(value)


def _read_float(config: Mapping[str, Any], path: Sequence[str], legacy_key: Optional[str], default: float) -> float:
    value = _nested_get(config, path)
    if value is None and legacy_key:
        value = config.get(legacy_key)
    if value is None:
        value = default
    return float(value)


def _read_str(config: Mapping[str, Any], path: Sequence[str], legacy_key: Optional[str], default: str) -> str:
    value = _nested_get(config, path)
    if value is None and legacy_key:
        value = config.get(legacy_key)
    if value is None:
        value = default
    return str(value)


def _read_bool(config: Mapping[str, Any], path: Sequence[str], legacy_key: Optional[str], default: bool) -> bool:
    value = _nested_get(config, path)
    if value is None and legacy_key:
        value = config.get(legacy_key)
    if value is None:
        value = default
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _read_schedule(
    config: Mapping[str, Any],
    path: Sequence[str],
    *,
    legacy_freq_key: Optional[str],
    legacy_start_key: Optional[str] = None,
    default_freq: int = 0,
    default_unit: str = "step",
) -> "LogSchedule":
    return LogSchedule(
        freq=_read_int(config, tuple(path) + ("freq",), legacy_freq_key, default_freq),
        unit=_read_str(config, tuple(path) + ("unit",), None, default_unit),
        start_step=_read_int(config, tuple(path) + ("start_step",), legacy_start_key, 0),
    )


@dataclass(frozen=True)
class LogSchedule:
    # freq=0 表示关闭该类日志；unit 明确 freq 的计数单位，避免 step/epoch/eval 混用
    freq: int = 0
    unit: str = "step"
    # start_step 在 unit=step 时表示 global step 起点；其他 unit 下作为通用起始 index
    start_step: int = 0

    def __post_init__(self) -> None:
        unit = str(self.unit or "step").lower()
        if unit not in {"step", "epoch", "eval"}:
            raise ValueError(f"不支持的日志调度单位: {self.unit}")
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "freq", int(self.freq))
        object.__setattr__(self, "start_step", int(self.start_step))

    def _enabled(self) -> bool:
        return int(self.freq) > 0

    def should_step(self, epoch: int, num_batches: int, batch_idx: int, *, by_global_step: bool, first_step: bool) -> bool:
        if self.unit != "step":
            return False
        return should_log_event(
            self.freq,
            epoch,
            num_batches,
            batch_idx,
            by_global_step,
            self.start_step,
            first_step,
        )

    def should_epoch(self, epoch: int) -> bool:
        if self.unit != "epoch" or not self._enabled():
            return False
        epoch_index = epoch + 1
        if epoch_index < self.start_step:
            return False
        return epoch_index % int(self.freq) == 0

    def should_eval(self, eval_index: int) -> bool:
        if self.unit != "eval" or not self._enabled():
            return False
        if eval_index < max(self.start_step, 1):
            return False
        return eval_index % int(self.freq) == 0


@dataclass(frozen=True)
class LoggingSchedules:
    # step 策略是所有日志调度共享的横轴语义
    by_global_step: bool
    first_step: bool
    train_metrics: LogSchedule
    train_behavior: LogSchedule
    train_optim: LogSchedule
    train_param_norm: LogSchedule
    media_train: LogSchedule
    media_eval: LogSchedule
    media_eval_trigger: str
    media_eval_policy: str
    eval_schedule: LogSchedule
    eval_fraction: float
    eval_behavior: LogSchedule
    runtime_perf: LogSchedule
    system_gpu: LogSchedule
    system_gpu_enabled: bool
    system_ddp_log_once: bool

    def should_log(self, schedule: LogSchedule, epoch: int, num_batches: int, batch_idx: int) -> bool:
        return schedule.should_step(
            epoch,
            num_batches,
            batch_idx,
            by_global_step=self.by_global_step,
            first_step=self.first_step,
        )

    def should_eval_epoch(self, epoch: int) -> bool:
        return self.eval_schedule.should_epoch(epoch)

    def should_eval_behavior(self, eval_index: int) -> bool:
        return self.eval_behavior.should_eval(eval_index)

    def should_eval_media(self, eval_index: int) -> bool:
        return self.media_eval_trigger == "eval" and self.media_eval.should_eval(eval_index)

    def should_system_gpu(self, epoch: int, num_batches: int, batch_idx: int) -> bool:
        return self.system_gpu_enabled and self.should_log(self.system_gpu, epoch, num_batches, batch_idx)


def build_logging_schedules(logging_config: Mapping[str, Any]) -> LoggingSchedules:
    # 新配置优先读取分组字段；legacy_key 只用于兼容旧 YAML 或外部脚本直接构造的配置
    logging_config = _as_mapping(logging_config)
    by_global_step = _read_bool(logging_config, ("step", "by_global_step"), "by_global_step", True)
    first_step = _read_bool(logging_config, ("step", "first_step"), "first_step", False)

    train_metrics = _read_schedule(
        logging_config,
        ("train", "metrics"),
        legacy_freq_key="metric_log_freq",
        default_freq=0,
        default_unit="step",
    )
    train_behavior = _read_schedule(
        logging_config,
        ("train", "behavior"),
        legacy_freq_key="heavy_metric_log_freq",
        legacy_start_key="heavy_metric_start_step",
        default_freq=train_metrics.freq,
        default_unit="step",
    )
    train_optim = _read_schedule(
        logging_config,
        ("train", "optim"),
        legacy_freq_key="optim_log_freq",
        default_freq=train_metrics.freq,
        default_unit="step",
    )
    train_param_norm = _read_schedule(
        logging_config,
        ("train", "param_norm"),
        legacy_freq_key="param_norm_log_freq",
        default_freq=0,
        default_unit="step",
    )
    media_train = _read_schedule(
        logging_config,
        ("media", "train"),
        legacy_freq_key="image_log_freq",
        legacy_start_key="image_start_step",
        default_freq=0,
        default_unit="step",
    )
    eval_schedule = _read_schedule(
        logging_config,
        ("eval", "schedule"),
        legacy_freq_key="eval_freq",
        default_freq=1,
        default_unit="epoch",
    )
    eval_fraction = _read_float(logging_config, ("eval", "schedule", "fraction"), "eval_fraction", 1.0)
    eval_behavior = _read_schedule(
        logging_config,
        ("eval", "behavior"),
        legacy_freq_key=None,
        default_freq=1,
        default_unit="eval",
    )
    media_eval = _read_schedule(
        logging_config,
        ("media", "eval"),
        legacy_freq_key=None,
        default_freq=1,
        default_unit="eval",
    )
    media_eval_trigger = _read_str(logging_config, ("media", "eval", "trigger"), None, "eval")
    media_eval_policy = _read_str(logging_config, ("media", "eval", "policy"), None, "last_batch_per_eval")

    runtime_perf = _read_schedule(
        logging_config,
        ("runtime", "perf"),
        legacy_freq_key="perf_log_freq",
        default_freq=0,
        default_unit="step",
    )
    system_gpu_enabled = _read_bool(logging_config, ("system", "gpu", "enabled"), None, True)
    system_gpu = _read_schedule(
        logging_config,
        ("system", "gpu"),
        legacy_freq_key=None,
        default_freq=runtime_perf.freq,
        default_unit="step",
    )
    system_ddp_log_once = _read_bool(logging_config, ("system", "ddp", "log_once"), None, True)

    return LoggingSchedules(
        by_global_step=by_global_step,
        first_step=first_step,
        train_metrics=train_metrics,
        train_behavior=train_behavior,
        train_optim=train_optim,
        train_param_norm=train_param_norm,
        media_train=media_train,
        media_eval=media_eval,
        media_eval_trigger=media_eval_trigger,
        media_eval_policy=media_eval_policy,
        eval_schedule=eval_schedule,
        eval_fraction=eval_fraction,
        eval_behavior=eval_behavior,
        runtime_perf=runtime_perf,
        system_gpu=system_gpu,
        system_gpu_enabled=system_gpu_enabled,
        system_ddp_log_once=system_ddp_log_once,
    )
