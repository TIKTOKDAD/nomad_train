# ============================================================
# Trainer - common training and evaluation loop
# ============================================================
# 本文件是 training_base 的主调度器：
# 1. Trainer 不直接知道 GNM/ViNT/NoMaD 的内部细节，只调用 Algorithm 协议
# 2. 训练阶段负责前向、AMP、反向、优化器更新、日志、重指标、可视化、性能统计
# 3. 评估阶段负责抽样评估、分布式聚合、最后一个 batch 可视化和调度器指标汇总
import itertools
import inspect
import time

import torch
import tqdm

from training_base.callbacks import CallbackManager
from training_base.core.checkpoint import restore_rng_state
from training_base.core.config import prepare_logging_config, save_run_configs
from training_base.loggers import Recorder
from training_base.loggers.key_format import format_metric_logs
from training_base.loggers.metric_store import reduce_metric_logs_distributed
from training_base.loggers.schedule import build_logging_schedules
from training_base.core.native_utils import (
    autocast,
    distributed_barrier,
    make_grad_scaler,
    rank0_tqdm_enabled,
    scale_backward_step,
)
from training_base.core.runtime import RuntimeContext, wrap_distributed_model


# 训练器：对算法与数据模块进行统一调度
class Trainer:
    """Paper-agnostic high-performance training loop for navigation algorithms."""

    # 初始化训练器：保存配置与上下文，构建日志与回调管理器
    def __init__(self, config, algorithm, datamodule, context: RuntimeContext) -> None:
        # config 是已合并/规范化后的完整配置，后续 runtime 字段会被补充实际 batch/路径信息
        self.config = config
        # algorithm 封装模型差异，Trainer 只认 build/step/metrics/visualize 等统一接口
        self.algorithm = algorithm
        self.datamodule = datamodule
        self.context = context
        # global_step 用于日志横轴；断点恢复后会从 checkpoint 中恢复
        self.global_step = 0
        self.recorder = None
        self.callbacks = None
        self._prepare_batch_accepts_config = "config" in inspect.signature(self.algorithm.prepare_batch).parameters

    def _save_effective_config(self) -> None:
        runtime = self.config["runtime"]
        if self.context.is_main_process:
            runtime["config_artifact_paths"] = save_run_configs(self.config, runtime["project_folder"])
        distributed_barrier()

    def _init_services(self) -> None:
        prepare_logging_config(self.config)
        self.recorder = Recorder(self.config, self.context)
        self.callbacks = CallbackManager(self.config, self.context)

    # 将聚合后的指标打印到控制台（仅主进程）
    def _print_store(self, store, mode: str, epoch: int, batch_idx: int, num_batches: int) -> None:
        if not self.context.is_main_process:
            return
        message = store.display_latest()
        if message:
            self.recorder.log_status(f"(轮次 {epoch}) (批次 {batch_idx}/{num_batches - 1}) {mode}: {message}")

    def _run_train_step(
        self,
        *,
        model,
        optimizer,
        batch,
        transform,
        device,
        config,
        epoch,
        batch_idx,
        num_batches,
        grad_scaler,
        state,
        schedules,
    ):
        # 单个训练 step 的核心路径：prepare batch -> forward/loss -> backward/step -> hook/log stats
        runtime = config["runtime"]
        # 只有主进程且命中媒体日志频率时才准备可视化图片，减少无谓 resize
        should_images = self.context.is_main_process and schedules.should_log(schedules.media_train, epoch, num_batches, batch_idx)
        prepared = self._prepare_batch(batch, transform, device, mode="train", should_log_images=should_images, config=config)

        # AMP 上下文由 runtime 配置控制，CPU 或关闭 AMP 时退化为普通上下文
        with autocast(device, bool(runtime.get("amp", False)), runtime.get("amp_dtype", "fp16")):
            result = self.algorithm.train_step(model, prepared, state, config)

        # 优化器统计可能需要额外遍历梯度/参数，因此单独按 schedule 控制
        should_optim_stats = self.context.is_main_process and schedules.should_log(
            schedules.train_optim,
            epoch,
            num_batches,
            batch_idx,
        )
        should_param_norm = self.context.is_main_process and schedules.should_log(
            schedules.train_param_norm,
            epoch,
            num_batches,
            batch_idx,
        )
        # scale_backward_step 封装 backward、梯度裁剪、optimizer.step 和 AMP scaler 更新
        optimizer_stats = scale_backward_step(
            result.loss,
            optimizer,
            grad_scaler,
            model=model,
            clip_config=config.get("optimizer", {}).get("gradient_clip", {}),
            collect_stats=should_optim_stats,
            collect_param_norm=should_param_norm,
        )
        # 算法和回调都可以在 optimizer.step 后维护额外状态，例如 NoMaD EMA
        self.algorithm.after_optimizer_step(model, state, config)
        self.callbacks.call("after_optimizer_step", model=model, algorithm=self.algorithm, state=state, config=config)
        self.global_step += 1
        if should_optim_stats or should_param_norm:
            # 优化器健康指标只在主进程记录，避免 DDP 多 rank 重复写日志
            self.callbacks.call(
                "log_optimizer_step",
                recorder=self.recorder,
                optimizer=optimizer,
                stats=optimizer_stats,
                global_step=self.global_step,
            )
        return prepared, result, should_images

    def _prepare_batch(self, batch, transform, device, *, mode: str, should_log_images: bool, config):
        # 兼容旧 Algorithm.prepare_batch 签名：有些算法不接收 config 参数
        kwargs = {
            "batch": batch,
            "transform": transform,
            "device": device,
            "mode": mode,
            "should_log_images": should_log_images,
        }
        if self._prepare_batch_accepts_config:
            kwargs["config"] = config
        return self.algorithm.prepare_batch(**kwargs)

    def _log_train_light_metrics(self, *, model, prepared, result, state, config, epoch, batch_idx, num_batches, schedules, metric_store, iterator, show_tqdm) -> None:
        # 轻量指标跟随 train_metrics schedule，通常每 N step 写一次
        if not schedules.should_log(schedules.train_metrics, epoch, num_batches, batch_idx):
            return
        metric_logs = dict(result.logs)
        metric_logs.update(self.algorithm.light_metrics(model, prepared, result, state, config, mode="train"))
        # DDP 下先跨 rank 平均，再写入主进程的 MetricStore
        metric_logs = reduce_metric_logs_distributed(metric_logs, self.context.device)
        if not self.context.is_main_process:
            return
        metric_store.update(metric_logs)
        if show_tqdm and result.loss is not None:
            iterator.set_postfix(loss=float(result.loss.detach().float().item()))
        self._print_store(metric_store, "train", epoch, batch_idx, num_batches)
        self.recorder.log_metrics(
            format_metric_logs(metric_store.latest(), "train"),
            step=self.global_step,
            commit=True,
        )

    def _log_train_heavy_metrics(self, *, model, prepared, state, config, epoch, batch_idx, num_batches, schedules, heavy_store) -> None:
        # 重指标只在主进程、命中 train_behavior schedule 时运行
        if not self.context.is_main_process or not schedules.should_log(schedules.train_behavior, epoch, num_batches, batch_idx):
            return
        with torch.inference_mode():
            # 重指标不参与训练，使用 eval/EMA 模型并关闭梯度
            metric_model = self.algorithm.model_for_eval(model, state)
            metric_logs = self.algorithm.heavy_metrics(metric_model, prepared, state, config, mode="train")
        if not metric_logs:
            return
        heavy_store.update(metric_logs)
        self._print_store(heavy_store, "train_heavy", epoch, batch_idx, num_batches)
        self.recorder.log_metrics(
            format_metric_logs(heavy_store.latest(), "train", kind="behavior"),
            step=self.global_step,
            commit=True,
        )

    def _log_train_visualization(self, *, model, prepared, result, state, config, project_folder, epoch, batch_idx, num_batches, should_images) -> None:
        # should_images 在 _run_train_step 中提前计算，避免算法重复判断 schedule
        if not should_images:
            return
        with torch.inference_mode():
            # 可视化不影响训练状态，默认同样使用 eval/EMA 模型
            viz_model = self.algorithm.model_for_eval(model, state)
            self.algorithm.visualize(
                model=viz_model,
                prepared=prepared,
                result=result,
                state=state,
                config=config,
                mode="train",
                project_folder=project_folder,
                epoch=epoch,
                batch_idx=batch_idx,
                num_batches=num_batches,
                global_step=self.global_step,
                recorder=self.recorder,
            )

    def _log_train_runtime_perf(self, *, result, device, epoch, batch_idx, num_batches, schedules, data_time, compute_time, step_time) -> None:
        # 训练性能指标走回调系统，便于按需接入控制台/W&B/系统监控
        if schedules.should_log(schedules.runtime_perf, epoch, num_batches, batch_idx):
            self.callbacks.call(
                "log_perf",
                recorder=self.recorder,
                mode="train",
                epoch=epoch,
                batch_idx=batch_idx,
                batch_size=result.batch_size or 0,
                data_time=data_time,
                compute_time=compute_time,
                step_time=step_time,
                device=device,
                global_step=self.global_step,
            )
        if schedules.should_system_gpu(epoch, num_batches, batch_idx):
            # GPU 显存等系统信息单独由 system.gpu schedule 控制
            self.callbacks.call(
                "log_system_gpu",
                recorder=self.recorder,
                device=device,
                global_step=self.global_step,
            )

    def _run_eval_step(self, *, eval_model, batch, transform, device, config, state, eval_type, should_images):
        # 评估 step 与训练 step 共用 prepare_batch，保证图像预处理和标签搬运一致
        runtime = config["runtime"]
        prepared = self._prepare_batch(batch, transform, device, mode=eval_type, should_log_images=should_images, config=config)
        with autocast(device, bool(runtime.get("amp", False)), runtime.get("amp_dtype", "fp16")):
            result = self.algorithm.eval_step(eval_model, prepared, state, config)
        return prepared, result

    def _restore_grad_scaler(self, grad_scaler, resume_state) -> None:
        # 只有 AMP scaler 启用且 checkpoint 中确实保存了状态才恢复
        if grad_scaler is None or not grad_scaler.is_enabled():
            return
        if not isinstance(resume_state.latest_checkpoint, dict):
            return
        scaler_state = resume_state.latest_checkpoint.get("grad_scaler")
        if scaler_state:
            grad_scaler.load_state_dict(scaler_state)

    def _log_eval_metrics(self, *, metric_store, heavy_store, eval_type) -> None:
        # 评估日志写平均值；heavy_store 会被打到 behavior 命名空间
        data_log = format_metric_logs(metric_store.average(), eval_type)
        data_log.update(format_metric_logs(heavy_store.average(), eval_type, kind="behavior"))
        if data_log:
            self.recorder.log_metrics(data_log, step=self.global_step, commit=False)

    def _log_eval_visualization(self, *, eval_model, prepared, result, state, config, eval_type, project_folder, epoch, num_batches) -> None:
        # 评估可视化统一伪装成最后一个 batch，便于 visualizer 复用命名逻辑
        self.algorithm.visualize(
            model=eval_model,
            prepared=prepared,
            result=result,
            state=state,
            config=config,
            mode=eval_type,
            project_folder=project_folder,
            epoch=epoch,
            batch_idx=max(num_batches - 1, 0),
            num_batches=num_batches,
            global_step=self.global_step,
            recorder=self.recorder,
        )

    # 单个训练 epoch：包含前向、反向、优化、日志、可视化与性能统计
    def _train_epoch(self, *, model, optimizer, dataloader, transform, device, project_folder, config, epoch, grad_scaler, state):
        model.train()
        logging = config["logging"]
        schedules = build_logging_schedules(logging)
        num_batches = len(dataloader)

        # 轻量指标与重指标分别统计，避免重计算影响日志频率
        metric_store = self.recorder.metric_store()
        heavy_store = self.recorder.metric_store()
        show_tqdm = rank0_tqdm_enabled(self.context.is_main_process)
        end_time = time.perf_counter()

        # 迭代训练数据，按配置控制日志与可视化节奏
        with tqdm.tqdm(
            dataloader,
            desc=f"{self.algorithm.name} 训练轮次 {epoch}",
            leave=False,
            disable=not show_tqdm,
            dynamic_ncols=True,
        ) as iterator:
            for batch_idx, batch in enumerate(iterator):
                # 统计数据加载时间与计算时间
                data_time = time.perf_counter() - end_time
                compute_start = time.perf_counter()
                prepared, result, should_images = self._run_train_step(
                    model=model,
                    optimizer=optimizer,
                    batch=batch,
                    transform=transform,
                    device=device,
                    config=config,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    num_batches=num_batches,
                    grad_scaler=grad_scaler,
                    state=state,
                    schedules=schedules,
                )

                # 轻量指标日志
                # 轻量指标只复用当前 step 已经算出的预测，不额外跑完整采样或可视化
                self._log_train_light_metrics(
                    model=model,
                    prepared=prepared,
                    result=result,
                    state=state,
                    config=config,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    num_batches=num_batches,
                    schedules=schedules,
                    metric_store=metric_store,
                    iterator=iterator,
                    show_tqdm=show_tqdm,
                )

                # 重计算指标日志（推理模式）
                # NoMaD 行为指标等会执行反向扩散采样，因此用独立频率 heavy_freq 控制
                self._log_train_heavy_metrics(
                    model=model,
                    prepared=prepared,
                    state=state,
                    config=config,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    num_batches=num_batches,
                    schedules=schedules,
                    heavy_store=heavy_store,
                )

                # 可视化输出（如轨迹、动作分布等）
                # 可视化统一使用 eval/EMA 模型，避免 dropout/BN 训练状态影响图像解释
                self._log_train_visualization(
                    model=model,
                    prepared=prepared,
                    result=result,
                    state=state,
                    config=config,
                    project_folder=project_folder,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    num_batches=num_batches,
                    should_images=should_images,
                )

                compute_time = time.perf_counter() - compute_start
                step_time = data_time + compute_time
                # 性能监控回调：数据/计算/步长耗时
                # data_time 近似 DataLoader 等待时间，compute_time 包含 prepare_batch 之后的训练计算
                self._log_train_runtime_perf(
                    result=result,
                    device=device,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    num_batches=num_batches,
                    schedules=schedules,
                    data_time=data_time,
                    compute_time=compute_time,
                    step_time=step_time,
                )
                end_time = time.perf_counter()

    # 评估流程：支持部分采样评估与重指标汇总
    def _evaluate(self, *, eval_type, model, dataloader, transform, device, project_folder, config, epoch, eval_index, distributed, state):
        # 评估模型可能是 EMA averaged_model，也可能是 unwrap 后的即时模型
        eval_model = self.algorithm.model_for_eval(model, state)
        eval_model.eval()
        logging = config["logging"]
        schedules = build_logging_schedules(logging)
        dataloader_len = len(dataloader)
        # logging.eval.schedule.fraction 允许只评估一部分 batch，最少保留 1 个 batch，避免空评估
        num_batches = min(max(int(dataloader_len * float(schedules.eval_fraction)), 1), dataloader_len) if dataloader_len > 0 else 0
        if schedules.media_eval_trigger != "eval":
            raise ValueError(f"不支持的 eval 图片触发源: {schedules.media_eval_trigger}")
        if schedules.media_eval_policy != "last_batch_per_eval":
            raise ValueError(f"不支持的 eval 图片策略: {schedules.media_eval_policy}")
        should_eval_media = self.context.is_main_process and self.algorithm.visualize_eval_last and schedules.should_eval_media(eval_index)
        should_eval_behavior = schedules.should_eval_behavior(eval_index)
        metric_store = self.recorder.metric_store()
        heavy_store = self.recorder.metric_store()
        last_prepared = None
        last_result = None

        # 评估过程不计算梯度
        with torch.inference_mode():
            show_tqdm = rank0_tqdm_enabled(self.context.is_main_process)
            iterator = tqdm.tqdm(
                itertools.islice(dataloader, num_batches),
                total=num_batches,
                dynamic_ncols=True,
                desc=f"正在评估 {eval_type} 第 {epoch} 轮",
                leave=False,
                disable=not show_tqdm,
            )
            for batch_idx, batch in enumerate(iterator):
                # 默认只给最后一个评估 batch 生成图片，控制磁盘和 W&B 开销
                should_images = should_eval_media and batch_idx == num_batches - 1
                # 与训练一致的 batch 准备流程
                prepared, result = self._run_eval_step(
                    eval_model=eval_model,
                    batch=batch,
                    transform=transform,
                    device=device,
                    config=config,
                    state=state,
                    eval_type=eval_type,
                    should_images=should_images,
                )
                # eval_step 的 logs 是基础损失，light_metrics 是按配置追加的轻量评估项
                metric_logs = dict(result.logs)
                metric_logs.update(self.algorithm.light_metrics(eval_model, prepared, result, state, config, mode=eval_type))
                metric_store.update(metric_logs)
                if should_eval_behavior and batch_idx == num_batches - 1:
                    # eval behavior 按 eval 次数触发，只在最后一个 batch 采样，控制扩散采样开销
                    heavy_logs = self.algorithm.heavy_metrics(eval_model, prepared, state, config, mode=eval_type)
                    if heavy_logs:
                        heavy_store.update(heavy_logs)
                last_prepared = prepared
                last_result = result

        # 分布式场景下合并各进程统计
        if distributed:
            metric_store.reduce_distributed(device)
            heavy_store.reduce_distributed(device)

        # 仅主进程记录评估日志与可视化
        if self.context.is_main_process and num_batches > 0:
            # eval_type 是数据集名，如 huron_test；格式化后进入 eval/{dataset}/...
            self._log_eval_metrics(metric_store=metric_store, heavy_store=heavy_store, eval_type=eval_type)

            if should_eval_media and last_prepared is not None and last_result is not None:
                self._log_eval_visualization(
                    eval_model=eval_model,
                    prepared=last_prepared,
                    result=last_result,
                    state=state,
                    config=config,
                    eval_type=eval_type,
                    project_folder=project_folder,
                    epoch=epoch,
                    num_batches=num_batches,
                )

        summary = metric_store.average()
        summary.update(heavy_store.average())
        return summary

    # 总训练入口：构建模型/目标/优化器，执行多轮 epoch
    def fit(self) -> None:
        # setup 会构建 Dataset/DataLoader，并把 dataset_metadata 写回 config["data"]
        self.datamodule.setup(build_lmdb_only=False)
        self._save_effective_config()
        self._init_services()
        loader_summary = getattr(self.datamodule, "loader_summary", None)
        if loader_summary:
            self.recorder.log_status(loader_summary)
        runtime = self.config["runtime"]
        schedules = build_logging_schedules(self.config["logging"])
        # system/runtime 静态配置交给性能回调记录，Trainer 只负责触发生命周期 hook
        self.callbacks.call("log_runtime_config", recorder=self.recorder, config=self.config, global_step=self.global_step)

        # 构建模型、损失目标与优化器/调度器
        model, model_extras = self.algorithm.build_model(self.config)
        objective = self.algorithm.build_objective(self.config)
        optimizer, scheduler = self.algorithm.configure_optimizers(model, self.config)
        # 断点恢复：加载模型/优化器状态，并恢复全局步数
        resume_state = self.algorithm.prepare_resume(model=model, optimizer=optimizer, scheduler=scheduler, config=self.config, device=self.context.device)
        self.global_step = int((resume_state.extra or {}).get("global_step", 0))
        if isinstance(resume_state.latest_checkpoint, dict):
            self.callbacks.load_state_dict(resume_state.latest_checkpoint.get("callback_state", {}))

        # 设备转移与分布式封装
        # 顺序很关键：先把模型搬到 device，再包 DDP，再创建算法状态/EMA
        model = model.to(self.context.device)
        model = wrap_distributed_model(model, runtime, self.context)
        state = self.algorithm.create_state(model, model_extras, objective, self.config, self.context.device, resume_state)
        if isinstance(resume_state.latest_checkpoint, dict):
            restore_rng_state(resume_state.latest_checkpoint.get("rng_state"), path=resume_state.load_project_folder or "<checkpoint>")

        # AMP 梯度缩放器（仅在 CUDA + AMP 时启用）
        grad_scaler = make_grad_scaler(self.context.device, bool(runtime.get("amp", False)), bool(runtime.get("use_grad_scaler", True)))
        self._restore_grad_scaler(grad_scaler, resume_state)
        # runtime.epochs 表示目标总 epoch 数；resume 后继续跑到该总数，而不是额外追加 epochs 轮。
        end_epoch = int(runtime["epochs"])

        try:
            if resume_state.current_epoch >= end_epoch:
                if self.context.is_main_process:
                    self.recorder.log_status(
                        f"恢复点已经到达目标训练轮次: current_epoch={resume_state.current_epoch}, "
                        f"runtime.epochs={end_epoch}。无需继续训练。"
                    )
                return
            # 训练与评估主循环
            # eval_index 是 1-based 的 eval 事件计数；断点恢复时从已完成 epoch 推导，保持 unit=eval 的节奏稳定
            eval_index = sum(1 for completed_epoch in range(resume_state.current_epoch) if schedules.should_eval_epoch(completed_epoch))
            for epoch in range(resume_state.current_epoch, end_epoch):
                # 分布式训练时需要为采样器设置 epoch 保证洗牌一致
                if hasattr(self.datamodule, "set_train_epoch"):
                    self.datamodule.set_train_epoch(epoch)
                elif self.datamodule.train_sampler is not None:
                    self.datamodule.train_sampler.set_epoch(epoch)

                # 训练阶段
                if bool(runtime["train"]):
                    if self.context.is_main_process:
                        self.recorder.log_status(f"开始 {self.algorithm.name} 训练轮次 {epoch}/{end_epoch - 1}")
                    self._train_epoch(
                        model=model,
                        optimizer=optimizer,
                        dataloader=self.datamodule.train_loader,
                        transform=self.datamodule.transform,
                        device=self.context.device,
                        project_folder=runtime["project_folder"],
                        config=self.config,
                        epoch=epoch,
                        grad_scaler=grad_scaler,
                        state=state,
                    )

                # 评估阶段（可配置评估频率与分布式评估）
                eval_summaries = {}
                distributed_eval = bool(runtime.get("distributed_eval", False))
                should_eval = schedules.should_eval_epoch(epoch)
                # 默认只在主进程评估；distributed_eval=True 时每个进程评估自己的子集再聚合
                if should_eval and (self.context.is_main_process or (self.context.distributed and distributed_eval)):
                    eval_index += 1
                    if hasattr(self.datamodule, "set_eval_epoch"):
                        self.datamodule.set_eval_epoch(eval_index)
                    for dataset_type, loader in self.datamodule.test_dataloaders.items():
                        if self.context.is_main_process:
                            self.recorder.log_status(f"开始 {dataset_type} 测试轮次 {epoch}/{end_epoch - 1}")
                        eval_summaries[dataset_type] = self._evaluate(
                            eval_type=dataset_type,
                            model=model,
                            dataloader=loader,
                            transform=self.datamodule.transform,
                            device=self.context.device,
                            project_folder=runtime["project_folder"],
                            config=self.config,
                            epoch=epoch,
                            eval_index=eval_index,
                            distributed=self.context.distributed and distributed_eval,
                            state=state,
                        )

                # 根据评估结果更新调度器
                if bool(runtime["train"]) and scheduler is not None:
                    self.algorithm.step_scheduler(scheduler, eval_summaries, self.config)

                # 调度器可能在 epoch 末尾更新学习率，具体日志 key 与主进程判断由 optim_monitor 负责
                self.callbacks.call("log_epoch_optimizer", recorder=self.recorder, optimizer=optimizer, global_step=self.global_step)

                # 回调：epoch 结束
                self.callbacks.call(
                    "on_epoch_end",
                    epoch=epoch,
                    global_step=self.global_step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    algorithm=self.algorithm,
                    state=state,
                    config=self.config,
                    eval_summaries=eval_summaries,
                    callback_manager=self.callbacks,
                    grad_scaler=grad_scaler,
                )
                # 进程同步，避免不同步进入下一轮
                distributed_barrier()
        finally:
            if self.callbacks is not None:
                self.callbacks.close()
            if self.context.is_main_process and self.recorder is not None:
                self.recorder.close()
