# ============================================================
# Trainer - common training and evaluation loop
# ============================================================
# 本文件是 training_base 的主调度器：
# 1. Trainer 不直接知道 GNM/ViNT/NoMaD 的内部细节，只调用 Algorithm 协议
# 2. 训练阶段负责前向、AMP、反向、优化器更新、日志、重指标、可视化、性能统计
# 3. 评估阶段负责抽样评估、分布式聚合、最后一个 batch 可视化和调度器指标汇总
import itertools
import time

import torch
import tqdm

from training_base.callbacks import CallbackManager
from training_base.loggers import Recorder
from training_base.loggers.key_format import format_metric_logs
from training_base.core.native_utils import (
    autocast,
    distributed_barrier,
    make_grad_scaler,
    rank0_tqdm_enabled,
    scale_backward_step,
    should_log_event,
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
        self.recorder = Recorder(config, context)
        self.callbacks = CallbackManager(config, context)

    # 将聚合后的指标打印到控制台（仅主进程）
    def _print_store(self, store, mode: str, epoch: int, batch_idx: int, num_batches: int) -> None:
        if not self.context.is_main_process:
            return
        message = store.display_latest()
        if message:
            print(f"(轮次 {epoch}) (批次 {batch_idx}/{num_batches - 1}) {mode}: {message}")

    # 单个训练 epoch：包含前向、反向、优化、日志、可视化与性能统计
    def _train_epoch(self, *, model, optimizer, dataloader, transform, device, project_folder, config, epoch, grad_scaler, state):
        model.train()
        runtime = config["runtime"]
        logging = config["logging"]
        num_batches = len(dataloader)
        # 只让主进程负责普通日志和图片，避免 DDP 多进程重复写入
        print_freq = int(logging.get("metric_log_freq", 0)) if self.context.is_main_process else 0
        image_freq = int(logging.get("image_log_freq", 0)) if self.context.is_main_process else 0
        heavy_freq = int(logging.get("heavy_metric_log_freq", print_freq)) if self.context.is_main_process else 0
        # 性能日志允许所有进程计算，但 Recorder/Sink 会按自身启用状态决定是否写出
        perf_freq = int(logging.get("perf_log_freq", 0))
        log_by_global_step = bool(logging.get("by_global_step", True))
        log_first_step = bool(logging.get("first_step", False))
        image_start = int(logging.get("image_start_step", 0))
        heavy_start = int(logging.get("heavy_metric_start_step", 0))

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
                # should_images 同时控制 batch 准备是否保留 CPU 可视化图，避免每步都 resize
                should_images = should_log_event(image_freq, epoch, num_batches, batch_idx, log_by_global_step, image_start, log_first_step)
                # 数据准备由算法自行处理（如张量化、归一化、标签构造）
                prepared = self.algorithm.prepare_batch(batch, transform, device, mode="train", should_log_images=should_images)

                # AMP 环境下执行前向与损失计算
                with autocast(device, bool(runtime.get("amp", False)), runtime.get("amp_dtype", "fp16")):
                    result = self.algorithm.train_step(model, prepared, state, config)

                # 反向传播 + 可选梯度裁剪 + 参数更新
                # scale_backward_step 内部负责 zero_grad、GradScaler、clip_grad 和 optimizer.step
                scale_backward_step(
                    result.loss,
                    optimizer,
                    grad_scaler,
                    model=model,
                    clip_config=config.get("optimizer", {}).get("gradient_clip", {}),
                )
                self.algorithm.after_optimizer_step(model, state, config)
                # 回调：允许外部逻辑在更新后介入
                self.callbacks.call("after_optimizer_step", model=model, algorithm=self.algorithm, state=state, config=config)
                self.global_step += 1

                # 轻量指标日志
                # 轻量指标只复用当前 step 已经算出的预测，不额外跑完整采样或可视化
                if should_log_event(print_freq, epoch, num_batches, batch_idx, log_by_global_step, 0, log_first_step):
                    metric_logs = dict(result.logs)
                    metric_logs.update(self.algorithm.light_metrics(model, prepared, result, state, config, mode="train"))
                    metric_store.update(metric_logs)
                    if show_tqdm and result.loss is not None:
                        iterator.set_postfix(loss=float(result.loss.detach().float().item()))
                    self._print_store(metric_store, "train", epoch, batch_idx, num_batches)
                    # 轻量指标只做命名格式化，不触发额外前向/采样/画图
                    self.recorder.log_metrics(
                        format_metric_logs(metric_store.latest(), "train"),
                        step=self.global_step,
                        commit=True,
                    )

                # 重计算指标日志（推理模式）
                # NoMaD 行为指标等会执行反向扩散采样，因此用独立频率 heavy_freq 控制
                if should_log_event(heavy_freq, epoch, num_batches, batch_idx, log_by_global_step, heavy_start, log_first_step):
                    with torch.inference_mode():
                        metric_model = self.algorithm.model_for_eval(model, state)
                        metric_logs = self.algorithm.heavy_metrics(metric_model, prepared, state, config, mode="train")
                    if metric_logs:
                        heavy_store.update(metric_logs)
                        self._print_store(heavy_store, "train_heavy", epoch, batch_idx, num_batches)
                        # 重指标单独走 behavior 命名空间，和普通 loss/action 曲线分离
                        self.recorder.log_metrics(
                            format_metric_logs(heavy_store.latest(), "train", kind="behavior"),
                            step=self.global_step,
                            commit=True,
                        )

                # 可视化输出（如轨迹、动作分布等）
                # 可视化统一使用 eval/EMA 模型，避免 dropout/BN 训练状态影响图像解释
                if should_images:
                    with torch.inference_mode():
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

                compute_time = time.perf_counter() - compute_start
                step_time = data_time + compute_time
                # 性能监控回调：数据/计算/步长耗时
                # data_time 近似 DataLoader 等待时间，compute_time 包含 prepare_batch 之后的训练计算
                if should_log_event(perf_freq, epoch, num_batches, batch_idx, log_by_global_step, 0, log_first_step):
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
                    )
                end_time = time.perf_counter()

    # 评估流程：支持部分采样评估与重指标汇总
    def _evaluate(self, *, eval_type, model, dataloader, transform, device, project_folder, config, epoch, distributed, state):
        # 评估模型可能是 EMA averaged_model，也可能是 unwrap 后的即时模型
        eval_model = self.algorithm.model_for_eval(model, state)
        eval_model.eval()
        runtime = config["runtime"]
        logging = config["logging"]
        heavy_freq = int(logging.get("heavy_metric_log_freq", logging.get("metric_log_freq", 0)))
        heavy_start = int(logging.get("heavy_metric_start_step", 0))
        eval_heavy_every_eval = bool(logging.get("eval_heavy_every_eval", True))
        log_by_global_step = bool(logging.get("by_global_step", True))
        log_first_step = bool(logging.get("first_step", False))
        dataloader_len = len(dataloader)
        # eval_fraction 允许只评估一部分 batch，最少保留 1 个 batch，避免空评估
        num_batches = min(max(int(dataloader_len * float(runtime["eval_fraction"])), 1), dataloader_len) if dataloader_len > 0 else 0
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
                should_images = self.context.is_main_process and self.algorithm.visualize_eval_last and batch_idx == num_batches - 1
                # 与训练一致的 batch 准备流程
                prepared = self.algorithm.prepare_batch(batch, transform, device, mode=eval_type, should_log_images=should_images)
                with autocast(device, bool(runtime.get("amp", False)), runtime.get("amp_dtype", "fp16")):
                    result = self.algorithm.eval_step(eval_model, prepared, state, config)
                # eval_step 的 logs 是基础损失，light_metrics 是按配置追加的轻量评估项
                metric_logs = dict(result.logs)
                metric_logs.update(self.algorithm.light_metrics(eval_model, prepared, result, state, config, mode=eval_type))
                metric_store.update(metric_logs)
                should_heavy = should_log_event(
                    heavy_freq,
                    epoch,
                    num_batches,
                    batch_idx,
                    log_by_global_step,
                    heavy_start,
                    log_first_step,
                )
                if eval_heavy_every_eval and batch_idx == num_batches - 1:
                    # 评估阶段保证每个数据集至少有一次 behavior 曲线，避免频率未对齐导致面板缺失
                    should_heavy = True
                if should_heavy:
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
            data_log = format_metric_logs(metric_store.average(), eval_type)
            data_log.update(format_metric_logs(heavy_store.average(), eval_type, kind="behavior"))
            if data_log:
                self.recorder.log_metrics(data_log, step=self.global_step, commit=False)

            if last_prepared is not None and last_result is not None:
                self.algorithm.visualize(
                    model=eval_model,
                    prepared=last_prepared,
                    result=last_result,
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

        summary = metric_store.average()
        summary.update(heavy_store.average())
        return summary

    # 总训练入口：构建模型/目标/优化器，执行多轮 epoch
    def fit(self) -> None:
        # setup 会构建 Dataset/DataLoader，并把 dataset_metadata 写回 config["data"]
        self.datamodule.setup(build_lmdb_only=False)
        runtime = self.config["runtime"]
        # system/runtime 静态配置交给性能回调记录，Trainer 只负责触发生命周期 hook
        self.callbacks.call("log_runtime_config", recorder=self.recorder, config=self.config, global_step=self.global_step)

        # 构建模型、损失目标与优化器/调度器
        model, model_extras = self.algorithm.build_model(self.config)
        objective = self.algorithm.build_objective(self.config)
        optimizer, scheduler = self.algorithm.configure_optimizers(model, self.config)
        # 断点恢复：加载模型/优化器状态，并恢复全局步数
        resume_state = self.algorithm.prepare_resume(model=model, optimizer=optimizer, scheduler=scheduler, config=self.config, device=self.context.device)
        self.global_step = int((resume_state.extra or {}).get("global_step", 0))

        # 设备转移与分布式封装
        # 顺序很关键：先把模型搬到 device，再包 DDP，再创建算法状态/EMA
        model = model.to(self.context.device)
        model = wrap_distributed_model(model, runtime, self.context)
        state = self.algorithm.create_state(model, model_extras, objective, self.config, self.context.device, resume_state)

        # AMP 梯度缩放器（仅在 CUDA + AMP 时启用）
        grad_scaler = make_grad_scaler(self.context.device, bool(runtime.get("amp", False)), bool(runtime.get("use_grad_scaler", True)))
        end_epoch = resume_state.current_epoch + int(runtime["epochs"])

        try:
            # 训练与评估主循环
            for epoch in range(resume_state.current_epoch, end_epoch):
                # 分布式训练时需要为采样器设置 epoch 保证洗牌一致
                if self.datamodule.train_sampler is not None:
                    self.datamodule.train_sampler.set_epoch(epoch)

                # 训练阶段
                if bool(runtime["train"]):
                    if self.context.is_main_process:
                        print(f"开始 {self.algorithm.name} 训练轮次 {epoch}/{end_epoch - 1}")
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
                should_eval = (epoch + 1) % int(runtime["eval_freq"]) == 0
                # 默认只在主进程评估；distributed_eval=True 时每个进程评估自己的子集再聚合
                if should_eval and (self.context.is_main_process or (self.context.distributed and distributed_eval)):
                    for dataset_type, loader in self.datamodule.test_dataloaders.items():
                        if self.context.is_main_process:
                            print(f"开始 {dataset_type} 测试轮次 {epoch}/{end_epoch - 1}")
                        eval_summaries[dataset_type] = self._evaluate(
                            eval_type=dataset_type,
                            model=model,
                            dataloader=loader,
                            transform=self.datamodule.transform,
                            device=self.context.device,
                            project_folder=runtime["project_folder"],
                            config=self.config,
                            epoch=epoch,
                            distributed=self.context.distributed and distributed_eval,
                            state=state,
                        )

                # 根据评估结果更新调度器
                if bool(runtime["train"]) and scheduler is not None:
                    self.algorithm.step_scheduler(scheduler, eval_summaries, self.config)

                # 记录当前学习率
                if self.context.is_main_process:
                    # 学习率属于优化器/运行时状态，放在 runtime/optim 分区
                    self.recorder.log_metrics({"runtime/optim/lr": optimizer.param_groups[0]["lr"]}, step=self.global_step, commit=False)

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
                )
                # 进程同步，避免不同步进入下一轮
                distributed_barrier()
        finally:
            self.callbacks.close()
            if self.context.is_main_process:
                self.recorder.close()
