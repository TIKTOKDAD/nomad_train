import itertools
import time

import torch
import tqdm

from training_base.callbacks import CallbackManager
from training_base.loggers import Recorder
from training_base.core.native_utils import (
    autocast,
    distributed_barrier,
    make_grad_scaler,
    rank0_tqdm_enabled,
    scale_backward_step,
    should_log_event,
)
from training_base.core.runtime import RuntimeContext, wrap_distributed_model


class Trainer:
    """Paper-agnostic high-performance training loop for navigation algorithms."""

    def __init__(self, config, algorithm, datamodule, context: RuntimeContext) -> None:
        self.config = config
        self.algorithm = algorithm
        self.datamodule = datamodule
        self.context = context
        self.global_step = 0
        self.recorder = Recorder(config, context)
        self.callbacks = CallbackManager(config, context)

    def _print_store(self, store, mode: str, epoch: int, batch_idx: int, num_batches: int) -> None:
        if not self.context.is_main_process:
            return
        message = store.display_latest()
        if message:
            print(f"(epoch {epoch}) (batch {batch_idx}/{num_batches - 1}) {mode}: {message}")

    def _train_epoch(self, *, model, optimizer, dataloader, transform, device, project_folder, config, epoch, grad_scaler, state):
        model.train()
        runtime = config["runtime"]
        logging = config["logging"]
        num_batches = len(dataloader)
        print_freq = int(logging.get("metric_log_freq", 0)) if self.context.is_main_process else 0
        image_freq = int(logging.get("image_log_freq", 0)) if self.context.is_main_process else 0
        heavy_freq = int(logging.get("heavy_metric_log_freq", print_freq)) if self.context.is_main_process else 0
        perf_freq = int(logging.get("perf_log_freq", 0))
        log_by_global_step = bool(logging.get("by_global_step", True))
        log_first_step = bool(logging.get("first_step", False))
        image_start = int(logging.get("image_start_step", 0))
        heavy_start = int(logging.get("heavy_metric_start_step", 0))

        metric_store = self.recorder.metric_store()
        heavy_store = self.recorder.metric_store()
        show_tqdm = rank0_tqdm_enabled(self.context.is_main_process)
        end_time = time.perf_counter()

        with tqdm.tqdm(
            dataloader,
            desc=f"{self.algorithm.name} train epoch {epoch}",
            leave=False,
            disable=not show_tqdm,
            dynamic_ncols=True,
        ) as iterator:
            for batch_idx, batch in enumerate(iterator):
                data_time = time.perf_counter() - end_time
                compute_start = time.perf_counter()
                should_images = should_log_event(image_freq, epoch, num_batches, batch_idx, log_by_global_step, image_start, log_first_step)
                prepared = self.algorithm.prepare_batch(batch, transform, device, mode="train", should_log_images=should_images)

                with autocast(device, bool(runtime.get("amp", False)), runtime.get("amp_dtype", "fp16")):
                    result = self.algorithm.train_step(model, prepared, state, config)

                scale_backward_step(
                    result.loss,
                    optimizer,
                    grad_scaler,
                    model=model,
                    clip_config=config.get("optimizer", {}).get("gradient_clip", {}),
                )
                self.algorithm.after_optimizer_step(model, state, config)
                self.callbacks.call("after_optimizer_step", model=model, algorithm=self.algorithm, state=state, config=config)
                self.global_step += 1

                if should_log_event(print_freq, epoch, num_batches, batch_idx, log_by_global_step, 0, log_first_step):
                    metric_logs = dict(result.logs)
                    metric_logs.update(self.algorithm.light_metrics(model, prepared, result, state, config, mode="train"))
                    metric_store.update(metric_logs)
                    if show_tqdm and result.loss is not None:
                        iterator.set_postfix(loss=float(result.loss.detach().float().item()))
                    self._print_store(metric_store, "train", epoch, batch_idx, num_batches)
                    self.recorder.log_metrics(metric_store.latest(prefix="train/"), step=self.global_step, commit=True)

                if should_log_event(heavy_freq, epoch, num_batches, batch_idx, log_by_global_step, heavy_start, log_first_step):
                    with torch.inference_mode():
                        metric_model = self.algorithm.model_for_eval(model, state)
                        metric_logs = self.algorithm.heavy_metrics(metric_model, prepared, state, config, mode="train")
                    if metric_logs:
                        heavy_store.update(metric_logs)
                        self._print_store(heavy_store, "train_heavy", epoch, batch_idx, num_batches)
                        self.recorder.log_metrics(heavy_store.latest(prefix="train/"), step=self.global_step, commit=True)

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
                            recorder=self.recorder,
                        )

                compute_time = time.perf_counter() - compute_start
                step_time = data_time + compute_time
                if should_log_event(perf_freq, epoch, num_batches, batch_idx, log_by_global_step, 0, log_first_step):
                    self.callbacks.call(
                        "log_perf",
                        recorder=self.recorder,
                        mode=f"{self.algorithm.name}_train",
                        epoch=epoch,
                        batch_idx=batch_idx,
                        batch_size=result.batch_size or 0,
                        data_time=data_time,
                        compute_time=compute_time,
                        step_time=step_time,
                        device=device,
                    )
                end_time = time.perf_counter()

    def _evaluate(self, *, eval_type, model, dataloader, transform, device, project_folder, config, epoch, distributed, state):
        eval_model = self.algorithm.model_for_eval(model, state)
        eval_model.eval()
        runtime = config["runtime"]
        logging = config["logging"]
        heavy_freq = int(logging.get("heavy_metric_log_freq", logging.get("metric_log_freq", 0)))
        heavy_start = int(logging.get("heavy_metric_start_step", 0))
        log_by_global_step = bool(logging.get("by_global_step", True))
        log_first_step = bool(logging.get("first_step", False))
        dataloader_len = len(dataloader)
        num_batches = min(max(int(dataloader_len * float(runtime["eval_fraction"])), 1), dataloader_len) if dataloader_len > 0 else 0
        metric_store = self.recorder.metric_store()
        heavy_store = self.recorder.metric_store()
        last_prepared = None
        last_result = None

        with torch.inference_mode():
            show_tqdm = rank0_tqdm_enabled(self.context.is_main_process)
            iterator = tqdm.tqdm(
                itertools.islice(dataloader, num_batches),
                total=num_batches,
                dynamic_ncols=True,
                desc=f"Evaluating {eval_type} epoch {epoch}",
                leave=False,
                disable=not show_tqdm,
            )
            for batch_idx, batch in enumerate(iterator):
                should_images = self.context.is_main_process and self.algorithm.visualize_eval_last and batch_idx == num_batches - 1
                prepared = self.algorithm.prepare_batch(batch, transform, device, mode=eval_type, should_log_images=should_images)
                with autocast(device, bool(runtime.get("amp", False)), runtime.get("amp_dtype", "fp16")):
                    result = self.algorithm.eval_step(eval_model, prepared, state, config)
                metric_logs = dict(result.logs)
                metric_logs.update(self.algorithm.light_metrics(eval_model, prepared, result, state, config, mode=eval_type))
                metric_store.update(metric_logs)
                if should_log_event(heavy_freq, epoch, num_batches, batch_idx, log_by_global_step, heavy_start, log_first_step):
                    heavy_logs = self.algorithm.heavy_metrics(eval_model, prepared, state, config, mode=eval_type)
                    if heavy_logs:
                        heavy_store.update(heavy_logs)
                last_prepared = prepared
                last_result = result

        if distributed:
            metric_store.reduce_distributed(device)
            heavy_store.reduce_distributed(device)

        if self.context.is_main_process and num_batches > 0:
            data_log = metric_store.average(prefix=f"{eval_type}/")
            data_log.update(heavy_store.average(prefix=f"{eval_type}/"))
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
                    recorder=self.recorder,
                )

        summary = metric_store.average()
        summary.update(heavy_store.average())
        return summary

    def fit(self) -> None:
        self.datamodule.setup(build_lmdb_only=False)
        runtime = self.config["runtime"]

        model, model_extras = self.algorithm.build_model(self.config)
        objective = self.algorithm.build_objective(self.config)
        optimizer, scheduler = self.algorithm.configure_optimizers(model, self.config)
        resume_state = self.algorithm.prepare_resume(model=model, optimizer=optimizer, scheduler=scheduler, config=self.config, device=self.context.device)
        self.global_step = int((resume_state.extra or {}).get("global_step", 0))

        model = model.to(self.context.device)
        model = wrap_distributed_model(model, runtime, self.context)
        state = self.algorithm.create_state(model, model_extras, objective, self.config, self.context.device, resume_state)

        grad_scaler = make_grad_scaler(self.context.device, bool(runtime.get("amp", False)), bool(runtime.get("use_grad_scaler", True)))
        end_epoch = resume_state.current_epoch + int(runtime["epochs"])

        try:
            for epoch in range(resume_state.current_epoch, end_epoch):
                if self.datamodule.train_sampler is not None:
                    self.datamodule.train_sampler.set_epoch(epoch)

                if bool(runtime["train"]):
                    if self.context.is_main_process:
                        print(f"Start {self.algorithm.name} Training Epoch {epoch}/{end_epoch - 1}")
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

                eval_summaries = {}
                distributed_eval = bool(runtime.get("distributed_eval", False))
                should_eval = (epoch + 1) % int(runtime["eval_freq"]) == 0
                if should_eval and (self.context.is_main_process or (self.context.distributed and distributed_eval)):
                    for dataset_type, loader in self.datamodule.test_dataloaders.items():
                        if self.context.is_main_process:
                            print(f"Start {dataset_type} Testing Epoch {epoch}/{end_epoch - 1}")
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

                if bool(runtime["train"]) and scheduler is not None:
                    self.algorithm.step_scheduler(scheduler, eval_summaries, self.config)

                if self.context.is_main_process:
                    self.recorder.log_metrics({"lr": optimizer.param_groups[0]["lr"]}, step=self.global_step, commit=False)

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
                distributed_barrier()
        finally:
            self.callbacks.close()
            if self.context.is_main_process:
                self.recorder.close()
