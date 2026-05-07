
import wandb
import os
import numpy as np
from typing import List, Optional, Dict
from prettytable import PrettyTable


from vint_train.training.train_utils import train, evaluate
from vint_train.training.train_utils import train_nomad, evaluate_nomad


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms


from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel


def _strip_module_prefix(state_dict: dict) -> dict:
    """
    Remove the `module.` prefix produced by DataParallel/DDP checkpoints.

    This keeps checkpoint loading compatible across single-GPU, DataParallel,
    and DDP launches.
    """
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        return state_dict
    if all(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _distributed_barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _make_grad_scaler(device: torch.device, amp_enabled: bool, use_grad_scaler: bool):
    enabled = bool(amp_enabled and use_grad_scaler and device.type == "cuda")
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _should_save_epoch(epoch: int, freq: int) -> bool:
    # numbered checkpoint 可以低频保存；latest checkpoint 仍可每个 epoch 保存以保证断点恢复。
    return freq > 0 and (epoch + 1) % freq == 0


def train_eval_loop(
    train_model: bool,
    model: nn.Module,
    optimizer: Adam,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    dataloader: DataLoader,
    test_dataloaders: Dict[str, DataLoader],
    transform: transforms,
    epochs: int,
    device: torch.device,
    project_folder: str,
    normalized: bool,
    wandb_log_freq: int = 10,
    print_log_freq: int = 100,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    current_epoch: int = 0,
    alpha: float = 0.5,
    learn_angle: bool = True,
    use_wandb: bool = True,
    eval_fraction: float = 0.25,
    train_sampler=None,
    distributed: bool = False,
    is_main_process: bool = True,
    amp_enabled: bool = False,
    amp_dtype: str = "fp16",
    use_grad_scaler: bool = True,
    log_by_global_step: bool = True,
    log_first_step: bool = False,
    image_log_start_step: int = 0,
    perf_log_freq: int = 0,
    distributed_eval: bool = False,
    save_latest_every_epoch: bool = True,
    checkpoint_freq: int = 1,
):
    """
    Train and evaluate ViNT/GNM for the configured epoch range.

    The loop preserves the existing evaluation cadence and keeps all wandb
    logging on the main process when DDP is active.
    """

    assert 0 <= alpha <= 1

    latest_path = os.path.join(project_folder, f"latest.pth")
    grad_scaler = _make_grad_scaler(device, amp_enabled, use_grad_scaler)


    for epoch in range(current_epoch, current_epoch + epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if train_model:
            if is_main_process:
                print(f"Start ViNT Training Epoch {epoch}/{current_epoch + epochs - 1}")
            train(
                model=model,
                optimizer=optimizer,
                dataloader=dataloader,
                transform=transform,
                device=device,
                project_folder=project_folder,
                normalized=normalized,
                epoch=epoch,
                alpha=alpha,
                learn_angle=learn_angle,
                print_log_freq=print_log_freq if is_main_process else 0,
                wandb_log_freq=wandb_log_freq,
                image_log_freq=image_log_freq if is_main_process else 0,
                num_images_log=num_images_log,
                use_wandb=use_wandb and is_main_process,
                use_tqdm=is_main_process,
                grad_scaler=grad_scaler,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                log_by_global_step=log_by_global_step,
                log_first_step=log_first_step,
                image_log_start_step=image_log_start_step,
                perf_log_freq=perf_log_freq,
            )

        avg_total_loss = float("nan")
        # distributed_eval=True 时各 rank 评估自己的分片，最后在 evaluate 内聚合指标。
        if is_main_process or (distributed and distributed_eval):
            avg_total_test_loss = []
            for dataset_type in test_dataloaders:
                if is_main_process:
                    print(f"Start {dataset_type} ViNT Testing Epoch {epoch}/{current_epoch + epochs - 1}")
                loader = test_dataloaders[dataset_type]
                _, _, total_eval_loss = evaluate(
                    eval_type=dataset_type,
                    model=_unwrap_model(model),
                    dataloader=loader,
                    transform=transform,
                    device=device,
                    project_folder=project_folder,
                    normalized=normalized,
                    epoch=epoch,
                    alpha=alpha,
                    learn_angle=learn_angle,
                    num_images_log=num_images_log,
                    use_wandb=use_wandb and is_main_process,
                    eval_fraction=eval_fraction,
                    use_tqdm=is_main_process,
                    distributed=distributed and distributed_eval,
                    is_main_process=is_main_process,
                )
                avg_total_test_loss.append(total_eval_loss)
            if len(avg_total_test_loss) > 0:
                avg_total_loss = float(np.mean(avg_total_test_loss))

        if distributed:
            metric_obj = [avg_total_loss if is_main_process else None]
            dist.broadcast_object_list(metric_obj, src=0)
            avg_total_loss = metric_obj[0]

        if train_model and scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_total_loss)
            else:
                scheduler.step()

        if is_main_process:
            checkpoint = {
                "epoch": epoch,
                "model": _unwrap_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "avg_total_test_loss": avg_total_loss,
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
            }

            if use_wandb:
                wandb.log({
                    "avg_total_test_loss": avg_total_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                }, commit=False)

            if save_latest_every_epoch:
                torch.save(checkpoint, latest_path)
            # numbered checkpoint 低频保存，减少磁盘 IO 和其他 rank 的 barrier 等待时间。
            if _should_save_epoch(epoch, checkpoint_freq):
                numbered_path = os.path.join(project_folder, f"{epoch}.pth")
                torch.save(checkpoint, numbered_path)
        _distributed_barrier()


    if is_main_process and use_wandb:
        wandb.log({})
    if is_main_process:
        print()

def train_eval_loop_nomad(
    train_model: bool,
    model: nn.Module,
    optimizer: Adam,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    noise_scheduler: DDPMScheduler,
    train_loader: DataLoader,
    test_dataloaders: Dict[str, DataLoader],
    transform: transforms,
    goal_mask_prob: float,
    epochs: int,
    device: torch.device,
    project_folder: str,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    current_epoch: int = 0,
    alpha: float = 1e-4,
    use_wandb: bool = True,
    eval_fraction: float = 0.25,
    eval_freq: int = 1,
    ema_state_dict: Optional[dict] = None,
    train_sampler=None,
    distributed: bool = False,
    is_main_process: bool = True,
    heavy_metric_log_freq: int = 1000,
    heavy_metric_start_step: int = 0,
    num_action_samples_log: int = 8,
    amp_enabled: bool = False,
    amp_dtype: str = "fp16",
    use_grad_scaler: bool = True,
    log_by_global_step: bool = True,
    log_first_step: bool = False,
    image_log_start_step: int = 0,
    perf_log_freq: int = 0,
    distributed_eval: bool = False,
    save_latest_every_epoch: bool = True,
    checkpoint_freq: int = 1,
    ema_checkpoint_freq: int = 1,
    optimizer_checkpoint_freq: int = 1,
    scheduler_checkpoint_freq: int = 1,
):
    """
    Train and evaluate NoMaD for the configured epoch range.

    Heavy diffusion metrics and image visualizations remain observable, but are
    controlled by lower-frequency config switches so they do not dominate the
    hot training path.
    """
    latest_path = os.path.join(project_folder, f"latest.pth")

    ema_model = EMAModel(model=_unwrap_model(model), power=0.75)
    if ema_state_dict is not None:
        ema_model.averaged_model.load_state_dict(
            _strip_module_prefix(ema_state_dict),
            strict=False,
        )
    grad_scaler = _make_grad_scaler(device, amp_enabled, use_grad_scaler)
    

    for epoch in range(current_epoch, current_epoch + epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if train_model:
            if is_main_process:
                print(f"Start ViNT DP Training Epoch {epoch}/{current_epoch + epochs - 1}")

            train_nomad(
                model=model,
                ema_model=ema_model,
                optimizer=optimizer,
                dataloader=train_loader,
                transform=transform,
                device=device,
                noise_scheduler=noise_scheduler,
                goal_mask_prob=goal_mask_prob,
                project_folder=project_folder,
                epoch=epoch,
                print_log_freq=print_log_freq if is_main_process else 0,
                wandb_log_freq=wandb_log_freq,
                image_log_freq=image_log_freq if is_main_process else 0,
                num_images_log=num_images_log,
                use_wandb=use_wandb and is_main_process,
                alpha=alpha,
                use_tqdm=is_main_process,
                heavy_metric_log_freq=heavy_metric_log_freq if is_main_process else 0,
                heavy_metric_start_step=heavy_metric_start_step,
                num_action_samples_log=num_action_samples_log,
                grad_scaler=grad_scaler,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                log_by_global_step=log_by_global_step,
                log_first_step=log_first_step,
                image_log_start_step=image_log_start_step,
                perf_log_freq=perf_log_freq,
            )

        if train_model and lr_scheduler is not None:
            lr_scheduler.step()

        if is_main_process:
            ema_latest_path = os.path.join(project_folder, f"ema_latest.pth")
            if save_latest_every_epoch:
                torch.save(ema_model.averaged_model.state_dict(), ema_latest_path)
                print(f"Saved EMA model to {ema_latest_path}")
            # EMA/model/optimizer/scheduler 的 numbered 文件按各自频率保存，训练数值不受影响。
            if _should_save_epoch(epoch, ema_checkpoint_freq):
                ema_numbered_path = os.path.join(project_folder, f"ema_{epoch}.pth")
                torch.save(ema_model.averaged_model.state_dict(), ema_numbered_path)

            checkpoint = {
                "epoch": epoch,
                "model": _unwrap_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "ema_model": ema_model.averaged_model.state_dict(),
            }
            if lr_scheduler is not None:
                checkpoint["scheduler"] = lr_scheduler.state_dict()

            if save_latest_every_epoch:
                torch.save(checkpoint, latest_path)
            if _should_save_epoch(epoch, checkpoint_freq):
                numbered_path = os.path.join(project_folder, f"{epoch}.pth")
                torch.save(checkpoint, numbered_path)
                print(f"Saved model checkpoint to {numbered_path}")

            latest_optimizer_path = os.path.join(project_folder, f"optimizer_latest.pth")
            if save_latest_every_epoch:
                torch.save(optimizer.state_dict(), latest_optimizer_path)
            if _should_save_epoch(epoch, optimizer_checkpoint_freq):
                numbered_path = os.path.join(project_folder, f"optimizer_{epoch}.pth")
                torch.save(optimizer.state_dict(), numbered_path)

            if lr_scheduler is not None:
                latest_scheduler_path = os.path.join(project_folder, f"scheduler_latest.pth")
                if save_latest_every_epoch:
                    torch.save(lr_scheduler.state_dict(), latest_scheduler_path)
                if _should_save_epoch(epoch, scheduler_checkpoint_freq):
                    numbered_path = os.path.join(project_folder, f"scheduler_{epoch}.pth")
                    torch.save(lr_scheduler.state_dict(), numbered_path)

        if (epoch + 1) % eval_freq == 0 and (is_main_process or (distributed and distributed_eval)):
            for dataset_type in test_dataloaders:
                if is_main_process:
                    print(
                        f"Start {dataset_type} NoMaD Testing Epoch {epoch}/{current_epoch + epochs - 1}"
                    )
                loader = test_dataloaders[dataset_type]

                evaluate_nomad(
                    eval_type=dataset_type,
                    ema_model=ema_model,
                    dataloader=loader,
                    transform=transform,
                    device=device,
                    noise_scheduler=noise_scheduler,
                    goal_mask_prob=goal_mask_prob,
                    project_folder=project_folder,
                    epoch=epoch,
                    print_log_freq=print_log_freq,
                    image_log_freq=image_log_freq if is_main_process else 0,
                    num_images_log=num_images_log,
                    wandb_log_freq=wandb_log_freq,
                    use_wandb=use_wandb and is_main_process,
                    eval_fraction=eval_fraction,
                    heavy_metric_log_freq=heavy_metric_log_freq,
                    heavy_metric_start_step=heavy_metric_start_step,
                    num_action_samples_log=num_action_samples_log,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                    log_by_global_step=log_by_global_step,
                    log_first_step=log_first_step,
                    image_log_start_step=image_log_start_step,
                    distributed=distributed and distributed_eval,
                    is_main_process=is_main_process,
                )
        

        if is_main_process and use_wandb:
            wandb.log({
                "lr": optimizer.param_groups[0]["lr"],
            }, commit=False)


        if is_main_process and use_wandb:
            wandb.log({}, commit=False)


        _distributed_barrier()


    if is_main_process and use_wandb:
        wandb.log({})
    if is_main_process:
        print()

def load_model(
    model,
    model_type,
    checkpoint: dict,
) -> None:
    """
    Load model weights from either legacy or structured checkpoints.
    """
    if model_type == "nomad":



        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        state_dict = _strip_module_prefix(state_dict)
        model.load_state_dict(state_dict, strict=False)
    else:

        loaded_model = checkpoint["model"]
        if isinstance(loaded_model, dict):
            state_dict = loaded_model
        elif hasattr(loaded_model, "module"):
            state_dict = loaded_model.module.state_dict()
        else:
            state_dict = loaded_model.state_dict()
        model.load_state_dict(_strip_module_prefix(state_dict), strict=False)


def load_ema_model(
    ema_model,
    state_dict: dict,
) -> None:
    """
    Load EMA weights from a state dictionary.
    """
    ema_model.load_state_dict(state_dict)


def count_parameters(
    model,
):
    """
    Count and print the number of trainable parameters.
    """

    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params


    print(f"Total Trainable Params: {total_params/1e6:.2f}M")
    return total_params
