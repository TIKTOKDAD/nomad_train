import os
import time

import torch

from training_base.core.config import apply_cli_overrides, build_arg_parser, load_config
from training_base.core.runtime import (
    barrier,
    broadcast_string,
    cleanup_distributed,
    is_torchrun,
    setup_cudnn,
    setup_runtime,
    setup_seed,
)
from training_base.data.data_module import NavigationDataModule, handle_build_lmdb_only, preflight_navigation_data
from training_base.registry import algorithm_registry, register_builtins
from training_base.trainer import Trainer


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")


def _resolve_config_path(path: str) -> str:
    if os.path.isabs(path) or os.path.exists(path):
        return path
    candidate = os.path.join(CONFIG_DIR, os.path.basename(path))
    if os.path.exists(candidate):
        return candidate
    return path


def _prepare_build_lmdb_only(config) -> None:
    runtime = config["runtime"]
    if not bool(runtime.get("build_lmdb_only", False)):
        return
    if is_torchrun():
        raise RuntimeError(
            "build_lmdb_only must run as a single process. "
            "Use: python -m training_base.cli -c training_base/configs/nomad_retrain.yaml --build-lmdb-only"
        )
    runtime["distributed"] = False
    runtime["require_ddp_for_multigpu"] = False
    runtime["gpu_ids"] = [int(runtime.get("build_lmdb_gpu_id", 0))]
    for sink in config["logging"].get("sinks", []):
        sink["enabled"] = False


def _prepare_project_folder(config, context) -> None:
    runtime = config["runtime"]
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S") if context.is_main_process else ""
    timestamp = broadcast_string(timestamp)
    runtime["run_name"] += "_" + timestamp
    runtime["project_folder"] = os.path.join("logs", runtime["project_name"], runtime["run_name"])
    if context.is_main_process:
        os.makedirs(runtime["project_folder"], exist_ok=True)
    barrier()


def _prepare_logging(config) -> None:
    for sink in config["logging"].get("sinks", []):
        sink.setdefault("project", config["runtime"]["project_name"])
        sink.setdefault("run_name", config["runtime"]["run_name"])
        sink.setdefault("config_path", config.get("config_path"))
        sink.setdefault("full_config", config)


def main(argv=None) -> None:
    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = load_config(_resolve_config_path("defaults.yaml"), _resolve_config_path(args.config))
    config = apply_cli_overrides(config, args)

    _prepare_build_lmdb_only(config)
    preflight_navigation_data(config)

    context = setup_runtime(config["runtime"])
    setup_seed(config["runtime"])
    setup_cudnn(config["runtime"])
    _prepare_project_folder(config, context)
    _prepare_logging(config)

    if context.is_main_process:
        print(config)

    try:
        if bool(config["runtime"].get("build_lmdb_only", False)):
            if not handle_build_lmdb_only(config, context):
                raise RuntimeError("--build-lmdb-only is not supported by the configured visual-navigation setup.")
            if context.is_main_process:
                print("FINISHED LMDB CACHE BUILD")
            return

        register_builtins()
        algorithm_cls = algorithm_registry.get(config["algorithm"]["name"])
        algorithm = algorithm_cls()
        datamodule = NavigationDataModule(config, context)
        Trainer(config=config, algorithm=algorithm, datamodule=datamodule, context=context).fit()
        if context.is_main_process:
            print("FINISHED TRAINING")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
