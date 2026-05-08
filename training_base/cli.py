import os
import time

# ============================================================
# CLI entrypoint - config, runtime, data preflight, and training
# ============================================================
# 本文件是 `python -m training_base.cli` 的入口：
# 1. 解析命令行并合并 defaults.yaml + 用户配置
# 2. 初始化 DDP/CUDA/随机种子/日志输出目录
# 3. 支持单进程 --build-lmdb-only 模式
# 4. 注册内置组件并启动 Trainer.fit()
# 训练入口脚本：负责配置加载、运行时初始化、数据检查与训练流程启动。

import torch

from training_base.core.config import (
    apply_cli_overrides,
    build_arg_parser,
    load_config,
    run_config_artifact_paths,
    safe_config_for_logging,
    save_run_configs,
)
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


# 工程根目录与默认配置目录（用于相对路径解析）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")


# 解析配置文件路径：优先使用传入路径，其次尝试默认配置目录
def _resolve_config_path(path: str) -> str:
    # 绝对路径或当前目录存在的路径直接使用
    if os.path.isabs(path) or os.path.exists(path):
        return path
    # 只传 nomad.yaml 这类文件名时，自动到 training_base/configs 下查找
    candidate = os.path.join(CONFIG_DIR, os.path.basename(path))
    if os.path.exists(candidate):
        return candidate
    return path


# 处理“仅构建 LMDB 缓存”的特殊运行模式
def _prepare_build_lmdb_only(config) -> None:
    runtime = config["runtime"]
    if not bool(runtime.get("build_lmdb_only", False)):
        return
    # 缓存构建要求单进程，避免多个 rank 同时写同一个 LMDB 目录
    if is_torchrun():
        raise RuntimeError(
            "build_lmdb_only 必须以单进程运行。"
            "请使用: python -m training_base.cli -c training_base/configs/nomad_retrain.yaml --build-lmdb-only"
        )
    runtime["distributed"] = False
    runtime["require_ddp_for_multigpu"] = False
    # 构建缓存时仍允许指定使用哪张 GPU 做图像预处理，但默认只用 0
    runtime["gpu_ids"] = [int(runtime.get("build_lmdb_gpu_id", 0))]
    # 缓存构建模式不需要日志 sink，避免产生无意义训练 run
    for sink in config["logging"].get("sinks", []):
        sink["enabled"] = False


# 初始化本次训练的输出目录（含时间戳，避免覆盖）
def _prepare_project_folder(config, context) -> None:
    runtime = config["runtime"]
    # 时间戳只由主进程生成，再广播给所有 rank，保证目录名一致
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S") if context.is_main_process else ""
    timestamp = broadcast_string(timestamp)
    runtime["run_name"] += "_" + timestamp
    runtime["project_folder"] = os.path.join("logs", runtime["project_name"], runtime["run_name"])
    if context.is_main_process:
        os.makedirs(runtime["project_folder"], exist_ok=True)
    barrier()


# 保存本次 run 的配置快照
def _prepare_run_config_artifacts(config, context) -> None:
    runtime = config["runtime"]
    paths = run_config_artifact_paths(runtime["project_folder"])
    runtime["config_artifact_paths"] = paths
    if context.is_main_process:
        runtime["config_artifact_paths"] = save_run_configs(config, runtime["project_folder"])
    barrier()


# 为各日志 sink 填充统一的默认字段
def _prepare_logging(config) -> None:
    full_config = safe_config_for_logging(config)
    for sink in config["logging"].get("sinks", []):
        # sink 自己只读 logging 子配置，这里把 runtime/config 的全局信息补进去
        sink.setdefault("project", config["runtime"]["project_name"])
        sink.setdefault("run_name", config["runtime"]["run_name"])
        sink.setdefault("config_path", config.get("config_path"))
        sink.setdefault("config_artifact_paths", config["runtime"].get("config_artifact_paths", {}))
        sink.setdefault("full_config", full_config)


# 主流程：加载配置 -> 运行时初始化 -> 数据准备 -> 训练/构建缓存
def main(argv=None) -> None:
    try:
        # Windows/DataLoader/DDP 场景下 spawn 更稳；已设置过则忽略 RuntimeError
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    # 解析命令行参数，并加载/合并配置
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = load_config(_resolve_config_path("defaults.yaml"), _resolve_config_path(args.config))
    config = apply_cli_overrides(config, args)

    # 根据参数调整运行模式，并做数据前置检查
    _prepare_build_lmdb_only(config)
    # DDP 模式会在 setup_runtime 之前检查 LMDB 是否就绪，尽早失败
    preflight_navigation_data(config)

    # 初始化分布式/设备/随机种子与 cuDNN
    context = setup_runtime(config["runtime"])
    setup_seed(config["runtime"])
    setup_cudnn(config["runtime"])
    _prepare_project_folder(config, context)
    _prepare_run_config_artifacts(config, context)
    _prepare_logging(config)

    # 主进程打印最终配置，便于复现实验
    if context.is_main_process:
        print(config)

    try:
        # 仅构建 LMDB：成功后直接退出
        if bool(config["runtime"].get("build_lmdb_only", False)):
            if not handle_build_lmdb_only(config, context):
                raise RuntimeError("--build-lmdb-only 不支持当前配置的视觉导航数据流程。")
            if context.is_main_process:
                print("LMDB 缓存构建完成")
            return

        # 常规训练流程：注册内置组件 -> 构建算法与数据模块 -> 启动训练
        register_builtins()
        # algorithm_registry 返回算法类；这里实例化后交给 Trainer 使用
        algorithm_cls = algorithm_registry.get(config["algorithm"]["name"])
        algorithm = algorithm_cls()
        datamodule = NavigationDataModule(config, context)
        Trainer(config=config, algorithm=algorithm, datamodule=datamodule, context=context).fit()
        if context.is_main_process:
            print("训练完成")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
