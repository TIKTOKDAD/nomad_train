# ============================================================
# WandB sink - optional experiment tracking backend
# ============================================================
# 本文件只封装 W&B 特有行为：
# 1. 仅主进程初始化 W&B run，避免 DDP 多进程重复创建实验
# 2. 上传 CLI/core 已经准备好的配置 artifact，不直接解析训练配置路径
# 3. 统一记录指标、图像，并把本地图像路径包装为 wandb.Image

import logging
import os

from training_base.registry import log_sink_registry


LOGGER = logging.getLogger(__name__)


# 规范化配置 artifact 路径集合，兼容 dict/list/tuple 三种传入形式
def _iter_config_artifact_paths(paths):
    if isinstance(paths, dict):
        return [path for path in paths.values() if path]
    if isinstance(paths, (list, tuple)):
        return [path for path in paths if path]
    return []


# 上传单个配置 artifact；上传失败只告警，不阻断训练主流程
def _save_config_artifact(wandb, path) -> None:
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        return
    rel_path = os.path.relpath(abs_path, os.getcwd())
    if rel_path == os.pardir or rel_path.startswith(os.pardir + os.sep):
        LOGGER.warning("跳过 W&B 配置 artifact 上传，路径不在当前工作目录内: %s", abs_path)
        return
    try:
        wandb.save(rel_path, policy="now")
    except Exception as exc:
        LOGGER.warning("跳过 W&B 配置 artifact 上传: %s", exc)


# 注册 W&B 日志输出
@log_sink_registry.register("wandb")
class WandBSink:
    # 初始化 W&B sink；只有主进程真正创建 run，其余 rank 保持空实现
    def __init__(self, config, context) -> None:
        self.enabled = bool(config.get("enabled", True)) and context.is_main_process
        self.strict = bool(config.get("strict", False))
        self.run = None
        if not self.enabled:
            return

        try:
            # 延迟导入 wandb，避免未启用 W&B 时仍要求环境安装/登录
            import wandb

            wandb.login()
            wandb.init(
                project=config["project"],
                settings=wandb.Settings(start_method="thread"),
                entity=config.get("entity"),
            )
            self.run = wandb
        except Exception as exc:
            self._handle_error("初始化 W&B 失败", exc)
            return

        try:
            # 上传 run 目录中的配置快照；配置快照由 core/cli 负责生成
            for artifact_path in _iter_config_artifact_paths(config.get("config_artifact_paths", {})):
                _save_config_artifact(wandb, artifact_path)
            if config.get("run_name"):
                wandb.run.name = config["run_name"]
            if wandb.run and config.get("full_config") is not None:
                # full_config 是去掉自引用后的最终合并配置，用于 W&B 面板复现实验参数
                wandb.config.update(config["full_config"], allow_val_change=True)
        except Exception as exc:
            self._handle_error("上传 W&B 配置 artifact 失败", exc, disable=False)

    def _handle_error(self, message: str, exc: Exception, *, disable: bool = True) -> None:
        if self.strict:
            raise RuntimeError(message) from exc
        LOGGER.warning("%s，已跳过 W&B 日志: %s", message, exc)
        if disable:
            self.run = None

    # 记录数值指标
    def log_metrics(self, data, *, step=None, commit=True) -> None:
        if self.run is not None and data:
            try:
                self.run.log(data, step=step, commit=commit)
            except Exception as exc:
                self._handle_error("写入 W&B 指标失败", exc)

    # 记录图像指标
    def log_images(self, data, *, step=None, commit=False) -> None:
        if self.run is not None and data:
            try:
                self.run.log(data, step=step, commit=commit)
            except Exception as exc:
                self._handle_error("写入 W&B 图像失败", exc)

    # 将本地图像路径包装为 W&B Image；未启用 W&B 时原样返回
    def image(self, path):
        if self.run is None:
            return path
        try:
            return self.run.Image(path)
        except Exception as exc:
            self._handle_error("创建 W&B 图像失败", exc)
            return path

    # 结束一次日志提交
    def close(self) -> None:
        if self.run is not None:
            run = self.run
            try:
                run.log({}, commit=True)
            except Exception as exc:
                self._handle_error("提交 W&B 日志失败", exc, disable=False)
            try:
                run.finish()
            except Exception as exc:
                self._handle_error("关闭 W&B 失败", exc, disable=False)
            finally:
                self.run = None
