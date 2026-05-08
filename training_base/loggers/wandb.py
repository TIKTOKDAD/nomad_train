# ============================================================
# WandB sink - optional experiment tracking backend
# ============================================================
# 本文件封装 W&B 日志输出：
# 1. 仅主进程初始化 W&B，避免 DDP 多进程重复创建 run
# 2. log_metrics/log_images 统一接收 Recorder 传入的数据
# 3. image(path) 把本地可视化图片包装为 wandb.Image

from training_base.registry import log_sink_registry


# 注册 W&B 日志输出
@log_sink_registry.register("wandb")
class WandBSink:
    # 初始化 W&B 运行（仅主进程）
    def __init__(self, config, context) -> None:
        self.enabled = bool(config.get("enabled", True)) and context.is_main_process
        self.run = None
        if not self.enabled:
            return
        # 延迟 import wandb，避免未启用 W&B 时仍要求环境安装/登录
        import wandb

        wandb.login()
        wandb.init(
            project=config["project"],
            settings=wandb.Settings(start_method="thread"),
            entity=config.get("entity"),
        )
        if config.get("config_path"):
            # 保存用户 YAML，便于从 W&B run 复现实验
            wandb.save(config["config_path"], policy="now")
        if config.get("run_name"):
            wandb.run.name = config["run_name"]
        if wandb.run and config.get("full_config") is not None:
            # full_config 是合并后的最终配置，比单独 YAML 更能说明实际运行值
            wandb.config.update(config["full_config"])
        self.run = wandb

    # 记录数值指标
    def log_metrics(self, data, *, step=None, commit=True) -> None:
        if self.run is not None and data:
            self.run.log(data, step=step, commit=commit)

    # 记录图像指标
    def log_images(self, data, *, step=None, commit=False) -> None:
        if self.run is not None and data:
            self.run.log(data, step=step, commit=commit)

    # 将图像路径包装为 W&B Image
    def image(self, path):
        if self.run is None:
            return path
        return self.run.Image(path)

    # 结束一次日志提交
    def close(self) -> None:
        if self.run is not None:
            self.run.log({}, commit=True)
