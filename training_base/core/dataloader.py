# ============================================================
# DataLoader helpers - worker and prefetch configuration
# ============================================================
# 本文件集中生成 torch DataLoader 参数：
# 1. num_workers=0 时不传 persistent_workers/prefetch_factor，避免 PyTorch 报错
# 2. DDP 场景下可把 YAML 中的全局 worker 数均分到每个 rank
# 3. 训练和评估分别读取 num_workers/test_num_workers，便于独立调参

# 组装 DataLoader 的通用参数
def loader_kwargs(num_workers: int, pin_memory: bool, prefetch_factor: int, persistent_workers: bool):
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        # 这两个参数只有多 worker DataLoader 才有效
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


# 计算每个进程应使用的 worker 数量
def workers_per_rank(config, distributed: bool, world_size: int, *, train: bool) -> int:
    # 训练与评估可配置不同的 worker 参数
    if train:
        base_num_workers = int(config.get("num_workers", 0))
        configured_workers = config.get("num_workers_per_rank")
        global_key = "num_workers_is_global"
    else:
        base_num_workers = int(config.get("test_num_workers", config.get("num_workers", 0)))
        configured_workers = None
        global_key = "test_num_workers_is_global"

    # 显式配置优先；否则根据是否分布式自动折算
    if configured_workers is not None:
        # num_workers_per_rank 表示每个 rank 的 worker 数，不再除 world_size
        return int(configured_workers)
    if distributed and bool(config.get(global_key, True)):
        # base_num_workers 若表示全局总数，则按 world_size 均分；0 仍保持 0
        workers = base_num_workers // world_size
        return max(1, workers) if base_num_workers > 0 else 0
    return base_num_workers
