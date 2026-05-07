def loader_kwargs(num_workers: int, pin_memory: bool, prefetch_factor: int, persistent_workers: bool):
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


def workers_per_rank(config, distributed: bool, world_size: int, *, train: bool) -> int:
    if train:
        base_num_workers = int(config.get("num_workers", 0))
        configured_workers = config.get("num_workers_per_rank")
        global_key = "num_workers_is_global"
    else:
        base_num_workers = int(config.get("test_num_workers", config.get("num_workers", 0)))
        configured_workers = None
        global_key = "test_num_workers_is_global"

    if configured_workers is not None:
        return int(configured_workers)
    if distributed and bool(config.get(global_key, True)):
        workers = base_num_workers // world_size
        return max(1, workers) if base_num_workers > 0 else 0
    return base_num_workers

