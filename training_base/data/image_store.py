# ============================================================
# LMDB image store - image loading facade for NavigationDataset
# ============================================================
# 本文件把 LMDB 缓存细节包成一个很小的读取器：
# 1. 初始化时按配置构建或打开 LMDB 环境
# 2. load() 根据轨迹名和时间步定位图像 key
# 3. 从缓存读取字节流后复用 data_utils 的图像预处理函数

import io

from training_base.data.data_utils import get_data_path, img_path_to_data
from training_base.data.lmdb_cache import build_or_open_lmdb_cache


# 基于 LMDB 的图像读取器，供 NavigationDataset 在 __getitem__ 中调用
class LmdbImageStore:
    """LMDB-backed image reader used by NavigationDataset."""

    # 打开或构建对应 split 的图像缓存
    def __init__(
        self,
        *,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        goals_index,
        image_size,
        image_aspect_ratio: float,
        lmdb_lock: bool,
        lmdb_readahead: bool,
        lmdb_meminit: bool,
        lmdb_max_readers: int,
        lmdb_map_size: int,
        lmdb_cache_mode: str,
        rebuild_incomplete_lmdb: bool,
        use_tqdm: bool = True,
    ) -> None:
        self.data_folder = data_folder
        self.dataset_name = dataset_name
        self.image_size = image_size
        self.image_aspect_ratio = float(image_aspect_ratio)
        # build_or_open_lmdb_cache 负责完整性校验、缺失构建和只读打开
        self.env = build_or_open_lmdb_cache(
            data_folder=data_folder,
            data_split_folder=data_split_folder,
            dataset_name=dataset_name,
            goals_index=goals_index,
            lmdb_lock=lmdb_lock,
            lmdb_readahead=lmdb_readahead,
            lmdb_meminit=lmdb_meminit,
            lmdb_max_readers=lmdb_max_readers,
            lmdb_map_size=lmdb_map_size,
            lmdb_cache_mode=lmdb_cache_mode,
            rebuild_incomplete_lmdb=rebuild_incomplete_lmdb,
            use_tqdm=use_tqdm,
        )

    # 读取单张图像并转换成训练用张量
    def load(self, trajectory_name, time):
        # key 使用原始图像路径，和构建 LMDB 时写入的 key 保持完全一致
        image_path = get_data_path(self.data_folder, trajectory_name, time)
        try:
            # 每次读取都开启只读事务；LMDB 读事务开销很低，且支持多 worker 并发
            with self.env.begin() as txn:
                image_buffer = txn.get(image_path.encode())
            if image_buffer is None:
                raise KeyError(image_path)
            # PIL 可以从 BytesIO 读取，避免把缓存内容落回临时文件
            image_bytes = io.BytesIO(bytes(image_buffer))
            return img_path_to_data(image_bytes, self.image_size, aspect_ratio=self.image_aspect_ratio)
        except Exception as exc:
            raise RuntimeError(
                f"无法从数据集 '{self.dataset_name}' 的 LMDB 缓存读取图像: {image_path}"
            ) from exc
