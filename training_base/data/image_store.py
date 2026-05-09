import io

from training_base.data.data_utils import get_data_path, img_path_to_data
from training_base.data.lmdb_cache import build_or_open_lmdb_cache


class LmdbImageStore:
    """LMDB-backed image reader used by NavigationDataset."""

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

    def load(self, trajectory_name, time):
        image_path = get_data_path(self.data_folder, trajectory_name, time)
        try:
            with self.env.begin() as txn:
                image_buffer = txn.get(image_path.encode())
            if image_buffer is None:
                raise KeyError(image_path)
            image_bytes = io.BytesIO(bytes(image_buffer))
            return img_path_to_data(image_bytes, self.image_size, aspect_ratio=self.image_aspect_ratio)
        except Exception as exc:
            raise RuntimeError(
                f"无法从数据集 '{self.dataset_name}' 的 LMDB 缓存读取图像: {image_path}"
            ) from exc
