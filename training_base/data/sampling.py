# ============================================================
# Data sampling helpers - deterministic epoch/index-aware sampling
# ============================================================
# 本文件解决“DataLoader worker + shuffle + 负样本采样”下的可复现问题：
# 1. sampler 把 epoch 一起传给 dataset
# 2. dataset 在 __getitem__ 期间写入 sample_context
# 3. labeling.py 根据 seed/epoch/index 派生稳定随机数，保证同一配置可复现

import torch
from torch.utils.data import Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler

from training_base.data.labeling import sample_context


# 创建带固定种子的 torch.Generator
def _seeded_generator(seed: int, offset: int = 0) -> torch.Generator:
    generator = torch.Generator()
    # offset 用于把不同 epoch 或不同用途的随机序列错开
    generator.manual_seed(int(seed) + int(offset))
    return generator


# 稳定抽取训练子集：同一 seed 下索引集合固定
def stable_subset_indices(dataset_size: int, subset_fraction: float, seed: int) -> list:
    if dataset_size <= 0:
        return []
    # 至少保留 1 个样本，避免很小 fraction 产生空训练集
    subset_size = max(1, int(dataset_size * float(subset_fraction)))
    subset_size = min(subset_size, dataset_size)
    return torch.randperm(dataset_size, generator=_seeded_generator(seed))[:subset_size].tolist()


class EpochAwareDataset(Dataset):
    """Wrap a dataset so each item receives seed/epoch/index sampling context."""

    def __init__(self, dataset: Dataset, *, seed: int) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        epoch = self.epoch
        sample_index = index
        # EpochAwareSampler 会传入 (epoch, index)，普通索引路径仍保持兼容
        if isinstance(index, tuple) and len(index) == 2:
            epoch, sample_index = index
        # 在样本构造期间注入上下文，使目标采样的随机性可按 epoch/index 复现
        with sample_context(seed=self.seed, epoch=int(epoch), index=int(sample_index)):
            return self.dataset[int(sample_index)]


class EpochAwareSampler(Sampler):
    """Sequential or shuffled sampler that carries epoch into dataset indexing."""

    def __init__(self, data_source: Dataset, *, shuffle: bool, seed: int, offset: int = 0) -> None:
        self.data_source = data_source
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.offset = int(offset)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        dataset_size = len(self.data_source)
        if self.shuffle:
            # 非分布式训练下按 seed+epoch 洗牌，确保每个 epoch 顺序不同但可复现
            indices = torch.randperm(dataset_size, generator=_seeded_generator(self.seed, self.offset + self.epoch)).tolist()
        else:
            indices = list(range(dataset_size))
        # 把 epoch 打包进索引，由 EpochAwareDataset 解包并写入 sample_context
        return iter((self.epoch, int(index)) for index in indices)

    def __len__(self) -> int:
        return len(self.data_source)


class EpochAwareDistributedSampler(DistributedSampler):
    """DistributedSampler variant that also passes epoch to the dataset."""

    def __iter__(self):
        # 复用 PyTorch DistributedSampler 的分片逻辑，只改变返回索引的形状
        return iter((self.epoch, int(index)) for index in super().__iter__())


# 根据是否分布式构建对应 sampler
def build_epoch_sampler(dataset: Dataset, *, distributed: bool, world_size: int, rank: int, seed: int):
    if distributed:
        # DDP 下由 DistributedSampler 负责按 rank 切分数据
        return EpochAwareDistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=int(seed),
            drop_last=False,
        )
    # 单进程训练使用自定义 sampler，仍保留 epoch-aware 的采样上下文
    return EpochAwareSampler(dataset, shuffle=True, seed=int(seed))
