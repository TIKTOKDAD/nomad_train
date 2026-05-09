# ============================================================
# Data sampling helpers - deterministic epoch/index-aware sampling
# ============================================================

import torch
from torch.utils.data import Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler

from training_base.data.labeling import sample_context


def _seeded_generator(seed: int, offset: int = 0) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed) + int(offset))
    return generator


def stable_subset_indices(dataset_size: int, subset_fraction: float, seed: int) -> list:
    if dataset_size <= 0:
        return []
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
        if isinstance(index, tuple) and len(index) == 2:
            epoch, sample_index = index
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
            indices = torch.randperm(dataset_size, generator=_seeded_generator(self.seed, self.offset + self.epoch)).tolist()
        else:
            indices = list(range(dataset_size))
        return iter((self.epoch, int(index)) for index in indices)

    def __len__(self) -> int:
        return len(self.data_source)


class EpochAwareDistributedSampler(DistributedSampler):
    """DistributedSampler variant that also passes epoch to the dataset."""

    def __iter__(self):
        return iter((self.epoch, int(index)) for index in super().__iter__())


def build_epoch_sampler(dataset: Dataset, *, distributed: bool, world_size: int, rank: int, seed: int):
    if distributed:
        return EpochAwareDistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=int(seed),
            drop_last=False,
        )
    return EpochAwareSampler(dataset, shuffle=True, seed=int(seed))
