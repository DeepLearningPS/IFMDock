
from functools import lru_cache

from . import BaseWrapperDataset


class LRUCacheDataset(BaseWrapperDataset):
    def __init__(self, dataset, token=None):
        super().__init__(dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, index):
        return self.dataset[index]

    @lru_cache(maxsize=16)
    def collater(self, samples):
        return self.dataset.collater(samples)
