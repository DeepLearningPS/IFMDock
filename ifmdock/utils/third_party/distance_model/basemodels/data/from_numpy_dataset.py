
import torch
from functools import lru_cache

from . import BaseWrapperDataset


class FromNumpyDataset(BaseWrapperDataset):
    def __init__(self, dataset):
        super().__init__(dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        try:
            return torch.from_numpy(self.dataset[idx])
        except Exception as e:
            return None


