
import numpy as np
import torch
from functools import lru_cache

from . import BaseWrapperDataset


class AppendTokenDataset(BaseWrapperDataset):
    def __init__(self, dataset, token=None):
        super().__init__(dataset)
        self.token = token

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        try:
            item = self.dataset[idx]
        except Exception as e:
            return None

        if item is None:
            return None
        
        if self.token is not None:
            item = torch.cat([item, torch.full_like(item[0], self.token).unsqueeze(0)], dim=0)
        return item

