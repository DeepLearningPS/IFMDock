
from functools import lru_cache
from Distance_model.basemodels.data import BaseWrapperDataset


class KeyDataset(BaseWrapperDataset):
    def __init__(self, dataset, key):
        self.dataset = dataset
        self.key = key

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        
        try:
            return self.dataset[idx][self.key]
        except Exception as e:
            return None
        
        
        
