
from functools import lru_cache

import torch
from Distance_model.basemodels.data import Dictionary
from functools import lru_cache
from . import BaseWrapperDataset


class TokenizeDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        dictionary: Dictionary,
        max_seq_len: int=512,
    ):
        self.dataset = dataset
        self.dictionary = dictionary
        self.max_seq_len = max_seq_len

    @lru_cache(maxsize=16)
    def __getitem__(self, index: int):
        try:
            raw_data = self.dataset[index]
            '''
            if raw_data is None:
            try:
                assert len(raw_data) < self.max_seq_len and len(raw_data) > 0
            except Exception as e:
                raise Exception(e)
            return torch.from_numpy(self.dictionary.vec_index(raw_data)).long()
            '''
            return torch.from_numpy(self.dictionary.vec_index(raw_data)).long()
        except Exception as e:
            return None