
from functools import lru_cache

import numpy as np
import torch

# This dataset is optional for the docking-distance model.  Keep importing the
# core UniMol modules possible on installations without the text tokenizer.
try:
    from tokenizers import BertWordPieceTokenizer
except ImportError:
    BertWordPieceTokenizer = None

from . import BaseWrapperDataset, LRUCacheDataset


class BertTokenizeDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        dict_path: str,
        max_seq_len: int=512,
    ):
        if BertWordPieceTokenizer is None:
            raise ImportError(
                "BertTokenizeDataset requires the optional 'tokenizers' package."
            )
        self.dataset = dataset
        self.tokenizer = BertWordPieceTokenizer(dict_path, lowercase=True)
        self.max_seq_len = max_seq_len

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return True

    def __getitem__(self, index: int):
        raw_str = self.dataset[index]
        raw_str = raw_str.replace('<unk>', '[UNK]')
        output = self.tokenizer.encode(raw_str)
        ret = torch.Tensor(output.ids).long()
        if ret.size(0) > self.max_seq_len:
            ret = ret[:self.max_seq_len]
        return ret
