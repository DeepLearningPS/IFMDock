
from . import UnicoreDataset


class NumSamplesDataset(UnicoreDataset):
    def __getitem__(self, index):
        return 1

    def __len__(self):
        return 0

    def collater(self, samples):
        return sum(samples)
