import numpy as np
import torch
from torch_geometric.transforms import BaseTransform


class SubsetTransform(BaseTransform):
    def __init__(self, n):
        self.n = n

    def __call__(self, data):
        conf_batch = np.random.choice(data["ligand"].ifmdock_pos.shape[0], self.n)
        data["ligand"].ifmdock_pos = data["ligand"].ifmdock_pos[conf_batch]
        data["atom"].ifmdock_pos = data["atom"].ifmdock_pos[conf_batch]
        data["receptor"].ifmdock_pos = data["atom"].ifmdock_pos[
            :, data["atom"].ca_mask
        ]
        return data


class MultiplicityTransform(BaseTransform):
    def __init__(self, n):
        self.n = n

    def __call__(self, data):
        conf_batch = torch.repeat_interleave(
            torch.arange(data["ligand"].ifmdock_pos.shape[0]), self.n
        )
        data["ligand"].ifmdock_pos = data["ligand"].ifmdock_pos[conf_batch]
        data["atom"].ifmdock_pos = data["atom"].ifmdock_pos[conf_batch]
        data["receptor"].ifmdock_pos = data["atom"].ifmdock_pos[
            :, data["atom"].ca_mask
        ]
        return data


class EuclideanNoiseTransform(BaseTransform):
    def __init__(self, lig_sigma=None, atom_sigma=None):
        self.lig_sigma = lig_sigma
        self.atom_sigma = atom_sigma

    def __call__(self, data):
        if data is None:
            return None

        if self.lig_sigma is not None:
            lig_updates = torch.normal(
                mean=0, std=self.lig_sigma, size=data["ligand"].ifmdock_pos.shape
            )
            data["ligand"].ifmdock_pos += lig_updates

        if self.atom_sigma is not None:
            atom_updates = torch.normal(
                mean=0, std=self.atom_sigma, size=data["atom"].ifmdock_pos.shape
            )
            data["atom"].ifmdock_pos += atom_updates

        return data
