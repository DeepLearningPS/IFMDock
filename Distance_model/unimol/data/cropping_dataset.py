# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from functools import lru_cache
import logging
from unicore.data import BaseWrapperDataset
from . import data_utils

logger = logging.getLogger(__name__)


class CroppingPocketDataset(BaseWrapperDataset):
    def __init__(self, dataset, seed, atoms, coordinates, holo_coordinates, max_atoms=300):
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.coordinates = coordinates
        self.holo_coordinates = holo_coordinates
        self.max_atoms = (
            max_atoms  # max number of atoms in a molecule, None indicates no limit.
        )
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        try:
            dd = self.dataset[index].copy()
        except Exception as e:
            #print(e)
            return None
        atoms = dd[self.atoms]
        coordinates = dd[self.coordinates]
        # residue = dd["residue"]
        holo_coordinates = dd[self.holo_coordinates]

        # crop atoms according to their distance to the center of pockets #根据距离质心的距离进行修剪，仅仅数量大于256时才截断
        if self.max_atoms and len(atoms) > self.max_atoms:
            with data_utils.numpy_seed(self.seed, epoch, index):
                distance = np.linalg.norm(
                    coordinates - coordinates.mean(axis=0), axis=1
                )

                def softmax(x):
                    x -= np.max(x)
                    x = np.exp(x) / np.sum(np.exp(x))
                    return x

                distance += 1  # prevent inf
                weight = softmax(np.reciprocal(distance))
                
                #这里在截断分子的时候，使用了随机数, 当然距离近权重更大，更容易被选中，为什么要随机？
                '''
                #np.random.choice(a, size=None, replace=True, p=None)
                a: 数组或整数。如果是整数，表示从 np.arange(a) 中选择。
                size: 输出样本的形状。默认为 None，返回单个值。如果指定为整数或元组，则返回对应大小的样本。
                replace: 布尔值，表示是否允许重复选择。默认为 True（允许重复）。
                p: 1D数组，表示每个元素被选择的概率分布。如果 None，则每个元素被均等选择。
                '''

                #随机选择，当原子数量不多的时候，可以使用
                #随机截断
                index = np.random.choice(len(atoms), self.max_atoms, replace=False, p=weight)



                '''
                #按距离最近的截断，这样可以保证40个构象的蛋白是一样的,
                #固定截断
                distance_index = np.argsort(distance)
                #print('distance.shape:', distance.shape)        # (370,)
                #print('distance_index:', distance_index.shape)  # (370,)
                index = distance_index[:self.max_atoms]
                #raise Exception('stop')
                '''

                atoms = atoms[index]
                coordinates = coordinates[index]
                # residue = residue[index]
                holo_coordinates = holo_coordinates[index]

        dd[self.atoms] = atoms
        dd[self.coordinates] = coordinates.astype(np.float32)
        # dd["residue"] = residue
        dd[self.holo_coordinates] = holo_coordinates.astype(np.float32)
        #print('atoms, self.coordinates, self.holo_coordinates:', len(atoms), dd[self.coordinates].shape, dd[self.holo_coordinates].shape)
        return dd

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)
