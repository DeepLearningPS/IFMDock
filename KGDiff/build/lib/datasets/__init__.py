import torch
from torch.utils.data import Subset
from .pl_pair_dataset import PocketLigandPairDataset, PDBPairDataset
from collections import defaultdict
import os

def get_dataset(config, data_flag = 'old_test', single_test = False, protein = None, ligand = None, data_name = 'pdbbind2020_r10', *args, **kwargs):
    name = config.name
    root = config.path
    data_flag = data_flag
    #print('name:', name) #name: pdbbind
    #raise Exception('stop01')
    print('初始化数据集------------------------------------------------')
    if name == 'pl':
        dataset = PocketLigandPairDataset(root, *args, **kwargs) 
    
    elif name == 'pdbbind':
        #raise Exception('stop1')
        dataset = PDBPairDataset(root, data_flag, single_test, protein, ligand, data_name, *args, **kwargs)
    else:
        raise NotImplementedError('Unknown dataset: %s' % name)
    print('a example data 6un3 ----------------------------------------------------------------------------------------')
    print('a example dataset[6un3]:', dataset['6un3'])
    print('len(dataset):', len(dataset)) #len(dataset): 16251, len(dataset): 16259, 由于我们zmats跳过了8个数据，可能导致在根据index找数据时，找不到，但没有报错，数据截止到报错
    
    print('划分训练验证测试---------------------------------------------------------------------------------------------')
    if 'split' in config: #训练用这个
        print('config.split:', config.split)
        split = torch.load(config.split) #以字典的形式给出 ，#存放的是对index.pkl文件的id编号，分训练、验证、测试
        exclude_index = dataset.exclude
        print('exclude_index:', exclude_index)
        #max_index = len(dataset.keys) -1 #方法2: 直接剔除超过数据库中的最大索引的值。方法2有效，但是会索引到原子需要大于17的分子，不行，还得第一方法

        for i in split:
            print('i:', i)
            #train: 13750
            #valid: 1240
            #test: 104
            try:
                print(f'max {i}: {max(split[i])}')
            except Exception:
                pass
            #max train: 9icd
            #max valid:   6v1c
            #max test:  6un3

            '''
            for j in split[i]:
                if exclude_index.get(j) != None: #方法2，剔除错误的索引，但前提保证dataset.keys的数据不能少，里面包含错误而生成的None，我们要跳过这些None
                #if j > max_index: #方法2，直接剔除最大索引
                    print('rm:', j)
                    split[i].remove(j)
            '''
            assert '0' not in split[i]
            print(f'{i}: {len(split[i])}')
            print(f'{i}: {type(split[i])}') #<class 'list'>
        #exit()
        #subsets = {k: Subset(dataset, indices=v) for k, v in split.items()} #形成训练、验证、测试,

        #exit()
        print('提取训练验证测试----------------------------------------------------------------------------------------')
        subsets = defaultdict(list)
        for k, v in split.items():
            for i in v:
                dt = dataset[i]
                #print('dt:', dt) # []
                subsets[k].extend(dt)
                #exit()
        

        #存在一个问题，如果dataset中没有对应的split缩需要的索引，不会报错，而是会截断到数据缺失的那个位置，这就导致返回的数据集只是一部分而已
        return dataset, subsets
    else:  #采样用这个
        return dataset
