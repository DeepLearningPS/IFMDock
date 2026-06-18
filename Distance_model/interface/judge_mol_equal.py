import argparse
import os
import shutil
import time
import sys
sys.path.append(os.path.abspath('./'))\



# EcConf
import numpy as np 
from rdkit import Chem

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from torch_scatter import scatter_sum, scatter_mean
from tqdm.auto import tqdm


import copy
from rdkit import Chem
from rdkit.Chem import AllChem
import copy
from tqdm import tqdm
from rdkit.Geometry.rdGeometry import Point3D
from collections import Counter
import matplotlib.pyplot as plt
import random 
import dill
import json

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# 读取SDF文件中的分子
def read_molecule_from_sdf(sdf_file):
    try:
        suppl = Chem.SDMolSupplier(sdf_file)
    except Exception as e:  #7nr8_ligand.mol2
        suppl = [Chem.MolFromMol2File(os.path.join(os.path.dirname(sdf_file), sdf_file.split('/')[-2] + '_ligand.mol2'), sanitize=False)]
    
    molecules = [Chem.RemoveHs(mol) for mol in suppl if mol is not None]
    return molecules

# 比较两个分子的原子数量和原子类型
def compare_molecules(mol1, mol2):
    # 比较原子数量
    if mol1.GetNumAtoms() != mol2.GetNumAtoms():
        return False
    
    # 比较原子类型
    atoms1 = [atom.GetSymbol() for atom in mol1.GetAtoms()]
    atoms2 = [atom.GetSymbol() for atom in mol2.GetAtoms()]
    
    return sorted(atoms1) == sorted(atoms2)

#判断两个分子的原子数量是否一致，以及对应的原子类型是否一样
def judge_mol(sdf_file1, sdf_file2):
    # 读取分子
    molecules1 = read_molecule_from_sdf(sdf_file1)
    molecules2 = read_molecule_from_sdf(sdf_file2)

    # 假设每个SDF文件中只有一个分子，测试第一个就可以了
    mol1 = molecules1[0]
    mol2 = molecules2[0]
    compare_molecules(mol1, mol2)
    """
    # 比较分子
    if compare_molecules(mol1, mol2):
        ##print("两个分子的原子数量和原子类型相同")
        pass
    else:
        #print('error:', sdf_file1, sdf_file2)
        #raise Exception("两个分子的原子数量或原子类型不同")
        w_file.write(f'sdf_file1: {sdf_file1}\n')
        w_file.write(f'sdf_file2: {sdf_file2}\n')
        w_file.write(f'\n')
    """
        


if __name__ == '__main__':
    #判断两个分子的原子类型是否一样
    w_file    = open('error_not_equal.txt', 'w')
    #base_path = '/mnt_191/fanzhiguang/47/mnt/unimol_docking_v2/interface/pdb2020_predict_sdf_boxsize10'
    base_path = '/data/fan_zg/MDocking/Docking_baseline/unimol_docking_v2/interface/BindingNetv2_High_predict_sdf_boxsize10'
    name_list = []

    for bs in os.listdir(base_path):
        if os.path.isdir(os.path.join(base_path, bs)):
            name_list.append(bs)

    reading_error = 0
    error_dict = {}
    for name in tqdm(name_list):
        try:
            sdf_file1 = os.path.join(base_path, name, f'{name}_ligand.sdf')
            sdf_file2 = os.path.join(base_path, name, f'gen_{name}.sdf')
            sdf_file3 = os.path.join(base_path, name, f'org_{name}.sdf')
            judge_mol(sdf_file1,sdf_file2)
            judge_mol(sdf_file1,sdf_file3)
        except Exception as e:
            print(f'reading error: {e}, name: {name}')
            #如果有问题，则删除，并记录
            #shutil.rmtree(os.path.join(base_path, name))
            reading_error += 1
            w_file.write(name + '\n')
            
            error_dict[name] = e
        
    #print('reading_error_num:', reading_error)

    if reading_error > 0:
        print('验证不通过， 两个分子的原子类型不是一样的，生成的分子顺序有乱')
    else:
        print('验证通过， 两个分子的原子类型是一样的，生成的分子顺序没有乱')
    w_file.close()
    print('error_dict:', error_dict)