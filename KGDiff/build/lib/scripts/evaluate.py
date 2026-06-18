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




def set_seed(seed):
    torch.manual_seed(seed)  # 设置 PyTorch 的随机数种子
    torch.cuda.manual_seed_all(seed)  # 设置所有 GPU 的随机数种子
    np.random.seed(seed)  # 设置 NumPy 的随机数种子
    random.seed(seed)  # 设置 Python 自带的随机数种子
    torch.backends.cudnn.deterministic = True  # 设置 CuDNN 算法为确定性算法
    torch.backends.cudnn.benchmark = True



'''
这里的评估代码作用：将已经生成好的保存成sdf或pdb等文件格式的配体拿过来，计算各种指标，与具体的模型无关。输入的数据源可以是包含rdkit mol对象pickle，
也可以是包含sdf文件的目录路径
'''



def rmsds_rdkit(truth_mol_list, gen_mol_list, num):
    #truth_mol是一维list，gen_mol是一个2维度list，存放整个测试集的结果
    #对于每一条数据，随机挑选num个进行测试
    #print('truth_mol_list:', len(truth_mol_list))
    #print('gen_mol_list:', gen_mol_list)
    index       = list(range(40)) #这里默认是40个中，随机挑选num个
    #print('index:', index)
    data_dict = {}
    try:
        tg_index    = random.sample(index, num) #diffdock采样出来，有失败的，因此不足40，最后一个要特殊处理，全取，否则这一步会报错
        truth_mols  = truth_mol_list #每一个数据的ground truth不用动，只有一个
        gen_mols    = []

        for mols in gen_mol_list:
            #sub_mol = [mols[i] for i in tg_index]
            random.shuffle(mols)
            sub_mol = mols[:num] #随机打乱，直接取前num个
            gen_mols.append(sub_mol)
    except Exception as e:
        truth_mols  = truth_mol_list #每一个数据的ground truth不用动，只有一个
        gen_mols    = gen_mol_list


    rmsd_dict = defaultdict(list)      #用于绘制箱线图等分布图
    rmsd_list = []
    rmsd_list_per = []  #存放每一个复合物的值，这是一个二维list
    for i, mols in enumerate(gen_mols):
        tmp = []
        for mol in mols:
            try:
                #考虑子结构匹配的rsmd, 下面两个是一个东西，GetBestRMS优先将两个分子对齐，然后找最佳的rmsd
                #AllChem.GetBestRMS会对两个分子进行对齐，然后计算rmsd
                #值得注意的是，参考的配体和生成的配体之间的原子顺序不一样，因此计算rmsd时，需要对齐
                #https://www.rdkit.org/docs/source/rdkit.Chem.rdMolAlign.html#rdkit.Chem.rdMolAlign.GetBestRMS
                rmsd = AllChem.GetBestRMS(Chem.RemoveHs(mol), Chem.RemoveHs(truth_mols[i]))
                #rmsd = Chem.rdMolAlign.GetBestRMS(Chem.RemoveHs(mol), Chem.RemoveHs(truth_mols[i]))  

                #rmsd = Chem.rdMolAlign.AlignMol(Chem.RemoveHs(mol), Chem.RemoveHs(truth_mols[i])) #这个结果比AllChem.GetBestRMS()要大

                rmsd = rmsd * np.sqrt(3)

                #仅仅计算两个坐标矩阵的rmsd
                #pre_pos  = Chem.RemoveHs(mol).GetConformer(0).GetPositions()
                #holo_pos = Chem.RemoveHs(truth_mols[i]).GetConformer(0).GetPositions()
                #rmsd = np.sqrt(np.sum((pre_pos - holo_pos) ** 2) / pre_pos.shape[0]) # unimol的计算rmsd方法，注意这里np.sum没有指定轴，所以计算的是全部, 这种方法可以计算两个向量
            except Exception as e:
                print('error:', e)
                #print('mol:', mol.GetConformer(0).GetPositions())
                #print('truth_mols:', truth_mols[i].GetConformer(0).GetPositions())
                continue
            tmp.append(rmsd)
            rmsd_list.append(rmsd)
        if tmp:
            rmsd_list_per.append(tmp)

            #计算num个数据点的统计结果
            rmsd_dict['rmsd_mean'].append(np.mean(tmp))
            rmsd_dict['rmsd_std'].append(np.std(tmp))
            rmsd_dict['rsmd_mid'].append(np.median(tmp))
            rmsd_dict['rmsd_max'].append(np.max(tmp))
            rmsd_dict['rmsd_min'].append(np.min(tmp))

            #如果rmsd小于2，则合格
            np_rmsd = np.array(tmp)
            all_num = np_rmsd.shape[0] #数据对，num
            indices = np_rmsd <= 2 
            sub_num = np.count_nonzero(indices)
            rmsd_dict['rmsd_rate'].append(sub_num / all_num)

            print('sub_num / all_num:', sub_num / all_num)
            #print('np.mean(np_rmsd <= 2):', np.mean(np_rmsd <= 2)) #结果一样



    print('all num:', len(truth_mol_list) * num)
    print('rmsd_list num:', len(rmsd_list))

    #这里最好先对每一个复合物的num个采样样本进行统计，然后再对100个复合物再统计, 注意再统计应该使用均值
    rmsd_mean = round(np.mean(rmsd_dict['rmsd_mean']), 4)
    rmsd_std  = round(np.mean(rmsd_dict['rmsd_std']), 4)
    rsmd_mid  = round(np.mean(rmsd_dict['rsmd_mid']),4)
    rmsd_max  = round(np.mean(rmsd_dict['rmsd_max']), 4)
    rmsd_min  = round(np.mean(rmsd_dict['rmsd_min']), 4)
    rmsd_rate      = round(np.mean(rmsd_dict['rmsd_rate']), 4)
    #print('new_rmsd_rate:', round(np.sum(np.array(rmsd_list) <= 2), 4))

    data_dict['data_per']   =  rmsd_list_per #每一条数据长度不一样，不能转numpy
    data_dict['data']       =  np.array(rmsd_list)
    data_dict['all']        = [rmsd_rate, rmsd_mean, rmsd_std, rsmd_mid, rmsd_max, rmsd_min]

    return data_dict




def rmsds(truth_mol_list, gen_mol_list, num, data_name_list):
    #truth_mol是一维list，gen_mol是一个2维度list，存放整个测试集的结果
    #对于每一条数据，随机挑选num个进行测试
    #print('truth_mol_list:', len(truth_mol_list))
    #print('gen_mol_list:', gen_mol_list)
    index       = list(range(40)) #这里默认是40个中，随机挑选num个
    #print('index:', index)
    data_dict = {}
    try:
        tg_index    = random.sample(index, num) #diffdock采样出来，有失败的，因此不足40，最后一个要特殊处理，全取，否则这一步会报错
        truth_mols  = truth_mol_list #每一个数据的ground truth不用动，只有一个
        gen_mols    = []

        for mols in gen_mol_list:
            #sub_mol = [mols[i] for i in tg_index]
            random.shuffle(mols)
            sub_mol = mols[:num] #随机打乱，直接取前num个
            gen_mols.append(sub_mol)
    except Exception as e:
        truth_mols  = truth_mol_list #每一个数据的ground truth不用动，只有一个
        gen_mols    = gen_mol_list


    rmsd_dict = defaultdict(list)      #用于绘制箱线图等分布图
    rmsd_list = []
    rmsd_list_per = []  #存放每一个复合物的值，这是一个二维list
    assert len(gen_mols) == len(data_name_list)
    for i, (mols, dt_name) in enumerate(zip(gen_mols, data_name_list)):
        tmp = []
        for mol in mols:
            try:
                #考虑子结构匹配的rsmd, 下面两个是一个东西，GetBestRMS优先将两个分子对齐，然后找最佳的rmsd
                #https://www.rdkit.org/docs/source/rdkit.Chem.rdMolAlign.html#rdkit.Chem.rdMolAlign.GetBestRMS
                #rmsd = AllChem.GetBestRMS(Chem.RemoveHs(mol), Chem.RemoveHs(truth_mols[i]))
                #rmsd = Chem.rdMolAlign.GetBestRMS(Chem.RemoveHs(mol), Chem.RemoveHs(truth_mols[i]))  


                #仅仅计算两个坐标矩阵的rmsd
                pre_pos  = Chem.RemoveHs(mol).GetConformer(0).GetPositions()
                holo_pos = Chem.RemoveHs(truth_mols[i]).GetConformer(0).GetPositions()

                assert pre_pos.shape == holo_pos.shape
                #rmsd = np.sqrt(np.mean(np.sum((pre_pos - holo_pos) ** 2, axis=-1)))  #标准的rmsd，这种方法不能计算两个向量
                rmsd = np.sqrt(np.sum((pre_pos - holo_pos) ** 2) / pre_pos.shape[0]) # unimol的计算rmsd方法，注意这里np.sum没有指定轴，所以计算的是全部, 这种方法可以计算两个向量

            
            except Exception as e:
                print('error:', e)
                #print('mol:', mol.GetConformer(0).GetPositions())
                #print('truth_mols:', truth_mols[i].GetConformer(0).GetPositions())
                continue
            tmp.append(rmsd)
            rmsd_list.append(rmsd)
        if tmp:
            rmsd_list_per.append(tmp)

            #计算num个数据点的统计结果
            rmsd_dict['rmsd_mean'].append(np.mean(tmp))
            rmsd_dict['rmsd_std'].append(np.std(tmp))
            rmsd_dict['rsmd_mid'].append(np.median(tmp))
            rmsd_dict['rmsd_max'].append(np.max(tmp))
            rmsd_dict['rmsd_min'].append(np.min(tmp))
            rmsd_dict['data_name'].append(dt_name)
            rmsd_dict['data_mol'].append(mols) #保存mol, 后面直接用于分析了，排序可以跟着均值走

            #如果rmsd小于2，则合格
            np_rmsd = np.array(tmp)
            all_num = np_rmsd.shape[0] #数据对，num
            indices = np_rmsd <= 2 
            sub_num = np.count_nonzero(indices)
            rmsd_dict['rmsd_rate'].append(sub_num / all_num)

            #print('sub_num / all_num:', sub_num / all_num)
            #print('np.mean(np_rmsd <= 2):', np.mean(np_rmsd <= 2)) #结果一样



    print('all num:', len(truth_mol_list) * num)
    print('rmsd_list num:', len(rmsd_list))

    #这里最好先对每一个复合物的num个采样样本进行统计，然后再对100个复合物再统计, 注意再统计应该使用均值
    rmsd_mean = round(np.mean(rmsd_dict['rmsd_mean']), 4)
    rmsd_std  = round(np.mean(rmsd_dict['rmsd_std']), 4)
    rsmd_mid  = round(np.mean(rmsd_dict['rsmd_mid']),4)
    rmsd_max  = round(np.mean(rmsd_dict['rmsd_max']), 4)
    rmsd_min  = round(np.mean(rmsd_dict['rmsd_min']), 4)
    rmsd_rate = round(np.mean(rmsd_dict['rmsd_rate']), 4)

    data_dict['data_per']   =  rmsd_list_per #每一条数据长度不一样，不能转numpy
    data_dict['data']       =  np.array(rmsd_list)
    data_dict['all']        = [rmsd_rate, rmsd_mean, rmsd_std, rsmd_mid, rmsd_max, rmsd_min]

    data_dict['rmsd_mean']  = rmsd_dict['rmsd_mean']
    data_dict['rmsd_std']   = rmsd_dict['rmsd_std']
    data_dict['rmsd_mid']   = rmsd_dict['rsmd_mid']
    data_dict['rmsd_max']   = rmsd_dict['rsmd_max']
    data_dict['rmsd_min']   = rmsd_dict['rsmd_min']
    data_dict['data_name']  = rmsd_dict['data_name']
    data_dict['data_name_rmsd_mean']  = {k: v for k, v in zip(rmsd_dict['data_name'], rmsd_dict['rmsd_mean'])}
    data_dict['data_name_mol']        = {k: v for k, v in zip(rmsd_dict['data_name'], rmsd_dict['data_mol'])}
    

    return data_dict



def rmsdsV2(truth_mol_list, gen_mol_list, num):
    #truth_mol是一维list，gen_mol是一个2维度list，存放整个测试集的结果
    #对于每一条数据，随机挑选num个进行测试
    #print('truth_mol_list:', len(truth_mol_list))
    #print('gen_mol_list:', gen_mol_list)
    index       = list(range(40)) #这里默认是40个中，随机挑选num个
    #print('index:', index)
    data_dict = {}
    try:
        tg_index    = random.sample(index, num) #diffdock采样出来，有失败的，因此不足40，最后一个要特殊处理，全取，否则这一步会报错
        truth_mols  = truth_mol_list #每一个数据的ground truth不用动，只有一个
        gen_mols    = []

        for mols in gen_mol_list:
            #sub_mol = [mols[i] for i in tg_index]
            random.shuffle(mols)
            sub_mol = mols[:num] #随机打乱，直接取前num个
            gen_mols.append(sub_mol)
    except Exception as e:
        truth_mols  = truth_mol_list #每一个数据的ground truth不用动，只有一个
        gen_mols    = gen_mol_list


    rmsd_dict = defaultdict(list)      #用于绘制箱线图等分布图
    rmsd_list = []
    rmsd_list_per = []  #存放每一个复合物的值，这是一个二维list
    for i, mols in enumerate(gen_mols):
        tmp = []
        for mol in mols:
            try:
                #考虑子结构匹配的rsmd, 下面两个是一个东西，GetBestRMS优先将两个分子对齐，然后找最佳的rmsd
                #https://www.rdkit.org/docs/source/rdkit.Chem.rdMolAlign.html#rdkit.Chem.rdMolAlign.GetBestRMS
                #rmsd = AllChem.GetBestRMS(Chem.RemoveHs(mol), Chem.RemoveHs(truth_mols[i]))
                #rmsd = Chem.rdMolAlign.GetBestRMS(Chem.RemoveHs(mol), Chem.RemoveHs(truth_mols[i]))  


                #仅仅计算两个坐标矩阵的rmsd
                pre_pos  = Chem.RemoveHs(mol).GetConformer(0).GetPositions()
                holo_pos = Chem.RemoveHs(truth_mols[i]).GetConformer(0).GetPositions()
                #rmsd = np.sqrt(np.mean(np.sum((pre_pos - holo_pos) ** 2, axis=-1)))  #标准的rmsd，这种方法不能计算两个向量
                rmsd = np.sqrt(np.sum((pre_pos - holo_pos) ** 2) / pre_pos.shape[0]) # unimol的计算rmsd方法，注意这里np.sum没有指定轴，所以计算的是全部, 这种方法可以计算两个向量

            
            except Exception as e:
                print('error:', e)
                #print('mol:', mol.GetConformer(0).GetPositions())
                #print('truth_mols:', truth_mols[i].GetConformer(0).GetPositions())
                continue
            tmp.append(rmsd)
            rmsd_list.append(rmsd)
        if tmp:
            rmsd_list_per.append(tmp)

            #计算num个数据点的统计结果
            rmsd_dict['rmsd_mean'].extend(tmp)
            rmsd_dict['rmsd_std'].extend(tmp)
            rmsd_dict['rsmd_mid'].extend(tmp)
            rmsd_dict['rmsd_max'].extend(tmp)
            rmsd_dict['rmsd_min'].extend(tmp)

            #如果rmsd小于2，则合格
            np_rmsd = np.array(tmp)
            all_num = np_rmsd.shape[0] #数据对，num
            indices = np_rmsd <= 2 
            sub_num = np.count_nonzero(indices)
            rmsd_dict['rmsd_rate'].append(sub_num / all_num)

            #print('sub_num / all_num:', sub_num / all_num)
            #print('np.mean(np_rmsd <= 2):', np.mean(np_rmsd <= 2)) #结果一样



    print('all num:', len(truth_mol_list) * num)
    print('rmsd_list num:', len(rmsd_list))

    #这里最好先对每一个复合物的num个采样样本进行统计，然后再对100个复合物再统计, 注意再统计应该使用均值
    rmsd_mean = round(np.mean(rmsd_dict['rmsd_mean']), 4)
    rmsd_std  = round(np.std(rmsd_dict['rmsd_std']), 4)
    rsmd_mid  = round(np.median(rmsd_dict['rsmd_mid']),4)
    rmsd_max  = round(np.max(rmsd_dict['rmsd_max']), 4)
    rmsd_min  = round(np.min(rmsd_dict['rmsd_min']), 4)
    #rmsd_rate      = round(np.mean(rmsd_dict['rmsd_rate']), 4)
    rmsd_rate       = round(np.mean(np.array(rmsd_list) <= 2.0), 4)

    data_dict['data_per']   =  rmsd_list_per #每一条数据长度不一样，不能转numpy
    data_dict['data']       =  np.array(rmsd_list)
    data_dict['all']        = [rmsd_rate, rmsd_mean, rmsd_std, rsmd_mid, rmsd_max, rmsd_min]

    return data_dict




def read_file(file_path, mode = None, flag = 'sdf', num = 1000, step = 24, model ='ecdock', name_list = None, poor_name_list = None):
    truth_mol_list, gen_mol_list = [], []
    failed_list = []
    data_name_list = []
    if model == 'ecdock':
        if flag == 'sdf':
            #/mnt/home/fanzhiguang/47/new_KGDiff-EcDock/ecdock_step25/result_0/step24/gen_ligand_0.sdf
            #Chem.rdmolfiles.SDMolSupplier(self.ligand_sdf)[0],这里要加[0], 因为返回的是一个list，如果使用Chem.MolFromMolFile， 则不用，这个仅限制于读取只有一个对象的sdf
            #for i in os.listdir(file_path)[:num]:
            for i in name_list[:num]:
            #for i in poor_name_list[:num]: #为了节约时间，ecdock可以先评估效果差的数据
                #print(num)
                #print(i)
                path = os.path.join(file_path, i)
                if os.path.exists(path) and os.path.isdir(path) and os.listdir(path): #目录存在且不空
                    #print('exist')
                    try:
                        base_path   = os.path.join(file_path, f'{i}/step{step}')
                        org_sup     = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, f'origin_ligand_{i}.sdf'))[0]
                        gen_sup     = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, f'gen_ligand_{i}.sdf'))
                        org_sup = Chem.RemoveHs(org_sup)
                        gen_sup = [Chem.RemoveHs(mol) for mol in gen_sup]
                        assert org_sup.GetNumAtoms() == gen_sup[0].GetNumAtoms()
                        truth_mol_list.append(org_sup) 
                        gen_mol_list.append(gen_sup) 
                    except Exception as e:
                        failed_list.append(i)
                        print('error:', e)
                        continue
                    data_name_list.append(i)
            #print('len(data_name_list):', len(data_name_list))
        else:
            with open(file_path + '/resault.pickle', 'rb') as file:
                data            = dill.load(file)
                truth_mol_list  = data['org']
                gen_mol_list    = data['gen']

    elif model == 'diffdock':
        #/mnt/home/fanzhiguang/47/DiffDock-main/results/user_predictions_small/complex_0/gen.sdf
        #for i in os.listdir(file_path)[:num]:
        for i in name_list[:num]:
            path = os.path.join(file_path, i)
            if os.path.exists(path) and os.path.isdir(path) and os.listdir(path): #目录存在且不空
                try:
                    base_path   = os.path.join(file_path, f'{i}')
                    org_sup     = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, 'origin.sdf'))[0]  #Chem.rdmolfiles.SDMolSupplier返回的时候一个list
                    gen_sup     = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, 'gen.sdf'))
                    org_sup = Chem.RemoveHs(org_sup)
                    gen_sup = [Chem.RemoveHs(mol) for mol in gen_sup]
                    assert org_sup.GetNumAtoms() == gen_sup[0].GetNumAtoms()
                    truth_mol_list.append(org_sup) 
                    gen_mol_list.append(gen_sup) 
                except Exception as e:
                    failed_list.append(i)
                    print('error:', e)
                    continue
                data_name_list.append(i)
        
    elif model == 'glide':
        #for i in os.listdir(file_path)[:num]:
        for i in name_list[:num]:
            path = os.path.join(file_path, i)
            #print('path:', path)
            if os.path.exists(path) and os.path.isdir(path) and os.listdir(path) and len(os.listdir(path)) > 1: #目录存在且不空
                base_path   = os.path.join(file_path, f'{i}')
                try:
                    file = os.path.join(base_path, f'{i}_ligand_config_lib.sdf')

                    if  not os.path.exists(file):
                        file = os.path.join(base_path, f'{i}_ligand-rdkit-glide.sdf')

                    gen_sup     = Chem.rdmolfiles.SDMolSupplier(file)
                except OSError:
                    failed_list.append(i)
                    continue #存在生成失败的情况
                org_sup     = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, f'{i}_ligand.sdf'))[0]  #Chem.rdmolfiles.SDMolSupplier返回的时候一个list
                org_sup = Chem.RemoveHs(org_sup)
                gen_sup = [Chem.RemoveHs(mol) for mol in gen_sup][:] #取最好的第一个
                try:
                    assert org_sup.GetNumAtoms() == gen_sup[0].GetNumAtoms() #存在原子数量不一样的情况
                except:
                    failed_list.append(i)
                    print(f'{org_sup.GetNumAtoms()} != {gen_sup[0].GetNumAtoms()}') #43 != 42
                    continue
                truth_mol_list.append(org_sup) 
                gen_mol_list.append(gen_sup) 
                data_name_list.append(i)
        
    elif model == 'KarmaDock':
        if mode == 'uncorrected':
            #for i in os.listdir(file_path)[:num]:
            for i in name_list[:num]:
                path = os.path.join(file_path, i)
                if os.path.exists(path) and os.path.isdir(path) and os.listdir(path): #目录存在且不空
                    try:
                        base_path   = os.path.join(file_path, f'{i}')
                        org_sup     = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, f'{i}_org.sdf'))[0]  #Chem.rdmolfiles.SDMolSupplier返回的时候一个list
                        gen_sup     = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, f'{i}_pred_uncorrected.sdf'))
                        org_sup = Chem.RemoveHs(org_sup)
                        gen_sup = [Chem.RemoveHs(mol) for mol in gen_sup]
                        assert org_sup.GetNumAtoms() == gen_sup[0].GetNumAtoms()
                        truth_mol_list.append(org_sup) 
                        gen_mol_list.append(gen_sup) 
                    except Exception as e:
                        failed_list.append(i) #reading failed_list: ['6mla', '6erv', '6r4k', '5nw8', '5zw6', '6hp5', '6jse', '6d3x', '6bvh', '6s07', '5wyq', '6a8n', '5ol3']
                        print('error:', e)
                        continue
                    data_name_list.append(i)
        elif mode == 'ff_corrected':
            #for i in os.listdir(file_path)[:num]:
            for i in name_list[:num]:
                path = os.path.join(file_path, i)
                if os.path.exists(path) and os.path.isdir(path) and os.listdir(path): #目录存在且不空
                    try:
                        base_path   = os.path.join(file_path, f'{i}')
                        org_sup     = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, f'{i}_org.sdf'))[0]  #Chem.rdmolfiles.SDMolSupplier返回的时候一个list
                        gen_sup     = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, f'{i}_pred_ff_corrected.sdf'))
                        org_sup = Chem.RemoveHs(org_sup)
                        gen_sup = [Chem.RemoveHs(mol) for mol in gen_sup]
                        assert org_sup.GetNumAtoms() == gen_sup[0].GetNumAtoms()
                        truth_mol_list.append(org_sup) 
                        gen_mol_list.append(gen_sup) 
                    except Exception as e:
                        failed_list.append(i) #['6mla', '6erv', '6r4k', '5nw8', '5zw6', '6hp5', '6jse', '6d3x', '6bvh', '6s07', '6a8n', '5ol3']
                        print('error:', e)
                        continue
                    data_name_list.append(i)
        elif mode == 'align_corrected':
            #for i in os.listdir(file_path)[:num]:
            for i in name_list[:num]:
                path = os.path.join(file_path, i)
                if os.path.exists(path) and os.path.isdir(path) and os.listdir(path): #目录存在且不空
                    try:
                        base_path   = os.path.join(file_path, f'{i}')
                        org_sup     = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, f'{i}_org.sdf'))[0]  #Chem.rdmolfiles.SDMolSupplier返回的时候一个list
                        gen_sup     = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, f'{i}_pred_align_corrected.sdf'))
                        org_sup = Chem.RemoveHs(org_sup)
                        gen_sup = [Chem.RemoveHs(mol) for mol in gen_sup]
                        assert org_sup.GetNumAtoms() == gen_sup[0].GetNumAtoms()
                        truth_mol_list.append(org_sup) 
                        gen_mol_list.append(gen_sup) 
                    except Exception as e:
                        failed_list.append(i) #['6mla', '6erv', '6r4k', '5nw8', '5zw6', '6hp5', '6jse', '6d3x', '6bvh', '6s07', '6a8n', '5ol3']
                        print('error:', e)
                        continue
                    data_name_list.append(i)

    elif model == 'unimol':
        if mode == 'uncorrected':
            #for i in os.listdir(file_path)[:num]:
            for i in name_list[:num]:
                path = os.path.join(file_path, i)
                if os.path.exists(path) and os.path.isdir(path) and os.listdir(path): #目录存在且不空
                    try:
                        base_path   = os.path.join(file_path, f'{i}')
                        org_sup     = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, f'org_{i}.sdf'))[0]  #Chem.rdmolfiles.SDMolSupplier返回的时候一个list
                        gen_sup     = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, f'gen_{i}.sdf'))
                        org_sup = Chem.RemoveHs(org_sup)
                        gen_sup = [Chem.RemoveHs(mol) for mol in gen_sup]
                        assert org_sup.GetNumAtoms() == gen_sup[0].GetNumAtoms()
                        truth_mol_list.append(org_sup) 
                        gen_mol_list.append(gen_sup) 
                    except Exception as e:
                        failed_list.append(i) #reading failed_list: ['6mla', '6erv', '6r4k', '5nw8', '5zw6', '6hp5', '6jse', '6d3x', '6bvh', '6s07', '5wyq', '6a8n', '5ol3']
                        print('error:', e)
                        continue
                    data_name_list.append(i)
        elif mode == 'corrected':
            #for i in os.listdir(file_path)[:num]:
            for i in name_list[:num]:
                path = os.path.join(file_path, i)
                if os.path.exists(path) and os.path.isdir(path) and os.listdir(path): #目录存在且不空
                    try:
                        base_path   = os.path.join(file_path, f'{i}')
                        org_sup     = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, f'org_{i}.sdf'))[0]  #Chem.rdmolfiles.SDMolSupplier返回的时候一个list
                        gen_sup     = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, f'clash_fix_optimize_gen_{i}.sdf'))
                        org_sup = Chem.RemoveHs(org_sup)
                        gen_sup_ = []
                        for mol in gen_sup:
                            try: 
                                Chem.RemoveHs(mol) #存在不正确的mol，如果坐标为nan的，这样去氢原子时会报错，正好过滤掉
                                gen_sup_.append(mol)
                            except Exception as e:
                                print('error:', e)
                                continue
                        #gen_sup = [Chem.RemoveHs(mol) for mol in gen_sup]
                        gen_sup = gen_sup_
                        assert org_sup.GetNumAtoms() == gen_sup[0].GetNumAtoms()
                        truth_mol_list.append(org_sup) 
                        gen_mol_list.append(gen_sup) 
                    except Exception as e:
                        failed_list.append(i) #reading failed_list: ['6mla', '6erv', '6r4k', '5nw8', '5zw6', '6hp5', '6jse', '6d3x', '6bvh', '6s07', '5wyq', '6a8n', '5ol3']
                        print('error:', e)
                        continue
                    data_name_list.append(i)

    
    print('reading failed_list:', failed_list)
    
    return truth_mol_list, gen_mol_list, data_name_list

def boxplot(boxplot_data_list, save_path, name):
    data = boxplot_data_list

    # 创建箱线图
    #plt.boxplot(data, vert=True, patch_artist=True)
    plt.boxplot(data,
                notch = True, 
                sym = 'b+', 			# 异常点绘制蓝色的加号
                vert = True, 
                #whis = 1, 
                positions = [1,2,3,4,5], 
                widths = 0.4, 
                showmeans = True
    )

    # 添加标题和标签
    plt.title(f'{name.upper()} RMSD')
    plt.xlabel('Sampling Num')
    plt.ylabel('Value')
    plt.xticks([1, 2, 3, 4, 5], [1, 5, 10, 25, 40])
    plt.show()
    plt.savefig(save_path)
    plt.close()




def histplot(plot_data_dict, base_save_path, name, model):
    name_list = plot_data_dict['data_name']
    for k_name in list(plot_data_dict.keys())[:-1]:
        #print('k_name:', k_name)
        data = plot_data_dict[k_name]
        data = np.array(data)

        if k_name == 'rmsd_mean':
            print('rmsd_mean <= 2 rate:', round(np.mean(data < 2.0), 4))
            print('rmsd_mean > 2 rate:', round(np.mean(data > 2.0), 4))
            print('rmsd_mean > 3 rate:', round(np.mean(data > 3.0), 4))
            print('rmsd_mean > 4 rate:', round(np.mean(data > 4.0), 4))
            print('rmsd_mean > 5 rate:', round(np.mean(data > 5.0), 4))

            #保存一下结果差的数据，之后评估这些就可以了
            if model == 'ecdock':
                with open(f'../CrossDocked2020/data/{data_name}_poor_name.txt', 'w') as f: #data_name是全局变量
                    assert len(name_list) == len(data)
                    for name, dt in zip(name_list, data > 2.0):
                        if dt:
                            f.write(f'{name}\n')


        # 绘制直方图
        sns.histplot(data, bins=30, kde=True, color='blue')

        # 添加标题和标签
        plt.title(f'{k_name.upper()} Histogram of {model.upper()}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')

        # 显示图形
        #plt.show()
        save_path = os.path.join(os.path.dirname(base_save_path), os.path.splitext(os.path.basename(base_save_path))[0] + f'_{k_name}.png')
        #print('save_path:', save_path)
        plt.savefig(f'{save_path}')

if __name__ == '__main__':
    #设置随机数种子
    set_seed(2024)

    '''
    #为diffdock生成测试的文件列表
    name_list = []
    with open('../CrossDocked2020/data/pdb2020_test_name.txt') as f:
        for i in f:
            name_list.append(i.strip())
    print('len(name_list):', len(name_list))

    with open('../CrossDocked2020/data/protein_ligand_example_csv.csv', 'w') as f:
        f.write('complex_name,protein_path,ligand_description,protein_sequence\n') #表头
        for i in name_list:
            #complex_name,protein_path,ligand_description,protein_sequence
            #,data/PDBBind_processed/5l8c/5l8c_protein_processed.pdb,data/PDBBind_processed/5l8c/5l8c_ligand.sdf,
            ps = f'../CrossDocked2020/data/pdb2020_test/{i}'
            pf = os.path.join(ps, f'{i}_pocket10_400.pdb')
            lf = os.path.join(ps, f'{i}_ligand.sdf')
            line = f',{pf},{lf},\n'
            f.write(line)

    '''

    model = 'ecdock'
    data_name = 'posebusters'  #new_pdb2020_test, posebusters, pdbbind2020_r10
    step = 25
    gnn  = 'equiformer'  #ecdock时，采用不同的神经网络, equiformer
    diffusion = 'cm' #ecdock时，采用不同的扩散模型， CM/DDPM
    mode = '' #不用赋值
    if model == 'ecdock':    #posebusters_ecdock_cm_equiformer_step1
        #file_path = f'/mnt/home/fanzhiguang/47/new_KGDiff-EcDock/{data_name}_ecdock_{diffusion}_{gnn}_step{step}' #这个目录下，可以存放是sdf也可以是pickle，路径别忘了改
        #file_path = 'posebusters_glide_ecdock_cm_equiformer_step10_atom_8i_glide'
        #file_path = 'pdbbind2020_r10_ecdock_cm_equiformer_step25_interaction_limit4.5ai_only_test_leak4.5_finetune'
        #file_path = '/mnt/home/fanzhiguang/47/new_KGDiff-EcDock/posebusters_ecdock_cm_equiformer_step25_interaction_limit4.5ai_retain_fine_test_3'
        #file_path = 'posebusters_ecdock_cm_equiformer_step25_interaction_limit4.5ai_retrain_fine'
        #file_path = 'posebusters_ecdock_cm_equiformer_step25_interaction_limit4.5ai_retrain'
        #file_path = 'posebusters_ecdock_cm_equiformer_step25_interaction_limit4.5ai_retain_fine_No_'
        file_path = 'posebusters_ecdock_cm_equiformer_step25_interaction_limit4.5ai_retrain_fine'

        model_name = data_name + '_' + model + '_' + diffusion + '_' + gnn + f'_step{step}' #记得改名字
        step = step - 1
    elif model == 'diffdock':
        file_path = '/mnt/home/fanzhiguang/47/DiffDock-main/results/new_user_predictions104' #user_predictions_full_protein, user_predictions_pocket400_protein
        model_name = model + '_full_protein' #记得改名字
    elif model == 'glide': #生成和参考的原子顺序不对，需要对齐
        #file_path = '/mnt/home/fanzhiguang/47/CrossDocked2020/data/pdb2020_test_glide/docking_baseline'  #docking_baseline, 
        file_path = '/mnt/home/fanzhiguang/47/CrossDocked2020/data/pdb2020_test_glide/docking_resault/posebusters/gen_docking_full'
        model_name = model #记得改名字
    elif model == 'KarmaDock':
        file_path = '/mnt/home/fanzhiguang/47/KarmaDock/pdbbind_result'
        mode = 'uncorrected'   #uncorrected/ff_corrected/align_corrected
        model_name = model + '_' + mode #记得改名字
    elif model == 'unimol':
        box_size = 10
        #file_path = f'/mnt/home/fanzhiguang/47/unimol_docking_v2/interface/{data_name}_predict_sdf_boxsize{box_size}_fix_protein_cutoff'
        file_path = f'/mnt/home/fanzhiguang/47/unimol_docking_v2/interface/{data_name}_predict_sdf_boxsize{box_size}'
        mode = 'corrected'   #uncorrected/corrected #, 修正优化，有3个失败的reading failed_list: ['7RPZ', '7U0U', '7ZXV']
        model_name = data_name + '_' + model + '_' + f'box_size{box_size}_{mode}' #记得改名字
        
    name_list = []
    with open(f'../CrossDocked2020/data/{data_name}_name.txt') as f:
        for i in f:
            name_list.append(i.strip())

    #效果差的，复杂的分子，为了节约时间，我们平常评估这些即可
    poor_name_list = []
    try:
        with open(f'../CrossDocked2020/data/{data_name}_poor_name.txt') as f:
            for i in f:
                poor_name_list.append(i.strip())
    except Exception as e:
        poor_name_list = None
        
    


    #读取配体的sdf文件,truth_mol是一维list，gen_mol是一个2维度list，存放整个测试集的结果
    truth_mol, gen_mol, data_name_list = read_file(file_path, mode, flag = 'sdf', num = 100, step = step, model = model, name_list = name_list, poor_name_list = poor_name_list)  #读取所有数据，并转化成rdkit mol对象, step值别忘了改
    print('truth_mol, gen_mol:', len(truth_mol), len(gen_mol))
    assert len(truth_mol) == len(gen_mol)
    

    #计算rmsd。从生成的40个分子的中随机选择1/3/5/10/40的，拿过来看rmsd成功率
    resault_dict = {}
    boxplot_data_list   = [] #保留1,5,40结果用于绘制箱线图
    histplot_data_dict  = {} #rmsd的分布图
    dt_name_dict = {}
    dt_name_mol_dict = {}
    for num in [1, 3, 5, 10, 25, 40][:]:
        data_dict = rmsds(truth_mol, gen_mol, num, data_name_list) #对于每一条数据，随机挑选num个进行测试
        resault_dict[num] = ['rate, rmsd_mean, rmsd_std, rsmd_mid, rmsd_max, rmsd_min:', data_dict['all']]
        if num in [1, 5, 10, 25, 40]:
            boxplot_data_list.append(data_dict['data'])

        if num == 40:
            histplot_data_dict['rmsd_mean'] = data_dict['rmsd_mean']
            histplot_data_dict['rmsd_min']  = data_dict['rmsd_min']
            histplot_data_dict['data_name'] = data_dict['data_name']
            dt_name_dict     = data_dict['data_name_rmsd_mean'] 
            dt_name_mol_dict = data_dict['data_name_mol']

    #对rmsd排序，找结果差的数据
    dt_name_sorted_dict = dict(sorted(dt_name_dict.items(), key=lambda item: item[1], reverse=False))
    #同步更新data_name_mol
    dt_name_mol_sorted_dict = {}
    for k in dt_name_sorted_dict:
        dt_name_mol_sorted_dict[k] = dt_name_mol_dict[k]
    print('rmsd sorted from smallest to biggest', list(dt_name_sorted_dict.keys()))

    with open(f'resault/{model}_rmsdmean_sorted.pkl', 'wb') as f:
        dill.dump(dt_name_sorted_dict, f)

    with open(f'resault/{model}_mol_sorted.pkl', 'wb') as f:
        dill.dump(dt_name_sorted_dict, f)

    #exit()
    print(resault_dict)
    #保存字典为JSON文件
    #path = 'resault'
    path = file_path
    os.makedirs(path, exist_ok=True)

    file_name = f'{model_name}_evaluate_resault.json'
    with open(os.path.join(path, file_name), 'w') as file:
        json.dump(resault_dict, file, indent=4)
    

    #绘制箱线图
    save_path = os.path.join(path, f'{model_name}_boxplot.png')
    boxplot(boxplot_data_list, save_path, model_name)

    #绘制rmsd的分布直方图
    save_path = os.path.join(path, f'{model_name}_histplot.png')
    histplot(histplot_data_dict, save_path, model_name, model)




    #exit()
    print(resault_dict)
    #保存字典为JSON文件
    path = 'resault'
    os.makedirs(path, exist_ok=True)

    file_name = f'{model_name}_evaluate_resault.json'
    with open(os.path.join(path, file_name), 'w') as file:
        json.dump(resault_dict, file, indent=4)
    

    #绘制箱线图
    save_path = os.path.join(path, f'{model_name}_boxplot.png')
    boxplot(boxplot_data_list, save_path, model_name)


    #绘制rmsd的分布直方图
    save_path = os.path.join(path, f'{model_name}_histplot.png')
    histplot(histplot_data_dict, save_path, model_name, model)



    