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
import collections


def read_sdf(path, name_list):
    gen_pos_list = [] #是一个2维list，内部是一个40的构象序列
    org_pos_list = [] #是一个2维list，内部是一个40的构象序列
    for name in name_list:
        base_path = os.path.join(path, name)
        mol_list = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, 'step24', f'gen_ligand_{name}.sdf'))
        pos_xyz = []
        for mol in mol_list:
            pos = Chem.RemoveHs(mol).GetConformer(0).GetPositions()
            pos_xyz.append(pos)
        
        gen_pos_list.append(pos_xyz)

    for name in name_list:
        base_path = os.path.join(path, name)
        mol_list = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, 'step24', f'origin_ligand_{name}.sdf'))
        pos_xyz = []
        for mol in mol_list:
            pos = Chem.RemoveHs(mol).GetConformer(0).GetPositions()
            pos_xyz.append(pos)
        
        org_pos_list.append(pos_xyz * 40) #因为参考的只有一个分子，所以这里复制40份
    
    return gen_pos_list, org_pos_list

def read_pickle(path, name_list):
    # 读取pickle数据
    unimol_org_ligand_pos_list   = [] #由于验证和ecdock的原子顺序
    unimol_gen_ligand_pos_list   = []
    unimol_org_protein_pos_list  = [] #由于蛋白原子要与unimol_predict_distance的保持一致，这里将unimol保存的org蛋白作为参考的蛋白
    unimol_predict_distance_list = []

    for name in name_list:
        with open(os.path.join(path, name, f'interaction_{name}.pkl'), 'rb') as f:
            data_dict = dill.load(f)
            unimol_org_ligand_pos_list.append(data_dict['holo_coords_list'])
            unimol_gen_ligand_pos_list.append(data_dict['coords_predict_list'])
            unimol_org_protein_pos_list.append(data_dict['pocket_coords_list'])
            unimol_predict_distance_list.append(data_dict['cross_distance_list'])

    return unimol_org_ligand_pos_list, unimol_gen_ligand_pos_list, unimol_org_protein_pos_list, unimol_predict_distance_list


def calculate_distance_matrix(ligand_pos_list, protein_pos_list):
    """
    计算两个坐标矩阵之间的欧氏距离

    参数:
    A (torch.Tensor): 大小为 (n, 3) 的坐标矩阵
    B (torch.Tensor): 大小为 (m, 3) 的坐标矩阵

    返回:
    torch.Tensor: 大小为 (n, m) 的距离矩阵
    """
    all_dist_matrix_list = []
    for As, Bs in zip(ligand_pos_list, protein_pos_list):
        dist_matrix_list = []
        # A 的形状为 (n, 3)，B 的形状为 (m, 3)
        # 计算 A 的每个点与 B 的每个点之间的距离
        for A, B in zip(As, Bs):
            #diff = A.unsqueeze(1) - B.unsqueeze(0)  # diff 的形状为 (n, m, 3)， numpy: np.expand_dims(A, axis=1)
            diff = np.expand_dims(A, axis=1) - np.expand_dims(B, axis=0)
            dist_matrix = np.sqrt(np.sum(diff**2, axis=2))  # dist_matrix 的形状为 (n, m)
            dist_matrix_list.append(dist_matrix)
        all_dist_matrix_list.append(dist_matrix_list)

    return all_dist_matrix_list



def histplot(plot_data_dict, base_save_path, name, model):
    for k_name in list(plot_data_dict.keys())[:]:

        if k_name in ['rmsd_min', 'rmsd_mean'][1:]:
            #print('k_name:', k_name)
            data = plot_data_dict[k_name]
            data = np.array(data)

            if k_name == 'rmsd_mean':
                print('rmsd_mean <= 2 rate:', round(np.mean(data <= 2.0), 4))
                print('rmsd_mean > 2 rate:', round(np.mean(data > 2.0), 4))
                print('rmsd_mean > 3 rate:', round(np.mean(data > 3.0), 4))
                print('rmsd_mean > 4 rate:', round(np.mean(data > 4.0), 4))
                print('rmsd_mean > 5 rate:', round(np.mean(data > 5.0), 4))

            # 绘制直方图
            sns.histplot(data, bins=30, kde=True, color='blue')

            # 添加标题和标签
            plt.title(f'{k_name.upper()} Histogram of {model.upper()}')
            plt.xlabel('Value')
            plt.ylabel('Frequency')

            # 显示图形
            #plt.show()
            save_path = os.path.join(base_save_path, name + f'_{k_name}.png')
            #print('save_path:', save_path)
            plt.savefig(f'{save_path}')
            plt.close()




def rmsds(distance_list1, distance_list2, save_path, name):
    #distance_list1, distance_list2是一个2维度list，存放整个测试集的结果
    data_dict = {}
    rmsd_dict = defaultdict(list)      #用于绘制箱线图等分布图
    rmsd_list = []
    rmsd_list_per = []  #存放每一个复合物的值，这是一个二维list

    assert len(distance_list1) == len(distance_list2)

    for i, (distance1, distance2) in enumerate(zip(distance_list1, distance_list2)):
        assert len(distance1) == len(distance2)
        tmp = []
        for d1, d2 in zip(distance1, distance2):
            try:
                #仅仅计算两个坐标矩阵的rmsd
                #rmsd = np.sqrt(np.mean(np.sum((d1, d2) ** 2, axis=-1)))  #标准的rmsd，这种方法不能计算两个向量
                rmsd = np.sqrt(np.sum((d1 - d2) ** 2) / d1.shape[0]) # unimol的计算rmsd方法，注意这里np.sum没有指定轴，所以计算的是全部, 这种方法可以计算两个向量
            except Exception as e:
                print('error:', e)
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

            #print('sub_num / all_num:', sub_num / all_num)
            #print('np.mean(np_rmsd <= 2):', np.mean(np_rmsd <= 2)) #结果一样



    print('all num:', len(distance_list1) * 40)
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

    #调用绘图函数，绘图分布图，并保存数据
    histplot(data_dict, save_path, name, name)

    with open(os.path.join(save_path, name + '.json'), 'w') as f:
        json.dump(data_dict['all'], f, indent=4)
    
    return data_dict






def rmsds_v2(distance_list1, distance_list2, save_path, name):
    #distance_list1, distance_list2是一个2维度list，存放整个测试集的结果
    data_dict = {}
    rmsd_dict = defaultdict(list)      #用于绘制箱线图等分布图
    rmsd_list = []
    rmsd_list_per = []  #存放每一个复合物的值，这是一个二维list

    assert len(distance_list1) == len(distance_list2)

    for i, (d1, d2) in enumerate(zip(distance_list1, distance_list2)):
        assert len(d1) == len(d2)
        tmp = []
        try:
            #仅仅计算两个坐标矩阵的rmsd
            #rmsd = np.sqrt(np.mean(np.sum((d1, d2) ** 2, axis=-1)))  #标准的rmsd，这种方法不能计算两个向量
            rmsd = np.sqrt(np.sum((d1 - d2) ** 2) / d1.shape[0]) # unimol的计算rmsd方法，注意这里np.sum没有指定轴，所以计算的是全部, 这种方法可以计算两个向量
        except Exception as e:
            print('error:', e)
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

            #print('sub_num / all_num:', sub_num / all_num)
            #print('np.mean(np_rmsd <= 2):', np.mean(np_rmsd <= 2)) #结果一样



    print('all num:', len(distance_list1))
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

    #调用绘图函数，绘图分布图，并保存数据
    histplot(data_dict, save_path, name, name)

    with open(os.path.join(save_path, name + '.json'), 'w') as f:
        json.dump(data_dict['all'], f, indent=4)
    
    return data_dict





def compuete_rmsd(ecdock_path, unimol_path, unimol_predict_distance_path):
    #计算计算ecdock，unimol配体到蛋白的距离d1，d2，计算参考配体到蛋白的距离d3, 以及unimol预测出来的距离矩阵d4，之后计算（d1, d3, d4), (d2, d3, d4)两两之间的rmsd， 
    # 但这里有一个问题，d4的维度和其余的不一样，要筛选
    name_list = []

    for i in os.listdir(ecdock_path):
        path = os.path.join(ecdock_path, i)
        if os.path.exists(path) and os.path.isdir(path) and os.listdir(path): #目录存在且不空
            name_list.append(i)
    
    #根据name_list读取三个路径下的配体和蛋白（只需要unimol_predict_distance_path，以这个的蛋白原子顺序为主）

    ecdock_gen_ligand_pos_list = []
    ecdock_org_ligand_pos_list = []

    unimol_org_ligand_pos_list   = [] #由于验证和ecdock的原子顺序
    unimol_gen_ligand_pos_list   = []
    unimol_org_protein_pos_list  = [] #由于蛋白原子要与unimol_predict_distance的保持一致，这里将unimol保存的org蛋白作为参考的蛋白
    unimol_predict_distance_list = []

    
    ecdock_gen_ligand_pos_list, ecdock_org_ligand_pos_list = read_sdf(ecdock_path, name_list)
    unimol_org_ligand_pos_list, unimol_gen_ligand_pos_list, unimol_org_protein_pos_list, unimol_predict_distance_list = read_pickle(unimol_predict_distance_path, name_list)

    """
    #读取一下unimol默认的最大原子数量，unimol_org_protein_pos_list
    unimol_org_protein_atom_num_set = set()
    unimol_org_protein_atom_num_list = list() 
    for i in unimol_org_protein_pos_list:
        for j in i:
            tg = j.shape[0]
            unimol_org_protein_atom_num_set.add(tg)
        unimol_org_protein_atom_num_list.append(tg) #每一个分子保存一个构象即可
    
    print('unimol_org_protein_atom_num_set sorted:', sorted(unimol_org_protein_atom_num_set, reverse=True)) 
    #最大的蛋白原子数量是256，但存在40左右数量多蛋白，是不是太少了。统计一下分布, 看一下蛋白原子数量少的原因是啥？是蛋白原子数量少，还是配体距离蛋白太远

    print('all num:', len(unimol_org_protein_atom_num_list))
    # 示例列表
    data = unimol_org_protein_atom_num_list

    # 计算元素出现的频率
    frequency = collections.Counter(data)

    sorted_by_values = dict(sorted(frequency.items(), key=lambda item: item[0], reverse=True))
    print('frequency sorted:', sorted_by_values)

    '''
    frequency sorted: {256: 130, 155: 6, 101: 5, 192: 5, 114: 5, 208: 4, 88: 4, 159: 4, 248: 4, 164: 4, 197: 3, 128: 3, 190: 3, 154: 3, 225: 3, 199: 3, 168: 3, 
    193: 3, 112: 3, 142: 3, 231: 3, 87: 3, 173: 3, 149: 3, 228: 3, 127: 3, 216: 3, 172: 3, 152: 3, 210: 3, 235: 3, 213: 3, 175: 3, 236: 3, 115: 2, 160: 2, 125: 2, 
    89: 2, 90: 2, 187: 2, 131: 2, 179: 2, 157: 2, 255: 2, 99: 2, 254: 2, 129: 2, 84: 2, 116: 2, 140: 2, 246: 2, 174: 2, 148: 2, 249: 2, 147: 2, 123: 2, 126: 2, 153: 2, 
    93: 2, 138: 2, 119: 2, 139: 2, 136: 2, 218: 2, 145: 2, 229: 2, 176: 2, 177: 2, 204: 2, 178: 2, 166: 2, 253: 2, 170: 2, 137: 2, 247: 2, 217: 2, 224: 2, 111: 2, 74: 2, 
    206: 2, 215: 2, 41: 2, 156: 2, 108: 2, 121: 2, 109: 2, 96: 2, 77: 2, 144: 2, 195: 2, 171: 2, 250: 1, 98: 1, 239: 1, 151: 1, 60: 1, 242: 1, 130: 1, 141: 1, 191: 1, 
    113: 1, 234: 1, 134: 1, 186: 1, 158: 1, 100: 1, 196: 1, 180: 1, 241: 1, 201: 1, 214: 1, 181: 1, 102: 1, 83: 1, 200: 1, 47: 1, 162: 1, 238: 1, 243: 1, 95: 1, 245: 1, 
    222: 1, 220: 1, 107: 1, 244: 1, 202: 1, 165: 1, 203: 1, 221: 1, 182: 1, 124: 1, 44: 1, 233: 1, 79: 1, 198: 1, 184: 1, 185: 1, 211: 1, 194: 1, 135: 1, 92: 1, 118: 1, 
    97: 1, 219: 1, 57: 1, 91: 1, 207: 1, 212: 1, 104: 1}
    '''

    '''
    frequency sorted: {256: 130, 255: 2, 254: 2, 253: 2, 250: 1, 249: 2, 248: 4, 247: 2, 246: 2, 245: 1, 244: 1, 243: 1, 242: 1, 241: 1, 239: 1, 238: 1, 236: 3, 
    235: 3, 234: 1, 233: 1, 231: 3, 229: 2, 228: 3, 225: 3, 224: 2, 222: 1, 221: 1, 220: 1, 219: 1, 218: 2, 217: 2, 216: 3, 215: 2, 214: 1, 213: 3, 212: 1, 211: 1, 
    210: 3, 208: 4, 207: 1, 206: 2, 204: 2, 203: 1, 202: 1, 201: 1, 200: 1, 199: 3, 198: 1, 197: 3, 196: 1, 195: 2, 194: 1, 193: 3, 192: 5, 191: 1, 190: 3, 187: 2, 
    186: 1, 185: 1, 184: 1, 182: 1, 181: 1, 180: 1, 179: 2, 178: 2, 177: 2, 176: 2, 175: 3, 174: 2, 173: 3, 172: 3, 171: 2, 170: 2, 168: 3, 166: 2, 165: 1, 164: 4, 
    162: 1, 160: 2, 159: 4, 158: 1, 157: 2, 156: 2, 155: 6, 154: 3, 153: 2, 152: 3, 151: 1, 149: 3, 148: 2, 147: 2, 145: 2, 144: 2, 142: 3, 141: 1, 140: 2, 139: 2, 
    138: 2, 137: 2, 136: 2, 135: 1, 134: 1, 131: 2, 130: 1, 129: 2, 128: 3, 127: 3, 126: 2, 125: 2, 124: 1, 123: 2, 121: 2, 119: 2, 118: 1, 116: 2, 115: 2, 114: 5, 
    113: 1, 112: 3, 111: 2, 109: 2, 108: 2, 107: 1, 104: 1, 102: 1, 101: 5, 100: 1, 99: 2, 98: 1, 97: 1, 96: 2, 95: 1, 93: 2, 92: 1, 91: 1, 90: 2, 89: 2, 88: 4, 
    87: 3, 84: 2, 83: 1, 79: 1, 77: 2, 74: 2, 60: 1, 57: 1, 47: 1, 44: 1, 41: 2}
    '''

    # 提取元素和对应的频率
    elements = list(frequency.keys())
    counts = list(frequency.values())

    # 绘制频率分布直方图
    plt.bar(elements, counts)

    # 设置标题和标签
    plt.title('unimol_org_protein_atom num Frequency Distribution')
    plt.xlabel('Element')
    plt.ylabel('Frequency')

    save_path = os.path.join('resault/compare_rmsd/distance_rmsd',  f'unimol_org_protein_atom_num.png')
    plt.savefig(f'{save_path}')

    # 显示图形
    #plt.show()
    plt.close()

    #exit()
    """



    #需要判断他们的原子顺序是否一致，一致则通过，否则报错
    #rtol（相对误差）: 控制两个数组中每个元素的相对公差。默认为 1e-05。
    #atol（绝对误差）: 控制最小允许的绝对公差。默认为 1e-08。
    #equal_nan: 如果设置为 True，则会将 NaN 视为相等。
    for A, B in zip(ecdock_org_ligand_pos_list, unimol_org_ligand_pos_list):
        #print('A.shape:', np.array(A).shape)
        #print('B.shape:', np.array(B).shape)
        assert np.allclose(np.array(A), np.array(B), rtol=0.01, atol=0.02)

    #获取距离矩阵，为了和uniml一致，采用unimol的方法，构建距离矩阵
    ecdock_gen_ligand_distance_list = calculate_distance_matrix(ecdock_gen_ligand_pos_list, unimol_org_protein_pos_list)
    ecdock_org_ligand_distance_list = calculate_distance_matrix(ecdock_org_ligand_pos_list, unimol_org_protein_pos_list)
    unimol_gen_ligand_distance_list = calculate_distance_matrix(unimol_gen_ligand_pos_list, unimol_org_protein_pos_list)

    np.set_printoptions(precision=2)

    #截断距离矩阵4.5，以参考的为基准
    cutoff_index_list = []
    for dt in ecdock_org_ligand_distance_list:
        #print('dt:', dt)
        cutoff_index = np.array(dt) <= 4.5 #存在一个问题，有些原子的4.5距离的是不存在的，所以每一个40个配体得到的距离矩阵变成向量来算rmsd
        cutoff_index_list.append(cutoff_index)
        #print('cutoff_index:', cutoff_index.shape)
    
    new_ecdock_gen_ligand_distance_list = []
    new_ecdock_org_ligand_distance_list = []
    new_unimol_gen_ligand_distance_list = []
    new_unimol_predict_distance_list    = []

    for cutoff_index, k1, k2, k3, k4 in zip(cutoff_index_list, ecdock_gen_ligand_distance_list, ecdock_org_ligand_distance_list, unimol_gen_ligand_distance_list, unimol_predict_distance_list):
        new_k1 = np.array(k1)
        new_k2 = np.array(k2)
        new_k3 = np.array(k3)
        new_k4 = np.array(k4)
        #print('cutoff_index:', cutoff_index.shape)
        #print('k1:', new_k2.shape)
        #shape = [new_k2.shape[0], -1]
        ecdock_gen_ligand_distance = new_k1[cutoff_index]
        ecdock_org_ligand_distance = new_k2[cutoff_index]
        unimol_gen_ligand_distance = new_k3[cutoff_index]
        unimol_predict_distance    = new_k4[cutoff_index]

        new_ecdock_gen_ligand_distance_list.append(ecdock_gen_ligand_distance)
        new_ecdock_org_ligand_distance_list.append(ecdock_org_ligand_distance)
        new_unimol_gen_ligand_distance_list.append(unimol_gen_ligand_distance)
        new_unimol_predict_distance_list.append(unimol_predict_distance)

    

    #计算各个距离矩阵之间的rmsd, 以及rmsd分布图，直接把获取的rmsd放入对应的文件夹保存
    save_path = 'resault/compare_rmsd/distance_rmsd'
    rmsds_v2(new_ecdock_gen_ligand_distance_list, new_ecdock_org_ligand_distance_list, save_path, name = 'ecdock_gen_to_org')
    rmsds_v2(new_ecdock_gen_ligand_distance_list, new_unimol_predict_distance_list, save_path, name = 'ecdock_gen_to_unimol-predict')

    rmsds_v2(new_unimol_gen_ligand_distance_list, new_ecdock_org_ligand_distance_list, save_path, name = 'unimol_gen_to_org')
    rmsds_v2(new_unimol_gen_ligand_distance_list, new_unimol_predict_distance_list, save_path, name = 'unimol_gen_to_unimol-predict')

    rmsds_v2(new_ecdock_org_ligand_distance_list, new_unimol_predict_distance_list, save_path, name = 'org_to_unimol-predict')



if __name__ == "__main__":
    ecdock_path                    = '/mnt/home/fanzhiguang/47/new_KGDiff-EcDock/posebusters_ecdock_cm_equiformer_step25_interaction_limit4.5ai_retrain_fine'
    unimol_path                    = '/mnt/home/fanzhiguang/47/unimol_docking_v2/interface/posebusters_predict_sdf_boxsize10_origin'
    unimol_predict_distance_path   = '/mnt/home/fanzhiguang/47/CrossDocked2020/data/posebusters/posebusters'

    compuete_rmsd(ecdock_path, unimol_path, unimol_predict_distance_path)