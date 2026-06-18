import numpy as np
import torch as th
from joblib import Parallel, delayed
import pandas as pd
import argparse
import os, sys
import MDAnalysis as mda
#sys.path.append("/home/shenchao/resdocktest2/rtmscore2")
sys.path.append(os.path.abspath(__file__).replace("rtmscore.py",".."))
from torch.utils.data import DataLoader
from RTMScore.data.data import VSDataset
from RTMScore.model.utils import collate, run_an_eval_epoch
from RTMScore.model.model2 import RTMScore, DGLGraphTransformer #LigandNet, TargetNet
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

#you need to set the babel libdir first if you need to generate the pocket
#os.environ["BABEL_LIBDIR"] = "/data/fan_zg/anaconda3/envs/torch2.1.0/lib/openbabel/3.1.0"


import copy
from rdkit import Chem
from rdkit.Chem import AllChem
import copy
import subprocess
import time
import multiprocessing

from collections import defaultdict
from ordered_set import OrderedSet
from biopandas.pdb import PandasPdb
from Bio import PDB

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
import math
import dill
from oddt.metrics import enrichment_factor as odd_enrichment_factor

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem
import numpy as np
from sklearn.decomposition import PCA


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from rdkit.Chem import Lipinski
from rdkit.Chem import Descriptors
    
def set_seed(seed):
    torch.manual_seed(seed)  # 设置 PyTorch 的随机数种子
    torch.cuda.manual_seed_all(seed)  # 设置所有 GPU 的随机数种子
    np.random.seed(seed)  # 设置 NumPy 的随机数种子
    random.seed(seed)  # 设置 Python 自带的随机数种子
    torch.backends.cudnn.deterministic = True  # 设置 CuDNN 算法为确定性算法
    torch.backends.cudnn.benchmark = True



def EF_old(per, data_path, refine_flag, score_flag, glide_skip):
    
    if score_flag == 'rtm':
        if refine_flag == 'refine':
            exit_file = f'{data_path}/{score_flag}_{refine_flag}_{glide_skip}_save_EF_dict_{per}.pkl'
        else:
            exit_file = f'{data_path}/{score_flag}_{refine_flag}_{glide_skip}_save_EF_dict_{per}.pkl'
        if os.path.exists(exit_file):
            with open(exit_file,'rb') as f:
                save_dict = dill.load(f)
                
            selected_active     = save_dict['selected_active']
            selected_molecules  = save_dict['selected_molecules']
            total_active        = save_dict['total_active']
            total_molecules     = save_dict['total_molecules']
                
            # 筛选出的活性分子比例
            hit_rate_selected = len(selected_active) / len(selected_molecules)
            
            # 总体中的活性分子比例
            hit_rate_total = len(total_active) / len(total_molecules)

            # 计算富集率
            if hit_rate_total == 0:
                return 0  # 避免除以零
            enrichment_factor = hit_rate_selected / hit_rate_total
        
        
        else:
            
            #计算富集率
            save_dict = {}
            data_name_list = []
            for name in list(os.listdir(data_path))[:]:
                path = os.path.join(data_path, name)
                if os.path.exists(path) and os.path.isdir(path) and os.listdir(path) and 'model' not in name: #目录存在且不空
                    data_name_list.append(name)

            loss_file_list = []
            rtm_data_dict  = {}
            total_active, total_molecules, selected_active, selected_molecules = [], [], [], []
            for name in data_name_list:
                try:
                    if refine_flag == 'refine':
                        with open(f'{data_path}/{name}/step4/refine_{name}_rtmscore.csv') as f:
                            rtm_data_dict[name] = float(f.readline().strip().split('\t')[1]) #读第一行第二列即可
                            total_molecules.append(name)
                            if 'active' in name:
                                total_active.append(name)
                    else:
                        with open(f'{data_path}/{name}/step4/{name}_rtmscore.csv') as f:
                            rtm_data_dict[name] = float(f.readline().strip().split('\t')[1]) #读第一行第二列即可
                            total_molecules.append(name)
                            if 'active' in name:
                                total_active.append(name)
                except Exception as e:
                    #print(e)
                    loss_file_list.append(name)
                    continue
            
            print('loss_file_list num / all num:', len(loss_file_list), len(data_name_list)) #loss_file_list num / all num: 642 1884
            print('total_active num / total_molecules num:', len(total_active), len(total_molecules))
            
            
            sort_rtm_data_dict  = dict(sorted(rtm_data_dict.items(), key=lambda item: item[1], reverse=True)) #从大到小
            sort_data_name_list = list(sort_rtm_data_dict.keys())
            print('len(sort_data_name_list):', len(sort_data_name_list))

            # 取TOP n
            top_n = math.ceil((len(sort_data_name_list) * per))
            print('top_n:', top_n) #9/18/94
            
            selected_molecules = sort_data_name_list[:top_n]

            for nm in selected_molecules:
                if 'active' in nm:
                    selected_active.append(nm)
                    
            print('selected_active num / selected_molecules num:', len(selected_active), len(selected_molecules))
            
            # 筛选出的活性分子比例
            hit_rate_selected = len(selected_active) / len(selected_molecules)
            
            # 总体中的活性分子比例
            hit_rate_total = len(total_active) / len(total_molecules)
            
            save_dict['selected_active']    = selected_active
            save_dict['selected_molecules'] = selected_molecules
            save_dict['total_active']       = total_active
            save_dict['total_molecules']    = total_molecules
                
            if refine_flag == 'refine':
                save_file = f'{data_path}/{score_flag}_{refine_flag}_{glide_skip}_save_EF_dict_{per}.pkl'
            else:
                save_file = f'{data_path}/{score_flag}_{refine_flag}_{glide_skip}_save_EF_dict_{per}.pkl'
            
            with open(f'{save_file}', 'wb') as f:
                dill.dump(save_dict, f)
            
            # 计算富集率
            if hit_rate_total == 0:
                return 0  # 避免除以零
            enrichment_factor = hit_rate_selected / hit_rate_total
        
        return enrichment_factor

    elif score_flag == 'glide':
        if refine_flag == 'refine':
            exit_file = f'{data_path}/{score_flag}_{refine_flag}_{glide_skip}_save_EF_dict_{per}.pkl'
        else:
            exit_file = f'{data_path}/{score_flag}_{refine_flag}_{glide_skip}_save_EF_dict_{per}.pkl'
        if os.path.exists(exit_file):
            with open(exit_file,'rb') as f:
                save_dict = dill.load(f)
                
            selected_active     = save_dict['selected_active']
            selected_molecules  = save_dict['selected_molecules']
            total_active        = save_dict['total_active']
            total_molecules     = save_dict['total_molecules']
                
            # 筛选出的活性分子比例
            hit_rate_selected = len(selected_active) / len(selected_molecules)
            
            # 总体中的活性分子比例
            hit_rate_total = len(total_active) / len(total_molecules)

            # 计算富集率
            if hit_rate_total == 0:
                return 0  # 避免除以零
            enrichment_factor = hit_rate_selected / hit_rate_total
    
        else:
            #计算富集率
            save_dict = {}
            data_name_list = []
            for name in list(os.listdir(data_path))[:]:
                path = os.path.join(data_path, name)
                if os.path.exists(path) and os.path.isdir(path) and os.listdir(path) and 'model' not in name: #目录存在且不空
                    data_name_list.append(name)

            loss_file_list = []
            rtm_data_dict  = {}
            total_active, total_molecules, selected_active, selected_molecules = [], [], [], []
            for name in data_name_list:
                if glide_skip == 'skip':
                    try:
                        if refine_flag == 'refine':
                            with open(f'{data_path}/{name}/step4/{name}_glide_score_refine.txt') as f: #active0_glide_score_refine
                                score_list = []
                                for line in f:
                                    score_list.append(float(line.strip().split('\t')[1]))
                                rtm_data_dict[name] = min(score_list) #读第一行第二列即可
                                
                                total_molecules.append(name)
                                if 'active' in name:
                                    total_active.append(name)
                        else:
                            with open(f'{data_path}/{name}/step4/{name}_glide_score.txt') as f:
                                score_list = []
                                for line in f:
                                    score_list.append(float(line.strip().split('\t')[1]))
                                rtm_data_dict[name] = min(score_list) #读第一行第二列即可
                                
                                total_molecules.append(name)
                                if 'active' in name:
                                    total_active.append(name)
                    except Exception as e:
                        #print(e)
                        loss_file_list.append(name)
                        continue
                else:
                    try:
                        if refine_flag == 'refine':
                            file = f'{data_path}/{name}/step4/{name}_glide_score_refine.txt'
                            if is_file_exist_and_not_empty(file):
                                with open(file) as f: #active0_glide_score_refine
                                    score_list = []
                                    for line in f:
                                        score_list.append(float(line.strip().split('\t')[1]))
                                    rtm_data_dict[name] = min(score_list) #读第一行第二列即可
                                    
                                    total_molecules.append(name)
                                    if 'active' in name:
                                        total_active.append(name)
                            
                            #不存在，则设置成最大值
                            else:
                                rtm_data_dict[name] = 100000 
                                total_molecules.append(name)
                                if 'active' in name:
                                    total_active.append(name)
                        else:
                            if is_file_exist_and_not_empty(file):
                                with open(f'{data_path}/{name}/step4/{name}_glide_score.txt') as f:
                                    score_list = []
                                    for line in f:
                                        score_list.append(float(line.strip().split('\t')[1]))
                                    rtm_data_dict[name] = min(score_list) #读第一行第二列即可
                                    total_molecules.append(name)
                                    if 'active' in name:
                                        total_active.append(name)
                            #不存在，则设置成最大值
                            else:
                                rtm_data_dict[name] = 100000 
                                total_molecules.append(name)
                                if 'active' in name:
                                    total_active.append(name)
                                    
                    except Exception as e:
                        #print(e)
                        loss_file_list.append(name)
                        continue
                    
            print('loss_file_list num / all num:', len(loss_file_list), len(data_name_list)) #loss_file_list num / all num: 642 1884
            print('total_active num / total_molecules num:', len(total_active), len(total_molecules))
            
            
            sort_rtm_data_dict  = dict(sorted(rtm_data_dict.items(), key=lambda item: item[1], reverse=False)) #从小到大，值是负的，越小越好
            sort_data_name_list = list(sort_rtm_data_dict.keys())
            print('len(sort_data_name_list):', len(sort_data_name_list))

            # 取TOP n
            top_n = math.ceil((len(sort_data_name_list) * per))
            print('top_n:', top_n) #9/18/94
            
            selected_molecules = sort_data_name_list[:top_n]

            for nm in selected_molecules:
                if 'active' in nm:
                    selected_active.append(nm)
                    
            print('selected_active num / selected_molecules num:', len(selected_active), len(selected_molecules))
            
            # 筛选出的活性分子比例
            hit_rate_selected = len(selected_active) / len(selected_molecules)
            
            # 总体中的活性分子比例
            hit_rate_total = len(total_active) / len(total_molecules)
            
            save_dict['selected_active']    = selected_active
            save_dict['selected_molecules'] = selected_molecules
            save_dict['total_active']       = total_active
            save_dict['total_molecules']    = total_molecules
                
            if refine_flag == 'refine':
                save_file = f'{data_path}/{score_flag}_{refine_flag}_{glide_skip}_save_EF_dict_{per}.pkl'
            else:
                save_file = f'{data_path}/{score_flag}_{refine_flag}_{glide_skip}_save_EF_dict_{per}.pkl'
            
            with open(f'{save_file}', 'wb') as f:
                dill.dump(save_dict, f)
            
            # 计算富集率
            if hit_rate_total == 0:
                return 0  # 避免除以零
            enrichment_factor = hit_rate_selected / hit_rate_total
        
        return enrichment_factor
    
    
    
    
def EF(per, data_path, refine_flag, score_flag, glide_skip, special_name_list = None):
    if refine_flag == 'refine':
        exit_file = f'{data_path}/{score_flag}_{refine_flag}_{glide_skip}_save_EF_dict_{per}.pkl'
    else:
        exit_file = f'{data_path}/{score_flag}_{refine_flag}_{glide_skip}_save_EF_dict_{per}.pkl'
    #if not os.path.exists(exit_file):
    '''
    if os.path.exists(exit_file):
        with open(exit_file,'rb') as f:
            save_dict = dill.load(f)
            
        selected_active     = save_dict['selected_active']
        selected_molecules  = save_dict['selected_molecules']
        total_active        = save_dict['total_active']
        total_molecules     = save_dict['total_molecules']
            
        # 筛选出的活性分子比例
        hit_rate_selected = len(selected_active) / len(selected_molecules)
        
        # 总体中的活性分子比例
        hit_rate_total = len(total_active) / len(total_molecules)

        # 计算富集率
        if hit_rate_total == 0:
            return 0  # 避免除以零
        enrichment_factor = hit_rate_selected / hit_rate_total
    
        return enrichment_factor
    '''
    

    
    #计算富集率
    save_dict = {}
    data_name_list = []
    if special_name_list:
        for name in special_name_list[:]:
            path = os.path.join(data_path, name)
            if os.path.exists(path) and os.path.isdir(path) and os.listdir(path) and 'model' not in name: #目录存在且不空
                data_name_list.append(name)
    else:
        for name in list(os.listdir(data_path))[:]:
            path = os.path.join(data_path, name)
            if os.path.exists(path) and os.path.isdir(path) and os.listdir(path) and 'model' not in name: #目录存在且不空
                data_name_list.append(name)
                

    loss_file_list = []
    rtm_data_dict  = {}
    total_active, total_molecules, selected_active, selected_molecules = [], [], [], []
    for name in data_name_list:
        try:
            if score_flag == 'glide':
                if refine_flag == 'refine':
                    file = f'{data_path}/{name}/{name}_glide_score_refine.txt'
                    #if not is_file_exist_and_not_empty(file):
                        #file = f'{data_path}/{name}/{name}_glide_score.txt'

                        
                    if is_file_exist_and_not_empty(file):
                        with open(file) as f:
                            score_list = []
                            for line in f:
                                score_list.append(float(line.strip().split('\t')[1]))
                            rtm_data_dict[name] = min(score_list) #读第一行第二列即可
                    else:
                        loss_file_list.append(name)
                        print(f'文件{file}不存在，设置为最差值')
                        rtm_data_dict[name] = 10000
                        
                        
                    total_molecules.append(name)
                    if 'active' in name:
                        total_active.append(name)
                            
                            
                else:
                    file = f'{data_path}/{name}/{name}_glide_score.txt'
                    #if not is_file_exist_and_not_empty(file):
                        #file = f'{data_path}/{name}/{name}_glide_score_refine.txt'

                        
                    if is_file_exist_and_not_empty(file):
                        with open(file) as f:
                            score_list = []
                            for line in f:
                                score_list.append(float(line.strip().split('\t')[1]))
                            rtm_data_dict[name] = min(score_list) #读第一行第二列即可
                    else:
                        loss_file_list.append(name)
                        print(f'文件{file}不存在，设置为最差值')
                        rtm_data_dict[name] = 10000
                        
                    total_molecules.append(name)
                    if 'active' in name:
                        total_active.append(name)
        
                            
            elif score_flag == 'rtm':
                if refine_flag == 'refine':
                    file = f'{data_path}/{name}/refine_{name}_rtmscore.csv'
                    #if not is_file_exist_and_not_empty(file):
                        #file = f'{data_path}/{name}/{name}_rtmscore.csv'

                    
                    
                    if is_file_exist_and_not_empty(file):
                        with open(file) as f:
                            score_list = []
                            for line in f:
                                score_list.append(float(line.strip().split('\t')[1]))
                            rtm_data_dict[name] = max(score_list) #读第一行第二列即可
                    else:
                        loss_file_list.append(name)
                        print(f'文件{file}不存在，设置为最差值')
                        rtm_data_dict[name] = -10000
                        
                    
                    total_molecules.append(name)
                    if 'active' in name:
                        total_active.append(name)
                        
                else:
                    file =f'{data_path}/{name}/{name}_rtmscore.csv'
                    #if not is_file_exist_and_not_empty(file):
                        #file = f'{data_path}/{name}/refine_{name}_rtmscore.csv'

                        
                    if is_file_exist_and_not_empty(file):
                        with open(file) as f:
                            score_list = []
                            for line in f:
                                score_list.append(float(line.strip().split('\t')[1]))
                            rtm_data_dict[name] = max(score_list) #读第一行第二列即可
                    else:
                        loss_file_list.append(name)
                        print(f'文件{file}不存在，设置为最差值')
                        rtm_data_dict[name] = -10000
                        
                    total_molecules.append(name)
                    if 'active' in name:
                        total_active.append(name)
                            
        except Exception as e:
            print(e)
            loss_file_list.append(name)
            continue
    
    print(f'loss num / all num: {len(loss_file_list)}/{len(data_name_list)}')
    #print('loss_file_list num / all num:', len(loss_file_list), len(data_name_list)) #loss_file_list num / all num: 642 1884
    #print('total_active num / total_molecules num:', len(total_active), len(total_molecules))
    
    if score_flag == 'rtm':
        sort_rtm_data_dict  = dict(sorted(rtm_data_dict.items(), key=lambda item: item[1], reverse=True)) #从大到小
    elif score_flag == 'glide':
        sort_rtm_data_dict  = dict(sorted(rtm_data_dict.items(), key=lambda item: item[1], reverse=False)) #从小到大
    sort_data_name_list = list(sort_rtm_data_dict.keys())
    #print('sort_rtm_data_dict:', sort_rtm_data_dict)
    #print('len(sort_data_name_list):', len(sort_data_name_list))

    # 取TOP n
    top_n = math.ceil((len(sort_data_name_list) * per))
    if top_n == 0:
        #print('top_n:', top_n)
        top_n = 1
    print('top_n:', top_n) #9/18/94
    
    selected_molecules = sort_data_name_list[:top_n]

    for nm in selected_molecules:
        if 'active' in nm:
            selected_active.append(nm)
            
    #print('selected_active num / selected_molecules num:', len(selected_active), len(selected_molecules))
    
    # 筛选出的活性分子比例
    hit_rate_selected = len(selected_active) / len(selected_molecules)
    
    # 总体中的活性分子比例
    assert len(total_active) != 0
    hit_rate_total = len(total_active) / len(total_molecules)
    
    save_dict['selected_active']    = selected_active
    save_dict['selected_molecules'] = selected_molecules
    save_dict['total_active']       = total_active
    save_dict['total_molecules']    = total_molecules
        
    if refine_flag == 'refine':
        save_file = f'{data_path}/{score_flag}_{refine_flag}_{glide_skip}_save_EF_dict_{per}.pkl'
    else:
        save_file = f'{data_path}/{score_flag}_{refine_flag}_{glide_skip}_save_EF_dict_{per}.pkl'
    
    with open(f'{save_file}', 'wb') as f:
        dill.dump(save_dict, f)
    
    # 计算富集率
    if hit_rate_total == 0:
        print('hit_rate_total:', hit_rate_total)
        return 0, len(loss_file_list) / len(data_name_list), loss_file_list # 避免除以零
    enrichment_factor = hit_rate_selected / hit_rate_total
    
    success_file_list = set(data_name_list) - set(loss_file_list)
        
    return enrichment_factor, len(loss_file_list) / len(data_name_list), loss_file_list, success_file_list
        

    

def EF2(per, data_path, refine_flag, score_flag, glide_skip, special_name_list = None, random_one = False):
    # 假设我们有以下数据：
    # scores - 化合物的预测得分列表（越高表示越可能是活性化合物）
    # labels - 对应的真实标签（1表示活性，0表示非活性）
    # cutoff - 前百分之多少的化合物（通常用0.01, 0.05等）
    
    if refine_flag == 'refine':
        exit_file = f'{data_path}/{score_flag}_{refine_flag}_{glide_skip}_save_EF_dict_{per}.pkl'
    else:
        exit_file = f'{data_path}/{score_flag}_{refine_flag}_{glide_skip}_save_EF_dict_{per}.pkl'
    #if not os.path.exists(exit_file):
    '''
    if os.path.exists(exit_file):
        with open(exit_file,'rb') as f:
            save_dict = dill.load(f)
            
        selected_active     = save_dict['selected_active']
        selected_molecules  = save_dict['selected_molecules']
        total_active        = save_dict['total_active']
        total_molecules     = save_dict['total_molecules']
            
        # 筛选出的活性分子比例
        hit_rate_selected = len(selected_active) / len(selected_molecules)
        
        # 总体中的活性分子比例
        hit_rate_total = len(total_active) / len(total_molecules)

        # 计算富集率
        if hit_rate_total == 0:
            return 0  # 避免除以零
        enrichment_factor = hit_rate_selected / hit_rate_total
        return envelope_factor
    '''
    

    
    #计算富集率
    save_dict = {}
    data_name_list = []
    if special_name_list:
        for name in special_name_list[:]:
            path = os.path.join(data_path, name)
            if os.path.exists(path) and os.path.isdir(path) and os.listdir(path) and 'model' not in name: #目录存在且不空
                data_name_list.append(name)
    else:
        for name in list(os.listdir(data_path))[:]:
            path = os.path.join(data_path, name)
            if os.path.exists(path) and os.path.isdir(path) and os.listdir(path) and 'model' not in name: #目录存在且不空
                data_name_list.append(name)

    
    loss_file_list = []
    rtm_data_dict  = {}
    label_data_dict  = {}
    total_active, total_molecules, selected_active, selected_molecules = [], [], [], []
    for name in data_name_list:
        try:
            if score_flag == 'glide':
                if refine_flag == 'refine':
                    file = f'{data_path}/{name}/{name}_glide_score_refine.txt'
                    #if not is_file_exist_and_not_empty(file):
                        #file = f'{data_path}/{name}/{name}_glide_score.txt'
                        
                    if is_file_exist_and_not_empty(file):
                        with open(file) as f:
                            score_list = []
                            for line in f:
                                score_list.append(float(line.strip().split('\t')[1]))
                            if random_one:
                                rtm_data_dict[name] = np.random.choice(score_list)
                            else:
                                rtm_data_dict[name] = min(score_list) #读第一行第二列即可
                    else:
                        loss_file_list.append(name)
                        print(f'文件{file}不存在，设置为最差值')
                        rtm_data_dict[name] = 10000 #不存在，则设置为最差的得分
                            
                    total_molecules.append(name)
                    if 'active' in name:
                        total_active.append(name)
                        label_data_dict[name] = 1
                    else:
                        label_data_dict[name] = 0
                            
                else:
                    file = f'{data_path}/{name}/{name}_glide_score.txt'
                    #if not is_file_exist_and_not_empty(file):
                        #file = f'{data_path}/{name}/{name}_glide_score_refine.txt'
                            
                            
                    if is_file_exist_and_not_empty(file):
                        with open(file) as f:
                            score_list = []
                            for line in f:
                                score_list.append(float(line.strip().split('\t')[1]))
                            if random_one:
                                rtm_data_dict[name] = np.random.choice(score_list)
                            else:
                                rtm_data_dict[name] = min(score_list) #读第一行第二列即可
                    else:
                        loss_file_list.append(name)
                        print(f'文件{file}不存在，设置为最差值')
                        rtm_data_dict[name] = 10000 #不存在，则设置为最差的得分
                            
                    total_molecules.append(name)
                    if 'active' in name:
                        total_active.append(name)
                        label_data_dict[name] = 1
                    else:
                        label_data_dict[name] = 0
                            
                            
            elif score_flag == 'rtm':    
                if refine_flag == 'refine':
                    file = f'{data_path}/{name}/refine_{name}_rtmscore.csv'
                    if not is_file_exist_and_not_empty(file):
                        file = f'{data_path}/{name}/{name}_rtmscore.csv'

                    '''
                    file2 = f'{data_path}/{name}/{name}_rtmscore.csv'
                    
                    score_list = []
                    
                    if is_file_exist_and_not_empty(file2):
                        with open(file2) as f:
                            for line in f:
                                score_list.append(float(line.strip().split('\t')[1])) 
                                    
                    if is_file_exist_and_not_empty(file):
                        with open(file) as f:
                            for line in f:
                                score_list.append(float(line.strip().split('\t')[1]))
                        rtm_data_dict[name] = max(score_list) #读第一行第二列即可
                    
                    else:
                        loss_file_list.append(name)
                        print(f'文件{file}不存在，设置为最差值')
                        rtm_data_dict[name] = -10000 #不存在，则设置为最差的得分
                        
                    '''
                    
                    
                    if is_file_exist_and_not_empty(file):
                        with open(file) as f:
                            score_list = []
                            for line in f:
                                score_list.append(float(line.strip().split('\t')[1]))
                            
                            if random_one:
                                rtm_data_dict[name] = np.random.choice(score_list)
                            else:
                                rtm_data_dict[name] = max(score_list) #读第一行第二列即可
                    else:
                        loss_file_list.append(name)
                        print(f'文件{file}不存在，设置为最差值')
                        rtm_data_dict[name] = -10000 #不存在，则设置为最差的得分
                    

                            
                    total_molecules.append(name)
                    if 'active' in name:
                        total_active.append(name)
                        label_data_dict[name] = 1
                    else:
                        label_data_dict[name] = 0
                            
                else:
                    file =f'{data_path}/{name}/{name}_rtmscore.csv'
                    #if not is_file_exist_and_not_empty(file):
                        #file = f'{data_path}/{name}/refine_{name}_rtmscore.csv'


                    if is_file_exist_and_not_empty(file):
                        with open(file) as f:
                            score_list = []
                            for line in f:
                                score_list.append(float(line.strip().split('\t')[1]))
                                
                            if random_one:
                                rtm_data_dict[name] = np.random.choice(score_list)
                            else:
                                rtm_data_dict[name] = max(score_list) #读第一行第二列即可
                    else:
                        loss_file_list.append(name)
                        print(f'文件{file}不存在，设置为最差值')
                        rtm_data_dict[name] = -10000 #不存在，则设置为最差的得分
                        
                        
                    total_molecules.append(name)
                    
                    
                    if 'active' in name:
                        total_active.append(name)
                        label_data_dict[name] = 1
                    else:
                        label_data_dict[name] = 0
                        
        except Exception as e:
            print(e)
            #exit()
            loss_file_list.append(name)
            continue
    

    
    print(f'loss num / all num: {len(loss_file_list)}/{len(data_name_list)}')
    scores = np.array(list(rtm_data_dict.values()))
    labels = np.array(list(label_data_dict.values()))
    if score_flag == 'rtm':
        sort_rtm_data_dict  = dict(sorted(rtm_data_dict.items(), key=lambda item: item[1], reverse=True)) #从大到小
    elif score_flag == 'glide':
        sort_rtm_data_dict  = dict(sorted(rtm_data_dict.items(), key=lambda item: item[1], reverse=False)) #从小到大
    
    #print('sort_rtm_data_dict:', sort_rtm_data_dict)
        
    if score_flag == 'rtm':
        #print('score_flag:', score_flag)
        ids = np.argsort(scores)[::-1] #从大到小排序（rtm是正值） #np.argsort(arr, kind='mergesort')  # 稳定排序
    elif score_flag == 'glide':
        #print('score_flag:', score_flag) ##从小到大排序（glide是负值）
        ids = np.argsort(scores)
    labels = labels[ids]
    #print('scores:', scores[:10])
    #print('labels:', labels[:10])
    #print('scores[labels]:', scores[ids][:10])
    #print('scores:', scores[:30])
    #print('labels:', labels[:30])
    cutoff = per # 计算前30%的富集率
    #print('cutoff:', cutoff)
    #print('scores:', scores[ids])

    ef = odd_enrichment_factor(labels, scores, percentage = cutoff, kind = 'fold') #scores没用， labels是经过得分排序后的
    
    success_file_list = set(data_name_list) - set(loss_file_list)
    return ef, len(loss_file_list) / len(data_name_list), loss_file_list, success_file_list       








def EF3(per, data_path, refine_flag, score_flag, glide_skip, special_name_list = None, random_one = False):
    # 假设我们有以下数据：
    # scores - 化合物的预测得分列表（越高表示越可能是活性化合物）
    # labels - 对应的真实标签（1表示活性，0表示非活性）
    # cutoff - 前百分之多少的化合物（通常用0.01, 0.05等）
    
    if refine_flag == 'refine':
        exit_file = f'{data_path}/{score_flag}_{refine_flag}_{glide_skip}_save_EF_dict_{per}.pkl'
    else:
        exit_file = f'{data_path}/{score_flag}_{refine_flag}_{glide_skip}_save_EF_dict_{per}.pkl'
    #if not os.path.exists(exit_file):
    '''
    if os.path.exists(exit_file):
        with open(exit_file,'rb') as f:
            save_dict = dill.load(f)
            
        selected_active     = save_dict['selected_active']
        selected_molecules  = save_dict['selected_molecules']
        total_active        = save_dict['total_active']
        total_molecules     = save_dict['total_molecules']
            
        # 筛选出的活性分子比例
        hit_rate_selected = len(selected_active) / len(selected_molecules)
        
        # 总体中的活性分子比例
        hit_rate_total = len(total_active) / len(total_molecules)

        # 计算富集率
        if hit_rate_total == 0:
            return 0  # 避免除以零
        enrichment_factor = hit_rate_selected / hit_rate_total
        return envelope_factor
    '''
    

    
    #计算富集率
    save_dict = {}
    data_name_list = []
    if special_name_list:
        for name in special_name_list[:]:
            path = os.path.join(data_path, name)
            if os.path.exists(path) and os.path.isdir(path) and os.listdir(path) and 'model' not in name: #目录存在且不空
                data_name_list.append(name)
    else:
        for name in list(os.listdir(data_path))[:]:
            path = os.path.join(data_path, name)
            if os.path.exists(path) and os.path.isdir(path) and os.listdir(path) and 'model' not in name: #目录存在且不空
                data_name_list.append(name)

    
    loss_file_list = []
    rtm_data_dict  = {}
    label_data_dict  = {}
    total_active, total_molecules, selected_active, selected_molecules = [], [], [], []
    for name in data_name_list:
        try:
            if score_flag == 'glide':
                if refine_flag == 'refine':
                    file = f'{data_path}/{name}/{name}_glide_score_refine.txt'
                    #if not is_file_exist_and_not_empty(file):
                        #file = f'{data_path}/{name}/{name}_glide_score.txt'
                        
                    if is_file_exist_and_not_empty(file):
                        with open(file) as f:
                            score_list = []
                            for line in f:
                                score_list.append(float(line.strip().split('\t')[1]))
                            if random_one:
                                rtm_data_dict[name] = np.random.choice(score_list)
                            else:
                                rtm_data_dict[name] = min(score_list) #读第一行第二列即可
                    else:
                        loss_file_list.append(name)
                        print(f'文件{file}不存在，设置为最差值')
                        rtm_data_dict[name] = 10000 #不存在，则设置为最差的得分
                            
                    total_molecules.append(name)
                    if 'active' in name:
                        total_active.append(name)
                        label_data_dict[name] = 1
                    else:
                        label_data_dict[name] = 0
                            
                else:
                    file = f'{data_path}/{name}/{name}_glide_score.txt'
                    #if not is_file_exist_and_not_empty(file):
                        #file = f'{data_path}/{name}/{name}_glide_score_refine.txt'
                            
                            
                    if is_file_exist_and_not_empty(file):
                        with open(file) as f:
                            score_list = []
                            for line in f:
                                score_list.append(float(line.strip().split('\t')[1]))
                            if random_one:
                                rtm_data_dict[name] = np.random.choice(score_list)
                            else:
                                rtm_data_dict[name] = min(score_list) #读第一行第二列即可
                    else:
                        loss_file_list.append(name)
                        print(f'文件{file}不存在，设置为最差值')
                        rtm_data_dict[name] = 10000 #不存在，则设置为最差的得分
                            
                    total_molecules.append(name)
                    if 'active' in name:
                        total_active.append(name)
                        label_data_dict[name] = 1
                    else:
                        label_data_dict[name] = 0
                            
                            
            elif score_flag == 'rtm':    
                if refine_flag == 'refine':
                    file = f'{data_path}/{name}/refine_{name}_rtmscore.csv'
                    if not is_file_exist_and_not_empty(file):
                        file = f'{data_path}/{name}/{name}_rtmscore.csv'

                    '''
                    file2 = f'{data_path}/{name}/{name}_rtmscore.csv'
                    
                    score_list = []
                    
                    if is_file_exist_and_not_empty(file2):
                        with open(file2) as f:
                            for line in f:
                                score_list.append(float(line.strip().split('\t')[1])) 
                                    
                    if is_file_exist_and_not_empty(file):
                        with open(file) as f:
                            for line in f:
                                score_list.append(float(line.strip().split('\t')[1]))
                        rtm_data_dict[name] = max(score_list) #读第一行第二列即可
                    
                    else:
                        loss_file_list.append(name)
                        print(f'文件{file}不存在，设置为最差值')
                        rtm_data_dict[name] = -10000 #不存在，则设置为最差的得分
                        
                    '''
                    
                    
                    if is_file_exist_and_not_empty(file):
                        with open(file) as f:
                            score_list = []
                            for line in f:
                                score_list.append(float(line.strip().split('\t')[1]))
                            
                            if random_one:
                                rtm_data_dict[name] = np.random.choice(score_list)
                            else:
                                rtm_data_dict[name] = max(score_list) #读第一行第二列即可
                    else:
                        loss_file_list.append(name)
                        print(f'文件{file}不存在，设置为最差值')
                        rtm_data_dict[name] = -10000 #不存在，则设置为最差的得分
                    

                            
                    total_molecules.append(name)
                    if 'active' in name:
                        total_active.append(name)
                        label_data_dict[name] = 1
                    else:
                        label_data_dict[name] = 0
                            
                else:
                    file =f'{data_path}/{name}/{name}_rtmscore.csv'
                    #if not is_file_exist_and_not_empty(file):
                        #file = f'{data_path}/{name}/refine_{name}_rtmscore.csv'


                    if is_file_exist_and_not_empty(file):
                        with open(file) as f:
                            score_list = []
                            for line in f:
                                score_list.append(float(line.strip().split('\t')[1]))
                                
                            if random_one:
                                rtm_data_dict[name] = np.random.choice(score_list)
                            else:
                                rtm_data_dict[name] = max(score_list) #读第一行第二列即可
                    else:
                        loss_file_list.append(name)
                        print(f'文件{file}不存在，设置为最差值')
                        rtm_data_dict[name] = -10000 #不存在，则设置为最差的得分
                        
                        
                    total_molecules.append(name)
                    
                    
                    if 'active' in name:
                        total_active.append(name)
                        label_data_dict[name] = 1
                    else:
                        label_data_dict[name] = 0
                        
        except Exception as e:
            print(e)
            #exit()
            loss_file_list.append(name)
            continue
    

    
    print(f'loss num / all num: {len(loss_file_list)}/{len(data_name_list)}')
    scores = np.array(list(rtm_data_dict.values()))
    labels = np.array(list(label_data_dict.values()))
    if score_flag == 'rtm':
        sort_rtm_data_dict  = dict(sorted(rtm_data_dict.items(), key=lambda item: item[1], reverse=True)) #从大到小
    elif score_flag == 'glide':
        sort_rtm_data_dict  = dict(sorted(rtm_data_dict.items(), key=lambda item: item[1], reverse=False)) #从小到大
    
    #print('sort_rtm_data_dict:', sort_rtm_data_dict)
        
    if score_flag == 'rtm':
        #print('score_flag:', score_flag)
        ids = np.argsort(scores)[::-1] #从大到小排序（rtm是正值） #np.argsort(arr, kind='mergesort')  # 稳定排序
    elif score_flag == 'glide':
        #print('score_flag:', score_flag) ##从小到大排序（glide是负值）
        ids = np.argsort(scores)
    labels = labels[ids]
    #print('scores:', scores[:10])
    #print('labels:', labels[:10])
    #print('scores[labels]:', scores[ids][:10])#print('scores:', scores[:30])
    #print('labels:', labels[:30])
    cutoff = per # 计算前30%的富集率
    #print('cutoff:', cutoff)
    #print('scores:', scores[ids])

    ef = odd_enrichment_factor(labels, scores, percentage = cutoff, kind = 'fold') #scores没用， labels是经过得分排序后的
    
    success_file_list = set(data_name_list) - set(loss_file_list)
    return ef, len(loss_file_list) / len(data_name_list), loss_file_list, success_file_list, sort_rtm_data_dict     





    


def is_file_exist_and_not_empty(filepath):
    # 检查文件是否存在
    if not os.path.exists(filepath):
        return False
    
    # 检查是否是文件（不是目录）
    if not os.path.isfile(filepath):
        return False
    
    # 检查文件大小是否大于0
    if os.path.getsize(filepath) > 0:
        return True
    else:
        return False







def EF_box():
    # ─── 数据加载 ────────────────────────────────────────
    EF_base = '/data/fan_zg/MDocking/EcDock_Evaluate/EF'
    files = {
        'Uni-Mol Docking V2+RTMScore':   'unimol_docking_v2_norefine_rtm.pickle',
        'KarmaDock Align+RTMScore':     'karmadock_norefine_rtm.pickle',
        'Glide+GlideScore':             'glide_norefine_glide.pickle',
        'Glide+RTMScore':               'glide_norefine_rtm.pickle',
        'CarsiDock+RTMScore':           'carsidock_norefine_rtm.pickle',
        'ECDock+GlideScore':            'current_best_mmf_ecdock_norefine_glide.pickle',
        'ECDock+RTMScore':              'ecdock_old_copy_norefine_rtm.pickle',
        'ECDock+Glide_Refine+GlideScore': 'current_best_mmf_ecdock_refine_glide.pickle',
        'ECDock+Glide_Refine+RTMScore': 'current_best_mmf_ecdock_refine_rtm.pickle'
    }
    EF_dict = {}
    for name, fname in files.items():
        try:
            with open(f'{EF_base}/{fname}', 'rb') as f:
                EF_dict[name] = dill.load(f)
        except Exception as e:
            print(f"WARNING: fail loading {fname}: {e}")

    # ─── 构造 DataFrame ─────────────────────────────────
    conc_map = {0.005: "0.5%", 0.01: "1.0%", 0.05: "5.0%"}
    records = []
    for model, d in EF_dict.items():
        for conc, label in conc_map.items():
            vals = np.asarray(d.get(conc, []))
            if vals.size:
                records += [{"Concentration": label, "Model": model, "EF": float(v)} for v in vals]
    df = pd.DataFrame(records)
    models = list(EF_dict.keys())
    if df.empty:
        raise RuntimeError("No EF data available!")

    # ─── 绘图配置 ───────────────────────────────────────
    sns.set_theme(
        style="whitegrid", context="notebook", font_scale=1.1,
        palette="tab10",
        rc={"axes.edgecolor": "black", "axes.linewidth": 1.2}
    )
    fig, ax = plt.subplots(figsize=(12, 6))

    # 箱线图 + 均值显示：白色菱形在每个箱体中央
    sns.boxplot(
        data=df, x="Concentration", y="EF", hue="Model",
        palette="tab10", ax=ax, showfliers=False, whis=[0, 95],
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "black",
            "markeredgecolor": "black",
            "markersize": 6
        },
        medianprops={"color": "firebrick", "linewidth": 2}
    )

    # 底部图例，每行最多4项
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles[:len(models)], labels=labels[:len(models)],
        title="Model", loc="upper center", bbox_to_anchor=(0.5, -0.15),
        ncol=4, frameon=True, fontsize=10, title_fontsize=12
    )

    # 去掉 x 轴标题、图表主标题
    ax.set_xlabel("")
    ax.set_title("")

    # 设置 y 轴标签
    ax.set_ylabel("Enrichment Factor", fontsize=15)

    # 边框与网格美化
    for sp in ax.spines.values():
        sp.set_edgecolor('black')
        sp.set_linewidth(1)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out_path = f"{EF_base}/EF.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"图已保存至：{out_path}")
    
    
    
def small_EF_box():
    # ─── 数据加载 ────────────────────────────────────────
    EF_base = '/data/fan_zg/MDocking/EcDock_Evaluate/EF'
    '''
    files = {
        'Uni-Mol Docking V2+RTMScore':   'unimol_docking_v2_norefine_rtm.pickle',
        'KarmaDock Align+RTMScore':     'karmadock_norefine_rtm.pickle',
        'Glide+GlideScore':             'glide_norefine_glide.pickle',
        'Glide+RTMScore':               'glide_norefine_rtm.pickle',
        'CarsiDock+RTMScore':           'carsidock_norefine_rtm.pickle',
        'ECDock+GlideScore':            'current_best_mmf_ecdock_norefine_glide.pickle',
        'ECDock+RTMScore':              'ecdock_old_copy_norefine_rtm.pickle',
        'ECDock+Glide_Refine+GlideScore': 'current_best_mmf_ecdock_refine_glide.pickle',
        'ECDock+Glide_Refine+RTMScore': 'current_best_mmf_ecdock_refine_rtm.pickle'
    }
    '''
    
    
    files = {
        #'Glide+GlideScore':             'glide_norefine_glide.pickle',
        'Glide':               'glide_norefine_rtm.pickle',
        'CarsiDock':           'carsidock_norefine_rtm.pickle',
        'KarmaDock(Align)':     'karmadock_norefine_rtm.pickle',
        'Uni-Mol Docking V2':   'unimol_docking_v2_norefine_rtm.pickle',
        'DiffDock':           'diffdock_norefine_rtm.pickle',
        'SurfDock':            'surfdock_norefine_rtm.pickle',
        #'ECDock+GlideScore':            'current_best_mmf_ecdock_norefine_glide.pickle',
        'EC-Dock':              'ecdock_old_copy_norefine_rtm.pickle',
        #'ECDock+Glide_Refine+GlideScore': 'current_best_mmf_ecdock_refine_glide.pickle',
        'EC-Dock(Glide-Refine)': 'current_best_mmf_ecdock_refine_rtm.pickle'
    }
    EF_dict = {}
    for name, fname in files.items():
        try:
            with open(f'{EF_base}/{fname}', 'rb') as f:
                EF_dict[name] = dill.load(f)
        except Exception as e:
            print(f"WARNING: fail loading {fname}: {e}")

    # ─── 构造 DataFrame ─────────────────────────────────
    conc_map = {0.005: "0.5%", 0.01: "1.0%", 0.05: "5.0%"}
    records = []
    for model, d in EF_dict.items():
        for conc, label in conc_map.items():
            vals = np.asarray(d.get(conc, []))
            if vals.size:
                records += [{"Concentration": label, "Model": model, "EF": float(v)} for v in vals]
    df = pd.DataFrame(records)
    models = list(EF_dict.keys())
    if df.empty:
        raise RuntimeError("No EF data available!")

    # ─── 绘图配置 ───────────────────────────────────────
    sns.set_theme(
        style="whitegrid", context="notebook", font_scale=1.1,
        palette="tab10",
        rc={"axes.edgecolor": "black", "axes.linewidth": 1.2}
    )
    fig, ax = plt.subplots(figsize=(12, 6))

    # 箱线图 + 均值显示：白色菱形在每个箱体中央
    sns.boxplot(
        data=df, x="Concentration", y="EF", hue="Model",
        palette="tab10", ax=ax, showfliers=False, whis=[0, 95],
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "black",
            "markeredgecolor": "black",
            "markersize": 6
        },
        medianprops={"color": "firebrick", "linewidth": 2}
    )

    # 底部图例，每行最多4项,ncol=3
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles[:len(models)], labels=labels[:len(models)],
        title="Model", loc="upper center", bbox_to_anchor=(0.5, -0.15),
        ncol=3, frameon=True, fontsize=14, title_fontsize=16
    )

    # 去掉 x 轴标题、图表主标题
    ax.set_xlabel("")
    ax.set_title("")

    # 设置 y 轴标签
    ax.set_ylabel("Enrichment Factor", fontsize=20)

    # 边框与网格美化
    for sp in ax.spines.values():
        sp.set_edgecolor('black')
        sp.set_linewidth(1)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out_path = f"{EF_base}/EF.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"图已保存至：{out_path}")




def analyze_top_EF(sdf_file):
    #sdf_file = 'your_file.sdf'  # 更改为你的SDF文件路径
    mol = read_sdf(sdf_file)[0]

    # 计算原子数量
    num_atoms = count_atoms(mol)
    #print(f"分子原子数量: {num_atoms}")

    # 计算摩根指纹
    morgan_fp = compute_morgan_fingerprint(mol)
    morgan_fp = list(morgan_fp.ToList())
    #print(f"摩根指纹: {morgan_fp}")
    #exit()

    # 计算ESP静电势（占位方法）
    #esp_values = compute_ESP(mol)
        #print(f"ESP静电势（占位值）: {esp_values}")
    
    return num_atoms, morgan_fp




def bar(atom_counts, save_file, x_title = None, bins=10):
    # 绘制直方图
    plt.figure(figsize=(8, 6))  # 设置图表大小
    n, bins, patches = plt.hist(atom_counts, bins=bins, edgecolor='black', alpha=0.6, color='#0d0b6f', density=True)

    # 拟合数据为正态分布
    mu, std = norm.fit(atom_counts)  # 获取正态分布的均值和标准差

    # 绘制拟合曲线（正态分布）
    xmin, xmax = plt.xlim()  # 获取x轴的范围
    x = np.linspace(xmin, xmax, 100)  # 在x轴上创建100个点
    p = norm.pdf(x, mu, std)  # 计算这些点对应的正态分布概率密度
    plt.plot(x, p, 'k', linewidth=2)  # 绘制拟合曲线

    # 标注拟合曲线的均值和标准差
    title = f'$\mu$ = {mu:.2f},  $\sigma$ = {std:.2f}'
    plt.title(title, fontsize=15)

    # 设置标签
    plt.xlabel(x_title, fontsize=15)
    plt.ylabel('Frequency', fontsize=15)

    # 保存图像
    plt.savefig(save_file + '_bar_with_fit.png', dpi=400, bbox_inches='tight')
    plt.close()  # 关闭图表，防止重复绘制




def box(atom_counts, save_file):
    # 数据字典：仅包含原子数量
    data = {
        'Atom Count': atom_counts
    }

    # 创建DataFrame
    df = pd.DataFrame(data)

    # 绘制箱线图
    plt.figure(figsize=(8, 6))  # 可选，设置图表大小
    sns.boxplot(y='Atom Count', data=df)  # 只有原子数量，纵向箱线图
    plt.title('Distribution of atom counts')

    # 保存图像
    plt.savefig(save_file+'_box.png', dpi=400, bbox_inches='tight')
    plt.close()  # 关闭图表，防止重复绘制



def PCAs(morgan_fps, atom_counts, save_file):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import numpy as np

    # 假设你有摩根指纹数据（每个分子一个高维向量）和原子数量数据
    #morgan_fps = np.random.rand(6, 50)  # 假设有6个分子，摩根指纹50维
    #atom_counts = [10, 12, 14, 16, 18, 20]  # 分子的原子数量

    # 使用PCA将摩根指纹降到二维
    pca = PCA(n_components=2)
    morgan_fps_pca = pca.fit_transform(morgan_fps)

    # 绘制PCA降维后的散点图
    plt.scatter(morgan_fps_pca[:, 0], morgan_fps_pca[:, 1], c=atom_counts, cmap='viridis')
    #具体来说，c=atom_counts 表示你希望根据 atom_counts 中的值来为每个点分配一个颜色
    #这种方式有助于直观地看到数据中原子数量与PCA降维结果之间的关系
    plt.xlabel("PCA Component 1", fontsize=15)
    plt.ylabel("PCA Component 2", fontsize=15)
    plt.title("PCA for Morgan fingerprints", fontsize=15)
    # 添加 colorbar
    cbar = plt.colorbar(label="Atom number")

    # 设置 colorbar 标签字体大小
    cbar.ax.yaxis.label.set_fontsize(15)  # 设置字体大小为 15

    #plt.show()
    plt.savefig(save_file+'_PCA_scatter.png', dpi=400, bbox_inches='tight')
    plt.close()  # 关闭图表，防止重复绘制


def KNN(morgan_fps, save_file, n_clusters = 5):
    from sklearn.cluster import KMeans

    # 使用 PCA 降维，将摩根指纹从2048维降到2维
    pca = PCA(n_components=2)
    morgan_fps_pca = pca.fit_transform(morgan_fps)

    # 使用K-means聚类，将数据分为5个簇
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(morgan_fps_pca)

    # 绘制聚类后的散点图
    plt.scatter(morgan_fps_pca[:, 0], morgan_fps_pca[:, 1], c=kmeans.labels_, cmap='viridis')
    plt.xlabel("PCA Component 1", fontsize=15)
    plt.ylabel("PCA Component 2", fontsize=15)
    plt.title("K-means for Morgan fingerprints", fontsize=15)
    # 添加 colorbar
    cbar = plt.colorbar(label="Cluster")

    # 设置 colorbar 标签字体大小
    cbar.ax.yaxis.label.set_fontsize(15)  # 设置字体大小为 15
    #plt.show()
    plt.savefig(save_file+'_KNN_scatter.png', dpi=400, bbox_inches='tight')
    plt.close()  # 关闭图表，防止重复绘制



def count_hydrogen_bond_donors_and_acceptors(mol):
    # 初始化计数器
    donors = 0
    acceptors = 0
    
    # 获取分子的氢键供体（包含-NH, -OH, -NH2等基团）
    for atom in mol.GetAtoms():
        # 氢键供体的原子通常是含氢的氧（-OH）、氮（-NH2）
        if atom.GetSymbol() == 'N' and atom.GetDegree() == 1:  # NH2
            donors += 1
        elif atom.GetSymbol() == 'O' and atom.GetDegree() == 1:  # OH
            donors += 1
        elif atom.GetSymbol() == 'N' and atom.GetDegree() == 3:  # NH (NH group)
            donors += 1

    # 获取分子的氢键受体（孤对电子的电负性原子：氧、氮、氟）
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'O' or atom.GetSymbol() == 'N' or atom.GetSymbol() == 'F':
            if atom.GetNumImplicitHs() == 0:  # 孤对电子
                acceptors += 1

    return donors, acceptors


# 修改后的 bar 函数
def joint_bar(active_num_atoms_list, predict_num_atoms_list, save_file, x_title=None, bins=10):
    # 绘制两个分布的直方图
    plt.figure(figsize=(8, 6))  # 设置图表大小
    
    # 绘制 active_num_atoms_list 的直方图（轮廓线，透明度为0.6）
    #plt.hist(active_num_atoms_list, bins=bins, edgecolor='black', alpha=0.6, histtype='step', color='#0d0b6f',linewidth=2, label='Active')

    # 绘制 predict_num_atoms_list 的直方图（轮廓线，透明度为0.6）
    #plt.hist(predict_num_atoms_list, bins=bins, edgecolor='black', alpha=0.6, histtype='step', color="#0b6f65", linewidth=2, label='Predict')

    # 拟合 active_num_atoms_list 为正态分布
    #mu_active, std_active = norm.fit(active_num_atoms_list)
    #n, bins, patches = plt.hist(active_num_atoms_list, bins=bins, edgecolor='black', alpha=0.6, color='#0d0b6f', density=True)
    mu, std = norm.fit(active_num_atoms_list)  # 获取正态分布的均值和标准差
    # 绘制拟合曲线（正态分布）
    
    new_list = active_num_atoms_list + predict_num_atoms_list
    xmin, xmax = min(new_list), max(new_list)  # 获取两个列表的最小值和最大值，作为x轴范围
    
    print('xmin, xmax:', xmin, xmax)
    
    #xmin, xmax = plt.xlim()  # 获取x轴的范围
    x = np.linspace(xmin, xmax, 100)  # 在x轴上创建100个点
    p = norm.pdf(x, mu, std)  # 计算这些点对应的正态分布概率密度
    plt.plot(x, p, linewidth=2, label=f'Active-N: $\mu$={mu:.2f}, $\sigma$={std:.2f}')
    

    # 拟合 predict_num_atoms_list 为正态分布
    #mu_predict, std_predict = norm.fit(predict_num_atoms_list)
    #n, bins, patches = plt.hist(predict_num_atoms_list, bins=bins, edgecolor='black', alpha=0.6, color="#6f0b2b", density=True)
    mu, std = norm.fit(predict_num_atoms_list)  # 获取正态分布的均值和标准差
    # 绘制拟合曲线（正态分布）
    #xmin, xmax = plt.xlim()  # 获取x轴的范围
    #x = np.linspace(xmin, xmax, 100)  # 在x轴上创建100个点
    p = norm.pdf(x, mu, std)  # 计算这些点对应的正态分布概率密度
    plt.plot(x, p, linewidth=2, label=f'Top-N: $\mu$={mu:.2f}, $\sigma$={std:.2f}')
    
    

    # 设置标题和标签
    #plt.title('Atom Number Distribution with Fits', fontsize=15)
    plt.xlabel(x_title, fontsize=15)
    plt.ylabel('Frequency', fontsize=15)

    # 添加图例
    plt.legend()

    # 保存图像
    plt.savefig(save_file + '_bar_with_fit_joint.png', dpi=400, bbox_inches='tight')
    plt.close()  # 关闭图表，防止重复绘制
    

def EF_case_analy(active_dict, predict_TopN_dict, EF_base, data_base, flag_name = None):
    # ─── 数据加载 ───────────────────────────────────────

    active_diff_files  = []
    predict_diff_files = []
    
    for key in active_dict.keys():
        for key2 in active_dict[key]:
            file = os.path.join(data_base, key, key, key2, f'{key2}_ligand.sdf')
            active_diff_files.append(file)
            #print('file1:', file)
    for key in predict_TopN_dict.keys():
        for key2 in predict_TopN_dict[key]:
            file = os.path.join(data_base, key, key, key2, f'{key2}_ligand.sdf')
            predict_diff_files.append(file)
    
        
        #exit()
    
    print('active_diff_files:',  len(active_diff_files))   # pos num = 35, neg num = 56
    print('predict_diff_files:', len(predict_diff_files))
    
    
    '''读取molecule文件，计算分子特征'''
    active_mol_list  = []
    predict_mol_list = []
    
    for sdf_file in active_diff_files:
        mol = read_sdf(sdf_file)[0]
        active_mol_list.append(mol)
    
    for sdf_file in predict_diff_files:
        mol = read_sdf(sdf_file)[0]
        predict_mol_list.append(mol)
    


    '''计算分子量，以及摩根指纹的PCA'''
    
    active_num_atoms_list = []
    active_morgan_fp_list = []
    for sdf_file in active_diff_files:
        try:
            num_atoms, morgan_fp = analyze_top_EF(sdf_file)
        except Exception as e:
            print(f"Error processing {sdf_file}: {e}")
            continue
        active_num_atoms_list.append(num_atoms)
        active_morgan_fp_list.append(morgan_fp)


    predict_num_atoms_list = []
    predict_morgan_fp_list = []
    for sdf_file in predict_diff_files:
        try:
            num_atoms, morgan_fp = analyze_top_EF(sdf_file)
        except Exception as e:
            print(f"Error processing {sdf_file}: {e}")
            continue
        predict_num_atoms_list.append(num_atoms)
        predict_morgan_fp_list.append(morgan_fp)
    
    
    save_path = f'{EF_base}/3-all-sample-{flag_name}-active'
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, 'all_atom_number')
    #bar(active_num_atoms_list, save_file, x_title = 'The number of atom', bins = 10)
    #box(active_morgan_fp_list, save_file)
    
    print('new_pos_morgan_fp_list.shape:', np.array(active_num_atoms_list).shape) # (2103, 2048)
    print('new_pos_num_atoms_list.shape:', np.array(active_morgan_fp_list).shape) # (2103,)
    
    save_file = os.path.join(save_path, 'all')
    #PCAs(active_morgan_fp_list, active_num_atoms_list, save_file)
    #KNN(active_morgan_fp_list, save_file, n_clusters = 7)
        


    save_path = f'{EF_base}/3-all-sample-{flag_name}-predict'
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, 'all_atom_number')
    #bar(predict_num_atoms_list, save_file, x_title = 'The number of atom', bins = 10)
    #box(predict_morgan_fp_list, save_file)
    
    
    save_file = os.path.join(save_path, 'all')
    #PCAs(predict_morgan_fp_list, predict_num_atoms_list, save_file)
    #KNN(predict_morgan_fp_list, save_file, n_clusters = 7)
    
    
    # 直方图合并
    # 创建保存路径
    save_path = f'{EF_base}/3-all-sample-{flag_name}-active-predict'
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, 'all_atom_number')

    # 调用 bar 函数绘制图像
    joint_bar(active_num_atoms_list, predict_num_atoms_list, save_file, x_title='The number of atoms', bins=10)
    
    
    
    
    
    '''计算可旋转键数量的分布'''
    active_bond_list  = []
    predict_bond_list = []

    for mol1, mol2 in zip(active_mol_list, predict_mol_list):
        rotatable_bonds1 = Lipinski.NumRotatableBonds(mol1)
        rotatable_bonds2 = Lipinski.NumRotatableBonds(mol2)
        active_bond_list.append(rotatable_bonds1)
        predict_bond_list.append(rotatable_bonds2)
        
    print(f"可旋转键数量active：{active_bond_list}")
    print(f"可旋转键数量predict：{predict_bond_list}")

    save_path = f'{EF_base}/3-all-sample-{flag_name}-active'
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, 'all_rotatable_bond')
    #bar(active_bond_list, save_file, x_title = 'The number of rotatable bonds', bins = 10)
    
    
    save_path = f'{EF_base}/3-all-sample-{flag_name}-predict'
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, 'all_rotatable_bond')
    #bar(predict_bond_list, save_file, x_title = 'The number of rotatable bonds',  bins = 10)

    # 直方图合并
    # 创建保存路径
    save_path = f'{EF_base}/3-all-sample-{flag_name}-active-predict'
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, 'all_rotatable_bond')

    # 调用 bar 函数绘制图像
    joint_bar(active_bond_list, predict_bond_list, save_file, x_title='The number of rotatable bonds', bins=10)
    
    

    '''计算机氢键受体、供体的分布'''
    active_donors_list  = []
    predict_donors_list = []
    
    active_acceptors_list  = []
    predict_acceptors_list = []

    for mol1, mol2 in zip(active_mol_list, predict_mol_list):
        donors1, acceptors1 = count_hydrogen_bond_donors_and_acceptors(mol1)
        donors2, acceptors2 = count_hydrogen_bond_donors_and_acceptors(mol2)
        active_donors_list.append(donors1)
        active_acceptors_list.append(acceptors1)
        predict_donors_list.append(donors2)
        predict_acceptors_list.append(acceptors2)


    print(f"氢键供体数量：{active_donors_list}")
    print(f"氢键受体数量：{active_acceptors_list}")


    save_path = f'{EF_base}/3-all-sample-{flag_name}-active'
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, 'all_donors_H')
    #bar(active_donors_list, save_file, x_title = 'The number of hydrogen bond donors', bins = 10)
    
    save_file = os.path.join(save_path, 'all_acceptors_H')
    #bar(active_acceptors_list, save_file, x_title = 'The number of hydrogen bond acceptors', bins = 10)



    save_path = f'{EF_base}/3-all-sample-{flag_name}-predict'
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, 'all_donors_H')
    #bar(predict_donors_list, save_file, x_title = 'The number of hydrogen bond donors', bins = 10)
    
    save_file = os.path.join(save_path, 'all_acceptors_H')
    #bar(predict_acceptors_list, save_file, x_title = 'The number of hydrogen bond acceptors', bins = 10)




    # 直方图合并
    # 创建保存路径
    save_path = f'{EF_base}/3-all-sample-{flag_name}-active-predict'
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, 'all_donors_H')

    # 调用 bar 函数绘制图像
    joint_bar(active_donors_list, predict_donors_list, save_file, x_title='The number of hydrogen bond donors', bins=10)
    



    # 直方图合并
    # 创建保存路径
    save_path = f'{EF_base}/3-all-sample-{flag_name}-active-predict'
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, 'all_acceptors_H')

    # 调用 bar 函数绘制图像
    joint_bar(active_acceptors_list, predict_acceptors_list, save_file, x_title='The number of hydrogen bond acceptors', bins=10)
    
    
    
    '''计算logp分布，即rdkit的疏水性指数'''
    
    active_logP_list  = []
    predict_logP_list = []
    

    for mol1, mol2 in zip(active_mol_list, predict_mol_list):
        # 计算 LogP 值
        logP1 = Descriptors.MolLogP(mol1)
        logP2 = Descriptors.MolLogP(mol2)
        active_logP_list.append(logP1)
        predict_logP_list.append(logP2)
        
    

    # 直方图合并
    # 创建保存路径
    save_path = f'{EF_base}/3-all-sample-{flag_name}-active-predict'
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, 'all_LogP-Hydrophobicity')

    # 调用 bar 函数绘制图像
    joint_bar(active_logP_list, predict_logP_list, save_file, x_title='The RDKit LogP Hydrophobicity', bins=10)   
    

    exit()
    
    
    # ─── 构造 DataFrame ─────────────────────────────────
    conc_map = {0.005: "0.5%", 0.01: "1.0%", 0.05: "5.0%"}
    records = []
    for model, d in EF_dict.items():
        for conc, label in conc_map.items():
            vals = np.asarray(d.get(conc, []))
            if vals.size:
                records += [{"Concentration": label, "Model": model, "EF": float(v)} for v in vals]
    df = pd.DataFrame(records)
    models = list(EF_dict.keys())
    if df.empty:
        raise RuntimeError("No EF data available!")

    # ─── 绘图配置 ───────────────────────────────────────
    sns.set_theme(
        style="whitegrid", context="notebook", font_scale=1.1,
        palette="tab10",
        rc={"axes.edgecolor": "black", "axes.linewidth": 1.2}
    )
    fig, ax = plt.subplots(figsize=(12, 6))

    # 箱线图 + 均值显示：白色菱形在每个箱体中央
    sns.boxplot(
        data=df, x="Concentration", y="EF", hue="Model",
        palette="tab10", ax=ax, showfliers=False, whis=[0, 95],
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "black",
            "markeredgecolor": "black",
            "markersize": 6
        },
        medianprops={"color": "firebrick", "linewidth": 2}
    )

    # 底部图例，每行最多4项,ncol=3
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles[:len(models)], labels=labels[:len(models)],
        title="Model", loc="upper center", bbox_to_anchor=(0.5, -0.15),
        ncol=3, frameon=True, fontsize=14, title_fontsize=16
    )

    # 去掉 x 轴标题、图表主标题
    ax.set_xlabel("")
    ax.set_title("")

    # 设置 y 轴标签
    ax.set_ylabel("Enrichment Factor", fontsize=20)

    # 边框与网格美化
    for sp in ax.spines.values():
        sp.set_edgecolor('black')
        sp.set_linewidth(1)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out_path = f"{EF_base}/EF.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"图已保存至：{out_path}")


def EF_box_old():
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    
    EF_base = '/data/fan_zg/MDocking/EcDock_Evaluate/EF'
    refine_flag     = 'norefine' #refine/norefine
    score_flag      = 'rtm'  # rtm/glide
    glide_skip      = 'noskip'   #skip/noskip #glide对接打分或rfine会失败，其概率比较高，怎么处理？一种是去掉失败的，另一种视为得分最差，设置为100000，是不是没有使用ligprep处理配体？
    
    model = 'carsidock' # ecdock_100conf_nodistance_step10 / current_best_mmf_ecdock / ecdock_old_copy / ecdock / ecdock_25conf / 
                #mmf_ecdock / current_best_mmf_ecdock/ ecdock_old_copy / glide / carsidock / unimol_docking_v2 / karmadock / vina
    # rtm没问题，glide有问题，两种方法结果不一致
    #['Glide', 'KarmaDock Align', 'DiffDock', 'CarsiDock', 'Uni-Mol Docking V2', 'ECDock', 'ECDock_MMFF']
    EF_dict = {}
    
    
    with open(f'{EF_base}/unimol_docking_v2_norefine_rtm.pickle', 'rb') as f:
        dt = dill.load(f)
        EF_dict['Uni-Mol Docking V2+RTMScore'] = dt 
        

    with open(f'{EF_base}/karmadock_norefine_rtm.pickle', 'rb') as f:
        dt = dill.load(f)
        EF_dict['KarmaDock Align+RTMScore'] = dt 
        
        
    with open(f'{EF_base}/glide_norefine_glide.pickle', 'rb') as f:
        dt = dill.load(f)
        EF_dict['Glide+GlideScore'] = dt
        
        
    with open(f'{EF_base}/glide_norefine_rtm.pickle', 'rb') as f:
        dt = dill.load(f)
        EF_dict['Glide+RTMScore'] = dt 
        
        
    with open(f'{EF_base}/carsidock_norefine_rtm.pickle', 'rb') as f:
        dt = dill.load(f)
        EF_dict['CarsiDock+RTMScore'] = dt 


    with open(f'{EF_base}/current_best_mmf_ecdock_norefine_glide.pickle', 'rb') as f:
        dt = dill.load(f)
        EF_dict['ECDock+GlideScore'] = dt 
        
        
        
    with open(f'{EF_base}/ecdock_old_copy_norefine_rtm.pickle', 'rb') as f:
        dt = dill.load(f)
        EF_dict['ECDock+RTMScore'] = dt 
        
        
    with open(f'{EF_base}/current_best_mmf_ecdock_refine_glide.pickle', 'rb') as f:
        dt = dill.load(f)
        EF_dict['ECDock+Glide_Refine+GlideScore'] = dt 
        
        
    with open(f'{EF_base}/current_best_mmf_ecdock_refine_rtm.pickle', 'rb') as f:
        dt = dill.load(f)
        EF_dict['ECDock+Glide_Refine+TRMScore'] = dt 
        
    
    # 示例数据生成
    models = list(EF_dict.keys())
    data = []
    
    for conc, conc_str in zip([0.005, 0.01, 0.05], ["0.5%", "1.0%", "5.0%"]):
        for model in models:
            #print(EF_dict[model].keys())
            #print(EF_dict[model][conc])
            #exit()
            tmp = {"Concentration": conc_str, "Model": model,  "EF": np.array(EF_dict[model][conc])}
            data.append(tmp)
    
    
    df = pd.DataFrame(data)

    # 绘图
    #sns.set(style="whitegrid", font_scale=1.1) #drop out
    sns.set_theme(
    style="whitegrid",        # 样式，可选 darkgrid, white, ticks 等 :contentReference[oaicite:1]{index=1}
    context="notebook",       # 用于控制大小语境
    font_scale=1.1,           # 字体缩放
    palette="tab10",          # 默认调色板
    rc={"axes.edgecolor": "black", "axes.linewidth": 1.2}
    )
    
    fig, ax = plt.subplots(figsize=(12, 6))

    # 箱线图
    sns.boxplot(data=df, x="Concentration", y="EF", hue="Model",
                palette="tab10", ax=ax, showfliers=False)

    # 均值 “点” 标记（无连线）
    sns.pointplot(data=df, x="Concentration", y="EF", hue="Model",
                dodge=0.6, markers="D", scale=0.6,
                errwidth=0, ci=None, palette="tab10",
                ax=ax, legend=False, join=False)

    # 图例底部，每行最多4项
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[:len(models)], labels=labels[:len(models)],
            title="Model", loc="upper center",
            bbox_to_anchor=(0.5, -0.15), ncol=4,
            frameon=True, fontsize=10, title_fontsize=12)

    # 去掉 x 轴标题和主图标题
    ax.set_xlabel("")
    ax.set_title("")

    # 设置 y 轴标签
    ax.set_ylabel("Enrichment Factor", fontsize=14)

    # 美化边框
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

    # 网格
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig("/data/fan_zg/MDocking/EcDock_Evaluate/EF.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("图已保存至：/data/fan_zg/MDocking/EcDock_Evaluate/EF.png")






# 读取SDF文件
def read_sdf(file_path):
    supplier = Chem.SDMolSupplier(file_path)
    mols = [mol for mol in supplier if mol is not None]
    return mols

# 计算原子数量
def count_atoms(mol):
    return mol.GetNumAtoms()

# 计算摩根指纹
def compute_morgan_fingerprint(mol, radius=2, nBits=2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)

# 计算分子的电荷分布（此处为占位，实际需要量子化学计算）
def compute_ESP(mol):
    # RDKit没有直接提供计算ESP的方法
    # 一般需要使用量子化学软件计算ESP
    # 假设返回一个空数组作为占位
    return np.zeros(mol.GetNumAtoms())




if __name__ == '__main__':
    '''
    data_name_file = '/data/fan_zg/MDocking/data_name.txt'
    data_name_dict = {}
    with open(data_name_file) as f:
        for line in f:
            tg = line.strip().split('-')
            data_name_dict[tg[0]] = line.strip()
            
    print('len(data_name_dict):', len(data_name_dict))
            
    df = pd.read_excel('/data/fan_zg/MDocking/new_VSDS/vsds_gap_71.xlsx', engine='openpyxl')  # 默认读取第一个 Sheet
    # 获取第一列数据（假设第一列没有标题名）
    first_column = df.iloc[:, 0].tolist()  # iloc[:, 0] 表示所有行的第 0 列

    dt_name_list = first_column[:]
    print('len(dt_name_list):', len(dt_name_list))
    
    with open('/data/fan_zg/MDocking/new_VSDS/drop_vsds_gap_71.txt', 'w') as f:
        for i in data_name_dict:
            if i not in set(dt_name_list):
                f.write(data_name_dict[i]+'\n')
        
        
            
    
    exit()
    '''
    
    
    set_seed(2025)

    refine_flag     = 'norefine' #refine/norefine
    score_flag      = 'rtm'  # rtm/glide
    glide_skip      = 'noskip'   #skip/noskip #glide对接打分或rfine会失败，其概率比较高，怎么处理？一种是去掉失败的，另一种视为得分最差，设置为100000，是不是没有使用ligprep处理配体？
    
    model = 'carsidock' # ecdock_100conf_nodistance_step10 / current_best_mmf_ecdock / ecdock_old_copy / ecdock / ecdock_25conf / 
                #mmf_ecdock / current_best_mmf_ecdock/ ecdock_old_copy / glide / carsidock / unimol_docking_v2 / karmadock / vina
    # rtm没问题，glide有问题，两种方法结果不一致
    
    #高的原因是refine丢失的文件多导致的，尤其是哪些丢失率高的靶点
    data_name_file  = '/data/fan_zg/MDocking/new_VSDS/data_name.txt'
    #data_name_file  = '/data/fan_zg/MDocking/new_VSDS/glide_refine_success_rate_upper_0.2_data_name.txt'
    #data_name_file  = '/data/fan_zg/MDocking/new_VSDS/glide_refine_success_rate_upper_0.9_data_name.txt'
    #data_name_file  = '/data/fan_zg/MDocking/new_VSDS/vsds_gap_71.txt'
    #data_name_file  = '/data/fan_zg/MDocking/new_VSDS/drop_vsds_gap_71.txt'
    
    base_dir        = '/data/fan_zg/MDocking/new_VSDS'
    
    EF_base = '/data/fan_zg/MDocking/EcDock_Evaluate/EF'
        
    data_name_list = []
    with open(data_name_file) as f:
        for line in f:
            tg = line.strip()
            data_name_list.append(tg)
    

    
    EF_dict         = defaultdict(list)
    use_EF2 = True # 是否启动第二种打分方法
    count_fail = 0
    loss_file_rate_dict = {}
    success_file_rate_dict = {}
    #exist_data_name = data_name_list[70:100]
    loss_name_dict = {}
    success_name_dict = {}
    name_ef = {}
    #data_name_list = ['O75469-6TFI', 'P00338-5W8J'] 
    

    
    EF_base = '/data/fan_zg/MDocking/EcDock_Evaluate/EF/EF-case'
    EF_dict = {}
    with open(f'{EF_base}/{model}.pickle', 'rb') as f:
        EF_dict = dill.load(f)
            
    pos_diff_names = list(EF_dict.keys())[:20]
    neg_diff_names = list(EF_dict.keys())[-20:]  
    
    #pos_diff_names = ['P00747-5UGD', 'P07550-6PS2', 'P53350-2RKU']
    #neg_diff_names = ['Q9UBN7-8G44', 'Q9UHL4-3N0T', 'Q9Y233-2OUR']
    
    #data_name_list = pos_diff_names
    data_name_list = neg_diff_names
    
    #flag_name = f'pos-{model}'
    flag_name = f'neg-{model}'
    
    
    predict_TopN_dict = {}
    active_dict       = {}
    
    for data_name in tqdm(data_name_list[0:150]):
        special_name_list = []

        random.shuffle(special_name_list)  # 直接打乱原列表
        
        print('data_name:', data_name)
        success_file_list = []
        
        '''是否随机取一个，即测试只采样一个时的情况'''
        
        random_one = False
        
        '''返回富集率dict，然后排序，前N个， N等于实际的活性分子的数量'''
        
        data_path = os.path.join(base_dir , model, f'{data_name}')
        
        N = 0
        
        active_list = []
        
        for i in os.listdir(data_path):
            if 'active' in i:
                N += 1
                active_list.append(i)
        
        print('active N:', N)
        
        _, _, _, _, sort_rtm_data_dict = EF3(N, data_path, refine_flag, score_flag, glide_skip, special_name_list, random_one)
        
        #print('sort_rtm_data_dict:', sort_rtm_data_dict)
        
        '''只取前N个'''
        sort_rtm_data_TopN = []
        
        ct = 0
        for key in sort_rtm_data_dict:
            ct += 1
            if ct > N:
                break
            sort_rtm_data_TopN.append(key)
        
        predict_TopN_dict[data_name] = sort_rtm_data_TopN
        active_dict[data_name]       = active_list
    print('predict_TopN_dict:', predict_TopN_dict)
    print('active_dict:', active_dict)
        

    
    
    with open(f'/data/fan_zg/MDocking/EcDock_Evaluate/EF/EF-case/{model}_{flag_name}_topN_name.pickle', 'wb') as f:
        dill.dump(predict_TopN_dict, f)

    with open(f'/data/fan_zg/MDocking/EcDock_Evaluate/EF/EF-case/{model}_{flag_name}_activte_name.pickle', 'wb') as f:
        dill.dump(active_dict, f)
    
    #exit()


    
    #sorted_name_ef = dict(sorted(name_ef.items(), key=lambda item: item[1], reverse=True))
    
    
    '''分析ecdock'''
    
    EF_base   = '/data/fan_zg/MDocking/EcDock_Evaluate/EF/EF-case'
    data_base = '/data/fan_zg/MDocking/VSDS_DTEBV-D/data'
    EF_case_analy(active_dict, predict_TopN_dict, EF_base, data_base, flag_name)
        
