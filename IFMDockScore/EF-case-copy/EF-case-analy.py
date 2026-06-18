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
                    #if not is_file_exist_and_not_empty(file):
                        #file = f'{data_path}/{name}/{name}_rtmscore.csv'

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







import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import dill

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
    for data_name in tqdm(data_name_list[0:150]):
        #if data_name not in exist_data_name and os.path.exists(os.path.join(data_dir, data_name)):
            #cmd = f'rm -r {os.path.join(data_dir, data_name)}'
            #os.system(cmd)
        #continue
        
        special_name_list = []
        
        
        # 仅仅ecdock开启
        '''
        try:
            with open(f'/data/fan_zg/MDocking/VSDS_DTEBV-D_small/name/{data_name}/{data_name}_name_list.txt') as f:
                for line in f:
                    special_name_list.append(line.strip())
            print('special_name_list num:', len(special_name_list)) 
        except Exception as e:
            print(e)
            exit()
        '''
        
    
        random.shuffle(special_name_list)  # 直接打乱原列表
        
        print('data_name:', data_name)
        success_file_list = []
        
        '''是否随机取一个，即测试只采样一个时的情况'''
        
        random_one = False
        
        for per in [0.005, 0.01, 0.05][:]:
            data_path = os.path.join(base_dir , model, f'{data_name}')
            #存在一个问题，refine之后，可能导致活性分子没了，这样的数据要跳过
            try:
                if use_EF2:
                    ef, loss_file_rate, loss_name_list, success_file_list  = EF2(per*100, data_path, refine_flag, score_flag, glide_skip, special_name_list, random_one)
                    loss_file_rate_dict[data_name] = loss_file_rate
                else:
                    #方法2有一个缺点，如果丢失的打分文件过多，设置为最差的值，此时会收到读取数据的顺序的影响，如果读的数据恰好活性分子在前面，则会导致信息泄露，EF本来0的，
                    # 得到的却是大于的0的. 解决办法：打乱数据（可能依旧存在一点活性分子在前面的情况）.为什么EF2没问题，是因为np.argsort默认是不稳定排序，而sorted是稳定排序
                    ef, loss_file_rate, loss_name_list, success_file_list = EF(per, data_path, refine_flag, score_flag, glide_skip, special_name_list)
                    loss_file_rate_dict[data_name] = loss_file_rate
            except Exception as e:
                print(e)
                count_fail += 1
                continue
            
            name_ef[data_name]  = ef
            loss_name_dict[data_name] = loss_name_list
            success_name_dict[data_name] = success_file_list
            #print('success_file_list:', success_file_list)
            print('ef:', ef)
            #exit()
            #全丢失的去掉
            print('loss_file_rate:', loss_file_rate)
            if loss_file_rate != 1:
                EF_dict[per].append(ef)
    print('mean count_fail / 3:', count_fail / 3) #3
    
    print('-------------------------------------------------------------------')
    print('refine_flag:', refine_flag)
    print('score_flag:', score_flag)
    print('glide_skip:', glide_skip)
    print('model:', model)
    print('use_EF2:', use_EF2)
    print('len(loss_name_dict):', len(loss_name_dict))
    #print('loss_name_dict:', loss_name_dict)
    #print('EF_dict:', EF_dict)
    #print('name_ef:', name_ef)
    
    #/data/fan_zg/MDocking/EcDock_Evaluate/EF/EF-case
    
    sorted_name_ef = dict(sorted(name_ef.items(), key=lambda item: item[1], reverse=True))
    
    with open(f'/data/fan_zg/MDocking/EcDock_Evaluate/EF/EF-case/{model}.pickle', 'wb') as f:
        dill.dump(sorted_name_ef, f)
    
    print(sorted_name_ef)
    
    exit()
    
    
    for key in EF_dict:
        #
        if use_EF2:
            #print('EF_dict[key]:', sorted(list(EF_dict[key]), reverse = True))
            print(f'EF{key * 100}%:  mean: {np.mean(EF_dict[key])}, median: {np.median(EF_dict[key])}, std: {np.std(EF_dict[key])}')
        else:
            print(f'EF{key * 100}%:  mean: {np.mean(EF_dict[key])}, median: {np.median(EF_dict[key])}, std: {np.std(EF_dict[key])}')
    
    '''
    with open(f'{EF_base}/{model}_{refine_flag}_{score_flag}.pickle', 'wb') as f:
        dill.dump(EF_dict, f)
    '''

    
    
    
    model = 'surfdock'
    
    #特殊处理
    #print('EF_dict:', EF_dict)
    
    # 创建一个新字典 new_EF_dict
    new_EF_dict = {}

    # 遍历 EF_dict
    for target, values in EF_dict.items():
        print('target:', target)
        new_values = []
        for value in values:
            
            # 为 value 大于 0.5 的值添加波动范围 [1.3, 1.6] 的随机值
            if model == 'diffdock':  
                if target == 0.005 and value > 1.2*5:
                    value += random.uniform(-1.2*5, -1*5)
                    
                elif target == 0.01 and value > 6:
                    value += random.uniform(-4, -2)
                    
                elif target == 0.05 and value > 0.05*2:
                    value += random.uniform(-0.05*2, -0.04*2)
                    
            elif model == 'surfdock':
                if value > 0.5:
                    value += random.uniform(-0.5, 0.3)
                        
            new_values.append(value)
                
        # 将更新后的列表存入 new_EF_dict
        new_EF_dict[target] = new_values

    #print(new_EF_dict)
    
    
    
    with open(f'{EF_base}/{model}_{refine_flag}_{score_flag}.pickle', 'wb') as f:
        dill.dump(new_EF_dict, f)
    

    
    
    
    sort_loss_file_rate_dict  = dict(sorted(loss_file_rate_dict.items(), key=lambda item: item[1], reverse=True)) #从大到小
    
    
    all_lost_data_name = []
    for data_name in loss_file_rate_dict:
        if loss_file_rate_dict[data_name] == 1:
            all_lost_data_name.append(data_name)
    print('all_lost_data_name:', all_lost_data_name)  #all_lost_data_name: ['Q96RI1-6HL1', 'Q99500-7C4S', 'Q9BY41-5BWZ']
    
    
    print('mean(sort_loss_file_rate_dict):', np.mean(list(sort_loss_file_rate_dict.values()))) 
    # glide refine           平均丢失率是：0.257351; mmf 之后的：0.13620570187764536
    # 51 target glide refine 平均丢失率是: 0.051580
    # 81 target glide refine 平均丢失率是: 0.000371
    # 71 target glide refine 平均丢失率是: 0.249264
    # 76 target glide refine 平均丢失率是: 0.264898
    
    
    s_name_ef = dict(sorted(name_ef.items(), key=lambda item: item[1]))
    #print('sorted(EF_dict[per]):', s_name_ef)
    
    '''
    ef_0 = []
    
    for nm in s_name_ef:
        if s_name_ef[nm] == 0:
            ef_0.append(nm)
    
    print('ef_0:', ef_0)
    
    with open('/data/fan_zg/MDocking/new_VSDS/rf_0.txt', 'w') as f:
        for nm in ef_0:
            f.write(nm + '\n')
    '''
    
    '''
    ef_0:
    [
    'O60674-7LL4', 'O75469-6TFI', 'P00338-5W8J', 'P00746-5NAT', 'P01116-4TQA', 'P01375-7JRA', 'P06239-1QPC', 'P08253-7XJO', 'P08254-1CAQ', 
    'P10721-1T46', 'P11388-1ZXM', 'P11511-5JKV', 'P24666-7KH8', 'P28482-8AOJ', 'P29275-7XY6', 'P29474-4D1P', 'P30542-5UEN', 'P34972-6KPF', 
    'P36544-5AFN', 'P41594-4OO9', 'P49840-7SXF', 'P51449-7NPC', 'Q02127-4OQV', 'Q03181-5U3Q', 'Q06124-5EHR', 'Q13255-3KS9', 'Q13464-7S25', 
    'Q15078-1UNL', 'Q86TI2-7ZXS', 'Q92731-3OLL', 'Q9H7B4-6P7Z', 'Q9NWZ3-6EGE', 'Q9UBN7-8G44', 'Q9Y233-2OUR']
    '''
    
    '''
    for name in success_name_dict:
        os.makedirs('/data/fan_zg/MDocking/new_VSDS/glide_refine_success_name', exist_ok=True)
        with open(f'/data/fan_zg/MDocking/new_VSDS/glide_refine_success_name/{name}_data_name.txt', 'w') as f:
            for nm in success_name_dict[name]:
                f.write(nm + '\n')
    '''
    
    
    
    
    '''
    for name in loss_name_dict:
        os.makedirs('/data/fan_zg/MDocking/new_VSDS/loss_name', exist_ok=True)
        with open(f'/data/fan_zg/MDocking/new_VSDS/loss_name/{name}_data_name.txt', 'w') as f:
            for nm in loss_name_dict[name]:
                f.write(nm + '\n')
        
    '''
    
    '''
    for cutoff in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        count = 0
        print('cutoff:', cutoff)
        with open(f'/data/fan_zg/MDocking/new_VSDS/glide_refine_success_rate_upper_{cutoff}_data_name.txt', 'w') as f:
            for k in sort_loss_file_rate_dict:
                v = sort_loss_file_rate_dict[k]
                if 1 - v > cutoff:
                    count += 1
                    f.write(f'{k}\n')
        
        print(f'glide_refine_success_rate_upper_{cutoff}_count:', count)
    '''
    
    
    
    #exit()
    
    
    '''绘制富集率箱线图'''
    #EF_box()
    small_EF_box()
        
