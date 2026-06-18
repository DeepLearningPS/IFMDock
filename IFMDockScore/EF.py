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
                            if os.path.exists(file):
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
                            if os.path.exists(file):
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
    
    
    
    
def EF(per, data_path, refine_flag, score_flag, glide_skip):
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
                    with open(f'{data_path}/{name}/step4/{name}_glide_score_refine.txt') as f:
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
                        #print('-----------')
                        #print('name:', name)
                        #print('score_list:', score_list)
                        #exit()
                        
                        total_molecules.append(name)
                        if 'active' in name:
                            total_active.append(name)
            
                            
            elif score_flag == 'rtm':
                if refine_flag == 'refine':
                    with open(f'{data_path}/{name}/step4/refine_{name}_rtmscore.csv') as f:
                        score_list = []
                        for line in f:
                            score_list.append(float(line.strip().split('\t')[1]))
                        rtm_data_dict[name] = max(score_list) #读第一行第二列即可
                        
                        total_molecules.append(name)
                        if 'active' in name:
                            total_active.append(name)
                else:
                    with open(f'{data_path}/{name}/step4/{name}_rtmscore.csv') as f:
                        score_list = []
                        for line in f:
                            score_list.append(float(line.strip().split('\t')[1]))
                        rtm_data_dict[name] = max(score_list) #读第一行第二列即可
                        
                        total_molecules.append(name)
                        if 'active' in name:
                            total_active.append(name)
                            
        except Exception as e:
            print(e)
            loss_file_list.append(name)
            continue
    
    print('len(loss_file_list):', len(loss_file_list))
    #print('loss_file_list num / all num:', len(loss_file_list), len(data_name_list)) #loss_file_list num / all num: 642 1884
    #print('total_active num / total_molecules num:', len(total_active), len(total_molecules))
    
    if score_flag == 'rtm':
        sort_rtm_data_dict  = dict(sorted(rtm_data_dict.items(), key=lambda item: item[1], reverse=True)) #从大到小
    elif score_flag == 'glide':
        sort_rtm_data_dict  = dict(sorted(rtm_data_dict.items(), key=lambda item: item[1], reverse=False)) #从大到小
    sort_data_name_list = list(sort_rtm_data_dict.keys())
    #print('len(sort_data_name_list):', len(sort_data_name_list))

    # 取TOP n
    top_n = math.ceil((len(sort_data_name_list) * per))
    if top_n == 0:
        #print('top_n:', top_n)
        top_n = 1
    #print('top_n:', top_n) #9/18/94
    
    selected_molecules = sort_data_name_list[:top_n]

    for nm in selected_molecules:
        if 'active' in nm:
            selected_active.append(nm)
            
    #print('selected_active num / selected_molecules num:', len(selected_active), len(selected_molecules))
    
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
        print('hit_rate_total:', hit_rate_total)
        return 0  # 避免除以零
    enrichment_factor = hit_rate_selected / hit_rate_total
    
        
    return enrichment_factor
        

    

def EF2(per, data_path, refine_flag, score_flag, glide_skip):
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
                    with open(f'{data_path}/{name}/step4/{name}_glide_score_refine.txt') as f:
                        score_list = []
                        for line in f:
                            score_list.append(float(line.strip().split('\t')[1]))
                        rtm_data_dict[name] = min(score_list) #读第一行第二列即可
                            
                        total_molecules.append(name)
                        if 'active' in name:
                            total_active.append(name)
                            label_data_dict[name] = 1
                        else:
                            label_data_dict[name] = 0
                            
                else:
                    with open(f'{data_path}/{name}/step4/{name}_glide_score.txt') as f:
                        score_list = []
                        for line in f:
                            score_list.append(float(line.strip().split('\t')[1]))
                        rtm_data_dict[name] = min(score_list) #读第一行第二列即可
                            
                        total_molecules.append(name)
                        if 'active' in name:
                            total_active.append(name)
                            label_data_dict[name] = 1
                        else:
                            label_data_dict[name] = 0
                            
                            
            elif score_flag == 'rtm':    
                if refine_flag == 'refine':
                    with open(f'{data_path}/{name}/step4/refine_{name}_rtmscore.csv') as f:
                        score_list = []
                        for line in f:
                            score_list.append(float(line.strip().split('\t')[1]))
                        rtm_data_dict[name] = max(score_list) #读第一行第二列即可
                            
                        total_molecules.append(name)
                        if 'active' in name:
                            total_active.append(name)
                            label_data_dict[name] = 1
                        else:
                            label_data_dict[name] = 0
                            
                else:
                    with open(f'{data_path}/{name}/step4/{name}_rtmscore.csv') as f:
                        score_list = []
                        for line in f:
                            score_list.append(float(line.strip().split('\t')[1]))
                        rtm_data_dict[name] = max(score_list) #读第一行第二列即可
                            
                        total_molecules.append(name)
                        if 'active' in name:
                            total_active.append(name)
                            label_data_dict[name] = 1
                        else:
                            label_data_dict[name] = 0
                        
        except Exception as e:
            print(e)
            loss_file_list.append(name)
            continue
    

    
    print('len(loss_file_list):', len(loss_file_list))
    scores = np.array(list(rtm_data_dict.values()))
    labels = np.array(list(label_data_dict.values()))
    if score_flag == 'rtm':
        #print('score_flag:', score_flag)
        ids = np.argsort(scores)[::-1] #从大到小排序（rtm是正值）
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

    ef = odd_enrichment_factor(labels, scores, percentage = cutoff, kind = 'fold') #scores没用， labels是经过得分排序后的
    return ef                         
    
    

if __name__ == '__main__':
    set_seed(2025)
    data_dir       = '/data/fan_zg/MDocking/VSDS_ECDock_Gen'
    data_name_file = '/data/fan_zg/MDocking/data_name.txt'
    
    refine_flag     = 'norefine' #refine/norefine
    score_flag      = 'rtm'  # rtm/glide
    glide_skip      = 'noskip'   #skip/noskip #glide对接打分或rfine会失败，其概率比较高，怎么处理？一种是去掉失败的，另一种视为得分最差，设置为100000，是不是没有使用ligprep处理配体？
    
    # rtm没问题，glide有问题，两种方法结果不一致
    
    data_name_list = []
    with open(data_name_file) as f:
        for line in f:
            tg = line.strip()
            data_name_list.append(tg)
            
            
    EF_dict         = defaultdict(list)
    use_EF2 = True
    count_fail = 0
    for data_name in tqdm(data_name_list[:]):
        print('data_name:', data_name)
        for per in [0.005, 0.01, 0.05]:
            data_path = os.path.join(data_dir, f'{data_name}_ecdock_cm_equiformer_step5_interaction_limit4.5ai_307_step5_unimol_distance')
            
            try:
                if use_EF2:
                    ef = EF2(per*100, data_path, refine_flag, score_flag, glide_skip)
                else:
                    ef = EF(per, data_path, refine_flag, score_flag, glide_skip)
            except Exception as e:
                print(e)
                count_fail += 1
                continue
            
            #print('ef:', ef)
            EF_dict[per].append(ef)
    print('mean count_fail / 3:', count_fail / 3) #3
    
    print('-------------------------------------------------------------------')
    for key in EF_dict:
        #
        if use_EF2:
            print(f'EF{key * 100}%:  mean: {np.mean(EF_dict[key])}, median: {np.median(EF_dict[key])}')
        else:
            print(f'EF{key * 100}%:  mean: {np.mean(EF_dict[key])}, median: {np.median(EF_dict[key])}')
