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


import glob
import shutil
import os


def dt_move(src_dir, dst_dir):
    # 确保目标目录存在
    os.makedirs(dst_dir, exist_ok=True)

    # 遍历所有 .txt 和 .csv 文件
    for ext in ["*.txt", "*.csv"]:
        for file_path in glob.glob(os.path.join(src_dir, ext)):
            if os.path.isfile(file_path):  # 确保是文件而不是目录
                shutil.copy(file_path, dst_dir)
                #print(f"Copied: {file_path} -> {dst_dir}")


def data_move(data_path, t_dir):
    data_name_list = []
    for name in list(os.listdir(data_path))[:]:
        path = os.path.join(data_path, name)
        if os.path.exists(path) and os.path.isdir(path) and os.listdir(path) and 'model' not in name: #目录存在且不空
            data_name_list.append(name)


    for name in data_name_list: 
        if model == 'ecdock':
            src_dir = f'{data_path}/{name}/step4'
        elif model == 'glide':
            src_dir = f'{data_path}/{name}'
            #/mnt_191/fanzhiguang/47/VSDS/VSDS_Glide/glide/O00329-6PYR/model/ecdock_step5/gen_docking/active0/active0_glide_score.txt
        elif model in ['carsidock', 'kamadock', 'unimol']:
            src_dir = f'{data_path}/{name}'

        dst_dir = f'{t_dir}/{name}'
        dt_move(src_dir, dst_dir)





def data_joint(data_path, t_dir):
    data_name_list = []
    for name in list(os.listdir(t_dir))[:]:
        path = os.path.join(t_dir, name)
        if os.path.exists(path) and os.path.isdir(path) and os.listdir(path) and 'model' not in name: #目录存在且不空
            data_name_list.append(name)


    for name in data_name_list: 
        if model == 'ecdock':
            dst_dir = f'{data_path}/{name}/step4'
        elif model == 'glide':
            dst_dir = f'{data_path}/{name}'
            #/mnt_191/fanzhiguang/47/VSDS/VSDS_Glide/glide/O00329-6PYR/model/ecdock_step5/gen_docking/active0/active0_glide_score.txt
        elif model in ['carsidock', 'kamadock', 'unimol']:
            dst_dir = f'{data_path}/{name}'

        
        src_dir = f'{t_dir}/{name}'
        os.makedirs(src_dir, exist_ok=True)
        dt_move(src_dir, dst_dir)


if __name__ == '__main__':
    '数据移动，只要分数，用于数据合并'
    refine_flag     = 'norefine' #refine/norefine
    score_flag      = 'rtm'  # rtm/glide
    glide_skip      = 'noskip'   #skip/noskip #glide对接打分或rfine会失败，其概率比较高，怎么处理？一种是去掉失败的，另一种视为得分最差，设置为100000，是不是没有使用ligprep处理配体？
    st_id  = 50
    en_id  = 100
    model  = 'unimol' # ecdock / glide / carsidock / unimol / kamadock / vina
    # rtm没问题，glide有问题，两种方法结果不一致
    
    #高的原因是refine丢失的文件多导致的，尤其是哪些丢失率高的靶点
    
    if model == 'ecdock':
        data_dir       = '/data/fan_zg/MDocking/VSDS_ECDock_Gen' #ecdock的rtm得分在本地服务器上
        data_name_file = '/data/fan_zg/MDocking/data_name.txt'
    elif model == 'glide':
        data_dir       = '/mnt_191/fanzhiguang/47/VSDS/VSDS_Glide/glide' #在本地服务器上
        data_name_file = '/mnt_191/fanzhiguang/47/VSDS/data_name.txt'
    elif model == 'carsidock':
        data_dir       = '/data/fan_zg/MDocking/Docking_baseline/CarsiDock/outputs/new_vsds' #166[70:100], 68[100:150]; 176[0:70]
        data_name_file = '/data/fan_zg/MDocking/data_name.txt'
    elif model == 'unimol':
        data_dir       = '/data/fan_zg/MDocking/Docking_baseline/unimol_docking_v2/interface/valid_vsds' #166[0:50; 100:150], 176[50:100]
        data_name_file = '/data/fan_zg/MDocking/data_name.txt'
    elif model == 'kamadock':
        data_dir       = '/data/fan_zg/MDocking/Docking_baseline/KarmaDock/vsds_gen' #本地服务器[0:150]
        data_name_file = '/data/fan_zg/MDocking/VSDS_DTEBV-D/data_name.txt'
    elif model == 'vina':
        data_dir       = None
        data_name_file = None
        
        
        
    data_name_list = []
    with open(data_name_file) as f:
        for line in f:
            tg = line.strip()
            data_name_list.append(tg)
    
    for data_name in tqdm(data_name_list[st_id: en_id]):
        #if data_name not in exist_data_name and os.path.exists(os.path.join(data_dir, data_name)):
            #cmd = f'rm -r {os.path.join(data_dir, data_name)}'
            #os.system(cmd)
        #continue
        print('data_name:', data_name)
        if model == 'ecdock':
            data_path = os.path.join(data_dir, f'{data_name}_ecdock_cm_equiformer_step5_interaction_limit4.5ai_307_step5_unimol_distance')
            t_dir     = os.path.join(f'{data_dir}_score', f'{data_name}_ecdock_cm_equiformer_step5_interaction_limit4.5ai_307_step5_unimol_distance')
        elif model == 'glide':
            data_path = os.path.join(data_dir, f'{data_name}', 'model/ecdock_step5/gen_docking')
            t_dir = os.path.join(f'{data_dir}_score', f'{data_name}', 'model/ecdock_step5/gen_docking')
        elif model == 'carsidock':
            data_path = os.path.join(data_dir, f'{data_name}')
            t_dir = os.path.join(f'{data_dir}_score_176', f'{data_name}')
        elif model == 'unimol':
            data_path = os.path.join(data_dir, f'{data_name}')
            t_dir = os.path.join(f'{data_dir}_score', f'{data_name}')
        elif model == 'kamadock':
            data_path = os.path.join(data_dir, f'{data_name}')
            t_dir = os.path.join(f'{data_dir}_score', f'{data_name}')
        elif model == 'vina':
            data_path = os.path.join(data_dir, f'{data_name}')
            t_dir = os.path.join(f'{data_dir}_score', f'{data_name}')
        
                    

        data_move(data_path, t_dir)
        #data_joint(data_path, t_dir)