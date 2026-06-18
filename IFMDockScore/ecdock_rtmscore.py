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
#os.environ["BABEL_LIBDIR"] = '/mnt_191/fanzhiguang/47/47_anaconda3/new_torch2.1.0/lib/openbabel/3.1.0' # "/data/fan_zg/anaconda3/envs/torch2.1.0/lib/openbabel/3.1.0"


import copy
from rdkit import Chem
from rdkit.Chem import AllChem
import copy
import subprocess
import time
#import multiprocessin
from tqdm import tqdm 

from rtmscore import one_data, is_file_exist_and_not_empty


    
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default = '/data/fan_zg/MDocking/VSDS_ECDock_Gen', help='Input data dir')
    
    p.add_argument('--data_name', default = '/data/fan_zg/MDocking/data_name.txt', help='Input data name')

    p.add_argument('--refine_flag', default = 'norefine', help='是否是经过Glide refine后的数据')

    p.add_argument('-p','--prot', default = None,
                                    help='Input protein file (.pdb)')


    p.add_argument('-l','--lig', default = None,
                                    help='Input ligand file (.sdf/.mol2)')

    p.add_argument('-m','--model', default="../trained_models/rtmscore_model1.pth",
                                    help='trained model path (default: "../trained_models/rtmscore_model1.pth")')

    p.add_argument('-o','--outprefix', default="out",
                                    help='the prefix of output file (default: "out")，输出目录')
    p.add_argument('-gen_pocket','--gen_pocket', action="store_true", default=False,
                                    help='whether to generate the pocket，如果提供的是整个蛋白，则需要生成口袋，同一个靶点的口袋是一样的。因此第一次生成口袋时保存，之后直接加载即可')

    p.add_argument('-c','--cutoff', default=10.0, type=float,
                                    help='the cutoff the define the pocket and interactions within the pocket (default: 10.0)，口袋长度')

    p.add_argument('-rl','--reflig', default = None,
                help='the reference ligand to determine the pocket(.sdf/.mol2)， 参考晶体结构，用于寻找口袋质心。如果配体已经平移到了质心，则用配体即可')

    p.add_argument('-pl', '--parallel', default=False, action="store_true",
                        help='whether to obtain the graphs in parallel (When the dataset is too large,\
                        it may be out of memory when conducting the parallel mode).')

    p.add_argument('-ac', '--atom_contribution', default=False, action="store_true",
                        help='whether to obtain the atom contrubution of the score.')

    p.add_argument('-rc', '--res_contribution', default=False, action="store_true",
                        help='whether to obtain the residue contrubution of the score.')
    
    
    p.add_argument('--st_i', default=0, type=int)
    p.add_argument('--en_i', default=-1, type=int)

    inargs = p.parse_args()
    #if inargs.gen_pocket:
        #if inargs.reflig is None:
            #raise ValueError("if pocket is generated, the reference ligand should be provided.")

    #计算贡献度时，只能选一个，是原子还是残基。建议用原子
    if inargs.atom_contribution and inargs.res_contribution:
        raise ValueError("only one of the atom_contribution and res_contribution can be supported")
    
    step = 5
    
    data_name_list = []
    with open(inargs.data_name) as f:
        for line in f:
            data_name_list.append(line.strip())

    
    
    for dt_name in data_name_list[inargs.st_i: inargs.en_i]:
        #dt_dir = os.path.join(inargs.data_dir, f'{dt_name}_ecdock_cm_equiformer_step5_interaction_limit4.5ai_307_step5_unimol_distance')
        dt_dir = os.path.join(inargs.data_dir, f'{dt_name}_ecdock_cm_equiformer_step5_interaction_limit4.5ai_3Dmultidistance')
        
        for dt in tqdm(list(sorted(list(os.listdir(dt_dir))))[:]):
            path = os.path.join(dt_dir, dt)
            if os.path.exists(path) and os.path.isdir(path) and os.listdir(path) and 'model' not in dt: #目录存在且不空
                ##需要改路径
                p_file = os.path.join('/data/fan_zg/MDocking/VSDS_DTEBV-D/data', f'{dt_name}', f'{dt_name}', dt, f'{dt}_protein.pdb') #蛋白名字改一下
                
                if inargs.refine_flag == 'refine':
                    l_file = os.path.join(path, f'step{step-1}', f'glide_refine_gen_{dt}_addh.sdf')  #glide_refine_gen_5SB2_addh.sdf
                    if not os.path.exists(l_file):
                        print('refine is 不存在')
                        l_file = os.path.join(path, f'step{step-1}', f'{dt}_addh_gen_ligand.sdf')
                        if not os.path.exists(l_file):
                            l_file = os.path.join(path, f'step{step-1}', f'gen_ligand_{dt}.sdf') 
                else:
                    l_file = os.path.join(path, f'step{step-1}', f'{dt}_addh_gen_ligand.sdf') 
                    if not os.path.exists(l_file):
                        l_file = os.path.join(path, f'step{step-1}', f'gen_ligand_{dt}.sdf') 

                        
                    
                #保存按rmt得分排序的分子，可以用于posebuses数据集
                origin_gen_l_file = l_file
                #print('origin_gen_l_file:', origin_gen_l_file)
                if inargs.refine_flag == 'refine':
                    rtm_score_l_file   = os.path.join(path, f'step{step-1}', f'refine_rtm_sort_gen_ligand_{dt}.sdf')
                else:
                    rtm_score_l_file   = os.path.join(path, f'step{step-1}', f'rtm_sort_gen_ligand_{dt}.sdf')
                
                ref_lig_file = l_file
                
                if inargs.refine_flag == 'refine':
                    if inargs.atom_contribution:
                        out_file = os.path.join(path, f'step{step-1}', f'refine_{dt}_rtmscore_ac')
                    elif inargs.res_contribution:
                        out_file = os.path.join(path, f'step{step-1}', f'refine_{dt}_rtmscore_rc')
                    else:
                        out_file = os.path.join(path, f'step{step-1}', f'refine_{dt}_rtmscore')
                else:
                    if inargs.atom_contribution:
                        out_file = os.path.join(path, f'step{step-1}', f'{dt}_rtmscore_ac')
                    elif inargs.res_contribution:
                        out_file = os.path.join(path, f'step{step-1}', f'{dt}_rtmscore_rc')
                    else:
                        out_file = os.path.join(path, f'step{step-1}', f'{dt}_rtmscore')
                    
                
                
                try:
                    print('dt_name:', dt_name)
                    #print(f"文件 {out_file} 存在且不为空")
                    if is_file_exist_and_not_empty(out_file + '.csv'):
                        print(f"文件 {out_file}.csv 存在且不为空")  
                        continue
                    else:
                        one_data(p_file, l_file, ref_lig_file, out_file, origin_gen_l_file, rtm_score_l_file, inargs)
                except Exception as e:
                    print('error:', e)


        
            




