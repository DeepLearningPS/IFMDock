import lmdb
import pickle
import pandas as pd
from predictor import UnimolPredictor
from sklearn.metrics import roc_auc_score
import argparse
import os
from tqdm import tqdm
import time
import torch
import numpy as np
import random

from unicore_cli.train import cli_main

# 设置MKL线程层为INTEL，确保使用Intel MKL的并行计算，而不是GNU OpenMP
#os.environ['MKL_THREADING_LAYER'] = 'INTEL'

# 可以设置OMP线程数量为1来避免与libgomp冲突
#os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'GNU'



def set_seed(seed):
    torch.manual_seed(seed)  # 设置 PyTorch 的随机数种子
    torch.cuda.manual_seed_all(seed)  # 设置所有 GPU 的随机数种子
    np.random.seed(seed)  # 设置 NumPy 的随机数种子
    random.seed(seed)  # 设置 Python 自带的随机数种子
    torch.backends.cudnn.deterministic = True  # 设置 CuDNN 算法为确定性算法
    torch.backends.cudnn.benchmark = True

def main(args):
    set_seed(2024)
    if args.mode == 'single':
        start_time = time.time()
        clf = UnimolPredictor.build_predictors(args.model_dir, 
                                               args.mode, 
                                               args.nthreads, 
                                               args.conf_size, 
                                               use_current_ligand_conf=args.use_current_ligand_conf,
                                               steric_clash_fix=args.steric_clash_fix)
        (input_protein, 
         input_ligand, 
         input_docking_grid, 
         output_ligand) = clf.predict_sdf(
            input_protein=args.input_protein, 
            input_ligand=args.input_ligand, 
            input_docking_grid = args.input_docking_grid,
            output_ligand_name = args.output_ligand_name, 
            output_ligand_dir = args.output_ligand_dir,
            output_ligand_dir2 = [args.output_ligand_dir]
         )
        end_time = time.time()
        execution_time = end_time - start_time
        #print('output ligand path:\n', output_ligand)
        #print("total time: ", execution_time, "sec.")
        #print('All processes done!')

    elif args.mode == 'batch_one2many':
        start_time = time.time()
        # get input info form csv file
        input_batch_info = pd.read_csv(args.input_batch_file)
        input_ligand = list(input_batch_info['input_ligand'].values)
        input_docking_grid = list(input_batch_info['input_docking_grid'].values)
        output_ligand_name = list(input_batch_info['output_ligand_name'].values)
        output_ligand_name = [str(ligand_name) for ligand_name in output_ligand_name]
        assert len(input_ligand) == len(input_docking_grid) and len(input_ligand) == len(output_ligand_name)
        # batch predict
        clf = UnimolPredictor.build_predictors(args.model_dir, args.mode, args.nthreads, args.conf_size, args.cluster, 
            use_current_ligand_conf=args.use_current_ligand_conf, steric_clash_fix=args.steric_clash_fix)
        (input_protein, 
         input_ligand, 
         input_docking_grid, 
         output_ligand) = clf.predict_sdf(
            input_protein=args.input_protein, 
            input_ligand=input_ligand, 
            input_docking_grid = input_docking_grid,
            output_ligand_name = output_ligand_name,
            output_ligand_dir = output_ligand_name,
            batch_size= args.batch_size,
            )
        #print('output ligands path:\n', output_ligand)
        end_time = time.time()
        execution_time = end_time - start_time
        #print("Average time: ", execution_time/len(input_ligand), "sec.")
        #print("Total time: ", execution_time, "sec.")
        #print('All processes done!')

    elif args.mode == 'batch_one2one':
        start_time = time.time()
        # get input info form csv file
        input_batch_info = pd.read_csv(args.input_batch_file)
        #记录错误的分子范围，是在哪个批次出现的，之后把这个批次的分子全部去掉 #第7,8个有问题
        num1 = 0
        num2 = 1000000000000  #第三个分子有问题，其下标是2, 如果数据中有一个错误，则都错误，什么情况？要分段处理数据，否则打开文件过多？
        input_protein = list(input_batch_info['input_protein'].values)[num1:num2]
        input_ligand = list(input_batch_info['input_ligand'].values)[num1:num2]
        input_docking_grid = list(input_batch_info['input_docking_grid'].values)[num1:num2]
        output_ligand_name = list(input_batch_info['output_ligand_name'].values)[num1:num2]
        output_ligand_dir2  = list(input_batch_info['output_ligand_dir2'].values)[num1:num2]
        ##print('output_ligand_dir2:', len(output_ligand_dir2))
        assert len(input_ligand) == len(input_docking_grid) and len(input_ligand) == len(output_ligand_name)
        # batch predict
        clf = UnimolPredictor.build_predictors(args.model_dir, args.mode, args.nthreads, args.conf_size, args.cluster, 
            use_current_ligand_conf=args.use_current_ligand_conf, steric_clash_fix=args.steric_clash_fix)
        
        (input_protein, 
         input_ligand, 
         input_docking_grid, 
         output_ligand) = clf.predict_sdf(
            input_protein=input_protein, 
            input_ligand=input_ligand, 
            input_docking_grid = input_docking_grid,
            output_ligand_name = output_ligand_name,
            output_ligand_dir = args.output_ligand_dir,
            output_ligand_dir2 = output_ligand_dir2,
            batch_size= args.batch_size,
            start_idx = args.start_idx,
            end_idx = args.end_idx,
            new_batch_data_name = args.new_batch_data_name,
            gpu = args.gpu
            )
        print('All processes done!')


def main_cli():

    parser = argparse.ArgumentParser(description='unimol docking run entry')
    parser.add_argument(
        "--model-dir",
        type=str,
        default='../weights/run0_pose_new_PDBbind_pose_recycling_4_lr_0.0003_bs_32_dist_th_8.0_epoch_200_wp_0.06/checkpoint_best.pt',
        help='dir of the model'
    )
    parser.add_argument(
        "--input-protein",
        type=str,
        default='protein.pdb',
        help='path of the protein pdb file',
    )
    parser.add_argument(
        "--input-ligand",
        type=str,
        default='ligand.sdf',
        help='path of the ligand sdf file',
    )
    parser.add_argument(
        "--input-batch-file",
        type=str,
        default='input_batch.csv',
        help='path of thr input file in batch mode, one line for each ligand, each line contains the input ligand path, the input docking grid path, and the output ligand name',
    )
    parser.add_argument(
        "--input-docking-grid",
        type=str,
        default='docking_grid.json',
        help='name of the docking grid json file',
    )
    parser.add_argument(
        "--output-ligand-name",
        type=str,
        default='ligand_predict',
        help='name of the ligand sdf file',
    )
    parser.add_argument(
        "--output-ligand-dir",
        type=str,
        default='./predict_sdf',
        help='name of the ligand sdf dir',
    )
    parser.add_argument(
        "--mode",
        type=str,
        default='single',
        help='docking running mode, single and batch, \
            batch_one2one represents batch_protein_to_single_ligand, \
            batch_one2many represents batch_protein_to_many_ligands,',
        choices=['single', 'batch_one2one', 'batch_one2many'],
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--nthreads", 
        type=int, 
        default=4, 
        help="num of threads for data preprocessing"
    )
    parser.add_argument(
        "--conf-size",
        default=10,
        type=int,
        help="number of conformers generated with each molecule",
    )
    parser.add_argument(
        "--cluster",
        action="store_true",
        help="whether preform conformer clustering when data preprocess",
    )
    parser.add_argument(
        "--use_current_ligand_conf", 
        action='store_true',
    )
    parser.add_argument(
        "--steric-clash-fix", 
        action='store_true',
        help="Whether to perform steric clash fix on Unimol docking results"
    )
    
    parser.add_argument(
        "--start_idx",
        type=int, 
        default=0
    )
    
    parser.add_argument(
        "--end_idx", 
        type=int,
        default=None
    )


    parser.add_argument(
        "--gpu", 
        type=int,
        default=0
    )



    parser.add_argument(
        "--new_batch_data_name",
        type=str, 
        default=None
    )
    
    
    args = parser.parse_args()
    
    
    
    #print(args)
    main(args)


if __name__ == "__main__":
    main_cli()