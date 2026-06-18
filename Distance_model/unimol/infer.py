#!/usr/bin/env python3 -u
# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import pickle
import torch
from unicore import checkpoint_utils, distributed_utils, options, utils
from unicore.logging import progress_bar
from unicore import tasks
import shutil
from tqdm import tqdm

import torch
import numpy as np
import random

from unicore.data.nested_dictionary_dataset import  BuildDataset


def set_seed(seed):
    torch.manual_seed(seed)  # 设置 PyTorch 的随机数种子
    torch.cuda.manual_seed_all(seed)  # 设置所有 GPU 的随机数种子
    np.random.seed(seed)  # 设置 NumPy 的随机数种子
    random.seed(seed)  # 设置 Python 自带的随机数种子
    torch.backends.cudnn.deterministic = True  # 设置 CuDNN 算法为确定性算法
    torch.backends.cudnn.benchmark = True


set_seed(2024)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("unimol.inference")


def main(args):

    assert (
        args.batch_size is not None
    ), "Must specify batch size either with --batch-size"

    use_cuda = torch.cuda.is_available() and not args.cpu
    use_fp16 = args.fp16 and use_cuda

    if use_cuda:
        torch.cuda.set_device(args.device_id)

    if args.distributed_world_size > 1:
        data_parallel_world_size = distributed_utils.get_data_parallel_world_size()
        data_parallel_rank = distributed_utils.get_data_parallel_rank()
    else:
        data_parallel_world_size = 1
        data_parallel_rank = 0

    # Load model
    logger.info("loading model(s) from {}".format(args.path))
    state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
    task = tasks.setup_task(args)  #创建一个任务对象，task是系统包
    model = task.build_model(args) #创建任务后，使用自定义的数据对象，即unicore_task.py文件
    model.load_state_dict(state["model"], strict=False)
    #print('ok?')

    # Move models to GPU
    if use_fp16:
        model.half()
    if use_cuda:
        model.cuda()

    # #print args
    logger.info(args)

    # Build loss
    loss = task.build_loss(args)
    loss.eval()

    #if os.path.exists('protein_fail.txt'):
        #os.remove('protein_fail.txt')

    #pkt_data_path和mol_data_path文件，是必要的，且每一个复合物，有不同的这样的文件，不能共享？看看具体用来干什么的？用于生成dataloader，作为原子数量的约束#
    #构建数据集的时候，数据不要打乱
    #print("args.valid_subset:", args.valid_subset) #batch_data
    for subset in args.valid_subset.split(","):
        #print('subset:', subset) #batch_data
        #print('args.conf_size:', args.conf_size) #10,
        
        #raise Exception('test')
        print('args.start_idx, args.end_idx:', args.start_idx, args.end_idx)
        try:
            task.load_dataset(subset, start_idx=args.start_idx, end_idx=args.end_idx, combine=False, epoch=1) #加载配体和蛋白文件
            dataset = task.dataset(subset)
            #print('dataset:', dataset) #<unicore.data.nested_dictionary_dataset.NestedDictionaryDataset object at 0x151fadfb0310>
        except KeyError:
            raise Exception("Cannot find dataset: " + subset)

        #print('ok2')

        #print('len(dataset):', len(dataset)) #2470, 没意义，是自定义的数据类型
        #print('type(dataset):', type(dataset)) #2470
        #raise Exception('test')
        
        """
        #遍历数据集，然后去除None
        count = 0
        new_dataset = []
        #print('type(dataset):', type(dataset)) # dataset: <unicore.data.nested_dictionary_dataset.NestedDictionaryDataset object at 0x1535ba92c850>
        #exit()
        for ddt in dataset: 
            #遍历这个数据集时，会出错，数据集中有None，不是已经过滤掉了？在unicore/data/nested_dictionary_dataset.py加异常，返回None即可。因此在dataset = task.dataset(subset)设置
            #数据集后，再遍历数据集，去掉None，之后再形成batch
            #print('count:', count)

                
            if ddt['net_input.mol_src_distance'].shape[-1] != ddt['net_input.mol_src_edge_type'].shape[-1]:
                continue
            
            if ddt != None:
                new_dataset.append(ddt)

            '''
            mol_src_tokens: torch.Size([10, 36])
            mol_src_distance: torch.Size([10, 21, 21])
            mol_src_coord: torch.Size([10, 21, 3])
            mol_src_coord[:2]: tensor([[ 0.0000,  0.0000,  0.0000],
                    [ 1.3372,  0.4923, -4.3993]], device='cuda:0')
            mol_src_edge_type: torch.Size([10, 36, 36])
            pocket_src_tokens: torch.Size([10, 160])
            '''
            
            ##print('ddt:', ddt)
            '''
            ddt.keys: odict_keys(['net_input.mol_src_tokens', 'net_input.mol_src_coord', 'net_input.mol_src_distance', 'net_input.mol_src_edge_type', 
            'net_input.pocket_src_tokens', 'net_input.pocket_src_coord', 'net_input.pocket_src_distance', 'net_input.pocket_src_edge_type', 'target.distance_target', 
            'target.holo_coord', 'target.holo_distance_target', 'target.holo_coord_pocket', 'smi_name', 'pocket_name', 'holo_coord', 'holo_coord_pocket', 
            'holo_center_coordinates'])
            '''
            count += 1
            #if count == 2:
                #break
        #print('all count:', count)
        #print('len(new_dataset):', len(new_dataset))
        #raise Exception('test')
        
        
        
        #重构数据，因为之前删除了一些，数量减少了
        dataset = BuildDataset(new_dataset)
        """
        
        if not os.path.exists(args.results_path):
            os.makedirs(args.results_path)
        
        #print('args.results_path:', args.results_path) #/mnt/home/fanzhiguang/47/Uni-Mol/unimol_docking_v2/interface/pdb2020_predict_sdf
        #print('args.valid_subset:', args.valid_subset) #ligand_predict_example #batch_data
        #print("subset + .pkl:", subset + '.pkl') #batch_data.pkl, 数据要保存的
        save_path = os.path.join(args.results_path, subset + ".pkl") #看看这个文件是怎么生成的，具体内容是啥？每一个文件分子都准备一个还是只需要共享一个即可？
        # Initialize data iterator
        ##print('dataset:', list(dataset)[7000:7001])

        #这个迭代器有问题，如果有一个失败，会导致其余也失败，需要改源代码,
        #
        itr = task.get_batch_iterator(
            dataset=dataset,
            batch_size=args.batch_size,
            #ignore_invalid_inputs=True,
            ignore_invalid_inputs=False,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=data_parallel_world_size,
            shard_id=data_parallel_rank,
            num_workers=args.num_workers,
            data_buffer_size=args.data_buffer_size,
        ).next_epoch_itr(shuffle=False)

        #print('len(itr):', len(itr)) #能计算长度？2 * 4 = 8, 可以计算
        #print('args.batch_size:', args.batch_size)

        #使用迭代器的好处就是能够防止数据太多而导致打开文件过多出错
        '''
        new_itr = []
        for i in range(len(itr)):
            ##print('iter:', i) #能走到这一步
            try:
                new_itr.append(next(itr)) #在生成数据的时候，需要加载处理好的配体文件.lmdb, 但里面有None，所以会报错，因此应该跳过
            #except StopIteration: #因为不知道到底有数据有多少，所以只能等到报StopIteration才结束
                ##print('迭代结束')
                #break
            except Exception as e:
                #print(f"Error processing batch {i}: {str(e)}")
                #print('dataloader error, skip')
                new_itr.append(None)
        '''
        
        ##print('len(new_itr):', len(new_itr)) # 1

        
        progress = progress_bar.progress_bar(
            #new_itr,
            itr,
            log_format=args.log_format,
            log_interval=args.log_interval,
            prefix=f"valid on '{subset}' subset",
            default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
        )
        
        #print('pkt_data_path和mol_data_path文件1')
        log_outputs = []
        #print('len(progress):', len(progress)) #1,在这就已经过滤掉了错误了分子了
        for i, sample in enumerate(tqdm(progress)): #progress错误, 卡住是因为数据加载问题
            #print('pkt_data_path和mol_data_path文件2')
            ##print('sample:', sample)
            if sample == None:  #这里有问题，如果有一个复合物出错，导致其余都错了
                #print('sample == None')
                log_outputs.append(None)
            elif len(sample) == 0:
                #print('len(sample) == 0, skip')
                log_outputs.append(None)
                continue
                #log_outputs.append(None) #在处理数据的时候，部分数据出错，但是在output_ligand_name中未剔除，所以需要标记一下
            else:
                #print('-----------------------time0---------------------------------')
                #print("sample['pocket_name'][0]:", sample['pocket_name'][0])
                sample = utils.move_to_cuda(sample) if use_cuda else sample #看一下在哪里形成配体-蛋白连接表的？这个表怎么来的？要么是在批量形成过程中，要么是在模型里面

                #print('-----------------------time00---------------------------------')
                _, _, dt = task.valid_step(sample, model, loss, test=True) #调用模型进行预测，并返回结果
                #print('-----------------------time000---------------------------------')
                log_output = dt

                if log_output == None:
                    #print('log_output, protein_name:', log_output, sample['pocket_name'][0])
                    ##print('sample:', sample)
                    #raise Exception('error')
                    #模型推理时出错
                    with open('protein_fail.txt', 'w') as f:
                        f.write(sample['pocket_name'][0] + '\n')
                    
                    #print('error protein_name[0]:', sample['pocket_name'][0])
                    
                else:
                    progress.log({}, step=i)
                    log_outputs.append(log_output)
                
                #print('protein_name[0]:', sample['pocket_name'][0])
            
        #print('pkl log_outputs num:', len(log_outputs)) #307， 存放的神经网络预测出来的分子数据
        ##print('pkl log_outputs[0].keys():', log_outputs[0].keys())
        '''
        pkl log_outputs[0].keys(): dict_keys(['loss', 'cross_distance_loss', 'distance_loss', 'coord_loss', 'prmsd_loss', 'prmsd_score', 
        'bsz', 'sample_size', 'coord_predict', 'coord_target', 'smi_name', 'pocket_name', 'atoms', 'pocket_atoms', 'coordinates', 'holo_coordinates',
        'pocket_coordinates', 'holo_center_coordinates'])
        '''
        #print('len(log_outputs):', len(log_outputs))
        
        '''
        loss_list = []
        #print('log_outputs:', type(log_outputs)) # list
        for log in log_outputs:
            ##print('log:', log)
            #print('type(log):', type(log)) #dict
            
            #exit()
            tg = log.get("loss", 0)
            loss_list.append(tg)
        '''
        
        
        
        #raise Exception('stop') #为啥没有被中断？
        pickle.dump(log_outputs, open(save_path, "wb")) #如果出错，这里是没有数据存储的
        logger.info("Done inference! ")
        #exit()

    #return None


def cli_main():
    parser = options.get_validation_parser()
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)

    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
