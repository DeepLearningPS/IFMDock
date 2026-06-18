import os
import sys
import subprocess
import warnings
from tqdm import tqdm
warnings.simplefilter("ignore")  # 更彻底的忽略方式


data_name_file = '/data/fan_zg/MDocking/new_VSDS/data_name.txt'

data_name_list = []
with open(data_name_file) as f:
    for line in f:
        data_name_list.append(line.strip())
print('len(data_name_list):', len(data_name_list)) 


'''
cd /data/fan_zg/MDocking/RTMScore-main
conda activate torch2.1.0
python vsds_batch_deal.py

'''

'''
#/data/fan_zg/MDocking/VSDS_ECDock_Gen
#/data/fan_zg/MDocking/new_VSDS/bingnetv2_EcDock_sample_5conf_dir
model_list = ['bingnetv2_EcDock_sample_5conf_dir', 'VSDS_ECDock_Gen', 'ecdock_25conf', 'carsidock', 'diffdock', 'ecdock', 'ecdock_MMF', 'ecdock_refine', 'glide', 'karmadock', 'unimol', 'vina']
#/data/fan_zg/MDocking/origin_VSDS/ecdock_25conf
refine = False
for model in model_list[:1]:
    # model = 'carsidock'
    for dataset in tqdm(data_name_list[0:80]):
        if 'refine' in model or refine:
            cmd = f'CUDA_VISIBLE_DEVICES=7 python rtmscore_general_ecdock.py --refine_flag refine --data_dir /data/fan_zg/MDocking/{model}/{dataset}_ecdock_cm_equiformer_step5_interaction_limit4.5ai_307_step5_unimol_distance \
                --data_name /data/fan_zg/MDocking/new_VSDS/data_name.txt -gen_pocket -c 10.0 -m trained_models/rtmscore_model1.pth'
        else:
            cmd = f'CUDA_VISIBLE_DEVICES=7 python rtmscore_general_ecdock.py --refine_flag norefine --data_dir /data/fan_zg/MDocking/{model}/{dataset}_ecdock_cm_equiformer_step5_interaction_limit4.5ai_307_step5_unimol_distance \
                --data_name /data/fan_zg/MDocking/new_VSDS/data_name.txt -gen_pocket -c 10.0 -m trained_models/rtmscore_model1.pth'
        os.system(cmd)
'''



## 166跑[0:50], 176跑[50:100], 68跑[100:150]

##['carsidock', 'diffdock', 'ecdock', 'ecdock_100conf', 'ecdock_3d', 'ecdock_MMF', 'ecdock_multidistance', 'ecdock_refine', 'glide', 'karmadock', 'unimol', 'vina']
model_list = ['bingnetv2_EcDock_sample_5conf_not_mmff', 'bingnetv2_EcDock_sample_5conf_dir', 'ecdock_100conf_nodistance_step10', 'ecdock_25conf', 'carsidock', 'diffdock', 'ecdock', 'ecdock_MMF', 'ecdock_refine', 'glide', 'karmadock', 'unimol', 'vina']
#/data/fan_zg/MDocking/origin_VSDS/ecdock_25conf
#data_name_list = ['O75469-6TFI', 'P00338-5W8J'] 
for model in model_list[:1]:
    # model = 'carsidock'
    for dataset in data_name_list[60:160]:
        if 'refine' in model:
            cmd = f'CUDA_VISIBLE_DEVICES=4 python rtmscore_general.py --refine_flag refine --data_dir /data/fan_zg/MDocking/origin_VSDS/{model}/{dataset} \
                --data_name /data/fan_zg/MDocking/new_VSDS/data_name.txt -gen_pocket -c 10.0 -m trained_models/rtmscore_model1.pth'
        else:
            cmd = f'CUDA_VISIBLE_DEVICES=4 python rtmscore_general.py --refine_flag norefine --data_dir /data/fan_zg/MDocking/new_VSDS/{model}/{dataset} \
                --data_name /data/fan_zg/MDocking/new_VSDS/data_name.txt -gen_pocket -c 10.0 -m trained_models/rtmscore_model1.pth'
        os.system(cmd)

    


