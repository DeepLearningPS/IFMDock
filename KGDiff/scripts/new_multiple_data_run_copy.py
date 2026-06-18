import os 
import sys



#虚拟筛选多个数据集执行

'''
cd /data/fan_zg/MDocking/new_KGDiff-EcDock
conda activate torch2.1.0
python KGDiff/scripts/new_multiple_data_run.py
'''


name_list = []

with open('/data/fan_zg/MDocking/VSDS_DTEBV-D/data_name.txt') as f:
    for line in f:
        name_list.append(line.strip())
        
#python KGDiff/scripts/new_multiple_data_run.py, 数据正在生成在[60,70], [80, 90], [100, 110]
# 166跑[0:50], 176跑[50:100], 68跑[100:150]
# 176能使用的卡0,1
for name in name_list[67:70]: #ec7跑[60,70], ec8跑[80,90]
    cmd = f'CUDA_VISIBLE_DEVICES=3 python  KGDiff/scripts/new_sample_diffusion.py --config ./configs/sampling.yml -i 0 --guide_mode pdbbind_random --type_grad_weight 100 \
            --pos_grad_weight 25 --result_path new_ecdock_step15 --data_flag new_test --data_name {name} --diffusion cm --gnn equiformer --sample_num 500 \
            --conf_num 25 --si 0 --ei 200000 --test_name 3Dmultidistance'
    
    os.system(cmd)
    
    
    #/data/fan_zg/MDocking/VSDS_DTEBV-D/data/O00329-6PYR/O00329-6PYR
    
    '''
    CUDA_VISIBLE_DEVICES=5 python  KGDiff/scripts/new_sample_diffusion.py --config ./configs/sampling.yml -i 0 --guide_mode pdbbind_random --type_grad_weight 100 \
    --pos_grad_weight 25 --result_path new_ecdock_step15 --data_flag new_test --data_name posebustersv1 --diffusion cm --gnn equiformer --sample_num 500 \
    --conf_num 3 --si 0 --ei 20 --test_name true_distance
    '''