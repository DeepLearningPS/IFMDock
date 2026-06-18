import os 
import sys
import shutil 



#虚拟筛选多个数据集执行




name_list = []

with open('/data/fan_zg/MDocking/VSDS_DTEBV-D/data_name.txt') as f:
    for line in f:
        name_list.append(line.strip())
        
#python KGDiff/scripts/new_multiple_data_run.py 数据正在生成在[60,70], [80, 90], [100, 110]
# 166跑[0:50], 176跑[50:110], 68跑[110:150]
s_base_dir = '/data/fan_zg/MDocking/VSDS_DTEBV-D/data'
t_base_dir = '/data/fan_zg/MDocking/VSDS_DTEBV-D/sub_data/data'
for st_id, end_id in [[60,70], [80, 90], [100, 110]]:
    print('st_id, end_id:', st_id, end_id)
    for name in name_list[st_id:end_id]: 
        s_dir = os.path.join(s_base_dir, name)
        t_dir = os.path.join(t_base_dir, name)
        os.makedirs(t_dir, exist_ok=True)
        for i in range(5):
            s_file = os.path.join(s_dir, f'{name}_processed_final_{name}_interaction_gen_split3_5_{i}_v33.lmdb')
            t_file = os.path.join(t_dir, f'{name}_processed_final_{name}_interaction_gen_split3_5_{i}_v33.lmdb')
            shutil.copy2(s_file, t_file)


print('success')
            
        