import os
import sys
import subprocess
import warnings
warnings.simplefilter("ignore")  # 更彻底的忽略方式

data_name_file = '/data/fan_zg/MDocking/data_name.txt'


'''
cd /data/fan_zg/MDocking/RTMScore-main
conda activate torch2.1.0
python batch_deal.py

'''

## 166跑[0:50], 176跑[50:100], 68跑[100:150]

'''ecdock'''

st  = 0
en  = 50

for i in list(range(st, en)):
    cmd = f'CUDA_VISIBLE_DEVICES=4 python ecdock_rtmscore.py --refine_flag norefine --data_dir /data/fan_zg/MDocking/EcDock_sample_dir \
        --data_name /data/fan_zg/MDocking/data_name.txt -gen_pocket -c 10.0 -m trained_models/rtmscore_model1.pth \
            --st_i {i} --en_i {i+1}'
    os.system(cmd)
    #process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
    #process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    #print(f"Command is running in the background with PID: {process.pid}")

exit()



'''glide'''
'''
st  = 2
en  = 3
#160:50:65, 188:65:95, 186:95:125, 47:125:150
for i in list(range(st, en)):
    cmd = f'CUDA_VISIBLE_DEVICES=0 python glide_rtmscore.py --refine_flag norefine --data_dir /mnt_191/fanzhiguang/47/VSDS/VSDS_Glide/glide \
        --data_name /mnt_191/fanzhiguang/47/VSDS/VSDS_Glide/data_name.txt -gen_pocket -c 10.0 -m trained_models/rtmscore_model1.pth \
            --st_i {i} --en_i {i+1}'
    os.system(cmd)
    #process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
    #process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    #print(f"Command is running in the background with PID: {process.pid}")


exit()
'''


'''carsidock'''
'''
#carsidock优化的时候加氢了，所以不用再加氢
#70~100
st  = 90
en  = 100
#160:50:65, 188:65:95, 186:95:125, 47:125:150
for i in list(range(st, en)):
    cmd = f'CUDA_VISIBLE_DEVICES=6 python carsidock_rtmscore.py --refine_flag norefine --data_dir /data/fan_zg/MDocking/Docking_baseline/CarsiDock/outputs/new_vsds \
        --data_name /data/fan_zg/MDocking/VSDS_Glide/data_name.txt -gen_pocket -c 10.0 -m trained_models/rtmscore_model1.pth \
            --st_i {i} --en_i {i+1}'
    os.system(cmd)
    #process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
    #process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    #print(f"Command is running in the background with PID: {process.pid}")
'''


'''unimol'''
#carsidock优化的时候加氢了，所以不用再加氢
#0~50;100~150

all_lost_data_name =  ['Q96RI1-6HL1', 'Q99500-7C4S', 'Q9BY41-5BWZ']


data_name_list = []
with open(data_name_file) as f:
    for line in f:
        data_name_list.append(line.strip())
print('len(data_name_list):', len(data_name_list))     
lost_index = []
for i in all_lost_data_name:
    j = data_name_list.index(i)
    lost_index.append(j)
print('lost_index:', lost_index)
for ids in lost_index[0:1]:
    st  = ids
    en  = ids + 1
    #160:50:65, 188:65:95, 186:95:125, 47:125:150
    for i in list(range(st, en)):
        cmd = f'CUDA_VISIBLE_DEVICES=7 python unimol_rtmscore.py --refine_flag norefine --data_dir /data/fan_zg/MDocking/Docking_baseline/unimol_docking_v2/interface/valid_vsds \
            --data_name /data/fan_zg/MDocking/VSDS_Glide/data_name.txt -gen_pocket -c 10.0 -m trained_models/rtmscore_model1.pth \
                --st_i {i} --en_i {i+1}'
        os.system(cmd)
        #process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
        #process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        #print(f"Command is running in the background with PID: {process.pid}")