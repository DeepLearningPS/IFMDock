import shutil
import os
import csv
from tqdm import tqdm

from pathlib import Path

# 获取当前目录
CURRENT_DIR = os.getcwd()

# 切换工作目录
#os.chdir(CURRENT_DIR)
# 验证
print("当前工作目录:", CURRENT_DIR)



s_path = 'Distance_model/interface/tmpdata_predict_sdf_boxsize10'
t_path = f'{CURRENT_DIR}/tmpdata/tmpdata'
count = 0
for i in os.listdir(s_path):
    path = os.path.join(s_path, i)
    if os.path.exists(path) and os.path.isdir(path) and os.listdir(path): #目录存在且不空
        #print('i:', i)
        s_file = os.path.join(s_path, i, f'interaction_{i}.pkl')
        t_file = os.path.join(t_path, i, f'interaction_{i}_v2.pkl')
        shutil.copy2(s_file, t_file)
        count += 1  


print('success num:', count) #428

path = Path(s_path)
if path.is_dir():  # 确保它是目录
        shutil.rmtree(path)
        print(f"目录 {s_path} 已删除")








