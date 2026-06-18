import os
import shutil
def move_data(base_path, step = 15):
    #ecdock生成的文件太多，不方便移动，减少点
    step = step - 1
    if os.path.exists('../../tmp'):
        shutil.rmtree('../../tmp')
    for name in os.listdir(base_path):
        path = os.path.join(base_path, name)
        if os.path.exists(path) and os.path.isdir(path) and os.listdir(path) and 'model' not in name:
            s_dir1 = os.path.join(path, f'step{step}')
            s_file = os.path.join(path, f'{name}_protein.pdb')
            s_file2 = os.path.join(path, f'{name}_ligand_docking_grid_boxsize10.json')
            t_dir = f'../../tmp/{name}'

            os.makedirs(t_dir, exist_ok=True)
            #shutil.copytree(s_dir1, t_dir + f'/step{step}', dirs_exist_ok=True)
            #shutil.copy(s_file, t_dir)
            shutil.copy(s_file2, t_dir)
            






if __name__ == '__main__':
    path = '/data/fan_zg/MDocking/EcDock_sample_dir/posebusters_ecdock_cm_equiformer_step15_interaction_limit4.5ai_cm_equiformer_gen_split_3.5_test_243_new_cross0_force0_optim0_conf100'
    move_data(path)