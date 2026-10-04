import argparse
import os
import dill
import shutil
from pathlib import Path



def gen_test_csv_pdb2020_box20(base_path):

    name_list = []
    with open(os.path.join(base_path, 'new_pdb2020_test_name.txt')) as f:
        for i in f:
            name_list.append(i.strip('\n'))
    

    data_list = []
    data_list.append('input_protein,input_ligand,input_docking_grid,output_ligand_name,output_ligand_dir2')

    for name in name_list:
        tg = f'{base_path}/new_pdb2020_test/{name}/{name}_protein.pdb,{base_path}/new_pdb2020_test/{name}/{name}_ligand.sdf,{base_path}/new_pdb2020_test/{name}/{name}_ligand_docking_grid_boxsize20.json,{name},pdb2020_predict_sdf_boxsize20/{name}'
        data_list.append(tg)
    

    with open('pdb2020_input_batch_one2one_boxsize20.csv', 'w') as f:
        for i in data_list:
            f.write(i + '\n')




def gen_test_csv_pdb2020_box10(base_path):

    name_list = []
    with open(os.path.join(base_path, 'new_pdb2020_test_name.txt')) as f:
        for i in f:
            name_list.append(i.strip('\n'))
    

    data_list = []
    data_list.append('input_protein,input_ligand,input_docking_grid,output_ligand_name,output_ligand_dir2')

    for name in name_list:
        tg = f'{base_path}/new_pdb2020_test/{name}/{name}_protein.pdb,{base_path}/new_pdb2020_test/{name}/{name}_ligand.sdf,{base_path}/new_pdb2020_test/{name}/{name}_ligand_docking_grid_boxsize10.json,{name},pdb2020_predict_sdf_boxsize10/{name}'
        data_list.append(tg)
    

    with open('pdb2020_input_batch_one2one_boxsize10.csv', 'w') as f:
        for i in data_list:
            f.write(i + '\n')



def gen_train_csv_pdb2020_box10(base_path, name_file, data_name = 'new_pdbbind2020', error_file = None):

    name_list = []
    with open(name_file, 'r') as f:
        for line in f:
            name_list.append(line.strip())
    error_list = []
    if error_file:
        with open(error_file, 'r') as f:
            for line in f:
                error_list.append(line.strip())


    with open('ligand_fail_file_name.txt', 'r') as f:
        for line in f:
            error_list.append(line.strip())


    data_list = []
    data_list.append('input_protein,input_ligand,input_docking_grid,output_ligand_name,output_ligand_dir2')
    success_count = 0
    for name in name_list:
        if name not in error_list:
            tg = f'{base_path}/{data_name}/{data_name}/{name}/{name}_protein.pdb,{base_path}/{data_name}/{data_name}/{name}/{name}_ligand.sdf,{base_path}/{data_name}/{data_name}/{name}/{name}_ligand_docking_grid_boxsize10.json,{name},new_pdb2020_predict_sdf_boxsize10/{name}'
            data_list.append(tg)
            success_count += 1
        

    with open('new_pdb2020_input_batch_one2one_boxsize10.csv', 'w') as f:
        for i in data_list:
            f.write(i + '\n')



def gen_test_csv_posebusters_box10(base_path):

    name_list = []
    with open(os.path.join(base_path, 'posebusters_name.txt')) as f:
        for i in f:
            name_list.append(i.strip('\n'))
    

    

    data_list = []
    data_list.append('input_protein,input_ligand,input_docking_grid,output_ligand_name,output_ligand_dir2')

    success_count = 0

    for name in name_list:
        tg = f'{base_path}/posebusters/{name}/{name}_protein.pdb,{base_path}/posebusters/{name}/{name}_ligand.sdf,{base_path}/posebusters/{name}/{name}_ligand_docking_grid_boxsize10.json,{name},posebusters_predict_sdf_boxsize10/{name}'
        data_list.append(tg)
        success_count += 1

    with open('posebusters_input_batch_one2one_boxsize10.csv', 'w') as f:
        for i in data_list:
            f.write(i + '\n')



def gen_test_csv_posebusters_box20(base_path):

    name_list = []
    with open(os.path.join(base_path, 'posebusters_name.txt')) as f:
        for i in f:
            name_list.append(i.strip('\n'))
    

    data_list = []
    data_list.append('input_protein,input_ligand,input_docking_grid,output_ligand_name,output_ligand_dir2')

    for name in name_list:
        tg = f'{base_path}/posebusters/{name}/{name}_protein.pdb,{base_path}/posebusters/{name}/{name}_ligand.sdf,{base_path}/posebusters/{name}/{name}_ligand_docking_grid_boxsize20.json,{name},posebusters_predict_sdf_boxsize20/{name}'
        data_list.append(tg)
    

    with open('posebusters_input_batch_one2one_boxsize20.csv', 'w') as f:
        for i in data_list:
            f.write(i + '\n')





def again_gen_train_csv_pdb2020_box10(base_path, name_file, data_name):



    error_protein = set()
    with open('protein_fail.txt') as f:
        for i in f:
            tg = i.split('/')[-2]   
            error_protein.add(tg)


    if os.path.exists('protein_fail.txt'):
        os.remove('protein_fail.txt')
    with open('protein_fail.txt', 'w') as file:
        pass 


    data_list = []
    data_list.append('input_protein,input_ligand,input_docking_grid,output_ligand_name,output_ligand_dir2')
    success_count = 0
    for name in error_protein:
        tg = f'{base_path}/{data_name}/{data_name}/{name}/{name}_protein.pdb,{base_path}/{data_name}/{data_name}/{name}/{name}_ligand.sdf,{base_path}/{data_name}/{data_name}/{name}/{name}_ligand_docking_grid_boxsize10.json,{name},new_pdb2020_predict_sdf_boxsize10/{name}'
        data_list.append(tg)
        success_count += 1
    with open('new_pdb2020_input_batch_one2one_boxsize10_again.csv', 'w') as f:
        for i in data_list:
            f.write(i + '\n')




def again_gen_train_csv_posebusters_box10(base_path):

    error_protein2 = set()
    with open('protein_fail.txt') as f:
        for i in f:
            tg = i.split('/')[-2]    
            error_protein2.add(tg)


    if os.path.exists('protein_fail.txt'):
        os.remove('protein_fail.txt')
    with open('protein_fail.txt', 'w') as file:
        pass 

    data_list = []
    data_list.append('input_protein,input_ligand,input_docking_grid,output_ligand_name,output_ligand_dir2')
    success_count = 0
    for name in error_protein2:
        tg = f'{base_path}/posebusters/posebusters/{name}/{name}_protein.pdb,{base_path}/posebusters/posebusters/{name}/{name}_ligand.sdf,{base_path}/posebusters/posebusters/{name}/{name}_ligand_docking_grid_boxsize10.json,{name},posebusters_predict_sdf_boxsize10/{name}'
        data_list.append(tg)
        success_count += 1
        

    with open('posebusters_input_batch_one2one_boxsize10_again.csv', 'w') as f:
        for i in data_list:
            f.write(i + '\n')



def gen_data_box10(base_path, name_file, data_name = 'new_pdbbind2020', error_file = None, data_s_id = 0, data_e_id = 1000000000, data_check = 1):

    name_list = []
    with open(name_file, 'r') as f:
        for line in f:
            name_list.append(line.strip())
    error_list = []




    data_list = []
    data_list.append('input_protein,input_ligand,input_docking_grid,output_ligand_name,output_ligand_dir2')
    success_count = 0
    for name in name_list[data_s_id: data_e_id]:
        if name not in error_list:
            tg = f'{base_path}/{data_name}/{data_name}/{name}/{name}_protein_256.pdb,{base_path}/{data_name}/{data_name}/{name}/{name}_ligand.sdf,{base_path}/{data_name}/{data_name}/{name}/{name}_ligand_docking_grid_boxsize10.json,{name},{data_name}_predict_sdf_boxsize10/{name}'
            data_list.append(tg)
            success_count += 1
    CURRENT_DIR = Path(__file__).resolve().parent
    os.chdir(CURRENT_DIR)
    print(":", os.getcwd())


    with open(f'{data_name}_input_batch_one2one_boxsize10.csv', 'w') as f:
        for i in data_list:
            f.write(i + '\n')
    print('len(data_list):', len(data_list) - 1)





def gen_data_box10_vsds(base_path, name_file, data_name = 'new_pdbbind2020', error_file = None, data_s_id = 0, data_e_id = 1000000, data_check = 1):

    name_list = []
    with open(name_file, 'r') as f:
        for line in f:
            name_list.append(line.strip())
    error_list = ['2r0z']
    
    
    if error_file:
        with open(error_file, 'r') as f:
            for line in f:
                error_list.append(line.strip())


    with open('ligand_fail_file_name.txt', 'r') as f:
        for line in f:
            error_list.append(line.strip())
    



    data_list = []
    data_list.append('input_protein,input_ligand,input_docking_grid,output_ligand_name,output_ligand_dir2')
    success_count = 0
    for name in name_list[data_s_id: data_e_id]:
        if name not in error_list:
            tg = f'{base_path}/{data_name}/{data_name}/{name}/{name}_protein.pdb,{base_path}/{data_name}/{data_name}/{name}/{name}_ligand.sdf,{base_path}/{data_name}/{data_name}/{name}/{name}_ligand_docking_grid_boxsize10.json,{name},vsds2/{data_name}_sdf/{name}'
            data_list.append(tg)
            success_count += 1
        
    if data_name == 'newer_pdbbind2020':
        with open(f'vsds2/{data_name}_csv.csv', 'w') as f:
            for i in data_list:
                f.write(i + '\n')
    else:
        with open(f'vsds2/{data_name}_csv.csv', 'w') as f:
            for i in data_list:
                f.write(i + '\n')






def again_gen_data_box10(base_path, name_file, data_name, data_s_id = 0, data_e_id = 1000000, data_check = 1):



    error_protein = set()
    with open('protein_fail.txt') as f:
        for i in f:
            tg = i.split('/')[-2]    
            error_protein.add(tg)

    if os.path.exists('protein_fail.txt'):
        os.remove('protein_fail.txt')
    with open('protein_fail.txt', 'w') as file:
        pass 


    with open('error_not_equal.txt') as f:
        for i in f:
            tg = i.strip()   
            error_protein.add(tg)


    data_list = []
    data_list.append('input_protein,input_ligand,input_docking_grid,output_ligand_name,output_ligand_dir2')
    success_count = 0
    for name in error_protein:
        tg = f'{base_path}/{data_name}/{data_name}/{name}/{name}_protein.pdb,{base_path}/{data_name}/{data_name}/{name}/{name}_ligand.sdf,{base_path}/{data_name}/{data_name}/{name}/{name}_ligand_docking_grid_boxsize10.json,{name},{data_name}_predict_sdf_boxsize10/{name}'
        data_list.append(tg)
        success_count += 1
    with open(f'{data_name}_input_batch_one2one_boxsize10_again.csv', 'w') as f:
        for i in data_list:
            f.write(i + '\n')





if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate the distance-model CSV for a prepared EC-Dock dataset."
    )
    parser.add_argument("--data_name", default="tmpdata")
    parser.add_argument(
        "--project_root",
        default=str(Path(__file__).resolve().parents[3]),
    )
    args = parser.parse_args()

    if not args.data_name or "/" in args.data_name or "\\" in args.data_name:
        parser.error("--data_name must be one directory name, not a path")

    base_path = str(Path(args.project_root).resolve())
    name_file = f'{base_path}/{args.data_name}/{args.data_name}_name.txt'
    error_file = f'{base_path}/{args.data_name}/error_ligand_list.txt'

    gen_data_box10(
        base_path=base_path,
        name_file=name_file,
        data_name=args.data_name,
        error_file=error_file,
    )
        


        
