
import os
from .processor import Processor
import shutil
from ordered_set import OrderedSet
import pathlib


import numpy as np
import lmdb
import pickle
import copy
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from multiprocessing import Pool
from typing import List
from sklearn.cluster import KMeans
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolAlign import AlignMolConformers
from biopandas.pdb import PandasPdb
import dill
import time


class GeneralMolPredictor:
    def __init__(self, model_dir, mode='single', nthreads=4, conf_size=1, cluster=False, use_current_ligand_conf=False, steric_clash_fix=False):
        self.model_dir = model_dir
        self.mode = mode
        self.nthreads = nthreads
        self.use_current_ligand_conf = use_current_ligand_conf
        self.cluster = cluster
        self.steric_clash_fix = steric_clash_fix
        self.conf_size = conf_size
        if self.use_current_ligand_conf:
            self.conf_size = 1
        else:
            self.conf_size = conf_size

    def preprocess(self, input_protein, input_ligand, input_docking_grid, output_ligand_name, output_ligand_dir):
        preprocessor = Processor.build_processors(
            self.mode, self.nthreads, conf_size=self.conf_size, cluster=self.cluster,
            use_current_ligand_conf=self.use_current_ligand_conf)
        processed_data = preprocessor.preprocess(input_protein, input_ligand, input_docking_grid, output_ligand_name, output_ligand_dir)
        return processed_data

    def is_file_exist_and_not_empty(self, filepath):
        if not os.path.exists(filepath):
            return False
        if not os.path.isfile(filepath):
            return False
        if os.path.getsize(filepath) > 1000:
            return True
        else:
            return False

    def predict(self, input_protein:str, 
                input_ligand:str, 
                input_docking_grid:str, 
                output_ligand_name:str, 
                output_ligand_dir:str, 
                batch_size:int,start_idx, end_idx, new_batch_data_name, gpu = 0):
        
        
        
        os.makedirs(os.path.abspath(output_ligand_dir), exist_ok=True)
        if os.path.exists(os.path.join(os.path.abspath(output_ligand_dir), 'batch_data.lmdb')):
            print(f"：{os.path.abspath(output_ligand_dir)}")
            pass
        else:
            lmdb_name = self.preprocess(input_protein, input_ligand, input_docking_grid, output_ligand_name, output_ligand_dir)

        
        lmdb_name = 'batch_data'
        
        pkt_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "example_data", "dict_pkt.txt")
        mol_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "example_data", "dict_mol.txt")
        script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "generalmodels", "infer.py")
        user_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "generalmodels")
        cmd = f' cp {pkt_data_path} {os.path.abspath(output_ligand_dir)} \n\
                cp {mol_data_path} {os.path.abspath(output_ligand_dir)} \n\
            CUDA_VISIBLE_DEVICES={gpu} python {script_path} --user-dir {user_dir} {os.path.abspath(output_ligand_dir)} --valid-subset {lmdb_name} \
            --results-path {os.path.abspath(output_ligand_dir)} \
            --num-workers 1 --ddp-backend=c10d --batch-size {batch_size} \
            --task docking --loss docking --arch docking \
            --conf-size {self.conf_size} \
            --dist-threshold 8.0 --recycling 4 \
            --path {self.model_dir}  \
            --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
            --log-interval 50 --log-format simple --required-batch-size-multiple 1 \
            --start_idx {start_idx} \
            --end_idx {end_idx}'

        os.system(cmd)
        
        
        

        lmdb_name = 'batch_data'
        pkl_file  = os.path.join(os.path.abspath(output_ligand_dir), lmdb_name + '.pkl')
        new_pkl_file  = os.path.join(os.path.abspath(output_ligand_dir), 'new_' + lmdb_name + '.pkl')
        lmdb_file = os.path.join(os.path.abspath(output_ligand_dir), lmdb_name+'.lmdb')
        new_lmdb_file = os.path.join(os.path.abspath(output_ligand_dir), 'new_' + lmdb_name + '.lmdb')
        
        

        with open(pkl_file, 'rb') as f:
            pkl_data = dill.load(f)


    
        pkl_data_dict = {}
        for dt in pkl_data:
            name = dt['pocket_name'][0].split('/')[-2]
            pkl_data_dict[name] = dt
        error_protein = OrderedSet()
        '''
        with open('protein_fail.txt') as f:
            for i in f:
                tg = i.split('/')[-2]
                error_protein.add(tg)
        '''
        data_dict = self.load_lmdb_data(lmdb_file, "mol_list", start_idx, end_idx)
        new_data_list = []
        new_pkl_list  = []
        name_list     = []
    
        current_name_list = list(pkl_data_dict.keys())       
        for name in current_name_list:
            try:
                assert pkl_data_dict[name]
                new_pkl_list.append(pkl_data_dict[name])
            except (KeyError, AssertionError) as e:
                print('KeyError:', name, e)
                continue                
            new_data_list.append(data_dict[name])
            name_list.append(name)
        if os.path.isfile(new_lmdb_file):
            os.remove(new_lmdb_file)
        self.write_lmdb(new_lmdb_file, new_data_list)

        if os.path.isfile(new_pkl_file):
            os.remove(new_pkl_file)
        
        with open(new_pkl_file, 'wb') as f:
            pkl_data = dill.dump(new_pkl_list, f)
        new_input_protein,  new_input_ligand, new_input_docking_grid, new_output_ligand_name = [], [], [], []

        assert len(input_protein) == len(input_ligand) and len(input_docking_grid) == len(output_ligand_name)

        for name in name_list:
            for i,j,k,l in zip(input_protein, input_ligand, input_docking_grid, output_ligand_name):
                nm = i.split('/')[-2]
                if nm == name and name in j and name in k and name in l:
                    new_input_protein.append(i)
                    new_input_ligand.append(j)
                    new_input_docking_grid.append(k)
                    new_output_ligand_name.append(l)
                    break
        
        
        return new_pkl_file, new_lmdb_file, new_input_protein,  new_input_ligand, new_input_docking_grid, new_output_ligand_name, output_ligand_dir, name_list


    def write_lmdb(self, outputfilename, mol_list, seed=42, result_dir="./results"):
        env_new = lmdb.open(
            outputfilename,
            subdir=False,
            readonly=False,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
            map_size=int(10e9),
        )
        txn_write = env_new.begin(write=True)

        ii = 0
        for inner_output in mol_list:
            txn_write.put(f"{ii}".encode("ascii"), pickle.dumps(inner_output))
            ii+=1

        txn_write.commit()
        env_new.close()




    def load_lmdb_data(self, lmdb_path, key, start_idx = 0, end_idx = None):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        with env.begin() as txn:
            cursor = txn.cursor()
            all_keys = [key for key, _ in cursor]
        total_len = len(all_keys)

        if end_idx is None or end_idx > total_len:
            end_idx = total_len

        _keys = all_keys[start_idx:end_idx]
        
        txn = env.begin()
        collects = []
        collects_dict = {}
        for idx in _keys:
            datapoint_pickled = txn.get(idx)
            data = pickle.loads(datapoint_pickled)
            name = data["pocket"].split('/')[-2]
            collects_dict[name] = data
            
            
        
        '''
        collects = []
        collects_dict = {}
        for idx in list(range(len(_keys)))[start_idx:end_idx]:
            datapoint_pickled = txn.get(f"{idx}".encode("ascii"))
            data = pickle.loads(datapoint_pickled)
            name = data["pocket"].split('/')[-2]
            collects_dict[name] = data
        '''
        return collects_dict
    def postprocess(self, output_pkl, output_lmdb, output_ligand_name, output_ligand_dir, output_ligand_dir2, input_ligand, input_protein):
        postprocessor = Processor.build_processors(self.mode, conf_size=self.conf_size)
        mol_list, smi_list, coords_predict_list, holo_coords_list, holo_center_coords_list, prmsd_score_list, fail_index, pocket_coords_list, cross_distance_list, holo_pocket_coords_list, ligand_emb_list, pocket_emb_list = postprocessor.postprocess_data_pre(output_pkl, output_lmdb)
        
        new_output_ligand_name = []
        for j in output_ligand_name:
            new_output_ligand_name.extend([j] * self.conf_size)

        mol_list = [n for i, n in enumerate(mol_list) if i not in fail_index]
        smi_list = [n for i, n in enumerate(smi_list) if i not in fail_index]
        coords_predict_list = [n for i, n in enumerate(coords_predict_list) if i not in fail_index]
        holo_center_coords_list = [n for i, n in enumerate(holo_center_coords_list) if i not in fail_index]
        prmsd_score_list = [n for i, n in enumerate(prmsd_score_list) if i not in fail_index]
        holo_coords_list = [n for i, n in enumerate(holo_coords_list) if i not in fail_index]
        pocket_coords_list = [n for i, n in enumerate(pocket_coords_list) if i not in fail_index]
        cross_distance_list = [n for i, n in enumerate(cross_distance_list) if i not in fail_index]
        holo_pocket_coords_list = [n for i, n in enumerate(holo_pocket_coords_list) if i not in fail_index]
        ligand_emb_list = [n for i, n in enumerate(ligand_emb_list) if i not in fail_index]
        pocket_emb_list = [n for i, n in enumerate(pocket_emb_list) if i not in fail_index]
        new_output_ligand_name = [n for i, n in enumerate(new_output_ligand_name) if i not in fail_index]
        output_ligand_name = list(OrderedSet(new_output_ligand_name))

        if not mol_list:
            return None
        else:
            output_ligand_sdf = postprocessor.get_sdf(mol_list, smi_list, coords_predict_list, holo_center_coords_list, prmsd_score_list, output_ligand_name, output_ligand_dir, output_ligand_dir2, holo_coords_list, pocket_coords_list, holo_pocket_coords_list, cross_distance_list, ligand_emb_list, pocket_emb_list, tta_times=self.conf_size)
            for i, j, k in zip(output_ligand_sdf, input_protein, input_ligand):
                tg_path = os.path.dirname(i)
                shutil.copy2(j, tg_path)
                shutil.copy2(k, tg_path)
            
                

            return output_ligand_sdf

    def predict_sdf(self, input_protein:str, 
                    input_ligand:str, input_docking_grid:str, 
                    output_ligand_name:str, output_ligand_dir:str, output_ligand_dir2,
                    batch_size:int = 4, start_idx = 0, end_idx = None, new_batch_data_name = None, gpu = 0):

        output_pkl, output_lmdb, input_protein,  input_ligand, input_docking_grid, output_ligand_name, output_ligand_dir, name_list = self.predict(input_protein, 
                                            input_ligand, 
                                            input_docking_grid, 
                                            output_ligand_name, 
                                            output_ligand_dir, 
                                            batch_size,
                                            start_idx, end_idx, new_batch_data_name, gpu = gpu)



        new_output_ligand_dir2 = []
        for name in name_list:
            for i in output_ligand_dir2:
                if name in i:
                    new_output_ligand_dir2.append(i)
                    break


        output_sdf = self.postprocess(output_pkl, 
                                    output_lmdb, 
                                    output_ligand_name, 
                                    output_ligand_dir,
                                    new_output_ligand_dir2,
                                    input_ligand,
                                    input_protein)
        return input_protein, input_ligand, input_docking_grid, output_sdf
    @classmethod
    def build_predictors(cls, model_dir, mode = 'batch_one2one', 
                         nthreads = 4, conf_size =1, 
                         cluster=False, use_current_ligand_conf=False, steric_clash_fix=False):
        return cls(model_dir, mode, nthreads, conf_size, 
                   cluster, use_current_ligand_conf=use_current_ligand_conf, steric_clash_fix=steric_clash_fix)    
