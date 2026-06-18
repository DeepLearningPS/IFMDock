# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
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
from functools import wraps

class FunctionTimeoutError(Exception):
    """自定义异常，当函数运行时间超过阈值时抛出"""
    #raise Exception('time out')
    pass

def measure_time(threshold):
    """装饰器，用于检测函数运行时间是否超过阈值
    
    Args:
        threshold (float): 时间阈值（秒）
    
    Returns:
        function: 装饰器函数
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)  # 执行原函数
            elapsed_time = time.time() - start_time
            
            if elapsed_time > threshold:
                raise FunctionTimeoutError(
                    f"Function '{func.__name__}' exceeded time threshold. "
                    f"Elapsed time: {elapsed_time:.2f}s, Threshold: {threshold}s"
                )
            return result
        return wrapper
    return decorator

# 使用示例
@measure_time(threshold=1.5)  # 设置阈值为1.5秒
def my_function():
    """模拟一个耗时函数"""
    time.sleep(2)  # 模拟耗时2秒的操作
    return "Done"



class Processor:
    def __init__(self, 
        mode:str='single', 
        nthreads:int=20, 
        conf_size:int=10, 
        cluster:bool=False, 
        main_atoms:List[str]=["N", "CA", "C", "O", "H"], 
        allow_pocket_atoms:List[str]=[['C', 'H', 'N', 'O', 'S']],
        use_current_ligand_conf:bool=False
    ):
        self.mode = mode
        self.nthreads = nthreads
        self.conf_size = conf_size
        self.cluster = cluster
        self.main_atoms = main_atoms
        self.allow_pocket_atoms = allow_pocket_atoms
        if self.mode in ['batch_one2one', 'batch_one2many']:
            self.lmdb_name = 'batch_data'
        self.use_current_ligand_conf = use_current_ligand_conf

    def preprocess(self, input_protein:str, input_ligand, input_docking_grid:str, output_ligand_name:str, out_lmdb_dir:str):
        seed = 42 
        if self.mode=='single':
            supp = Chem.SDMolSupplier(input_ligand)
            mol = [mol for mol in supp if mol][0]
            ori_smiles = Chem.MolToSmiles(mol)
            smiles_list = [ori_smiles]
            input_protein = [input_protein]
            input_ligand = [input_ligand]
            input_docking_grid = [input_docking_grid]
        elif self.mode in ['batch_one2one', 'batch_one2many']:
            if self.mode == 'batch_one2many':
                input_protein = [input_protein] * len(input_ligand)
            smiles_list = []
            error_count = 0
            error_name_list = []
            error_index_list = []
            for i in range(len(input_ligand)):
                try:
                    supp = Chem.SDMolSupplier(input_ligand[i])
                    mol = [mol for mol in supp if mol][0]
                    ori_smiles = Chem.MolToSmiles(mol)
                    mol  = Chem.RemoveHs(mol)
                    ligand_pos = np.array(mol.GetConformer(0).GetPositions())
                    smiles_list.append(ori_smiles)
                except Exception as e:
                    try:
                        #不标准化
                        supp = Chem.SDMolSupplier(input_ligand[i], sanitize=False)
                        mol = [mol for mol in supp if mol][0]
                        ori_smiles = Chem.MolToSmiles(mol)
                        
                        #mol  = Chem.RemoveHs(mol)
                        #try:
                            #mol  = Chem.RemoveHs(mol)
                        #except Exception as e:
                            #print('not RemoveHs')
                            
                        #mol_h = Chem.AddHs(mol, addCoords=True)
                        # 再去氢（此时结构更“合法”）
                        mol = Chem.RemoveHs(mol, sanitize=False)  #去除氢时，sanitize = False
                            
                        ligand_pos = np.array(mol.GetConformer(0).GetPositions())
                        smiles_list.append(ori_smiles)
                        input_ligand[i] = os.path.join(os.path.dirname(input_ligand[i]), 'origin_' + input_ligand[i].split('/')[-2] + '_ligand.sdf')
                    except Exception as e:
                        try:
                            mol = Chem.MolFromMol2File(os.path.join(os.path.dirname(input_ligand[i]), input_ligand[i].split('/')[-2] + '_ligand.mol2'), sanitize=False)
                            ori_smiles = Chem.MolToSmiles(mol)
                            mol  = Chem.RemoveHs(mol)
                            ligand_pos = np.array(mol.GetConformer(0).GetPositions())
                            smiles_list.append(ori_smiles)
                            input_ligand[i] = os.path.join(os.path.dirname(input_ligand[i]), input_ligand[i].split('/')[-2] + '_ligand.mol2')
                        except Exception as e:
                            error_count += 1
                            error_name_list.append(input_ligand[i].split('/')[-2])
                            #print(e)
                            #print('error ligand:', input_ligand[i])
                            error_index_list.append(i)


            with open('error_ligand.txt', 'a') as f:
                for name in error_name_list:
                    f.write(name + '\n')

                
            #print('lack error_count:', error_count)
            #exit()
        #删除无法生成smiles的数据，要同步删除所有的数据对应的索引
        print('error_index_list:', error_index_list)
        print('output_ligand_name:', output_ligand_name)
        for index_i in error_index_list:
            print('index_i:', index_i)
            del output_ligand_name[index_i]
            del input_protein[index_i]
            del input_ligand[index_i]
            del input_docking_grid[index_i]
            #del out_lmdb_dir[index_i]
            
        lmdb_name = self.write_lmdb(output_ligand_name, smiles_list, input_protein, input_ligand, input_docking_grid, seed=seed, result_dir=out_lmdb_dir)
        return lmdb_name

    def single_conf_gen(self, tgt_mol, num_confs=1000, seed=42, removeHs=True):
        mol = copy.deepcopy(tgt_mol)
        mol = Chem.AddHs(mol)
        allconformers = AllChem.EmbedMultipleConfs(
            mol, numConfs=num_confs, randomSeed=seed, clearConfs=True
        )
        sz = len(allconformers)
        for i in range(sz):
            try:
                AllChem.MMFFOptimizeMolecule(mol, confId=i)
            except:
                continue
        if removeHs:
            mol = Chem.RemoveHs(mol)
        return mol

    def single_conf_gen_no_MMFF(self, tgt_mol, num_confs=1000, seed=42, removeHs=True):
        mol = copy.deepcopy(tgt_mol)
        mol = Chem.AddHs(mol)
        allconformers = AllChem.EmbedMultipleConfs(
            mol, numConfs=num_confs, randomSeed=seed, clearConfs=True
        )
        if removeHs:
            mol = Chem.RemoveHs(mol)
        return mol

    @measure_time(threshold=10)
    def clustering_coords_copy(self, mol, M=1000, N=100, seed=42, cluster=False, removeHs=True, gen_mode='mmff'):
        # N是构象数量，M是每一个构象生成多少M个rdkit构象，这里是默认是10倍的构象数量，如果想生成40个构象，则生成40*10个rdkit构象，然后优化，找最好的，作为配体的初始坐标

        try:
            rdkit_coords_list = []
            if not cluster:
                M = N
            if gen_mode == 'mmff':
                rdkit_mol = self.single_conf_gen(mol, num_confs=M, seed=seed, removeHs=removeHs)
            elif gen_mode == 'no_mmff':
                rdkit_mol = self.single_conf_gen_no_MMFF(mol, num_confs=M, seed=seed, removeHs=removeHs)
            noHsIds = [
                rdkit_mol.GetAtoms()[i].GetIdx()
                for i in range(len(rdkit_mol.GetAtoms()))
                if rdkit_mol.GetAtoms()[i].GetAtomicNum() != 1
            ]
            ### exclude hydrogens for aligning
            AlignMolConformers(rdkit_mol, atomIds=noHsIds) #对齐构象，只作用于重原子
            sz = len(rdkit_mol.GetConformers()) #生成的构象数量
            for i in range(sz):
                _coords = rdkit_mol.GetConformers()[i].GetPositions().astype(np.float32)
                rdkit_coords_list.append(_coords)

            ### exclude hydrogens for clustering, pick closest to centroid:
            
            if cluster:
                # (num_confs, num_atoms, 3)
                rdkit_coords = np.array(rdkit_coords_list)[:, noHsIds]
                # (num_confa, num_atoms, 3) -> (num_confs, num_atoms*3)
                rdkit_coords_flatten = rdkit_coords.reshape(sz, -1)
                kmeans = KMeans(n_clusters=N, random_state=seed).fit(rdkit_coords_flatten) #分成N块，然后每一块取均值
                # (num_clusters, num_atoms, 3)
                center_coords = kmeans.cluster_centers_.reshape(N, -1, 3)
                # (num_cluster, num_confs)
                cdist = ((center_coords[:, None] - rdkit_coords[None, :])**2).sum(axis=(-1, -2))
                # (num_confs,)
                argmin = np.argmin(cdist, axis=-1)
                coords_list = [rdkit_coords_list[i] for i in argmin]
            else:
                coords_list = rdkit_coords_list
            

            #总是有问题，不聚类了
            #print('len(coords_list):', len(coords_list))
            if len(coords_list) != N:
                coords_list = coords_list + [coords_list[0]] * (N - len(coords_list)) 
        except Exception as e:
            #print(e)
            cluster = False
            rdkit_coords_list = []
            if not cluster:
                M = N
            if gen_mode == 'mmff':
                rdkit_mol = self.single_conf_gen(mol, num_confs=M, seed=seed, removeHs=removeHs)
            elif gen_mode == 'no_mmff':
                rdkit_mol = self.single_conf_gen_no_MMFF(mol, num_confs=M, seed=seed, removeHs=removeHs)
            noHsIds = [
                rdkit_mol.GetAtoms()[i].GetIdx()
                for i in range(len(rdkit_mol.GetAtoms()))
                if rdkit_mol.GetAtoms()[i].GetAtomicNum() != 1
            ]
            ### exclude hydrogens for aligning
            AlignMolConformers(rdkit_mol, atomIds=noHsIds) #对齐构象，只作用于重原子
            sz = len(rdkit_mol.GetConformers()) #生成的构象数量
            for i in range(sz):
                _coords = rdkit_mol.GetConformers()[i].GetPositions().astype(np.float32)
                rdkit_coords_list.append(_coords)

            ### exclude hydrogens for clustering, pick closest to centroid:
            
            if cluster:
                # (num_confs, num_atoms, 3)
                rdkit_coords = np.array(rdkit_coords_list)[:, noHsIds]
                # (num_confa, num_atoms, 3) -> (num_confs, num_atoms*3)
                rdkit_coords_flatten = rdkit_coords.reshape(sz, -1)
                kmeans = KMeans(n_clusters=N, random_state=seed).fit(rdkit_coords_flatten) #分成N块，然后每一块取均值
                # (num_clusters, num_atoms, 3)
                center_coords = kmeans.cluster_centers_.reshape(N, -1, 3)
                # (num_cluster, num_confs)
                cdist = ((center_coords[:, None] - rdkit_coords[None, :])**2).sum(axis=(-1, -2))
                # (num_confs,)
                argmin = np.argmin(cdist, axis=-1)
                coords_list = [rdkit_coords_list[i] for i in argmin]
            else:
                coords_list = rdkit_coords_list
            

            #总是有问题，不聚类了
            #print('len(coords_list):', len(coords_list))
            if len(coords_list) != N:
                coords_list = coords_list + [coords_list[0]] * (N - len(coords_list)) 

        return coords_list



    @measure_time(threshold=100)
    def clustering_coords(self, mol, M=1000, N=100, seed=42, cluster=False, removeHs=True, gen_mode='mmff'):
        # N是构象数量，M是每一个构象生成多少M个rdkit构象，这里是默认是10倍的构象数量，如果想生成40个构象，则生成40*10个rdkit构象，然后优化，找最好的，作为配体的初始坐标
        rdkit_coords_list = []
        if not cluster:
            M = N
        if gen_mode == 'mmff':
            rdkit_mol = self.single_conf_gen(mol, num_confs=M, seed=seed, removeHs=removeHs)
        elif gen_mode == 'no_mmff':
            rdkit_mol = self.single_conf_gen_no_MMFF(mol, num_confs=M, seed=seed, removeHs=removeHs)
        noHsIds = [
            rdkit_mol.GetAtoms()[i].GetIdx()
            for i in range(len(rdkit_mol.GetAtoms()))
            if rdkit_mol.GetAtoms()[i].GetAtomicNum() != 1
        ]
        ### exclude hydrogens for aligning
        AlignMolConformers(rdkit_mol, atomIds=noHsIds) #对齐构象，只作用于重原子
        sz = len(rdkit_mol.GetConformers()) #生成的构象数量
        for i in range(sz):
            _coords = rdkit_mol.GetConformers()[i].GetPositions().astype(np.float32)
            rdkit_coords_list.append(_coords)

        ### exclude hydrogens for clustering, pick closest to centroid:
        
        if cluster:
            # (num_confs, num_atoms, 3)
            rdkit_coords = np.array(rdkit_coords_list)[:, noHsIds]
            # (num_confa, num_atoms, 3) -> (num_confs, num_atoms*3)
            rdkit_coords_flatten = rdkit_coords.reshape(sz, -1)
            kmeans = KMeans(n_clusters=N, random_state=seed).fit(rdkit_coords_flatten) #分成N块，然后每一块取均值
            # (num_clusters, num_atoms, 3)
            center_coords = kmeans.cluster_centers_.reshape(N, -1, 3)
            # (num_cluster, num_confs)
            cdist = ((center_coords[:, None] - rdkit_coords[None, :])**2).sum(axis=(-1, -2))
            # (num_confs,)
            argmin = np.argmin(cdist, axis=-1)
            coords_list = [rdkit_coords_list[i] for i in argmin]
        else:
            coords_list = rdkit_coords_list
        

        #总是有问题，不聚类了
        ##print('len(coords_list):', len(coords_list))
        if len(coords_list) != N:
            coords_list = coords_list + [coords_list[0]] * (N - len(coords_list)) 
        

        return coords_list

    def find_residues_in_pocket(self, pocket: dict, pdf):
        """
        Given a pocket config and a residue df, 
        return a list of residues that are in the pocket
        """
        def _get_vertex(pocket: dict, axis: str) -> tuple:
            """
            Return the minimum and maximum values of the given axis

            Args:
            pocket (dict): pocket config
            axis (str): ["x", "y", "z"]

            Returns:
            A tuple of floats.
            """
            return (
                pocket["center_{}".format(axis)] \
                    - pocket["size_{}".format(axis)] / 2,
                pocket["center_{}".format(axis)] \
                    + pocket["size_{}".format(axis)] / 2
                )
        min_x, max_x = _get_vertex(pocket, "x")
        min_y, max_y = _get_vertex(pocket, "y")
        min_z, max_z = _get_vertex(pocket, "z")
        min_array = np.array([min_x, min_y, min_z]).reshape(1,3)
        max_array = np.array([max_x, max_y, max_z]).reshape(1,3)
        patoms, pcoords, residues = [], np.empty((0,3)), []
        for i in range(len(pdf)):
            atom_info = pdf.iloc[i]
            _rescoor = np.array(atom_info[['x_coord','y_coord','z_coord']].values).reshape(-1,3)
            mapping = (_rescoor > min_array) & (_rescoor < max_array)
            if (mapping.sum(-1) == 3).sum() > 0:
                patoms += [atom_info['atom_name']]
                pcoords = np.concatenate((pcoords, _rescoor), axis=0)
                residues += [str(atom_info['chain_id'])+str(atom_info['residue_number'])]
        return patoms, pcoords, residues

    def extract_pocket(self, input_protein, input_docking_grid):
        pmol = PandasPdb().read_pdb(input_protein) #提前去氢
        
        #去除氢原子
        ##print('pmol.df:', pmol.df)
        '''
        atom_df = pmol.df['ATOM']
        hetatm_df = pmol.df['HETATM']

        # 原子总数 = ATOM + HETATM 行数
        total_atoms = len(atom_df) + len(hetatm_df)
        #print(f"原子总数1：{total_atoms}")
        '''
        
        
        
        
        df_no_h = pmol.df['ATOM'][pmol.df['ATOM']['element_symbol'] != 'H']
        pmol.df['ATOM'] = df_no_h
        if 'HETATM' in pmol.df:
            pmol.df['HETATM'] = pmol.df['HETATM'][pmol.df['HETATM']['element_symbol'] != 'H']

        '''
        atom_df = pmol.df['ATOM']
        hetatm_df = pmol.df['HETATM']

        # 原子总数 = ATOM + HETATM 行数
        total_atoms = len(atom_df) + len(hetatm_df)
        #print(f"原子总数2：{total_atoms}")
        
        raise Exception('test')
        '''
        
        with open(input_docking_grid, "r") as file:
            box_dict = json.load(file)

        pdf = pmol.df['ATOM']
        patoms, pcoords, residues = self.find_residues_in_pocket(box_dict, pdf) #根据对接框来画范围的
        def _filter_pocketatoms(atom):
            if atom[:2] in ['Cd','Cs', 'Cn', 'Ce', 'Cm', 'Cf', 'Cl', 'Ca', 'Cr', 'Co', 'Cu', 'Nh', 'Nd', 'Np', 'No', 'Ne', 'Na', 'Ni', \
                'Nb', 'Os', 'Og', 'Hf', 'Hg', 'Hs', 'Ho', 'He', 'Sr', 'Sn', 'Sb', 'Sg', 'Sm', 'Si', 'Sc', 'Se']:
                return None
            if atom[0] >= '0' and atom[0] <= '9':
                return _filter_pocketatoms(atom[1:])
            if atom[0] in ['Z','M','P','D','F','K','I','B']:
                return None
            if atom[0] in self.allow_pocket_atoms:
                return atom
            return atom

        atoms, index, residues_tmp = [], [], []
        for i,a in enumerate(patoms):
            output = _filter_pocketatoms(a)
            if output is not None:
                index.append(True)
                atoms.append(output)
                residues_tmp.append(residues[i])
            else:
                index.append(False)
        coordinates = pcoords[index].astype(np.float32)
        residues = residues_tmp
        patoms = atoms
        pcoords = [coordinates]
        side = [0 if a in self.main_atoms else 1 for a in patoms]
        return patoms, pcoords, residues, side, box_dict

    def parser(self, content):
        try:
            smiles, input_protein, input_ligand, input_docking_grid, seed = content
            name = input_protein.split('/')[-2]
            
            tg = os.path.basename(input_protein).split('_')
            if len(tg) == 2:
                complex_name = tg[0]
            elif len(tg) == 3:
                complex_name = tg[1]


            #有一个问题，同一个蛋白文件，为啥40个构象得到的蛋白原子不一样，只有部分是交集呢？
            #现在有一个问题，这里只是生成1个蛋白口袋和40个配体的3d结构，40个的蛋白口袋是从哪里的？
            #也就是，40个蛋白口袋不是来自于这里的，不在data_lmdb.pkl形成这一部分
            #提前去除氢原子
            patoms, pcoords, residues, side, config = self.extract_pocket(input_protein, input_docking_grid) #从这里开始，提取出来的是口袋蛋白了，看一下提取的范围是多少ai
            #print('len(patoms):', len(patoms))
            #print('pcoords num:', len(pcoords[0])) #1
            #print('pcoords[0].shape:', pcoords[0].shape)
            #print('pcoords[0][:2]:', pcoords[0][:2])
            #print('pcoords[0]_centor:', np.mean(pcoords[0], axis = 0)) #蛋白是空，啥情况？

            #raise Exception('test')
            #pcoords[0]_centor: [-21.913183  12.446733  26.25213 ]
            #在这里是没有改变蛋白坐标的，还没减去质心什么的呢，在/mnt/home/fanzhiguang/47/unimol_docking_v2/unimol/data/normalize_dataset.py文件中，有对配体和蛋白的坐标的处理
            #即配体和蛋白坐标都减去了蛋白质心
            #raise Exception('stop')
            #pcoords num: 1
            #pcoords[0].shape: (128, 3) #这是输入的蛋白维度，但是我们得到的是(136, 3), 多出几个原子，这些是H原子，他们减去蛋白质心后对应的坐标是(0,0,0), 因此我们根据这个信息，要把蛋白
            #为0的坐标去掉
            #pcoords[0][:2]: [[-18.67   19.158  23.089]
            #[-17.539  18.199  22.696]]

            #口袋范围是根据对接框来算的，即分子的xyz轴方向的最大距离 + 10ai， 一般情况下分子的最大长度在10ai内，
            #所以提取的范围大概在10ai以内 = （xyz轴方向的最大距离 + 10ai） = （10 + 10）/ 2 = 10

            
            #print('input_protein:', input_protein) #input_protein: /mnt/home/fanzhiguang/47/CrossDocked2020/data/pdbbind2020_r10/pdbbind_connect/4u5o/4u5o_protein.pdb
            #print('complex_name:', complex_name) #complex_name: 4u5o
            #raise Exception('stop')

            # get ground truth conformation and generate ligand conformation, 默认读取sdf时，是需要凯库勒化的，但可能会报错，报错则使用原始文件或mol2
            if 'origin' in input_ligand:
                supp = Chem.SDMolSupplier(input_ligand, sanitize=False)
                mol = [Chem.RemoveHs(mol) for mol in supp if mol][0]
            elif '.mol2' in input_ligand:
                mol = Chem.MolFromMol2File(input_ligand, sanitize=False)
            else:
                supp = Chem.SDMolSupplier(input_ligand)
                mol = [Chem.RemoveHs(mol) for mol in supp if mol][0]
            
            ##print('ligand coords.shape:', mol.GetConformer().GetPositions().astype(np.float32).shape) #ligand coords.shape: (13, 3)， 配体也添加了H原子，我们也需要去掉
            #raise Exception('stop')
            self.use_current_ligand_conf = False #为了生成ecdock数据集，这里设置True
            if self.use_current_ligand_conf: #默认是false，所以才有了之后可能报错的可能，当然确实不应该使用参考的配体了。如果只是为了获取相互作用距离，则这个参数要开启，更精确
                #当然也可以使用下面的方法，随机rdkit, 此时设置构象数量为1
                return pickle.dumps(
                    {
                        "atoms": [atom.GetSymbol() for atom in mol.GetAtoms()],
                        "coordinates": [mol.GetConformer().GetPositions().astype(np.float32)],
                        "mol_list": [mol],
                        "pocket_atoms": patoms,
                        "pocket_coordinates": pcoords,
                        "side": side,
                        "residue": residues,
                        "config": config,
                        "holo_coordinates": [mol.GetConformer().GetPositions().astype(np.float32)],
                        "holo_mol": mol,
                        "holo_pocket_coordinates": pcoords,
                        "smi": smiles,
                        "pocket": input_protein,
                        "flag": 'success',
                        "name": name
                    },
                    protocol=-1,
                    ), True, input_ligand, complex_name, pcoords
            
            
            #mol = Chem.AddHs(mol) #addCoords=True
            #if mol == None:
                #raise Exception("mol is None")
            smiles = Chem.MolToSmiles(mol)
            latoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
            holo_coordinates = [mol.GetConformer().GetPositions().astype(np.float32)]
            holo_mol = mol
            N = self.conf_size # 训练集构象设置成1，不需要生成多个？
            M = self.conf_size * 10 #对于每一个分子，使用rdkit生成10个构象，然后优化，之后聚类找最好的，作为初始配体坐标
            mol_list = [mol] * N #值得注意的是holo表示参考的配体，这里的mol_list虽然是mol的多分复制，但是其坐标是通过其他方法获取的，并不是真实的
            #出错的一个重要原因是在生成40个rdkit构象时，实际生成的数量可能会小于指定的数量，这样就导致批量和我们所需要的不一致，导致了数据的填充，因此如果发现数量对不上，则重新生成
            coordinate_list = []
            try:
                coordinate_list = self.clustering_coords(mol, M=M, N=N, seed=seed, cluster = True, removeHs=True, gen_mode='mmff')
            except (Exception, FunctionTimeoutError) as e: 
                try:
                    coordinate_list = self.clustering_coords(mol, M=M, N=N, seed=seed, cluster = False, removeHs=True, gen_mode='mmff')
                except (Exception, FunctionTimeoutError) as e: 
                    try:
                        coordinate_list = self.clustering_coords(mol, M=1, N=1, seed=seed, cluster = False, removeHs=True, gen_mode='no_mmff')
                        coordinate_list = coordinate_list * N
                    except (Exception, FunctionTimeoutError) as e:
                            coordinate_list = holo_coordinates * N
                
            #数据少的原因是：/data/fan_zg/MDocking/Docking_baseline/unimol_docking_v2/unimol/tasks/docking_pose_v2.py
            
            '''
            try:
                coordinate_list = self.clustering_coords(mol, M=M, N=N, seed=seed, cluster = True, removeHs=True, gen_mode='mmff')
            except (Exception, FunctionTimeoutError) as e:
                try:
                    coordinate_list = self.clustering_coords(mol, M=M, N=N, seed=seed, cluster = False, removeHs=True, gen_mode='no_mmff')
                except (Exception, FunctionTimeoutError) as e:
                    try:
                        coordinate_list = self.clustering_coords(mol, M=1, N=1, seed=seed, cluster = False, removeHs=True, gen_mode='no_mmff')
                        coordinate_list = coordinate_list * N
                    except (Exception, FunctionTimeoutError) as e:
                        #print(e)
                        coordinate_list = holo_coordinates * N
            '''    

            assert len(coordinate_list) == N
            return pickle.dumps(
                {
                    "atoms": latoms,
                    "coordinates": coordinate_list, #rdkit坐标
                    "mol_list": mol_list,
                    "pocket_atoms": patoms,
                    "pocket_coordinates": pcoords,
                    "side": side,
                    "residue": residues,
                    "config": config,
                    "holo_coordinates": holo_coordinates, #ground truth
                    "holo_mol": holo_mol, #ground truth
                    "holo_pocket_coordinates": pcoords, #ground truth
                    "smi": smiles,
                    "pocket": input_protein,
                    "flag": 'success',
                    "name": name
                },
                protocol=-1,
                ), True, input_ligand, complex_name, pcoords
        except (Exception, FunctionTimeoutError) as e:
            #print(e)
            pcoords = 0
            return None, False,  input_ligand, complex_name, pcoords


                



    def write_lmdb(self, output_ligand_name, smiles_list, input_protein, input_ligand, input_docking_grid, seed=42, result_dir="./results"):
        #print('result_dir:', result_dir)
        os.makedirs(result_dir, exist_ok=True)
        if self.mode == 'single':
            outputfilename = os.path.join(result_dir, output_ligand_name + ".lmdb")
        elif self.mode in ['batch_one2one', 'batch_one2many']:
            outputfilename = os.path.join(result_dir, self.lmdb_name + ".lmdb")
            output_ligand_name = self.lmdb_name
        try:
            os.remove(outputfilename)
        except:
            pass
        env_new = lmdb.open(
            outputfilename,
            subdir=False,
            readonly=False,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
            map_size=int(100*(1024*1024*1024)), # 100GB
        )
        txn_write = env_new.begin(write=True)
        #print("Start preprocessing data...")
        #print(f'Number of ligands: {len(smiles_list)}')
        fail_file_list = []
        seed = [seed] * len(input_ligand)
        content_list = zip(smiles_list, input_protein, input_ligand, input_docking_grid, seed)

        #保存一下unimol使用的蛋白原子
        
        #多进程
        '''
        pcoords_dict = {}
        with Pool(self.nthreads) as pool:
            ii = 0
            failed_num = 0
            for inner_output, flag, file_ligand, complex_name, pcoords in tqdm(pool.imap(self.parser, content_list)): #遍历content_list
                #print('flag:', flag)
                if flag is True:
                    txn_write.put(f"{ii}".encode("ascii"), inner_output)
                    ii+=1
                    pcoords_dict[complex_name] = pcoords
                elif flag is False: 
                    #我们直接从源头过滤掉
                    fail_file_list.append(file_ligand)
                    failed_num += 1
                    continue
                    #txn_write.put(f"{i}".encode("ascii"), inner_output) #失败依旧添加数据，在其它地方过滤掉
                    #i+=1 #要添加上
                    #failed_num += 1
            txn_write.commit()
            env_new.close()
        '''
        
        #多进程
        pcoords_dict = {}
        with Pool(100) as pool:
            ii = 0
            failed_num = 0
            for inner_output, flag, file_ligand, complex_name, pcoords in tqdm(pool.imap(self.parser, content_list),total=len(smiles_list)): #遍历content_list
                ##print('flag:', flag)
                if flag is True:
                    txn_write.put(f"{ii}".encode("ascii"), inner_output)
                    ii+=1
                    pcoords_dict[complex_name] = pcoords
                elif flag is False: 
                    #我们直接从源头过滤掉
                    fail_file_list.append(file_ligand)
                    failed_num += 1
                    continue
                    #txn_write.put(f"{i}".encode("ascii"), inner_output) #失败依旧添加数据，在其它地方过滤掉
                    #i+=1 #要添加上
                    #failed_num += 1
            txn_write.commit()
            env_new.close()
        
        
        #单进程
        '''
        pcoords_dict = {}
        ii = 0
        failed_num = 0
        for smiles_i, input_protein_i, input_ligand_i, input_docking_grid_i, seed_i in tqdm(content_list, total=len(smiles_list)): #遍历content_list
            #print('ii:', ii)
            try:
                inner_output, flag, file_ligand, complex_name, pcoords = self.parser((smiles_i, input_protein_i, input_ligand_i, input_docking_grid_i, seed_i))
                #print('flag:', flag)
            
                txn_write.put(f"{ii}".encode("ascii"), inner_output)
                ii+=1
                pcoords_dict[complex_name] = pcoords
            except Exception as e: 
                #print('data deal fail:', e)
                #我们直接从源头过滤掉
                fail_file_list.append(input_ligand_i)
                failed_num += 1
                continue
                #txn_write.put(f"{i}".encode("ascii"), inner_output) #失败依旧添加数据，在其它地方过滤掉
                #i+=1 #要添加上
                #failed_num += 1
            
        txn_write.commit()
        env_new.close()
        '''

        return output_ligand_name #失败就记录一下哪一个文件失败了

    def load_lmdb_data(self, lmdb_path, key):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        txn = env.begin()
        _keys = list(txn.cursor().iternext(values=False))
        collects = []
        for idx in range(len(_keys)):
            datapoint_pickled = txn.get(f"{idx}".encode("ascii"))
            data = pickle.loads(datapoint_pickled)
            collects.append(data[key])
        return collects

    def postprocess_data_pre_copy(self, predict_file, lmdb_file):
        old_mol_list = self.load_lmdb_data(lmdb_file, "mol_list")
        #mol_list = [Chem.RemoveHs(mol) for items in mol_list for mol in items]
        fail_index = []
        mol_list = []
        num = 0
        #print('mol_list1:', old_mol_list) #这里是正常的
        for items in old_mol_list:
            for mol in items:
                try:
                    Chem.RemoveHs(mol)
                    mol_list.append(mol)
                except Exception as e:
                    #print(f"Failed to remove Hs from mol: {mol}, Error: {e}")
                    fail_index.append(num)
                    mol_list.append(None)
                
                num += 1

        #print('mol_list2:', mol_list) #这里是正常的

        #print('predict_file:', predict_file)
        predict = pd.read_pickle(predict_file) #这里加载的数据，已经把错误的给过滤了，不能过滤要保留，这此处过滤，防止顺序错误, 问题在于处理这个pkl文件时出错了，保存的全是None
        ##print('predict:', predict)
        #print('predict.keys():', predict[0].keys())
        '''
        predict.keys(): dict_keys(['loss', 'cross_distance_loss', 'distance_loss', 'coord_loss', 'prmsd_loss', 'prmsd_score', 'bsz', 'sample_size', 
        'coord_predict', 'coord_target', 'smi_name', 'pocket_name', 'atoms', 'pocket_atoms', 'coordinates', 'holo_coordinates', 'pocket_coordinates', 
        'holo_center_coordinates'])
        '''
        smi_list, pocket_list, coords_predict_list, holo_coords_list, holo_center_coords_list, prmsd_score_list = [],[],[],[],[],[]

        #新增，这些保存下来，用于我们的下游任务，相互作用预测
        cross_distance_list = []
        pocket_coords_list = []
        holo_pocket_coords_list = []

        #print('predict num:', len(predict)) #4

        for batch in predict:

            if batch == None:
                print('batch == None, skip') #这是跳过这个批量，感觉不行，另外一个批量里面有成功的，也有失败的，要区分开来，或者把批量设置成1
            else:
                #print("batch['atoms']:", batch['atoms']) #二维向量
                #print("batch['atoms']:", len(batch['atoms'])) #1
                #print("batch['atoms'].shape:", batch['atoms'].shape) #1
                sz = batch['atoms'].size(0) #1
                #print('batch num:', len(predict))
                #print('sz:', sz)
                #raise Exception("Invalid batch") #

                '''
                batch['atoms']: 40
                batch['atoms'].shape: torch.Size([40, 16])
                batch num: 2
                sz: 40
                '''
                
                for i in range(sz):
                    try:
                        smi_list.append(batch['smi_name'][i])
                        pocket_list.append(batch['pocket_name'][i])
                        prmsd_score_list.append(batch['prmsd_score'][i].numpy().astype(np.float32))

                        cross_distance_list.append(batch['cross_distance'][i].numpy().astype(np.float32))
                        pocket_coords_list.append(batch['pocket_coordinates'][i].numpy().astype(np.float32))
                        holo_pocket_coords_list.append(batch['holo_pocket_coordinates'][i].numpy().astype(np.float32))
                        ##print('holo_pocket_coords_list:', holo_pocket_coords_list)
                        #raise Exception('stop')
                        
                        token_mask = batch['atoms'][i]>2 #有效的词表下标是从3开始的，前3个是标志位？是这个意思？不清楚，只是知道这里用于剔除填充的标志位，对于配体总共3个

                        holo_coordinates = batch['holo_coordinates'][i]
                        holo_coordinates = holo_coordinates[token_mask,:]
                        holo_coordinates = holo_coordinates.numpy().astype(np.float32)

                        coord_predict = batch['coord_predict'][i]
                        coord_predict = coord_predict[token_mask,:]
                        coord_predict = coord_predict.numpy().astype(np.float32)

                        holo_center_coordinates = batch["holo_center_coordinates"][i][:3]
                        holo_center_coordinates.numpy().astype(np.float32)

                        holo_center_coords_list.append(holo_center_coordinates)        
                        coords_predict_list.append(coord_predict)
                        holo_coords_list.append(holo_coordinates)
                    except Exception as e:
                        #exit()
                        #print('error,skip:', e)
                        smi_list.append(None) 
                        pocket_list.append(None)
                        coords_predict_list.append(None)
                        holo_coords_list.append(None)
                        holo_center_coords_list.append(None) 
                        prmsd_score_list.append(None)


        return mol_list, smi_list, coords_predict_list, holo_coords_list, holo_center_coords_list, prmsd_score_list, fail_index, pocket_coords_list, cross_distance_list, holo_pocket_coords_list




    def postprocess_data_pre(self, predict_file, lmdb_file):
        #仅仅用于生成相互作用信息，因为在损失哪里已经去除填充的0了，所以这里就不再重复去除，否则导致数据长度不一样

        old_mol_list = self.load_lmdb_data(lmdb_file, "mol_list")
        #mol_list = [Chem.RemoveHs(mol) for items in mol_list for mol in items]
        fail_index = []
        mol_list = []
        num = 0
        ##print('mol_list1:', old_mol_list) #这里是正常的
        for items in old_mol_list:
            for mol in items:
                try:
                    new_mol = Chem.RemoveHs(mol)
                    mol_list.append(new_mol)
                except Exception as e:
                    #print(f"Failed to remove Hs from mol: {mol}, Error: {e}")
                    fail_index.append(num)
                    mol_list.append(None)
                
                num += 1

        ##print('mol_list2:', mol_list) #这里是正常的

        #print('predict_file:', predict_file)
        predict = pd.read_pickle(predict_file) #这里加载的数据，已经把错误的给过滤了，不能过滤要保留，这此处过滤，防止顺序错误, 问题在于处理这个pkl文件时出错了，保存的全是None
        ##print('predict:', predict)

        #print(f'assert {len(old_mol_list)} == {len(predict)}') #2934 == 4797
        assert len(old_mol_list) == len(predict)

        #print('predict.keys():', predict[0].keys())
        '''
        predict.keys(): dict_keys(['loss', 'cross_distance_loss', 'distance_loss', 'coord_loss', 'prmsd_loss', 'prmsd_score', 'bsz', 'sample_size', 
        'coord_predict', 'coord_target', 'smi_name', 'pocket_name', 'atoms', 'pocket_atoms', 'coordinates', 'holo_coordinates', 'pocket_coordinates', 
        'holo_center_coordinates'])
        '''
        smi_list, pocket_list, coords_predict_list, holo_coords_list, holo_center_coords_list, prmsd_score_list = [],[],[],[],[],[]

        #新增，这些保存下来，用于我们的下游任务，相互作用预测
        cross_distance_list = []
        pocket_coords_list = []
        holo_pocket_coords_list = []
        
        ligand_emb = []
        pocket_emb = []

        #print('predict num:', len(predict)) #4

        for batch in predict:

            if batch == None:
                #print('batch == None, skip') #这是跳过这个批量，感觉不行，另外一个批量里面有成功的，也有失败的，要区分开来，或者把批量设置成1
                raise Exception('error, None')
            else:
                #print("batch['atoms']:", batch['atoms']) #二维向量
                #print("batch['atoms']:", len(batch['atoms'])) #1
                #print("batch['atoms'].shape:", batch['atoms'].shape) #1
                sz = batch['atoms'].size(0) #1
                #print('batch num:', len(predict))
                #print('sz:', sz)
                #raise Exception("Invalid batch") #

                '''
                batch['atoms']: 40
                batch['atoms'].shape: torch.Size([40, 16])
                batch num: 2
                sz: 40
                '''
                #遍历批量内部的每一条数据
                for i in range(sz):
                    try:
                        smi_list.append(batch['smi_name'][i])
                        pocket_list.append(batch['pocket_name'][i])
                        prmsd_score_list.append(batch['prmsd_score'][i].numpy().astype(np.float32))

                        cross_distance_list.append(batch['cross_distance'][i].numpy().astype(np.float32))
                        pocket_coords_list.append(batch['pocket_coordinates'][i].numpy().astype(np.float32))
                        holo_pocket_coords_list.append(batch['holo_pocket_coordinates'][i].numpy().astype(np.float32))
                        #ligand_emb.append(batch['ligand_emb'][i].numpy())
                        #pocket_emb.append(batch['pocket_emb'][i].numpy())
                        ##print('holo_pocket_coords_list:', holo_pocket_coords_list)
                        #raise Exception('stop')
                        
                        #仅仅用于生成相互作用信息，因为在损失哪里已经去除填充的0了，所以这里就不再重复去除，否则导致数据长度不一样
                        #如果有问题，可以在这里去除填充的0
                        #token_mask = batch['atoms'][i]>2 #有效的词表下标是从3开始的，前3个是标志位？是这个意思？不清楚，只是知道这里用于剔除填充的标志位，对于配体总共3个

                        holo_coordinates = batch['holo_coordinates'][i]
                        #holo_coordinates = holo_coordinates[token_mask,:]
                        holo_coordinates = holo_coordinates.numpy().astype(np.float32)

                        coord_predict = batch['coord_predict'][i]
                        #coord_predict = coord_predict[token_mask,:]
                        coord_predict = coord_predict.numpy().astype(np.float32)

                        holo_center_coordinates = batch["holo_center_coordinates"][i][:3]
                        holo_center_coordinates.numpy().astype(np.float32)

                        holo_center_coords_list.append(holo_center_coordinates)        
                        coords_predict_list.append(coord_predict)
                        holo_coords_list.append(holo_coordinates)
                    except Exception as e:
                        #print('error,skip:', e)
                        raise Exception('error, skip')
                        smi_list.append(None) 
                        pocket_list.append(None)
                        coords_predict_list.append(None)
                        holo_coords_list.append(None)
                        holo_center_coords_list.append(None) 
                        prmsd_score_list.append(None)

        #下面这个能通过，也就是pocket_coords_list和holo_pocket_coords_list是一样的
        for i in range(len(pocket_coords_list)):
            assert np.allclose(np.array(pocket_coords_list[i]), np.array(holo_pocket_coords_list[i]), rtol=0.00, atol=0.00)

        '''
        np.set_#printoptions(precision=4, suppress=True) 
        #print('len(pocket_coords_list):', len(pocket_coords_list)) #一维度的list
        #print('len(holo_pocket_coords_list):', len(holo_pocket_coords_list)) #一维度的list
        
        #print(np.array(pocket_coords_list[0]).shape, np.array(pocket_coords_list[2]).shape)
        #print(np.array(holo_pocket_coords_list[0]).shape, np.array(holo_pocket_coords_list[2]).shape)

        A = np.array(pocket_coords_list[0])
        sorted_indices1 = np.lexsort((A[:, 2], A[:, 1], A[:, 0]))
        # 根据排序后的索引对坐标矩阵进行排序
        sorted_A = A[sorted_indices1]

        B = np.array(pocket_coords_list[2])
        sorted_indices2 = np.lexsort((B[:, 2], B[:, 1], B[:, 0]))
        # 根据排序后的索引对坐标矩阵进行排序
        sorted_B = B[sorted_indices2]

        # 将每一行转换为元组，以便进行集合操作
        coords1_tuples = {tuple(row) for row in A}
        coords2_tuples = {tuple(row) for row in B}

        # 找出交集
        intersection = np.array(list(coords1_tuples & coords2_tuples))

        #print("坐标矩阵之间的交集：")
        #print(intersection) #[] 有问题，蛋白的坐标矩阵竟然没有重叠？因为40个构象是单独采样的，得到的蛋白质心不同，这里是减去蛋白质心的，所以不一样，不要在这里测试

        
        #print('pocket_coords_list[0]\n:', sorted_A[:5])
        #print('pocket_coords_list[2]\n:', sorted_B[:5])

        #下面这个能通过，也就是pocket_coords_list和holo_pocket_coords_list是一样的
        for i in range(len(pocket_coords_list)):
            assert np.allclose(np.array(pocket_coords_list[i]), np.array(holo_pocket_coords_list[i]), rtol=0.00, atol=0.00)



        #print(np.sum(np.array(pocket_coords_list[0])), np.sum(np.array(pocket_coords_list[2]))) #因为蛋白在质心上，即原点，因此sum和mean操作得到的结果是接近0的,所以没啥意义
        assert np.allclose(sorted_A, sorted_B, rtol=0.01, atol=0.02)

        #print(np.sum(np.array(holo_pocket_coords_list[0])), np.sum(np.array(holo_pocket_coords_list[2])))
        assert np.allclose(np.array(holo_pocket_coords_list[0]), np.array(holo_pocket_coords_list[2]), rtol=0.01, atol=0.02)

        exit()
        '''
        return mol_list, smi_list, coords_predict_list, holo_coords_list, holo_center_coords_list, prmsd_score_list, fail_index, pocket_coords_list, cross_distance_list, holo_pocket_coords_list, ligand_emb, pocket_emb




    def set_coord(self, mol, coords):
        for i in range(coords.shape[0]):
            mol.GetConformer(0).SetAtomPosition(i, coords[i].tolist())
        return mol

    def add_coord(self, mol, xyz):
        x, y, z = xyz
        conf = mol.GetConformer(0)
        pos = conf.GetPositions()
        pos[:, 0] += x
        pos[:, 1] += y
        pos[:, 2] += z
        for i in range(pos.shape[0]):
            conf.SetAtomPosition(
                i, Chem.rdGeometry.Point3D(pos[i][0], pos[i][1], pos[i][2])
            )
        return mol
    

    def subtract_coord(self, mol, xyz):
        x, y, z = xyz
        conf = mol.GetConformer(0)
        pos = conf.GetPositions()
        pos[:, 0] -= x
        pos[:, 1] -= y
        pos[:, 2] -= z
        for i in range(pos.shape[0]):
            conf.SetAtomPosition(
                i, Chem.rdGeometry.Point3D(pos[i][0], pos[i][1], pos[i][2])
            )
        return mol
    
    def get_sdf(self, mol_list, smi_list, coords_predict_list, holo_center_coords_list, prmsd_score_list, output_ligand_name, output_ligand_dir, \
                output_ligand_dir2, holo_coords_list, pocket_coords_list, holo_pocket_coords_list, cross_distance_list, ligand_emb_list, pocket_emb_list, tta_times=10):
        #print("Start converting model predictions into sdf files...")
        output_ligand_list = []
        if self.mode == 'single':
            output_ligand_name = [output_ligand_name]
        #print('tta_times:', tta_times) #40
        #tta_times = 1
        ##print('mol_list:', mol_list)
        ##print('smi_list:', smi_list)
        #print('output_ligand_name:', output_ligand_name) #l
        #print('output_ligand_dir2:', output_ligand_dir2) #['predict_sdf_boxsize20/6erv']
        #outputfilename = os.path.join(output_ligand_dir, str(output_ligand_name[i]) + '.sdf')
        #try:
            #os.remove(outputfilename)
        #except:
            #pass
        new_holo_coords_lists    = []
        new_coords_predict_lists = []
        new_pocket_coords_lists  = []
        new_cross_distance_lists = []
        new_ligand_emb_lists     = []
        new_pocket_emb_lists     = []

        for i in tqdm(range(len(smi_list)//tta_times)): #将不同的分子数据分开，逐一遍历
            #print('===============================================================')
            #print('i:', i)
            coords_predict_tta = coords_predict_list[i*tta_times:(i+1)*tta_times]
            prmsd_score_tta = prmsd_score_list[i*tta_times:(i+1)*tta_times]
            mol_list_tta = mol_list[i*tta_times:(i+1)*tta_times]
            holo_center_coords_tta = holo_center_coords_list[i*tta_times:(i+1)*tta_times]

            holo_coords_tta   = holo_coords_list[i*tta_times:(i+1)*tta_times]
            pocket_coords_tta = pocket_coords_list[i*tta_times:(i+1)*tta_times]
            holo_pocket_coords_tta = holo_pocket_coords_list[i*tta_times:(i+1)*tta_times]
            cross_distance_tta = cross_distance_list[i*tta_times:(i+1)*tta_times]
            
            #ligand_emb_tta = ligand_emb_list[i*tta_times:(i+1)*tta_times]
            #pocket_emb_tta = pocket_emb_list[i*tta_times:(i+1)*tta_times]

            #保存所有的构象
            #idx = np.argmin(prmsd_score_tta) #这是找rmsd最小的保存
            #bst_predict_coords = coords_predict_tta[idx]
            #mol = mol_list_tta[idx]
            #print('mol_list_tta:', mol_list_tta)
            new_mol_list = []
            new_org_mol_list = []

            new_holo_coords_list    = []
            new_coords_predict_list = []
            new_pocket_coords_list  = []
            new_cross_distance_list = []
            new_ligand_emb_list     = []
            new_pocket_emb_list     = []

            for org_mol, mol, coords, centor, holo_coords, pocket_coords, holo_pocket_coords, cross_distance in zip(copy.deepcopy(mol_list_tta), \
                copy.deepcopy(mol_list_tta), coords_predict_tta, holo_center_coords_tta, copy.deepcopy(holo_coords_tta), copy.deepcopy(pocket_coords_tta), \
                    copy.deepcopy(holo_pocket_coords_tta), copy.deepcopy(cross_distance_tta)):
                #求holo_coords, pocket_coords质心，看看和holo_center_coords一样？生成的配体坐标和参考配体坐标都需要加上配体质心，除非直接mol_list中直接读取参考的坐标
                #蛋白口袋的坐标也减去了质心了？但为什么加上或减去质心后和源文件对不上？对不上的原因是，这里的蛋白坐标中的填充0，即(0,0,0), 这个需要过滤掉，但是0是在哪填充的呢？
                #pcoords[0].shape: (128, 3) #这是输入的蛋白维度，但是我们得到的是(136, 3), 多出几个原子，这些是H原子，他们减去蛋白质心后对应的坐标是(0,0,0), 因此我们根据这个信息，要把蛋白
                #为0的坐标去掉
                #配体的坐标也加了H原子，所以去掉
                orgin_pos = org_mol.GetConformer(0).GetPositions().astype(np.float32)
                holo_center_coords = centor

                holo_coords_c, pocket_coords_c, holo_pocket_coords_c = np.mean(holo_coords , axis = 0), np.mean(pocket_coords, axis = 0), np.mean(holo_pocket_coords, axis = 0)
                
                #值得注意的是，虽然配体与蛋白的口袋的质心很相近，但还是有那么一点微小差异的，不一样是正常的
                #print('值得注意的是，虽然配体与蛋白的口袋的质心很相近，但还是有那么一点微小差异的，不一样是正常的')
                #print('没有加质心的前的坐标均值')
                #print('holo_coords_c1:', holo_coords_c)
                #print('pocket_coords_c1:', pocket_coords_c) 
                #print('holo_pocket_coords_c1:', holo_pocket_coords_c)

                #print('holo_coords.shape1:', holo_coords.shape) #看看有没有(0,0,0),或者等于质心的holo_center_coords，因为要填充holo_coords.shape1: (13, 3), 已经去0了
                #print('orgin_pos.shape1:', orgin_pos.shape) #这里是有(0,0,0), orgin_pos.shape1: (22, 3), 没有去氢， #(22, 3)
                #print('predict_coords1:', coords.shape) #(13, 3)， 现在有一个问题：orgin_pos和coords长度对不上，下面是如何使用coords更新mol的呢？在处理数据时取氢操作，原子数量就一样了
                #print('pocket_coords.shape, holo_pocket_coords.shape1:', pocket_coords.shape, holo_pocket_coords.shape) #(136, 3) (136, 3)
                #print('cross_distance.shape:', cross_distance.shape) #cross_distance.shape: (16, 136), 现在有一个问题，这里的维度是16与配体的原子数量对不上，既不是13，也不是22

                #print('holo_coords[:2]1:', holo_coords[:2]) #看看有没有(0,0,0),或者等于质心的holo_center_coords，因为要填充，已经去0了，需要加质心
                #print('orgin_pos[:2]1:', orgin_pos[:2]) # 保存的是真实的坐标，不需要加质心

                #print('pocket_coords[:2]1:', pocket_coords[:2])
                #print('holo_pocket_coords[:2]1:', holo_pocket_coords[:2])


                #print('\n')

                #print('--------------------------------------------------------')
                #print('值得注意的是，虽然配体与蛋白的口袋的质心很相近，但还是有那么一点微小差异的，不一样是正常的')
                #print('加质心的后的坐标均值')
                #蛋白存填充0的情况，所以在损失那一部分，要去填充
                holo_coords, pocket_coords, holo_pocket_coords = holo_coords + np.array(holo_center_coords), pocket_coords + np.array(holo_center_coords), holo_pocket_coords + np.array(holo_center_coords)
                #holo_coords = holo_coords + np.array(holo_center_coords)

                #print('type(holo_center_coords):', type(holo_center_coords)) #<class 'torch.Tensor'>
                #print('holo_center_coords:', holo_center_coords)

                holo_coords_c, pocket_coords_c, holo_pocket_coords_c = np.mean(holo_coords , axis = 0), np.mean(pocket_coords, axis = 0), np.mean(holo_pocket_coords, axis = 0)
                #print('holo_coords_c2:', holo_coords_c)
                #print('pocket_coords_c2:', pocket_coords_c) #蛋白质心在0,0,0和配体不在一起，是否需要加配体质心holo_center_coords？
                #print('holo_pocket_coords_c2:', holo_pocket_coords_c) #

                
                orgin_pos = org_mol.GetConformer(0).GetPositions().astype(np.float32)
                #看看两者一样？必须要一样。 结果是一样, 允许差几位小数，因为有从sdf文件读取的坐标只有4位，所以有误差
                #print('holo_coords.shape2:', holo_coords.shape) #看看有没有(0,0,0),或者等于质心的holo_center_coords，因为要填充
                #print('orgin_pos.shape2:', orgin_pos.shape)

                #print('holo_coords[:]2:', holo_coords[:2]) #看看有没有(0,0,0),或者等于质心的holo_center_coords，因为要填充
                #print('orgin_pos[:]2:', orgin_pos[:2])

                #两者的结果是一样？
                #print('pocket_coords[:2]2:', pocket_coords[:2])
                #print('holo_pocket_coords[:2]2:', holo_pocket_coords[:2])

                #print('\n')
                #print('*****************************************************')
                #

                #print('值得注意的是，虽然配体与蛋白的口袋的质心很相近，但还是有那么一点微小差异的，不一样是正常的')
                #print('使用预测的坐标去更新mol，用于保存生成的sdf')
                #print('holo_center_coords:', holo_center_coords)
                #print('holo_coords_c:', holo_coords_c)
                
                new_org_mol_list.append(org_mol)
                #现在有一个问题：orgin_pos和coords长度对不上，下面是如何使用coords更新mol的呢？，因为填充的原子的顺序都在后面，所以即使长度不一样，那些填充的原子坐标没有变化
                #当然最好对mol取氢。在处理数据时取氢操作，原子数量就一样了
                new_mol = self.set_coord(copy.deepcopy(mol), coords)
                #print('new_mol1:', new_mol)
                #print('holo_center_coords.numpy():', holo_center_coords.numpy())
                try:
                    new_mol = self.add_coord(new_mol, holo_center_coords.numpy()) 
                    #print('new_mol2:', new_mol)
                except Exception as e:
                    print('e:', e)
                    with open('get_sdf_error.txt', 'a') as f:
                        input_ligand = os.path.join(output_ligand_dir2[i], 'org_' + str(output_ligand_name[i]) + '.sdf')
                        #print('input_ligand num:', len(input_ligand))
                        f.write(input_ligand + '\n')
                        continue

                new_pos = new_mol.GetConformer(0).GetPositions()
                new_centor = np.mean(new_pos, axis = 0)
                #print('new_mol centor:', new_centor)


                #
                #print('注意这里coords并没有加质心，所以质心对不上，但不影响我们使用，另外预测出来的配体质心未必要和参考的一致')
                #print('未加质心coords_centor:', np.mean(coords, axis = 0))
                #print('coords加质心')
                new_coords = coords + np.array(holo_center_coords)
                #print('加质心coords_centor:', np.mean(new_coords, axis = 0))


                #raise Exception('test')

                
                new_mol_list.append(new_mol)

                #print('tyep(holo_coords):', type(holo_coords)) #<class 'numpy.ndarray'>
                #print('type(coords):', type(new_coords))#<class 'numpy.ndarray'>
                #print('type(holo_pocket_coords):', type(holo_pocket_coords))#<class 'numpy.ndarray'>
                #print('type(cross_distance):', type(cross_distance))#<class 'numpy.ndarray'>

                #print('len(holo_coords):', holo_coords.shape)
                #print('len(coords):', new_coords.shape)
                #print('len(holo_pocket_coords):', holo_pocket_coords.shape)
                #print('len(cross_distance):', cross_distance.shape)

                new_holo_coords_list.append(holo_coords)
                new_coords_predict_list.append(new_coords)  #存在一个问题coords加了质心了？
                new_pocket_coords_list.append(holo_pocket_coords)
                new_cross_distance_list.append(cross_distance)
                #new_ligand_emb_list.append(ligand_emb)
                #new_pocket_emb_list.append(pocket_emb)


                #raise Exception('test')
            
            new_holo_coords_lists.append(new_holo_coords_list)
            new_coords_predict_lists.append(new_coords_predict_list)
            new_pocket_coords_lists.append(new_pocket_coords_list)
            new_cross_distance_lists.append(new_cross_distance_list)
            #new_ligand_emb_lists.append(new_ligand_emb_list)
            #new_pocket_emb_lists.append(new_pocket_emb_list)

            data_dict = {}
            data_dict['holo_coords_list'] = new_holo_coords_list
            data_dict['coords_predict_list'] = new_coords_predict_list
            data_dict['pocket_coords_list'] = new_pocket_coords_list
            data_dict['cross_distance_list'] = new_cross_distance_list
            #data_dict['ligand_emb_list'] = new_ligand_emb_list
            #data_dict['pocket_emb_list'] = new_pocket_emb_list


            if len(new_holo_coords_list) > 2:
                #判断参考的配体蛋白的原子顺序是否一样？
                for j in range(len(new_holo_coords_list))[1:]:
                    assert np.allclose(np.array(new_holo_coords_list[j-1]), np.array(new_holo_coords_list[j]), rtol=0.01, atol=0.02)


                np.set_printoptions(precision=4, suppress=True) 
                #print('len(pocket_coords_list):', len(new_pocket_coords_list)) #一维度的list
                
                #print(np.array(new_pocket_coords_list[0]).shape, np.array(new_pocket_coords_list[2]).shape)

                A = np.array(new_pocket_coords_list[0])
                sorted_indices1 = np.lexsort((A[:, 2], A[:, 1], A[:, 0]))
                # 根据排序后的索引对坐标矩阵进行排序
                sorted_A = A[sorted_indices1]

                B = np.array(new_pocket_coords_list[2])
                sorted_indices2 = np.lexsort((B[:, 2], B[:, 1], B[:, 0]))
                # 根据排序后的索引对坐标矩阵进行排序
                sorted_B = B[sorted_indices2]

                #print('pocket_coords_list[0]\n:', sorted_A[:5])
                #print('pocket_coords_list[2]\n:', sorted_B[:5])

                


                # 将每一行转换为元组，以便进行集合操作
                coords1_tuples = {tuple(row) for row in A}
                coords2_tuples = {tuple(row) for row in B}

                # 找出交集
                intersection = np.array(list(coords1_tuples & coords2_tuples))
                #print('A.shape:', A.shape)
                #print('B.shape:', B.shape)
                #print("坐标矩阵之间的交集 num：", len(intersection)) #171, 40个构象的蛋白原子是不一样的，是否有必要一样？故意为之，还是怎么的？
            

            '''
            assert np.allclose(sorted_A, sorted_B, rtol=0.01, atol=0.02)
            #蛋白的顺序不一样. 原因是原子不一样，虽然数量一样，但40个构象的蛋白是分别读取的，然后截取口袋的？
            #print(np.sum(np.array(new_pocket_coords_list[0])), np.sum(np.array(new_pocket_coords_list[-1]))) #-2764.335 -2859.1602
            assert np.allclose(np.sum(np.array(new_pocket_coords_list[0])), np.sum(np.array(new_pocket_coords_list[-1])), rtol=0.01, atol=0.02)
            assert np.allclose(np.array(new_pocket_coords_list[0]), np.array(new_pocket_coords_list[-1]), rtol=0.01, atol=0.02)
            '''
            
            #print('output_ligand_dir2[i]:', output_ligand_dir2[i]) #list index out of range
            os.makedirs(output_ligand_dir2[i], exist_ok=True)

            #可以直接在这里保存相互作用信息
            outputfilename = os.path.join(output_ligand_dir2[i], 'interaction_' + str(output_ligand_name[i]) + '.pkl')
            with open(outputfilename, "wb") as f:
                dill.dump(data_dict, f)

            outputfilename = os.path.join(output_ligand_dir2[i], 'org_' + str(output_ligand_name[i]) + '.sdf')
            #保存sdf
            w = Chem.SDWriter(outputfilename)
            #Chem.MolToMolFile(mol, outputfilename)
            ##print('new_mol_list:', len(new_mol_list))
            for mol in new_org_mol_list[:1]:
                # 去除氢原子
                new_mol = Chem.RemoveHs(mol)
                # 保存到 SDF 文件
                w.write(new_mol)
            w.close()



            outputfilename = os.path.join(output_ligand_dir2[i], 'gen_' + str(output_ligand_name[i]) + '.sdf')
            try:
                os.remove(outputfilename)
            except:
                pass

            #保存sdf
            w = Chem.SDWriter(outputfilename)
            #Chem.MolToMolFile(mol, outputfilename)
            ##print('new_mol_list:', len(new_mol_list))
            for mol in new_mol_list:
                # 去除氢原子
                new_mol = Chem.RemoveHs(mol)
                # 保存到 SDF 文件
                w.write(new_mol)
            w.close()

            output_ligand_list.append(outputfilename)

        

        #print("Done!")
        ##print('output_ligand_list:', output_ligand_list)
        if self.mode == 'single':
            return output_ligand_list[0]
        elif self.mode in ['batch_one2one', 'batch_one2many']:
            return output_ligand_list
    
    def single_clash_fix(self, input_content):
        input_ligand, output_ligand, label_ligand, pocket_mol = input_content
        script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "unimol", "scripts", "6tsr.py")
        cmd = "python {} --input-ligand {} --output-ligand {} --label-ligand {} --pocket-mol {} --num-6t-trials 5".format(
            script_path, input_ligand, output_ligand, label_ligand, pocket_mol
        )
        os.system(cmd)
        return True

    def clash_fix(self, predicted_ligand, input_protein, input_ligand):
        if self.mode=='batch_one2many':
            input_protein = [input_protein] * len(input_ligand)
        elif self.mode == 'single':
            input_ligand = [input_ligand]
            input_protein = [input_protein]
            predicted_ligand = [predicted_ligand]
        input_content = zip(predicted_ligand, predicted_ligand, input_ligand, input_protein)

        with Pool(self.nthreads) as pool:
            for inner_output in tqdm(
                pool.imap(self.single_clash_fix, input_content), total=len(input_ligand) if type(input_ligand) is list else 1
            ):
                if not inner_output:
                    print("fail to clash fix")
        return predicted_ligand

    @classmethod
    def build_processors(
        cls, 
        mode='single', 
        nthreads = 4, 
        conf_size = 1, 
        cluster=False,
        use_current_ligand_conf:bool=False
    ):
        return cls(
            mode, 
            nthreads, 
            conf_size=conf_size, 
            cluster=cluster, 
            use_current_ligand_conf=use_current_ligand_conf
        )