
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
import signal




import time
from functools import wraps

class FunctionTimeoutError(Exception):
    pass

def measure_time(threshold):
    """，
    
    Args:
        threshold (float): （）
    
    Returns:
        function: 
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            
            if elapsed_time > threshold:
                raise FunctionTimeoutError(
                    f"Function '{func.__name__}' exceeded time threshold. "
                    f"Elapsed time: {elapsed_time:.2f}s, Threshold: {threshold}s"
                )
            return result
        return wrapper
    return decorator
@measure_time(threshold=1.5)
def my_function():
    time.sleep(2)
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
                        supp = Chem.SDMolSupplier(input_ligand[i], sanitize=False)
                        mol = [mol for mol in supp if mol][0]
                        ori_smiles = Chem.MolToSmiles(mol)
                        mol = Chem.RemoveHs(mol, sanitize=False)
                            
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
                            error_index_list.append(i)


            with open('error_ligand.txt', 'a') as f:
                for name in error_name_list:
                    f.write(name + '\n')
        print('error_index_list:', error_index_list)
        print('output_ligand_name:', output_ligand_name)
        # Delete in reverse order; ascending deletion shifts later indices and
        # silently removes the wrong protein/ligand rows.
        for index_i in sorted(error_index_list, reverse=True):
            print('index_i:', index_i)
            del output_ligand_name[index_i]
            del input_protein[index_i]
            del input_ligand[index_i]
            del input_docking_grid[index_i]
            
        lmdb_name = self.write_lmdb(output_ligand_name, smiles_list, input_protein, input_ligand, input_docking_grid, seed=seed, result_dir=out_lmdb_dir)
        return lmdb_name

    def single_conf_gen(self, tgt_mol, num_confs=1000, seed=42, removeHs=True):
        mol = copy.deepcopy(tgt_mol)
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        params.clearConfs = True
        params.numThreads = 1  # multiprocessing already parallelizes molecules
        params.pruneRmsThresh = 0.1
        if hasattr(params, "timeout"):
            params.timeout = 30
        allconformers = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
        if not allconformers:
            raise ValueError("RDKit generated no conformer")
        # The old Python loop could spend minutes optimizing one difficult
        # conformer.  The batched API is faster and has an explicit iteration
        # ceiling.  Unsupported chemistries skip MMFF and retain ETKDG coords.
        if AllChem.MMFFHasAllMoleculeParams(mol):
            # A short relaxation is sufficient before conformer clustering;
            # UniMol consumes the geometry and performs its own refinement.
            # Long MMFF convergence dominates preprocessing for flexible
            # ligands without improving the requested candidate count.
            AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=1, maxIters=20)
        if removeHs:
            mol = Chem.RemoveHs(mol)
        return mol

    def single_conf_gen_no_MMFF(self, tgt_mol, num_confs=1000, seed=42, removeHs=True):
        mol = copy.deepcopy(tgt_mol)
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        params.clearConfs = True
        params.numThreads = 1
        params.pruneRmsThresh = 0.1
        if hasattr(params, "timeout"):
            params.timeout = 30
        allconformers = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
        if not allconformers:
            raise ValueError("RDKit generated no conformer")
        if removeHs:
            mol = Chem.RemoveHs(mol)
        return mol

    @measure_time(threshold=10)
    def clustering_coords_copy(self, mol, M=1000, N=100, seed=42, cluster=False, removeHs=True, gen_mode='mmff'):

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
            AlignMolConformers(rdkit_mol, atomIds=noHsIds)
            sz = len(rdkit_mol.GetConformers())
            for i in range(sz):
                _coords = rdkit_mol.GetConformers()[i].GetPositions().astype(np.float32)
                rdkit_coords_list.append(_coords)
            
            if not rdkit_coords_list:
                raise ValueError("RDKit generated no usable coordinates")
            if cluster:
                rdkit_coords = np.array(rdkit_coords_list)[:, noHsIds]
                rdkit_coords_flatten = rdkit_coords.reshape(sz, -1)
                cluster_count = min(N, sz)
                kmeans = KMeans(
                    n_clusters=cluster_count, random_state=seed, n_init=1,
                    max_iter=100,
                ).fit(rdkit_coords_flatten)
                center_coords = kmeans.cluster_centers_.reshape(cluster_count, -1, 3)
                cdist = ((center_coords[:, None] - rdkit_coords[None, :])**2).sum(axis=(-1, -2))
                argmin = np.argmin(cdist, axis=-1)
                coords_list = [rdkit_coords_list[i] for i in argmin]
            else:
                coords_list = rdkit_coords_list
            if len(coords_list) != N:
                coords_list = coords_list + [coords_list[0]] * (N - len(coords_list)) 
        except Exception as e:
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
            AlignMolConformers(rdkit_mol, atomIds=noHsIds)
            sz = len(rdkit_mol.GetConformers())
            for i in range(sz):
                _coords = rdkit_mol.GetConformers()[i].GetPositions().astype(np.float32)
                rdkit_coords_list.append(_coords)
            
            if cluster:
                rdkit_coords = np.array(rdkit_coords_list)[:, noHsIds]
                rdkit_coords_flatten = rdkit_coords.reshape(sz, -1)
                kmeans = KMeans(n_clusters=N, random_state=seed).fit(rdkit_coords_flatten)
                center_coords = kmeans.cluster_centers_.reshape(N, -1, 3)
                cdist = ((center_coords[:, None] - rdkit_coords[None, :])**2).sum(axis=(-1, -2))
                argmin = np.argmin(cdist, axis=-1)
                coords_list = [rdkit_coords_list[i] for i in argmin]
            else:
                coords_list = rdkit_coords_list
            if len(coords_list) != N:
                coords_list = coords_list + [coords_list[0]] * (N - len(coords_list)) 

        return coords_list



    @measure_time(threshold=100)
    def clustering_coords(self, mol, M=1000, N=100, seed=42, cluster=False, removeHs=True, gen_mode='mmff'):
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
        AlignMolConformers(rdkit_mol, atomIds=noHsIds)
        sz = len(rdkit_mol.GetConformers())
        for i in range(sz):
            _coords = rdkit_mol.GetConformers()[i].GetPositions().astype(np.float32)
            rdkit_coords_list.append(_coords)
        
        if not rdkit_coords_list:
            raise ValueError("RDKit generated no usable coordinates")
        if cluster:
            rdkit_coords = np.array(rdkit_coords_list)[:, noHsIds]
            rdkit_coords_flatten = rdkit_coords.reshape(sz, -1)
            cluster_count = min(N, sz)
            kmeans = KMeans(
                n_clusters=cluster_count, random_state=seed, n_init=1,
                max_iter=100,
            ).fit(rdkit_coords_flatten)
            center_coords = kmeans.cluster_centers_.reshape(cluster_count, -1, 3)
            cdist = ((center_coords[:, None] - rdkit_coords[None, :])**2).sum(axis=(-1, -2))
            argmin = np.argmin(cdist, axis=-1)
            coords_list = [rdkit_coords_list[i] for i in argmin]
        else:
            coords_list = rdkit_coords_list
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
        pmol = PandasPdb().read_pdb(input_protein)
        '''
        atom_df = pmol.df['ATOM']
        hetatm_df = pmol.df['HETATM']
        total_atoms = len(atom_df) + len(hetatm_df)
        '''
        
        
        
        
        df_no_h = pmol.df['ATOM'][pmol.df['ATOM']['element_symbol'] != 'H'].copy()
        # Match IFMDock/ParmEd alternate-location handling.  PandasPdb keeps
        # all alternate rows, whereas ParmEd retains the first altloc appearing
        # in the PDB (not necessarily the highest-occupancy one; e.g. 7XPO has
        # B before C).  Retain that same first physical atom identity.
        if 'alt_loc' in df_no_h.columns:
            identity = ['chain_id', 'residue_number', 'insertion', 'atom_name']
            df_no_h = df_no_h.drop_duplicates(identity, keep='first')
        pmol.df['ATOM'] = df_no_h
        if 'HETATM' in pmol.df:
            pmol.df['HETATM'] = pmol.df['HETATM'][pmol.df['HETATM']['element_symbol'] != 'H']

        '''
        atom_df = pmol.df['ATOM']
        hetatm_df = pmol.df['HETATM']
        total_atoms = len(atom_df) + len(hetatm_df)
        
        raise Exception('test')
        '''
        
        with open(input_docking_grid, "r") as file:
            box_dict = json.load(file)

        pdf = pmol.df['ATOM']
        patoms, pcoords, residues = self.find_residues_in_pocket(box_dict, pdf)
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
        input_ligand = ""
        complex_name = "unknown"
        try:
            # Runs inside a multiprocessing worker on Linux.  Unlike the old
            # post-hoc timer, SIGALRM interrupts a pathological molecule and
            # lets the existing fallback path keep the batch moving.
            def _timeout_handler(signum, frame):
                raise FunctionTimeoutError("ligand preprocessing exceeded 90 s")
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(90)
            smiles, input_protein, input_ligand, input_docking_grid, seed = content
            name = input_protein.split('/')[-2]
            complex_name = name
            
            tg = os.path.basename(input_protein).split('_')
            if len(tg) == 2:
                complex_name = tg[0]
            elif len(tg) == 3:
                complex_name = tg[1]
            patoms, pcoords, residues, side, config = self.extract_pocket(input_protein, input_docking_grid)
            if 'origin' in input_ligand:
                supp = Chem.SDMolSupplier(input_ligand, sanitize=False)
                mol = [Chem.RemoveHs(mol) for mol in supp if mol][0]
            elif '.mol2' in input_ligand:
                mol = Chem.MolFromMol2File(input_ligand, sanitize=False)
            else:
                supp = Chem.SDMolSupplier(input_ligand)
                mol = [Chem.RemoveHs(mol) for mol in supp if mol][0]
            self.use_current_ligand_conf = False
            if self.use_current_ligand_conf:
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
            smiles = Chem.MolToSmiles(mol)
            latoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
            holo_coordinates = [mol.GetConformer().GetPositions().astype(np.float32)]
            holo_mol = mol
            N = self.conf_size
            # Generate two candidates per retained cluster.  The original 10x
            # and previous 4x oversampling spend most preprocessing time in
            # ETKDG/MMFF; 2x still permits diversity-based selection while
            # halving that dominant cost.
            M = self.conf_size * 2
            mol_list = [mol] * N
            coordinate_list = []
            try:
                coordinate_list = self.clustering_coords(mol, M=M, N=N, seed=seed, cluster = True, removeHs=True, gen_mode='mmff')
            except FunctionTimeoutError:
                # The alarm has already fired; do not enter another expensive
                # RDKit call.  The input SDF conformer is a valid deterministic
                # fallback and preserves atom order.
                coordinate_list = holo_coordinates * N
            except Exception as e:
                try:
                    coordinate_list = self.clustering_coords(mol, M=M, N=N, seed=seed, cluster = False, removeHs=True, gen_mode='mmff')
                except FunctionTimeoutError:
                    coordinate_list = holo_coordinates * N
                except Exception as e:
                    try:
                        coordinate_list = self.clustering_coords(mol, M=1, N=1, seed=seed, cluster = False, removeHs=True, gen_mode='no_mmff')
                        coordinate_list = coordinate_list * N
                    except (Exception, FunctionTimeoutError) as e:
                            coordinate_list = holo_coordinates * N
            
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
                        coordinate_list = holo_coordinates * N
            '''    

            assert len(coordinate_list) == N
            output = pickle.dumps(
                {
                    "atoms": latoms,
                    "coordinates": coordinate_list,
                    "mol_list": mol_list,
                    "pocket_atoms": patoms,
                    "pocket_coordinates": pcoords,
                    "side": side,
                    "residue": residues,
                    "config": config,
                    "holo_coordinates": holo_coordinates,
                    "holo_mol": holo_mol,
                    "holo_pocket_coordinates": pcoords,
                    "smi": smiles,
                    "pocket": input_protein,
                    "flag": 'success',
                    "name": name
                },
                protocol=-1,
                ), True, input_ligand, complex_name, pcoords
            signal.alarm(0)
            return output
        except (Exception, FunctionTimeoutError) as e:
            signal.alarm(0)
            pcoords = 0
            with open("distance_preprocess_failures.tsv", "a") as handle:
                handle.write(f"{complex_name}\t{input_ligand}\t{type(e).__name__}: {e}\n")
            return None, False,  input_ligand, complex_name, pcoords


                



    def write_lmdb(self, output_ligand_name, smiles_list, input_protein, input_ligand, input_docking_grid, seed=42, result_dir="./results"):
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
            map_size=int(100*(1024*1024*1024)),
        )
        txn_write = env_new.begin(write=True)
        fail_file_list = []
        seed = [seed] * len(input_ligand)
        content_list = zip(smiles_list, input_protein, input_ligand, input_docking_grid, seed)
        '''
        pcoords_dict = {}
        with Pool(self.nthreads) as pool:
            ii = 0
            failed_num = 0
            for inner_output, flag, file_ligand, complex_name, pcoords in tqdm(pool.imap(self.parser, content_list)):
                if flag is True:
                    txn_write.put(f"{ii}".encode("ascii"), inner_output)
                    ii+=1
                    pcoords_dict[complex_name] = pcoords
                elif flag is False: 
                    fail_file_list.append(file_ligand)
                    failed_num += 1
                    continue
            txn_write.commit()
            env_new.close()
        '''
        pcoords_dict = {}
        with Pool(self.nthreads) as pool:
            ii = 0
            failed_num = 0
            # Unordered consumption prevents one slow input from withholding
            # already completed molecules behind it in the input sequence.
            for inner_output, flag, file_ligand, complex_name, pcoords in tqdm(pool.imap_unordered(self.parser, content_list),total=len(smiles_list)):
                if flag is True:
                    txn_write.put(f"{ii}".encode("ascii"), inner_output)
                    ii+=1
                    pcoords_dict[complex_name] = pcoords
                elif flag is False: 
                    fail_file_list.append(file_ligand)
                    failed_num += 1
                    continue
            txn_write.commit()
            env_new.close()
        '''
        pcoords_dict = {}
        ii = 0
        failed_num = 0
        for smiles_i, input_protein_i, input_ligand_i, input_docking_grid_i, seed_i in tqdm(content_list, total=len(smiles_list)):
            try:
                inner_output, flag, file_ligand, complex_name, pcoords = self.parser((smiles_i, input_protein_i, input_ligand_i, input_docking_grid_i, seed_i))
            
                txn_write.put(f"{ii}".encode("ascii"), inner_output)
                ii+=1
                pcoords_dict[complex_name] = pcoords
            except Exception as e: 
                fail_file_list.append(input_ligand_i)
                failed_num += 1
                continue
            
        txn_write.commit()
        env_new.close()
        '''

        return output_ligand_name

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
        fail_index = []
        mol_list = []
        num = 0
        for items in old_mol_list:
            for mol in items:
                try:
                    Chem.RemoveHs(mol)
                    mol_list.append(mol)
                except Exception as e:
                    fail_index.append(num)
                    mol_list.append(None)
                
                num += 1
        predict = pd.read_pickle(predict_file)
        '''
        predict.keys(): dict_keys(['loss', 'cross_distance_loss', 'distance_loss', 'coord_loss', 'prmsd_loss', 'prmsd_score', 'bsz', 'sample_size', 
        'coord_predict', 'coord_target', 'smi_name', 'pocket_name', 'atoms', 'pocket_atoms', 'coordinates', 'holo_coordinates', 'pocket_coordinates', 
        'holo_center_coordinates'])
        '''
        smi_list, pocket_list, coords_predict_list, holo_coords_list, holo_center_coords_list, prmsd_score_list = [],[],[],[],[],[]
        cross_distance_list = []
        pocket_coords_list = []
        holo_pocket_coords_list = []

        for batch in predict:

            if batch == None:
                print('batch == None, skip')
            else:
                sz = batch['atoms'].size(0)

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
                        
                        token_mask = batch['atoms'][i]>2

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
                        smi_list.append(None) 
                        pocket_list.append(None)
                        coords_predict_list.append(None)
                        holo_coords_list.append(None)
                        holo_center_coords_list.append(None) 
                        prmsd_score_list.append(None)


        return mol_list, smi_list, coords_predict_list, holo_coords_list, holo_center_coords_list, prmsd_score_list, fail_index, pocket_coords_list, cross_distance_list, holo_pocket_coords_list




    def postprocess_data_pre(self, predict_file, lmdb_file):

        old_mol_list = self.load_lmdb_data(lmdb_file, "mol_list")
        fail_index = []
        mol_list = []
        num = 0
        for items in old_mol_list:
            for mol in items:
                try:
                    new_mol = Chem.RemoveHs(mol)
                    mol_list.append(new_mol)
                except Exception as e:
                    fail_index.append(num)
                    mol_list.append(None)
                
                num += 1
        predict = pd.read_pickle(predict_file)
        assert len(old_mol_list) == len(predict)
        '''
        predict.keys(): dict_keys(['loss', 'cross_distance_loss', 'distance_loss', 'coord_loss', 'prmsd_loss', 'prmsd_score', 'bsz', 'sample_size', 
        'coord_predict', 'coord_target', 'smi_name', 'pocket_name', 'atoms', 'pocket_atoms', 'coordinates', 'holo_coordinates', 'pocket_coordinates', 
        'holo_center_coordinates'])
        '''
        smi_list, pocket_list, coords_predict_list, holo_coords_list, holo_center_coords_list, prmsd_score_list = [],[],[],[],[],[]
        cross_distance_list = []
        pocket_coords_list = []
        holo_pocket_coords_list = []
        
        ligand_emb = []
        pocket_emb = []

        for batch in predict:

            if batch == None:
                raise Exception('error, None')
            else:
                sz = batch['atoms'].size(0)

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

                        holo_coordinates = batch['holo_coordinates'][i]
                        holo_coordinates = holo_coordinates.numpy().astype(np.float32)

                        coord_predict = batch['coord_predict'][i]
                        coord_predict = coord_predict.numpy().astype(np.float32)

                        holo_center_coordinates = batch["holo_center_coordinates"][i][:3]
                        holo_center_coordinates.numpy().astype(np.float32)

                        holo_center_coords_list.append(holo_center_coordinates)        
                        coords_predict_list.append(coord_predict)
                        holo_coords_list.append(holo_coordinates)
                    except Exception as e:
                        raise Exception('error, skip')
                        smi_list.append(None) 
                        pocket_list.append(None)
                        coords_predict_list.append(None)
                        holo_coords_list.append(None)
                        holo_center_coords_list.append(None) 
                        prmsd_score_list.append(None)
        for i in range(len(pocket_coords_list)):
            assert np.allclose(np.array(pocket_coords_list[i]), np.array(holo_pocket_coords_list[i]), rtol=0.00, atol=0.00)

        '''
        np.set_#printoptions(precision=4, suppress=True) 

        A = np.array(pocket_coords_list[0])
        sorted_indices1 = np.lexsort((A[:, 2], A[:, 1], A[:, 0]))
        sorted_A = A[sorted_indices1]

        B = np.array(pocket_coords_list[2])
        sorted_indices2 = np.lexsort((B[:, 2], B[:, 1], B[:, 0]))
        sorted_B = B[sorted_indices2]
        coords1_tuples = {tuple(row) for row in A}
        coords2_tuples = {tuple(row) for row in B}
        intersection = np.array(list(coords1_tuples & coords2_tuples))
        for i in range(len(pocket_coords_list)):
            assert np.allclose(np.array(pocket_coords_list[i]), np.array(holo_pocket_coords_list[i]), rtol=0.00, atol=0.00)
        assert np.allclose(sorted_A, sorted_B, rtol=0.01, atol=0.02)
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
        output_ligand_list = []
        if self.mode == 'single':
            output_ligand_name = [output_ligand_name]
        new_holo_coords_lists    = []
        new_coords_predict_lists = []
        new_pocket_coords_lists  = []
        new_cross_distance_lists = []
        new_ligand_emb_lists     = []
        new_pocket_emb_lists     = []

        for i in tqdm(range(len(smi_list)//tta_times)):
            coords_predict_tta = coords_predict_list[i*tta_times:(i+1)*tta_times]
            prmsd_score_tta = prmsd_score_list[i*tta_times:(i+1)*tta_times]
            mol_list_tta = mol_list[i*tta_times:(i+1)*tta_times]
            holo_center_coords_tta = holo_center_coords_list[i*tta_times:(i+1)*tta_times]

            holo_coords_tta   = holo_coords_list[i*tta_times:(i+1)*tta_times]
            pocket_coords_tta = pocket_coords_list[i*tta_times:(i+1)*tta_times]
            holo_pocket_coords_tta = holo_pocket_coords_list[i*tta_times:(i+1)*tta_times]
            cross_distance_tta = cross_distance_list[i*tta_times:(i+1)*tta_times]
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
                orgin_pos = org_mol.GetConformer(0).GetPositions().astype(np.float32)
                holo_center_coords = centor

                holo_coords_c, pocket_coords_c, holo_pocket_coords_c = np.mean(holo_coords , axis = 0), np.mean(pocket_coords, axis = 0), np.mean(holo_pocket_coords, axis = 0)
                holo_coords, pocket_coords, holo_pocket_coords = holo_coords + np.array(holo_center_coords), pocket_coords + np.array(holo_center_coords), holo_pocket_coords + np.array(holo_center_coords)

                holo_coords_c, pocket_coords_c, holo_pocket_coords_c = np.mean(holo_coords , axis = 0), np.mean(pocket_coords, axis = 0), np.mean(holo_pocket_coords, axis = 0)

                
                orgin_pos = org_mol.GetConformer(0).GetPositions().astype(np.float32)
                
                new_org_mol_list.append(org_mol)
                new_mol = self.set_coord(copy.deepcopy(mol), coords)
                try:
                    new_mol = self.add_coord(new_mol, holo_center_coords.numpy()) 
                except Exception as e:
                    print('e:', e)
                    with open('get_sdf_error.txt', 'a') as f:
                        input_ligand = os.path.join(output_ligand_dir2[i], 'org_' + str(output_ligand_name[i]) + '.sdf')
                        f.write(input_ligand + '\n')
                        continue

                new_pos = new_mol.GetConformer(0).GetPositions()
                new_centor = np.mean(new_pos, axis = 0)
                new_coords = coords + np.array(holo_center_coords)

                
                new_mol_list.append(new_mol)

                new_holo_coords_list.append(holo_coords)
                new_coords_predict_list.append(new_coords)
                new_pocket_coords_list.append(holo_pocket_coords)
                new_cross_distance_list.append(cross_distance)
            
            new_holo_coords_lists.append(new_holo_coords_list)
            new_coords_predict_lists.append(new_coords_predict_list)
            new_pocket_coords_lists.append(new_pocket_coords_list)
            new_cross_distance_lists.append(new_cross_distance_list)

            data_dict = {}
            data_dict['holo_coords_list'] = new_holo_coords_list
            data_dict['coords_predict_list'] = new_coords_predict_list
            data_dict['pocket_coords_list'] = new_pocket_coords_list
            data_dict['cross_distance_list'] = new_cross_distance_list


            if len(new_holo_coords_list) > 2:
                for j in range(len(new_holo_coords_list))[1:]:
                    assert np.allclose(np.array(new_holo_coords_list[j-1]), np.array(new_holo_coords_list[j]), rtol=0.01, atol=0.02)


                np.set_printoptions(precision=4, suppress=True) 

                A = np.array(new_pocket_coords_list[0])
                sorted_indices1 = np.lexsort((A[:, 2], A[:, 1], A[:, 0]))
                sorted_A = A[sorted_indices1]

                B = np.array(new_pocket_coords_list[2])
                sorted_indices2 = np.lexsort((B[:, 2], B[:, 1], B[:, 0]))
                sorted_B = B[sorted_indices2]
                coords1_tuples = {tuple(row) for row in A}
                coords2_tuples = {tuple(row) for row in B}
                intersection = np.array(list(coords1_tuples & coords2_tuples))
            

            '''
            assert np.allclose(sorted_A, sorted_B, rtol=0.01, atol=0.02)
            assert np.allclose(np.sum(np.array(new_pocket_coords_list[0])), np.sum(np.array(new_pocket_coords_list[-1])), rtol=0.01, atol=0.02)
            assert np.allclose(np.array(new_pocket_coords_list[0]), np.array(new_pocket_coords_list[-1]), rtol=0.01, atol=0.02)
            '''
            os.makedirs(output_ligand_dir2[i], exist_ok=True)
            outputfilename = os.path.join(output_ligand_dir2[i], 'interaction_' + str(output_ligand_name[i]) + '.pkl')
            with open(outputfilename, "wb") as f:
                dill.dump(data_dict, f)

            outputfilename = os.path.join(output_ligand_dir2[i], 'org_' + str(output_ligand_name[i]) + '.sdf')
            w = Chem.SDWriter(outputfilename)
            for mol in new_org_mol_list[:1]:
                new_mol = Chem.RemoveHs(mol)
                w.write(new_mol)
            w.close()



            outputfilename = os.path.join(output_ligand_dir2[i], 'gen_' + str(output_ligand_name[i]) + '.sdf')
            try:
                os.remove(outputfilename)
            except:
                pass
            w = Chem.SDWriter(outputfilename)
            for mol in new_mol_list:
                new_mol = Chem.RemoveHs(mol)
                w.write(new_mol)
            w.close()

            output_ligand_list.append(outputfilename)
        if self.mode == 'single':
            return output_ligand_list[0]
        elif self.mode in ['batch_one2one', 'batch_one2many']:
            return output_ligand_list
    
    def single_clash_fix(self, input_content):
        input_ligand, output_ligand, label_ligand, pocket_mol = input_content
        script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "generalmodels", "scripts", "6tsr.py")
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
