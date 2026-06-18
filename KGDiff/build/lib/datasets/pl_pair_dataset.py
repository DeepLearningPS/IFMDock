import os
import sys
sys.path.append(os.path.abspath('./'))

import pickle
import lmdb
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from KGDiff.utils.data import PDBProtein, parse_sdf_file
from KGDiff.datasets.pl_data import ProteinLigandData, torchify_dict
from KGDiff.scripts.data_preparation.clean_crossdocked import TYPES_FILENAME
import torch
import numpy as np
from collections import defaultdict
from collections import defaultdict
from ordered_set import OrderedSet
import dill

import gzip
import shutil

from Bio.PDB import PDBParser, PDBIO, Select
from Bio import PDB


class PocketLigandPairDataset(Dataset):

    def __init__(self, raw_path, transform=None, version='final'):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')
        self.index_path = os.path.join(self.raw_path, 'index.pkl')
        self.processed_path = os.path.join(os.path.dirname(self.raw_path),
                                           os.path.basename(self.raw_path) + f'_processed_{version}.lmdb')
        self.raw_affinity_path = os.path.join('/data/qianhao', TYPES_FILENAME)
        self.affinity_path = os.path.join('data', 'affinity_info_complete.pkl')
        self.transform = transform
        self.db = None
        self.keys = None
        self.affinity_info = None
        
        if not os.path.exists(self.processed_path):
            print(f'{self.processed_path} does not exist, begin processing data')
            self._process()
        
        # len
        if self.db is None:
            self._connect_db()
        self.lengths =  len(self.keys)

        #dataset,提前把数据构建好
        self.datas_list = []
        for idx in range(self.lengths):
            data = self.get_ori_data(idx)
            if self.transform is not None:
                data = self.transform(data)
            
            self.datas_list.append(data)

            
    def _load_affinity_info(self):
        if self.affinity_info is not None:
            return
        if os.path.exists(self.affinity_path):
            with open(self.affinity_path, 'rb') as f:
                affinity_info = pickle.load(f)
        else:
            affinity_info = {}
            with open(self.raw_affinity_path, 'r') as f:
                for ln in tqdm(f.readlines()):
                    # <label> <pK> <RMSD to crystal> <Receptor> <Ligand> # <Autodock Vina score>
                    label, pk, rmsd, protein_fn, ligand_fn, vina = ln.split()
                    ligand_raw_fn = ligand_fn[:ligand_fn.rfind('.')]
                    affinity_info[ligand_raw_fn] = {
                        'label': float(label),
                        'rmsd': float(rmsd),
                        'pk': float(pk),
                        'vina': float(vina[1:])
                    }
            # save affinity info
            with open(self.affinity_path, 'wb') as f:
                pickle.dump(affinity_info, f)
        
        self.affinity_info = affinity_info
        
    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=False,
            subdir=False,
            readonly=False,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None
        
    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)

        num_skipped = 0
        with db.begin(write=True, buffers=True) as txn:
            for i, (pocket_fn, ligand_fn, *_) in enumerate(tqdm(index)):
                if pocket_fn is None: continue
                try:
                    data_prefix = self.raw_path
                    pocket_dict = PDBProtein(os.path.join(data_prefix, pocket_fn)).to_dict_atom()
                    ligand_dict = parse_sdf_file(os.path.join(data_prefix, ligand_fn))
                    data = ProteinLigandData.from_protein_ligand_dicts(
                        protein_dict=torchify_dict(pocket_dict),
                        ligand_dict=torchify_dict(ligand_dict),
                    )
                    data.protein_filename = pocket_fn
                    data.ligand_filename = ligand_fn
                    data = data.to_dict()  # avoid torch_geometric version issue
                    txn.put(
                        key=str(i).encode(),
                        value=pickle.dumps(data)
                    )
                except:
                    num_skipped += 1
                    print('Skipping (%d) %s' % (num_skipped, ligand_fn, ))
                    continue
        db.close()
    
    def __len__(self):
        return self.lengths

    def __getitem__(self, idx):
        return self.datas_list[idx]
    
    def _update(self, sid, affinity):
        if self.db is None:
            self._connect_db()
        txn = self.db.begin(write=True)
        data = pickle.loads(txn.get(sid))
        data.update({
            'affinity': affinity['vina'],
            'rmsd': affinity['rmsd'],
            'pk': affinity['pk'],
            'rmsd<2': affinity['label']
        })
        txn.put(
            key=sid,
            value=pickle.dumps(data)
        )
        txn.commit()

    def _inject_affinity(self, sid, ligand_path):
        if ligand_path[:-4] in self.affinity_info:
            affinity = self.affinity_info[ligand_path[:-4]]
            self._update(sid, affinity)
        else:
            raise AttributeError(f'affinity_info has no {ligand_path[:-4]}')
            
    def get_ori_data(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        if 'affinity' not in data:
            self._load_affinity_info()
            self._inject_affinity(key, data['ligand_filename'])
            data = pickle.loads(self.db.begin().get(key))
        
        data = ProteinLigandData(**data)
        data.id = idx
        assert data.protein_pos.size(0) > 0
        return data




def is_standard_amino_acid(residue):
    standard_aa = {
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
        'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
    }
    return residue.get_resname() in standard_aa

def is_metal_ion(residue):
    # Common metal ions in PDB files
    metal_ions = {'NA', 'K', 'MG', 'CA', 'MN', 'FE', 'CO', 'NI', 'CU', 'ZN'}
    return residue.get_resname() in metal_ions

def is_water(residue):
    return residue.get_resname() == 'HOH'




def pdb2020_filter_pdb(path, data_name):
    #base_path = data_name  #'posebusters'
    #data_name: posebusters/5SAK/5SAK_protein.pdb

        pdb_file = os.path.join(path, data_name)
        output_pdb_file = pdb_file


        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_file)
        
        io = PDB.PDBIO()
        io.set_structure(structure)
        
        class NonStandardResidueSelect(PDB.Select):
            def accept_residue(self, residue):
                return is_standard_amino_acid(residue) and not is_metal_ion(residue) and not is_water(residue)
        
        io.save(output_pdb_file, NonStandardResidueSelect())



class PDBPairDataset(Dataset): #from torch.utils.data import Dataset 不是来自于PyG

    def __init__(self, raw_path, data_flag='old_test', single_test = False, protein = None, ligand = None, data_name = 'pdbbind2020_r10', transform=None, version='final'):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')
        self.index_path = os.path.join(self.raw_path, 'index.pkl')
        self.data_flag = data_flag
        self.data_name = data_name

        self.processed_path = os.path.join(os.path.dirname(self.raw_path),
                                    os.path.basename(self.raw_path) + f'_processed_{version}_{self.data_name}_interaction_gen_split3_5_sub.lmdb')
                                    #os.path.basename(self.raw_path) + f'_processed_{version}_{self.data_name}.lmdb') V4代表使用了新的简化方法
        self.transform = transform
        self.db = None
        self.keys = None
        self.exclude = {} #剔除错误的索引
        self.exclude_name = OrderedSet() #错误的名字
        self.nameid2id_dict = {} #name_id:id

        if not os.path.exists(self.processed_path):
            print(f'{self.processed_path} does not exist, begin processing data')
            self._process()
        else:
            self._process()
        
        #现在的问题是如果跳过zmats错误的，则index读数据时会截断，如果不跳过，这些错误的数据没有zmats也不行，现在解决方法：一种纠正zmats错误，一种找到数据截断的地方，到底是如何截断
        #de,为什么没有报错

        name2id_dict_file = os.path.join(os.path.dirname(self.raw_path), f'{self.data_name}_name2id_dict.txt')


        with open(name2id_dict_file) as f:
            for i in f:
                tg = i.strip('\n').split('\t')
                if tg[1] == '0':
                    print("tg[1] == '0':", i)
                self.nameid2id_dict[tg[0]] = tg[1]

        self.name2nameid = defaultdict(list) #因为输入的是name，而这里数据库的key是name_id, 所以再映射一下，name:nameid_list 
        for i in self.nameid2id_dict:
            tg = i.split('_') #得到name, id
            assert len(tg) == 2
            if tg[1] == '0':
                print("tg[1] == '0':", i)
            self.name2nameid[tg[0]].append(i)
        
        assert '0' not in list(self.name2nameid.keys()) # '0'来自哪里？

        print('self.nameid2id_dict num:', len(self.nameid2id_dict))
        print('self.name2nameid num:', len(self.name2nameid))

        print('self.processed_path:', self.processed_path)

        '''
        # len
        if self.db is None: #这个必然执行的，即打开数据库
            self._connect_db()
        self.lengths =  len(self.keys)

        print('self.keys:', self.keys)
        print('self.keys:', type(self.keys)) #type: list
        print('self.keys:', len(self.keys)) #self.keys: 16251

        #dataset,提前把数据构建好
        self.datas_list = []
        for idx in range(self.lengths): #id可能不用应该range(self.lengths)，而是self.keys
            data = self.get_ori_data(idx)
            if self.transform is not None:
                data = self.transform(data) #如若这里添加芳香原子，那么zmats矩阵的时候也要扩充
            
            self.datas_list.append(data)
        '''

        #exit()
    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024*1024),   # 100GB #这个可能开小的，导致数据没有写进去.是的。如果数据量很大，务必把数据库给扩大
            create=False,
            subdir=False,
            readonly=False,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False)) #创建一个迭代器，values=False表示只返回键，不返回值。数组越界，不报错，可能是这里的迭代器比较特殊导致的？
            #self.keys = list(txn.cursor())
        
        with open(os.path.join(os.path.dirname(self.processed_path), 'exclude_index.txt'))as f:
            for i in f:
                self.exclude[int(i.strip('\n')) - 2] = int(i.strip('\n')) - 2

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None


    def _process(self):
        #更为直接简单的处理方法
        db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 100GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)

        error_file = open(os.path.join(os.path.dirname(self.processed_path), 'error_file.txt'), 'w')
        num_skipped = 0
        num_pocket  = 0
        num_zmats   = 0
        num_success = 0
        
        count = 0
        nameid2id_dict = {}


        with db.begin(write=True, buffers=True) as txn:
            error_list = [
                'posebusters/5SAK/5SAK_protein.pdb',
                'posebusters/6YDY/6YDY_protein.pdb',
                'posebusters/6YQV/6YQV_protein.pdb',
                'posebusters/6YRV/6YRV_protein.pdb',
                'posebusters/7BA0/7BA0_protein.pdb',
                'posebusters/7LMO/7LMO_protein.pdb',
                'posebusters/7POM/7POM_protein.pdb',
                'posebusters/7PRM/7PRM_protein.pdb',
                'posebusters/7U0U/7U0U_protein.pdb',
                'posebusters/7ZXV/7ZXV_protein.pdb',
                'posebusters/8D5D/8D5D_protein.pdb',
                'posebusters/8DP2/8DP2_protein.pdb',
                'posebusters/7S9H/7S9H_protein.pdb'

            ]
            for i, (pocket_fn, protein_fn, (pka, year, resl), ligand_fn, pdbid) in enumerate(tqdm(index[:1000])): #先使用前100个

                #if not os.path.isfile(os.path.join(self.raw_path, pocket_fn)):
                    #pocket_fn = protein_fn
            
                #if self.data_flag == 'new_test':
                    #pocket_fn = protein_fn #新数据集是没有口袋蛋白的，所以使用全蛋白，然后再获取口袋蛋白。另外新数据集没有亲和度，所以这里的亲和度是假的
                
                #因为我们还要再对蛋白进行切分，所以这里无论是否有蛋白口袋，我们都要按自己的标准再对全蛋白进行切分，所以使用全蛋白信息：
                pocket_fn = protein_fn

                print('pocket_fn:', pocket_fn)

                #if pocket_fn is None or pocket_fn in ['posebusters/5SAK/5SAK_protein.pdb', 'posebusters/6YDY/6YDY_protein.pdb', 'posebusters/6YQV/6YQV_protein.pdb']: 
                if pocket_fn is None:
                #if pocket_fn not in error_list:
                    continue
                
                select_list = [
                            'posebusters/5SAK/5SAK_protein.pdb', #配体的中一个氢，无法通过rdkit去除，导致和unimol保存的数量不一样
                            'posebusters/8D5D/8D5D_protein.pdb',
                            

                            'posebusters/6YRV/6YRV_protein.pdb', #连接为空，没有ON环，是个碳链


                            'posebusters/6YDY/6YDY_protein.pdb', #氨基酸不等量错误
                            'posebusters/6YQV/6YQV_protein.pdb',
                            'posebusters/7LMO/7LMO_protein.pdb',
                            'posebusters/7PRM/7PRM_protein.pdb',
                            'posebusters/7POM/7POM_protein.pdb',

                            


                            'posebusters/7U0U/7U0U_protein.pdb',  #rdkit 生成失败错误
                            'posebusters/7ZXV/7ZXV_protein.pdb',
                            'posebusters/7BA0/7BA0_protein.pdb',


                            'posebusters/8DP2/8DP2_protein.pdb', #unimol保存的蛋白坐标有重复，去掉
                            'posebusters/7S9H/7S9H_protein.pdb',

                            ]

                #if pocket_fn in select_list:
                    #continue
                    
                #如果不存在，则从数据库复制一份
                if self.data_name == 'pdbbind2020_r10' and not os.path.isfile(os.path.join(self.raw_path, pocket_fn)):
                    s_file = os.path.join('../CrossDocked2020/data/pdbbind2020/pdbbind2020',  '/'.join(pocket_fn.split('/')[:]))
                    t_file = os.path.join(self.raw_path, pocket_fn)
                    try:
                        shutil.copy(s_file, t_file)
                    except Exception as e:
                        print(e)
                        continue

                


                #清洗蛋白
                pdb2020_filter_pdb(self.raw_path, pocket_fn)



                try:
                    data_prefix = self.raw_path
                    ligand_dict = parse_sdf_file(os.path.join(data_prefix, ligand_fn), self.data_flag)
                    ligand_centor = np.mean(ligand_dict['pos'], axis=0)

                    #n = os.path.basename(pocket_fn).split('_')[0]
                    #try:
                    #unimol_pcoords = unimol_pcoords_dict[n] #找对应复合物的unimol生成的蛋白原子
                    #except Exception as e:
                        #print(e)
                        #print('unimol_pcoords_dict.keys:', list(unimol_pcoords_dict.keys()))
                        #print('error key:', n)
                        #exit()
                        #continue
                    #pocket_dict = PDBProtein(os.path.join(data_prefix, pocket_fn), ligand_centor, ligand_dict, unimol_pcoords).to_dict_atom_unimol() #读蛋白时可能出错 
                    
                    #pocket_dict = PDBProtein(os.path.join(data_prefix, pocket_fn), ligand_centor, ligand_dict, self.data_flag, unimol_pcoords = 
                                            #None).to_dict_atom_interaction() #读蛋白时可能出错
                    
                    #使用真实的连接表
                    #pocket_dict = PDBProtein(os.path.join(data_prefix, pocket_fn), ligand_centor, ligand_dict, self.data_flag, unimol_pcoords = 
                                            #None).to_dict_atom_interaction_org() #读蛋白时可能出错



                    #新的简化方法
                    #pocket_dict = PDBProtein(os.path.join(data_prefix, pocket_fn), ligand_centor, ligand_dict, self.data_flag, unimol_pcoords = 
                                            #None).to_dict_atom_interaction_v2() #读蛋白时可能出错, 包含参考的距离矩阵，另外我们在这里构建配体-蛋白连接表，
                    
                    #方法模仿配体连接表，只要保存蛋白原子顺序不乱，之后在模型构建knn时，把局部节点id替换换成全局即可。之后，要上40个距离矩阵，先考虑测试集

                    #使用真实的连接表
                    #pocket_dict = PDBProtein(os.path.join(data_prefix, pocket_fn), ligand_centor, ligand_dict, self.data_flag, unimol_pcoords = 
                                            #None).to_dict_atom_interaction_v2_org()


                    #新训练，划分<3.5和3.5~4.5两种类型
                    #划分相互作用连接表距离小于<3.5和3.5~4.5，两种集合，总共4种关系，配体到蛋白，和蛋白到配体，我们的键类型采样预扩充方法，总长度为20，可以容纳2种关系，
                    pocket_dict = PDBProtein(os.path.join(data_prefix, pocket_fn), ligand_centor, ligand_dict, self.data_flag, unimol_pcoords = 
                                            None).to_dict_atom_interaction_gen_split3_5()
                    

                    

                    data = ProteinLigandData.from_protein_ligand_dicts(
                        protein_dict=torchify_dict(pocket_dict),
                        ligand_dict=torchify_dict(ligand_dict),
                    )
                    data.protein_filename = pocket_fn
                    data.ligand_filename = ligand_fn
                    data.affinity = pka
                    n = os.path.basename(pocket_fn).split('_')[0]
                    data.name = n #记录复合物的名字
                    data = data.to_dict()  # avoid torch_geometric version issue
                    assert data['protein_pos'].size(0) > 0 #在未使用zmats之前，能报错的是这个地方


                    txn.put(
                        key=(n + '_' + str(i)).encode(), #lmdb要求把int索引字符串编码成字节串. 为了我们方便操作数据集，这里使用复合物的名字+id，同时保存id号到映射复合物的映射关系
                        value=pickle.dumps(data)
                    )
                    num_success += 1
                    nameid2id_dict[n + '_' + str(i)] = str(i)
                
                
                except (AssertionError, ValueError, TypeError, OSError, SystemExit) as e:
                #except Exception as e: #SystemExit不在Exception里面
                    self.exclude[count] = count
                    self.exclude_name.add(protein_fn)
                    data = None
                    n = os.path.basename(pocket_fn).split('_')[0]
                    txn.put(
                    key=(n + '_' + str(i)).encode(), #lmdb要求把int索引字符串编码成字节串
                    value=pickle.dumps(data)
                    )
                    count += 1
                    nameid2id_dict[n + '_' + str(i)] = str(i)


                    self.exclude_name.add(protein_fn)
                    error_file.write(f'error: {e}\n')
                    error_file.write(f'type(e): {type(e)}\n')   
                    print('error:', e)
                    print(f"异常类型: {type(e)}")
                    num_skipped += 1
                    error_file.write('Skipping ligand_fn (%d) %s \n' % (num_skipped, ligand_fn))
                    error_file.write('Skipping pocket_fn (%d) %s \n' % (num_skipped, pocket_fn))
                    print('Skipping (%d) %s' % (num_skipped, ligand_fn, ))
                    print('complex name:', n)
                    #exit()
                    continue
                    
            
                    
                
                
                
                
                
                
                
            
                

        error_file.write(f'num_skipped: {num_skipped}\n') 
        error_file.write(f'num_pocket: {num_pocket}\n')
        error_file.write(f'num_zmats: {num_zmats}\n')
        error_file.write(f'num_success: {num_success}\n')
        error_file.write(f'num_error: {count}\n')
        error_file.close()
        print('num_skipped:', num_skipped)
        print(f'num_pocket: {num_pocket}')
        print(f'num_zmats: {num_zmats}')
        print(f'num_success: {num_success}')
        print(f'num_error: {count}')

        self.nameid2id_dict = nameid2id_dict


        file_name = os.path.join(os.path.dirname(self.processed_path), f'{self.data_name}_name2id_dict.txt')


        with open(file_name, 'w')as f:
            for k, v in nameid2id_dict.items():
                f.write(k + '\t' + v + '\n')

        db.close()

        with open(os.path.join(os.path.dirname(self.processed_path), 'exclude_index.txt'), 'w') as f:
            for i in self.exclude.keys():
                f.write(str(i) + '\n')

        with open(os.path.join(os.path.dirname(self.processed_path), 'exclude_name.txt'), 'w') as f:
            for i in self.exclude_name:
                f.write(str(i) + '\n')
    
    '''
    def __len__(self):
        return self.lengths

    def __getitem__(self, idx):
        return self.datas_list[idx]
    #这是数据加载慢的一个重要原因，每读取一个数据，都要进行变换,我们把这部分操作挪到init()里面，进行预处理
    '''

    
    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def odl__getitem__(self, idx):
        data = self.get_ori_data(idx)
        if self.transform is not None and data is not None:
            try:
                data = self.transform(data)
            except Exception as e:
                print('data:', data) #[None]
                print('transform error:', e)
                data = data
        return data

    def __getitem__(self, idx):
        data_list = self.get_ori_data(idx)
        new_data_list = []
        for data in data_list: #list中可能存在None
            if self.transform is not None and data is not None:
                new_data = self.transform(data)
                new_data_list.append(new_data)
            else:
                new_data_list.append(data)

        return new_data_list



    def get_ori_data(self, idx):
        if self.db is None:
            self._connect_db()

        #print('self.name2nameid num 1:', len(self.name2nameid)) #self.name2nameid num: 16251
        key_list = self.name2nameid[idx] #给出复合物的名字，直接返回数据库中的所有信息key，此时data可能不止一个，尤其是crossdock，因为函数最终反复的是data_list,因此其它地方也要改
        #print('idx:', idx) # 0 ?
        #print('self.nameid2id_dict num:', len(self.nameid2id_dict)) #self.nameid2id_dict num: 16251
        #print('self.name2nameid num:', len(self.name2nameid)) #self.name2nameid num: 16251，

        #print('self.nameid2id_dict:', self.nameid2id_dict[list(self.nameid2id_dict.keys())[-1]])
        #print('self.name2nameid key:', list(self.name2nameid.keys())[-1])
        #print('self.name2nameid value:', self.name2nameid[list(self.name2nameid.keys())[-1]])
        #print('key_list:', key_list)  #[]

        data_list = [pickle.loads(self.db.begin().get(key.encode())) for key in key_list]
        #print('data_list:', data_list)
        
        new_data_list = []

        for data in data_list:
            try:
                data = ProteinLigandData(**data)
                data.id = idx
                assert data.protein_pos.size(0) > 0
            except Exception as e:
                #print('error id:', idx) #error id: 734
                print('error indices[idx]:', idx) #error indices[idx]: 14147 #这这里约束，不用修改torch data包
                data = None
                new_data_list.append(data)
                continue
        

            if max(data.protein_element) > 17 or max(data.ligand_element) > 17: 
                #print('batch.protein_element:', batch.protein_element)
                #print('batch.ligand_element:', batch.ligand_element)
                print('max(batch.protein_element), max(batch.ligand_element) > 17 ?:', max(data.protein_element), max(data.ligand_element))
                #raise Exception(f'>17')
                data = None
                new_data_list.append(data)
                continue
            

            '''
            'atom_isring': new_atom_isring[nonzero_indices],
            'atom_isO': new_atom_isO[nonzero_indices],
            'atom_isN': new_atom_isN[nonzero_indices],
            '''


            if len(data.protein_atom_isring) == 0 and len(data.protein_atom_isO) == 0 and len(data.protein_atom_isN) == 0:  #如果模糊作用不存在，则去掉
                data = None
                new_data_list.append(data)
                continue

        
            if len(data.ligand_atom_isring) == 0 and len(data.ligand_atom_isO) == 0 and len(data.ligand_atom_isN) == 0:  #如果模糊作用不存在，则去掉
                data = None
                new_data_list.append(data)
                continue
            
            new_data_list.append(data)
        
        return new_data_list 

    def get_ori_data_old(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx] #如果idx不在self.keys会如何？会不会出现idx不在self.keys中？会出现，并且这里不会报索引出错
        data = pickle.loads(self.db.begin().get(key))

        try:
            data = ProteinLigandData(**data)
            data.id = idx
            assert data.protein_pos.size(0) > 0
        except Exception as e:
            #print('error id:', idx) #error id: 734
            print('error indices[idx]:', idx) #error indices[idx]: 14147 #这这里约束，不用修改torch data包
            return None
    

        if max(data.protein_element) > 17 or max(data.ligand_element) > 17: 
            #print('batch.protein_element:', batch.protein_element)
            #print('batch.ligand_element:', batch.ligand_element)
            print('max(batch.protein_element), max(batch.ligand_element) > 17 ?:', max(data.protein_element), max(data.ligand_element))
            #raise Exception(f'>17')
            return None
        

        '''
        'atom_isring': new_atom_isring[nonzero_indices],
        'atom_isO': new_atom_isO[nonzero_indices],
        'atom_isN': new_atom_isN[nonzero_indices],
        '''
        if len(data.protein_atom_isring) == 0 and len(data.protein_atom_isO) == 0 and len(data.protein_atom_isN) == 0:  #如果模糊作用不存在，则去掉
            return None
    
        if len(data.ligand_atom_isring) == 0 and len(data.ligand_atom_isO) == 0 and len(data.ligand_atom_isN) == 0:  #如果模糊作用不存在，则去掉
            return None
        
        return data 
    


if __name__ == '__main__':

    dataset = PDBPairDataset('./data/pdbbind2020/')
    print(len(dataset), dataset[0])
