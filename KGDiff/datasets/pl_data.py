import torch
import torch_scatter
import numpy as np
from torch_geometric.data import Data
#from torch_geometric.data import DataLoader
from torch_geometric.loader import DataLoader
FOLLOW_BATCH = ('protein_element', 'ligand_element', 'ligand_bond_type', 'protein_link_t', 'protein_link_t_reverse') #这个表示在形成batch时，对这些数量进行标识，即区分是哪一个图上的


class ProteinLigandData(Data):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
        '''

        pocket_dict = {
        'element': np.array(self.element, dtype=np.int64), #原子序号
        'molecule_name': self.title, #固定为 pocket
        'pos': np.array(self.pos, dtype=np.float32), #所有原子坐标
        'is_backbone': np.array(self.is_backbone, dtype=np.bool_), #Boolean值，是否是主干原子
        'atom_name': self.atom_name, #名字不同于元素周期表中的化学符号
        'atom_to_aa_type': np.array(self.atom_to_aa_type, dtype=np.int64) ##每个原子所在的氨基酸残基
        }
        
        ligand_dict = {
        'smiles': Chem.MolToSmiles(rdmol),
        'element': element, #原子序号
        'pos': pos, #坐标，直接从rdkit mol 对象读取的
        'bond_index': edge_index, #已经排序了，不知道有什么作用？
        'bond_type': edge_type,
        'center_of_mass': center_of_mass,
        'atom_feature': feat_mat,
        'hybridization': hybridization #杂化
        }

        '''

    @staticmethod
    def from_protein_ligand_dicts(protein_dict=None, ligand_dict=None, **kwargs):
        instance = ProteinLigandData(**kwargs) #

        if protein_dict is not None:
            for key, item in protein_dict.items():
                instance['protein_' + key] = item

        if ligand_dict is not None:
            for key, item in ligand_dict.items():
                instance['ligand_' + key] = item
        
        ##print('instance1:', instance)
        #raise Exception('stop23')

        
        #这是什么？每一个原子的邻居。 ligand_bond_index == ligand_edge_index. 获取的是所有原子直接邻居，因为构建的是无向图（双向图），所以只需要遍历ligand_bond_index[0]的原子
        instance['ligand_nbh_list'] = {i.item(): [j.item() for k, j in enumerate(instance.ligand_bond_index[1])
                                                if instance.ligand_bond_index[0, k].item() == i]
                                    for i in instance.ligand_bond_index[0]} 


        '''
        #print('instance.ligand_bond_index[0]:', instance.ligand_bond_index[0])
        temp = {}
        for i in instance.ligand_bond_index[0]:
            #print('i:', i)
            tg = []
            for k, j in enumerate(instance.ligand_bond_index[1]):
                if instance.ligand_bond_index[0, k].item() == i:
                    tg.append(j.item())
            temp[i.item()] = tg
        '''

        
        ##print('instance2:', instance)
        #raise Exception('stop24')
        return instance

    def __inc__(self, key, value, *args, **kwargs): #这里的k和v来自哪里？和from_protein_ligand_dicts()有关系？

        ##print('key:', key)
        ##print('value:', value)
        '''
        在 Python 中，__inc__() 方法是 __index__() 方法的别名。这个方法用于将对象转换为整数索引。当对象被用作序列的索引时，
        Python 会尝试调用该对象的 __index__() 方法（如果存在）来获取其整数值。
        '''
        if key == 'ligand_bond_index':
            return self['ligand_element'].size(0)
        else:
            return super().__inc__(key, value)


class ProteinLigandDataLoader(DataLoader):

    def __init__(
            self,
            dataset,
            batch_size=1,
            shuffle=False,
            follow_batch=FOLLOW_BATCH,
            **kwargs
    ):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, follow_batch=follow_batch, **kwargs)


def torchify_dict(data):
    output = {}   #'mutiple_rd_pos', 'link_e', 'link_e_reverse',
    for k, v in data.items():
        if isinstance(v, np.ndarray) and k not in ['mutiple_rd_pos', 'cross_distance', 'cross_bond_index', 'cross_bond_type', 'cross_bond_index_reverse', 'cross_bond_type_reverse'][:3]:
            output[k] = torch.from_numpy(v)
        else:
            output[k] = v
    return output


def get_batch_connectivity_matrix(ligand_batch, ligand_bond_index, ligand_bond_type, ligand_bond_batch):
    batch_ligand_size = torch_scatter.segment_coo(
        torch.ones_like(ligand_batch),
        ligand_batch,
        reduce='sum',
    )
    batch_index_offset = torch.cumsum(batch_ligand_size, 0) - batch_ligand_size
    batch_size = len(batch_index_offset)
    batch_connectivity_matrix = []
    for batch_index in range(batch_size):
        start_index, end_index = ligand_bond_index[:, ligand_bond_batch == batch_index]
        start_index -= batch_index_offset[batch_index]
        end_index -= batch_index_offset[batch_index]
        bond_type = ligand_bond_type[ligand_bond_batch == batch_index]
        # NxN connectivity matrix where 0 means no connection and 1/2/3/4 means single/double/triple/aromatic bonds.
        connectivity_matrix = torch.zeros(batch_ligand_size[batch_index], batch_ligand_size[batch_index],
                                          dtype=torch.int)
        for s, e, t in zip(start_index, end_index, bond_type):
            connectivity_matrix[s, e] = connectivity_matrix[e, s] = t
        batch_connectivity_matrix.append(connectivity_matrix)
    return batch_connectivity_matrix
