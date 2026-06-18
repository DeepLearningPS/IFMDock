import math
from typing import Any, Callable, Iterable, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from tqdm.auto import tqdm



from typing import Iterator

from torch import Tensor, nn
from EcConf.comparm import *

import random
import numpy as np
from torch_scatter import scatter_sum, scatter_mean
import torch.nn.functional as F
import copy
import random

import networkx as nx
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Rotation as R
from rdkit.Chem import AllChem, rdMolTransforms
from rdkit import Geometry
from rdkit.Chem import AllChem, GetPeriodicTable, RemoveHs
#from models.molopt_score_model import center_pos, index_to_log_onehot, q_v_sample

from rdkit.Geometry.rdGeometry import Point3D
import time

from EcConf.utils.utils_torch import *

from TreeInvent2.model.consistency.consistency_models import opt_coords_moves,opt_complex_coords_moves


np.random.seed(2023)
torch.manual_seed(2023)
random.seed(2023)
torch.cuda.manual_seed_all(2023)



def Change_Mol_D3coord(inputmol,coords):
    '''
        用生成的坐标替换原始的构象坐标，inputmol分子构象，coords：坐标
    '''
    molobj=copy.deepcopy(inputmol)
    conformer=molobj.GetConformer()
    id=conformer.GetId()
    for cid,xyz in enumerate(coords):
        ##print(xyz[0],xyz[1],xyz[2],type(xyz))
        conformer.SetAtomPosition(cid,Point3D(float(xyz[0]),float(xyz[1]),float(xyz[2]))) #更新构象每个原子的坐标
    conf_id=molobj.AddConformer(conformer)
    molobj.RemoveConformer(id)
    molobj = Chem.RemoveHs(molobj)
    return molobj



def pad_dims_like(x: Tensor, other: Tensor) -> Tensor:
    """Pad dimensions of tensor `x` to match the shape of tensor `other`.

    Parameters
    ----------
    x : Tensor
        Tensor to be padded.
    other : Tensor
        Tensor whose shape will be used as reference for padding.

    Returns
    -------
    Tensor
        Padded tensor with the same shape as other.
    """
    ndim = other.ndim - x.ndim
    return x.view(*x.shape, *((1,) * ndim))


def _update_ema_weights_old(
    ema_weight_iter: Iterator[Tensor],
    online_weight_iter: Iterator[Tensor],
    ema_decay_rate: float,
) -> None:
    for ema_weight, online_weight in zip(ema_weight_iter, online_weight_iter):
        if ema_weight.data is None:
            ema_weight.data.copy_(online_weight.data)
        else:
            ema_weight.data.lerp_(online_weight.data, 1.0 - ema_decay_rate)





def _update_ema_weights(
    ema_weight_iter: Iterator[Tensor],
    online_weight_iter: Iterator[Tensor],
    ema_decay_rate: float,
    mode = None
) -> None:
    for ema_weight, online_weight in zip(ema_weight_iter, online_weight_iter):
        if ema_weight.data is None:
            ema_weight.data.copy_(online_weight.data)
        else:
            try:
                ema_weight.data.lerp_(online_weight.data, 1.0 - ema_decay_rate) #在更新非可学习参数时，存在整数参数，所以不可以通过小数来更新权重，这里直接跳过
            except Exception as e:
                if mode == 'buffers':
                    #print'error:', e)
                    #print'ema_decay_rate:', ema_decay_rate)
                    #print'ema_weight.data:', ema_weight.data)
                    #print'online_weight.data:', online_weight.data)
                    #print'skip update buffers')
                    #如果无法更新，则跳过
                    #if online_model != None:
                        ##print'online_model:', online_model)
                    #exit()
                    pass
                else:
                    #print'error:', e)
                    #print'ema_decay_rate:', ema_decay_rate)
                    #print'ema_weight.data:', ema_weight.data)
                    #print'online_weight.data:', online_weight.data)
                    #print'error update params')
                    #如果无法更新，则跳过
                    #if online_model != None:
                        ##print'online_model:', online_model)
                    exit()


def update_ema_model(
    ema_model: nn.Module, online_model: nn.Module, ema_decay_rate: float
) -> nn.Module:
    """Updates weights of a moving average model with an online/source model.

    Parameters
    ----------
    ema_model : nn.Module
        Moving average model.
    online_model : nn.Module
        Online or source model.
    ema_decay_rate : float
        Parameter that controls by how much the moving average weights are changed.

    Returns
    -------
    nn.Module
        Updated moving average model.
    """
    # Update parameters
    _update_ema_weights(
        ema_model.parameters(), online_model.parameters(), ema_decay_rate
    )
    #exit()
    ##print'online_model:', online_model)
    # Update buffers
    #为啥缓存区有long型数据呀，没法更新缓存区？？？
    _update_ema_weights(ema_model.buffers(), online_model.buffers(), ema_decay_rate, mode = 'buffers')

    return ema_model
    
def timesteps_schedule(
    current_training_step: int,
    total_training_steps: int,
    initial_timesteps: int = 2,
    final_timesteps: int = 25,
) -> int:
    """Implements the proposed timestep discretization schedule.

    Parameters
    ----------
    current_training_step : int
        Current step in the training loop.
    total_training_steps : int
        Total number of steps the model will be trained for.
    initial_timesteps : int, default=2
        Timesteps at the start of training.
    final_timesteps : int, default=150
        Timesteps at the end of training.

    Returns
    -------
    int
        Number of timesteps at the current point in training.
    """
    num_timesteps = final_timesteps**2 - initial_timesteps**2
    num_timesteps = current_training_step * num_timesteps / total_training_steps
    num_timesteps = math.ceil(math.sqrt(num_timesteps + initial_timesteps**2) - 1)

    return num_timesteps + 1  #返回值的取值在[initial_timesteps,final_timesteps],所以final_timesteps要改成25


def ema_decay_rate_schedule(
    num_timesteps: int, initial_ema_decay_rate: float = 0.95, initial_timesteps: int = 2
) -> float:
    """Implements the proposed EMA decay rate schedule.

    Parameters
    ----------
    num_timesteps : int
        Number of timesteps at the current point in training.
    initial_ema_decay_rate : float, default=0.95
        EMA rate at the start of training.
    initial_timesteps : int, default=2
        Timesteps at the start of training.

    Returns
    -------
    float
        EMA decay rate at the current point in training.
    """
    return math.exp(
        (initial_timesteps * math.log(initial_ema_decay_rate)) / num_timesteps
    )


def karras_schedule(
    num_timesteps: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    device: torch.device = None,
) -> Tensor:
    """Implements the karras schedule that controls the standard deviation of
    noise added.

    Parameters
    ----------
    num_timesteps : int
        Number of timesteps at the current point in training.
    sigma_min : float, default=0.002
        Minimum standard deviation.
    sigma_max : float, default=80.0
        Maximum standard deviation
    rho : float, default=7.0
        Schedule hyper-parameter.
    device : torch.device, default=None
        Device to generate the schedule/sigmas/boundaries/ts on.

    Returns
    -------
    Tensor
        Generated schedule/sigmas/boundaries/ts.
    """
    rho_inv = 1.0 / rho
    # Clamp steps to 1 so that we don't get nans
    steps = torch.arange(num_timesteps, device=device) / max(num_timesteps - 1, 1)
    sigmas = sigma_min**rho_inv + steps * (
        sigma_max**rho_inv - sigma_min**rho_inv
    )
    sigmas = sigmas**rho

    return sigmas


def skip_scaling(
    sigma: Tensor, sigma_data: float = 0.5, sigma_min: float = 0.002
) -> Tensor:
    """Computes the scaling value for the residual connection.

    Parameters
    ----------
    sigma : Tensor
        Current standard deviation of the noise.
    sigma_data : float, default=0.5
        Standard deviation of the data.
    sigma_min : float, default=0.002
        Minimum standard deviation of the noise from the karras schedule.

    Returns
    -------
    Tensor
        Scaling value for the residual connection.
    """
    return sigma_data**2 / ((sigma - sigma_min) ** 2 + sigma_data**2)


def output_scaling(
    sigma: Tensor, sigma_data: float = 0.5, sigma_min: float = 0.002
) -> Tensor:
    """Computes the scaling value for the model's output.

    Parameters
    ----------
    sigma : Tensor
        Current standard deviation of the noise.
    sigma_data : float, default=0.5
        Standard deviation of the data.
    sigma_min : float, default=0.002
        Minimum standard deviation of the noise from the karras schedule.

    Returns
    -------
    Tensor
        Scaling value for the model's output.
    """
    return (sigma_data * (sigma - sigma_min)) / (sigma_data**2 + sigma**2) ** 0.5


def model_forward_wrapper(
    model: nn.Module,
    feats: Tensor = None,
    adjs: Tensor = None,
    xyzs: Tensor = None,
    gmasks: Tensor = None,
    sigma: Tensor = None,
    sigma_data: float = 0.5,
    sigma_min: float = 0.002,

    protein_pos=None, #加了噪音
    protein_v=None, 
    batch_protein=None,

    init_ligand_pos=None, #加噪音了
    init_ligand_v=None,  #加噪音了
    batch_ligand=None,
    time_step=None,
    
    org_ligand_pos = None,
    org_protein_pos = None,
    ligand_bond_index = None, ligand_bond_type = None, ligand_bond_type_batch = None,
    protein_max_atom_num = None, ligand_max_atom_num  = None,
    protein_element = None, ligand_element = None,

    ligand_atom_isring  =  None,
    ligand_atom_isO     =  None,
    ligand_atom_isN     =  None,

    protein_atom_isring =  None,
    protein_atom_isO    =  None,
    protein_atom_isN    =  None,


    cross_lig_isring_flag   = None,
    cross_lig_isO_flag      = None,
    cross_lig_isN_flag      = None,

    cross_pro_isring_flag   = None,
    cross_pro_isO_flag      = None,
    cross_pro_isN_flag      = None,

    cross_ligand    = None,
    cross_protein   = None,
    cross_distance  = None,

    cross_bond_index = None, 
    cross_bond_type = None, 
    cross_bond_index_reverse = None, 
    cross_bond_type_reverse = None,

    protein_coords_predict = None,

    complex_mol = None,

    protein_element_batch = None,
    protein_link_t_batch = None,
    protein_link_t_reverse_batch = None,

    ligand_element_batch = None,

    rd_pos = None,

    sample=False,
    scale = True,
    rate = 10.0,
    scale_step = None, 

) -> Tensor:
    """Wrapper for the model call to ensure that the residual connection and scaling
    for the residual and output values are applied.

    Parameters
    ----------
    model : nn.Module
        Model to call.
    x : Tensor
        Input to the model, e.g: the noisy samples.
    sigma : Tensor
        Standard deviation of the noise. Normally referred to as t.
    sigma_data : float, default=0.5
        Standard deviation of the data.
    sigma_min : float, default=0.002
        Minimum standard deviation of the noise.
    **kwargs : Any
        Extra arguments to be passed during the model call.

    Returns
    -------
    Tensor
        Scaled output from the model with the residual connection applied.
    """
    c_skip = skip_scaling(sigma, sigma_data, sigma_min)
    c_out = output_scaling(sigma, sigma_data, sigma_min)

    # Pad dimensions as broadcasting will not work
    c_skip = c_skip.index_select(0, batch_ligand).view([-1, 1]).to(xyzs.device)
    c_out  = c_out.index_select(0, batch_ligand).view([-1, 1]).to(xyzs.device)
    #print ('*',c_out.shape,c_skip.shape)
    sigma=sigma.to(xyzs.device)


    st = time.perf_counter()
    preds = model(
        protein_pos=protein_pos, 
        protein_v=protein_v, 
        batch_protein=batch_protein,

        init_ligand_pos=init_ligand_pos, #加噪音了
        init_ligand_v=init_ligand_v,  #加噪音了
        batch_ligand=batch_ligand,
        time_step=time_step,
        sample = sample,
        org_ligand_pos = org_ligand_pos,
        org_protein_pos = org_protein_pos,
        ligand_bond_index = ligand_bond_index, ligand_bond_type = ligand_bond_type, ligand_bond_type_batch = ligand_bond_type_batch,
        sigmas = sigma,
        protein_element = protein_element, ligand_element = ligand_element,

        ligand_atom_isring  = ligand_atom_isring,
        ligand_atom_isO     = ligand_atom_isO,
        ligand_atom_isN     = ligand_atom_isN,

        protein_atom_isring = protein_atom_isring,
        protein_atom_isO    = protein_atom_isO,
        protein_atom_isN    = protein_atom_isN,

        cross_lig_isring_flag   = cross_lig_isring_flag,
        cross_lig_isO_flag      = cross_lig_isO_flag,
        cross_lig_isN_flag      = cross_lig_isN_flag,

        cross_pro_isring_flag   = cross_pro_isring_flag,
        cross_pro_isO_flag      = cross_pro_isO_flag,
        cross_pro_isN_flag      = cross_pro_isN_flag,

        cross_ligand    = cross_ligand,
        cross_protein   = cross_protein,
        cross_distance  = cross_distance,


        cross_bond_index = cross_bond_index, 
        cross_bond_type = cross_bond_type, 
        cross_bond_index_reverse = cross_bond_index_reverse, 
        cross_bond_type_reverse = cross_bond_type_reverse,

        protein_coords_predict = protein_coords_predict,

        complex_mol = complex_mol,

        protein_element_batch = protein_element_batch,
        protein_link_t_batch = protein_link_t_batch,
        protein_link_t_reverse_batch = protein_link_t_reverse_batch,
    
        ligand_element_batch = ligand_element_batch,

        rd_pos = rd_pos,


    )

    end = time.perf_counter()
    #print('a model time s:', round(end - st, 4)) #模型用了0.2秒，其它地方用了3秒

    #if sample == True:
        #print('cskip:', c_skip[0])
        #print('cout:',c_out[0])
        #print('---------------------------------------------')

    preds['pred_ligand_pos'] = c_skip * xyzs + c_out * preds['pred_ligand_pos'] 
    preds['final_pos'][preds['mask_ligand']] = preds['pred_ligand_pos'] #把配体和蛋白放在一起的坐标也更新了



    return preds



def center_pos(protein_pos, ligand_pos, batch_protein, batch_ligand, mode='protein'):
    if mode == 'none':
        offset = 0.
        pass
    elif mode == 'protein':
        offset = scatter_mean(protein_pos, batch_protein, dim=0) #分组减质心，最后对接或保存结构的时候是否需要再加上质心了？
        protein_pos = protein_pos - offset[batch_protein]
        ligand_pos = ligand_pos - offset[batch_ligand]
    elif mode == 'ligand':
        offset = scatter_mean(ligand_pos, batch_ligand, dim=0) #分组减质心，最后对接或保存结构的时候是否需要再加上质心了？
        protein_pos = protein_pos - offset[batch_protein]
        ligand_pos = ligand_pos - offset[batch_ligand]
    else:
        raise NotImplementedError
    return protein_pos, ligand_pos, offset





def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)  #多元分分类，ont-hot化，num——classes是原子类型数量
    # permute_order = (0, -1) + tuple(range(1, len(x.size())))
    # x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))   #取对数，并设置下限，也就是说把0设置成了1e-30，非常小的数
    return log_x



def log_normal(values, means, log_scales):
    var = torch.exp(log_scales * 2)
    log_prob = -((values - means) ** 2) / (2 * var) - log_scales - np.log(np.sqrt(2 * np.pi))
    return log_prob.sum(-1)


def log_sample_categorical(logits):
    uniform = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    # sample_onehot = F.one_hot(sample, self.num_classes)
    # log_sample = index_to_log_onehot(sample, self.num_classes)
    return gumbel_noise + logits


def log_1_min_a(a):
    return np.log(1 - np.exp(a) + 1e-40)


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def extract(coef, t, batch):
    out = coef[t][batch]
    return out.unsqueeze(-1)



def to_torch_const(x):
    ##print'x:', x)
    x = torch.from_numpy(x).float()
    x = nn.Parameter(x, requires_grad=False)
    return x




def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])

    alphas = np.clip(alphas, a_min=0.001, a_max=1.)

    # Use sqrt of this, so the alpha in our paper is the alpha_sqrt from the
    # Gaussian diffusion in Ho et al.
    alphas = np.sqrt(alphas)
    return alphas


def get_distance(pos, edge_index):
    return (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)



def log_onehot_to_index(log_x):
    return log_x.argmax(1)


def categorical_kl(log_prob1, log_prob2):  #kl_v = categorical_kl(log_v_true_prob, log_v_model_prob)  # [num_atoms, ] ，z0，zt
    kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
    return kl


def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    kl = 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + (mean1 - mean2) ** 2 * torch.exp(-logvar2))
    return kl.sum(-1)


def compose_context(h_protein, h_ligand, pos_protein, pos_ligand, batch_protein, batch_ligand, protein_element, ligand_element):
    # previous version has problems when ligand atom types are fixed
    # (due to sorting randomly in case of same element)

    #把配体和蛋白合并一起，并且有顺序
    batch_ctx = torch.cat([batch_protein, batch_ligand], dim=0) #批量，分别存放的是配体和蛋白的原点在批量中所属哪个图
    # sort_idx = batch_ctx.argsort()
    sort_idx = torch.sort(batch_ctx, stable=True).indices  #知道排序坐标，则可以通过sort_idx获取从原始的序列中获取排序好的值
    #获取排序坐标而不是真正对序列进行排序，好处很多

    #排序的结果使得，下标相同的情况下，配体在蛋白下面，即蛋白在前
    mask_ligand = torch.cat([
        torch.zeros([batch_protein.size(0)], device=batch_protein.device).bool(),
        torch.ones([batch_ligand.size(0)], device=batch_ligand.device).bool(),
    ], dim=0)[sort_idx]   #把蛋白掩码成0，配体掩码成1

    batch_ctx = batch_ctx[sort_idx]
    h_ctx = torch.cat([h_protein, h_ligand], dim=0)[sort_idx]  # (N_protein+N_ligand, H)
    pos_ctx = torch.cat([pos_protein, pos_ligand], dim=0)[sort_idx]  # (N_protein+N_ligand, 3)

    element_ctx = torch.cat([protein_element, ligand_element], dim=0)[sort_idx]  # (N_protein+N_ligand, 3)

    return h_ctx, pos_ctx, batch_ctx, element_ctx, mask_ligand



def add_random_offset(mol):
    for atom in mol.GetAtoms():
        pos = atom.GetIdx()
        for _ in range(3):
            atom_pos = mol.GetConformer().GetAtomPosition(pos)
            mol.GetConformer().SetAtomPosition(pos, (atom_pos.x + random.uniform(-0.01, 0.01), 
                                                    atom_pos.y + random.uniform(-0.01, 0.01), 
                                                    atom_pos.z + random.uniform(-0.01, 0.01)))


def GetDihedral(conf, atom_idx): #获取二面角
    #print('conf:', conf)
    #print('atom_idx:', atom_idx)
    #print('type(atom_idx[0]):', type(atom_idx[0]))
    #rdMolTransforms.GetDihedralRad需要的是mol的构象而不是mol
    #print('conf num:', len(conf.GetConformers()))#取第一个构象即可conf.GetConformers()[0]或者conf.GetConformer()

    torsion_degrees = rdMolTransforms.GetDihedralRad(conf.GetConformer(), atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3])#返回的弧度[0,2pi]
    #rdMolTransforms.GetDihedralDeg(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3]) #另一种计算二面角的方法，返回的角度[0,360]
        
    t = torsion_degrees // (2 * np.pi) #取整，求周期倍数
    torsion_degrees = torsion_degrees - (2 * np.pi) * t
    #print(f'Torsion {torsion.GetIdx()}: {torsion_degrees:.2f} degrees')

    return torsion_degrees

def SetDihedral(conf, atom_idx, new_vale):
    rdMolTransforms.SetDihedralRad(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3], new_vale)


def get_torsion_angles(mol): #获取二面角的顶点list
    torsions_list = []
    G = nx.Graph()
    for i, atom in enumerate(mol.GetAtoms()):
        G.add_node(i)
    nodes = set(G.nodes())
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        G.add_edge(start, end)
    for e in G.edges():
        G2 = copy.deepcopy(G)
        G2.remove_edge(*e)
        if nx.is_connected(G2): continue
        l = list(sorted(nx.connected_components(G2), key=len)[0])
        if len(l) < 2: continue
        n0 = list(G2.neighbors(e[0]))
        n1 = list(G2.neighbors(e[1]))
        torsions_list.append(
            (n0[0], e[0], e[1], n1[0])
        )
    return torsions_list


def compute_tor(mol):
    mol_ = copy.deepcopy(mol)
    mol_maybe_noh = RemoveHs(mol_, sanitize=True)
    rotable_bonds = get_torsion_angles(mol_maybe_noh) #二面角的4个顶点
    tor_list = []
    for i in rotable_bonds:
        tor = GetDihedral(mol_maybe_noh, i)
        tor_list.append(tor)

    tor = tor_list
    return tor



def internal_coordinate(mol_list):
    # 计算分子的键长
    bond_lists       = []
    angle_lists      = []
    torsion_lists    = []

    for mol in mol_list: #遍历每一个mol
        bond_list       = []
        angle_list      = []
        torsion_list    = []

        for bond in mol.GetBonds():
            bond_length = Chem.rdMolTransforms.GetBondLength(mol.GetConformer(), bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            #print(f'Bond {bond.GetIdx()}: {bond_length:.2f} angstroms')
            bond_list.append(bond_length)



        # 计算分子的键角
        for atom in mol.GetAtoms():
            atom_id = atom.GetIdx()
            atom_neighbors = atom.GetNeighbors()
            if len(atom_neighbors) == 2:
                angle_value = Chem.rdMolTransforms.GetAngleRad(mol.GetConformer(), atom_neighbors[0].GetIdx(), atom_id, atom_neighbors[1].GetIdx())
                angle_degrees = angle_value 
                #减去周期
                t = angle_degrees // (2 * np.pi) #取整，求周期倍数
                angle_degrees = angle_degrees - (2 * np.pi) * t
                #print(f'Angle {angle.GetIdx()}: {angle_degrees:.2f} degrees')
                angle_list.append(angle_degrees)


        #add_random_offset(mol)
        # 计算分子的二面角
        torsion_list = compute_tor(mol)

        
        bond_lists.extend(bond_list)
        angle_lists.extend(angle_list)
        torsion_lists.extend(torsion_list)
    
    return torch.stack(bond_lists), torch.stack(angle_lists), torch.stack(torsion_lists)



def calc_performance_stats(true_mols, model_mols):

    rmsd_list = []
    for tc, mc in zip(true_mols, model_mols):
        try:
            rmsd_val = AllChem.GetBestRMS(Chem.RemoveHs(tc), Chem.RemoveHs(mc))
        except RuntimeError:
            print('rmsd error, skip')
            pass
        rmsd_list.append(rmsd_val)


    rmsd = np.array(rmsd_list).mean()

    return rmsd


def distance(coords):
    #使用矩阵乘法，完全可以实现原子距离计算，不要循环

    A = coords.unsqueeze(0)  # 添加一维作为 batch 维度
    B = coords.unsqueeze(1)  # 添加一维作为另一个原子维度
    # 计算距离向量矩阵
    dist_vectors = A - B
    # 计算距离的平方
    dist_sq = torch.norm(dist_vectors, dim=-1)

    #print('dist_sq:', dist_sq.shape) #得到一个n * n的距离矩阵，我们将展平，用于连接多个矩阵
    return dist_sq.view(-1)


def py_rmsd(coords1, coords2):
    # 计算两个分子之间的平方距离
    squared_diff = torch.sum((coords1 - coords2) ** 2, dim=1)
    
    # 计算均方根偏差
    rmsd = torch.sqrt(torch.mean(squared_diff))
    
    return rmsd
    

def compute_rmsd(coords1, coords2):
    # 计算两个分子的质心
    center1 = torch.mean(coords1, dim=0)
    center2 = torch.mean(coords2, dim=0)
    
    # 将两个分子的坐标居中
    coords1_centered = coords1 - center1
    coords2_centered = coords2 - center2
    
    # 计算旋转矩阵
    U, _, Vt = torch.svd(torch.matmul(coords1_centered.t(), coords2_centered))
    rotation_matrix = torch.matmul(U, Vt)
    
    # 对一个分子进行旋转
    coords1_rotated = torch.matmul(coords1_centered, rotation_matrix)
    
    # 计算两个分子之间的平方距离
    squared_diff = torch.sum((coords1_rotated - coords2_centered) ** 2, dim=1)
    
    # 计算均方根偏差
    rmsd = torch.sqrt(torch.mean(squared_diff))
    
    return rmsd





def improved_timesteps_schedule(
    current_training_step: int,
    total_training_steps: int,
    initial_timesteps: int = 2,
    final_timesteps: int = 25,
) -> int:
    """Implements the improved timestep discretization schedule.

    Parameters
    ----------
    current_training_step : int
        Current step in the training loop.
    total_training_steps : int
        Total number of steps the model will be trained for.
    initial_timesteps : int, default=2
        Timesteps at the start of training.
    final_timesteps : int, default=150
        Timesteps at the end of training.

    Returns
    -------
    int
        Number of timesteps at the current point in training.

    References
    ----------
    [1] [Improved Techniques For Consistency Training](https://arxiv.org/pdf/2310.14189.pdf)
    """
    total_training_steps_prime = math.floor(
        total_training_steps
        / (math.log2(math.floor(final_timesteps / initial_timesteps)) + 1)
    )
    num_timesteps = initial_timesteps * math.pow(
        2, math.floor(current_training_step / total_training_steps_prime)
    )
    num_timesteps = min(num_timesteps, final_timesteps) + 1

    return num_timesteps


def improved_loss_weighting(sigmas: Tensor) -> Tensor:
    """Computes the weighting for the consistency loss.

    Parameters
    ----------
    sigmas : Tensor
        Standard deviations of the noise.

    Returns
    -------
    Tensor
        Weighting for the consistency loss.

    References
    ----------
    [1] [Improved Techniques For Consistency Training](https://arxiv.org/pdf/2310.14189.pdf)
    """
    return 1 / (sigmas[1:] - sigmas[:-1])



def pseudo_huber_loss(input: Tensor, target: Tensor, batch_ligand) -> Tensor:
    #伪huber损失
    """Computes the pseudo huber loss.

    Parameters
    ----------
    input : Tensor
        Input tensor.
    target : Tensor
        Target tensor.

    Returns
    -------
    Tensor
        Pseudo huber loss.
    """
    c = 0.00054 * math.sqrt(math.prod(input.shape[1:])) #用来计算list中的元素乘积, 实际上是去掉批量之后的元素相乘，可以不用动，那么就变成了固定的3了

    loss = scatter_mean((torch.sqrt((input - target) ** 2 + c**2) - c).sum(-1), batch_ligand, dim=0) #分组计算损失

    return loss





def lognormal_timestep_distribution(
    num_samples: int,
    sigmas: Tensor,
    mean: float = -1.1,
    std: float = 2.0,
) -> Tensor:
    """Draws timesteps from a lognormal distribution.

    Parameters
    ----------
    num_samples : int
        Number of samples to draw.
    sigmas : Tensor
        Standard deviations of the noise.
    mean : float, default=-1.1
        Mean of the lognormal distribution.
    std : float, default=2.0
        Standard deviation of the lognormal distribution.

    Returns
    -------
    Tensor
        Timesteps drawn from the lognormal distribution.

    References
    ----------
    [1] [Improved Techniques For Consistency Training](https://arxiv.org/pdf/2310.14189.pdf)
    """
    pdf = torch.erf((torch.log(sigmas[1:]) - mean) / (std * math.sqrt(2))) - torch.erf(
        (torch.log(sigmas[:-1]) - mean) / (std * math.sqrt(2))
    )
    pdf = pdf / pdf.sum()

    timesteps = torch.multinomial(pdf, num_samples, replacement=True)

    return timesteps




class ConsistencyTraining:
    """Implements the Consistency Training algorithm proposed in the paper.

    Parameters
    ----------
    sigma_min : float, default=0.002
        Minimum standard deviation of the noise.
    sigma_max : float, default=80.0
        Maximum standard deviation of the noise.
    rho : float, default=7.0
        Schedule hyper-parameter.
    sigma_data : float, default=0.5
        Standard deviation of the data.
    initial_timesteps : int, default=2
        Schedule timesteps at the start of training.
    final_timesteps : int, default=150
        Schedule timesteps at the end of training.
    initial_ema_decay_rate : float, default=0.95
        EMA rate at the start of training.
    """

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        sigma_data: float = 0.5,
        initial_timesteps: int = 2, #最小是2
        final_timesteps: int = 25, ##这里的self.final_timesteps步长有问题，默认是150，应该改一下，改成25/15之类的，不要太大，因为在构象生成中，不需要那么多步长

        lognormal_mean: float = -1.1,
        lognormal_std: float = 2.0,
    ) -> None:
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.sigma_data = sigma_data
        self.initial_timesteps = initial_timesteps
        self.final_timesteps = final_timesteps

        self.lognormal_mean = lognormal_mean
        self.lognormal_std = lognormal_std



    def q_v_sample(self, log_v0, t, batch):
        log_qvt_v0 = self.q_v_pred(log_v0, t, batch)
        sample_prob = log_sample_categorical(log_qvt_v0)
        sample_index = sample_prob.argmax(dim=-1) #返回最后一维度上的最大值索引，返回概率最大的索引
        log_sample = index_to_log_onehot(sample_index, self.num_classes) #one-hot化，概率索引
        return sample_index, log_sample

    # atom type generative process
    def q_v_posterior(self, log_v0, log_vt, t, batch):
        # q(vt-1 | vt, v0) = q(vt | vt-1, v0) * q(vt-1 | v0) / q(vt | v0)
        t_minus_1 = t - 1
        # Remove negative values, will not be used anyway for final decoder
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_qvt1_v0 = self.q_v_pred(log_v0, t_minus_1, batch)
        unnormed_logprobs = log_qvt1_v0 + self.q_v_pred_one_timestep(log_vt, t, batch)
        log_vt1_given_vt_v0 = unnormed_logprobs - torch.logsumexp(unnormed_logprobs, dim=-1, keepdim=True)
        return log_vt1_given_vt_v0
    


    # atom type diffusion process
    def q_v_pred_one_timestep(self, log_vt_1, t, batch):
        # q(vt | vt-1)
        log_alpha_t = extract(self.log_alphas_v, t, batch)
        log_1_min_alpha_t = extract(self.log_one_minus_alphas_v, t, batch)

        # alpha_t * vt + (1 - alpha_t) 1 / K
        log_probs = log_add_exp(
            log_vt_1 + log_alpha_t,
            log_1_min_alpha_t - np.log(self.num_classes)
        )
        return log_probs

    def q_v_pred(self, log_v0, t, batch):
        # compute q(vt | v0)
        log_cumprod_alpha_t = extract(self.log_alphas_cumprod_v, t, batch)
        log_1_min_cumprod_alpha = extract(self.log_one_minus_alphas_cumprod_v, t, batch)

        log_probs = log_add_exp(
            log_v0 + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - np.log(self.num_classes)
        )
        return log_probs


    def compute_v_Lt(self, log_v_model_prob, log_v0, log_v_true_prob, t, batch):
        kl_v = categorical_kl(log_v_true_prob, log_v_model_prob)  # [num_atoms, ]
        decoder_nll_v = -log_categorical(log_v0, log_v_model_prob)  # L0 #
        assert kl_v.shape == decoder_nll_v.shape
        mask = (t == 0).float()[batch]
        loss_v = scatter_mean(mask * decoder_nll_v + (1. - mask) * kl_v, batch, dim=0)
        return loss_v
    

    def calculate_distance_matrix(self, A, B):
        """
        计算两个坐标矩阵之间的欧氏距离

        参数:
        A (torch.Tensor): 大小为 (n, 3) 的坐标矩阵
        B (torch.Tensor): 大小为 (m, 3) 的坐标矩阵

        返回:
        torch.Tensor: 大小为 (n, m) 的距离矩阵
        """
        # A 的形状为 (n, 3)，B 的形状为 (m, 3)
        # 计算 A 的每个点与 B 的每个点之间的距离
        diff = A.unsqueeze(1) - B.unsqueeze(0)  # diff 的形状为 (n, m, 3)
        dist_matrix = torch.sqrt(torch.sum(diff**2, dim=2))  # dist_matrix 的形状为 (n, m)
        return dist_matrix



    def calculate_distance_matrix_batch(self, n, m):

        # 假设 n 和 m 是形状 [batch_size, n, 3] 和 [batch_size, m, 3] 的张量
        #batch_size = 4
        #n = 5
        #m = 6

        # 示例数据
        #n = torch.randn(batch_size, n, 3)  # 坐标矩阵 n, 形状为 [batch_size, n, 3]
        #m = torch.randn(batch_size, m, 3)  # 坐标矩阵 m, 形状为 [batch_size, m, 3]

        # 计算两组坐标点之间的欧几里得距离
        # 1. 扩展维度
        n_expanded = n.unsqueeze(2)  # 形状变为 [batch_size, n, 1, 3]
        m_expanded = m.unsqueeze(1)  # 形状变为 [batch_size, 1, m, 3]

        # 2. 计算两组坐标点之间的差值
        distance_matrix = torch.sqrt(torch.sum((n_expanded - m_expanded) ** 2, dim=-1))

        # 输出结果，形状为 [batch_size, n, m]，即每个 `n` 中的点到每个 `m` 中的点的距离
        #print(distance_matrix.shape)  # [batch_size, n, m]
        #print(distance_matrix)
        return distance_matrix



    def IC_Loss(self,pred,target,zmats,gmasks):
        #print('pred,target,zmats,gmasks:', pred.shape,target.shape,zmats.shape,gmasks.shape)
        #pred,target,zmats,gmasks: torch.Size([2, 250, 3]) torch.Size([2, 250, 3]) torch.Size([2, 250, 5]) torch.Size([2, 250])
        #exit()
        #pred,target,zmats,gmasks: torch.Size([900, 9, 3]) torch.Size([900, 9, 3]) torch.Size([900, 9, 5]) torch.Size([900, 9])
        pred_bonddis,pred_angle,pred_dihedral,j1,j2,j3=xyz2ic(pred,zmats) #笛卡尔转内坐标
        target_bonddis,target_angle,target_dihedral,j1,j2,j3=xyz2ic(target,zmats)
        pred_dismat=torch.cdist(pred,pred,compute_mode='donot_use_mm_for_euclid_dist')
        target_dismat=torch.cdist(target,target,compute_mode='donot_use_mm_for_euclid_dist')
        gmasks_2D=gmasks.unsqueeze(-1)*gmasks.unsqueeze(-1).permute(0,2,1)
        loss_angle=F.mse_loss(pred_angle[gmasks],target_angle[gmasks])
        loss_dismat=F.mse_loss(pred_dismat[gmasks_2D],target_dismat[gmasks_2D])
        loss_bonddis=F.mse_loss(pred_bonddis[gmasks],target_bonddis[gmasks])
        dihedral_diff=torch.abs(pred_dihedral[gmasks]-target_dihedral[gmasks])
        dihedral_diff=torch.where(dihedral_diff>math.pi,math.pi*2-dihedral_diff,dihedral_diff)
        loss_dihedral=torch.mean(torch.square(dihedral_diff))
        return loss_dismat,loss_bonddis,loss_angle,loss_dihedral



    def __call__(
        self,
        online_model: nn.Module,
        ema_model: nn.Module,
        current_training_step: int,
        total_training_steps: int, 
        args,
        config,
        protein_atom_feature_dim,
        ligand_atom_feature_dim,

        protein_pos=None,
        protein_v=None,
        affinity=None,
        batch_protein=None,
        
        ligand_pos=None,
        ligand_v=None,
        batch_ligand=None,

        ligand_bond_index=None,
        ligand_bond_type=None,
        ligand_bond_type_batch=None,

        protein_element=None, 
        ligand_element=None,

        ligand_mol=None,

        ligand_fill_coords =  None,
        ligand_fill_zmats  =  None,
        ligand_fill_masks  =  None,
        ligand_fill_atom_order = None,

        ligand_atom_isring  =  None,
        ligand_atom_isO     =  None,
        ligand_atom_isN     =  None,

        protein_atom_isring =  None,
        protein_atom_isO    =  None,
        protein_atom_isN    =  None,


        cross_lig_isring_flag = None,
        cross_lig_isO_flag = None,
        cross_lig_isN_flag = None,

        cross_pro_isring_flag = None,
        cross_pro_isO_flag = None,
        cross_pro_isN_flag = None,


        cross_ligand    = None,
        cross_protein   = None,
        cross_distance  = None,

        cross_bond_index = None, 
        cross_bond_type = None, 
        cross_bond_index_reverse = None, 
        cross_bond_type_reverse = None,

        protein_coords_predict = None,

        complex_mol = None,

        protein_element_batch = None,
        protein_link_t_batch = None,
        protein_link_t_reverse_batch = None,

        ligand_element_batch = None,

        rd_pos = None,
        
        ligand_emb = None,
        pocket_emb = None,



        rate = 10.0,
        scale = True,

    ) -> Tuple[Tensor, Tensor]:
        """Runs one step of the consistency training algorithm.

        Parameters
        ----------
        online_model : nn.Module
            Model that is being trained.
        ema_model : nn.Module
            An EMA of the online model.
        x : Tensor
            Clean data.
        current_training_step : int
            Current step in the training loop.
        total_training_steps : int
            Total number of steps in the training loop.
        **kwargs : Any
            Additional keyword arguments to be passed to the models.

        Returns
        -------
        (Tensor, Tensor)
            The predicted and target values for computing the loss.
        """



        self.num_classes = ligand_atom_feature_dim

        self.loss_v_weight = config.loss_v_weight
        self.loss_exp_weight = config.loss_exp_weight
        self.use_classifier_guide = config.use_classifier_guide

        # atom type diffusion schedule in log space
        alphas_v                = cosine_beta_schedule(self.final_timesteps, config.v_beta_s)
        log_alphas_v            = np.log(alphas_v)
        log_alphas_cumprod_v    = np.cumsum(log_alphas_v)
        self.log_alphas_v       = to_torch_const(log_alphas_v).cuda()
        self.log_one_minus_alphas_v         = to_torch_const(log_1_min_a(log_alphas_v)).cuda()
        self.log_alphas_cumprod_v           = to_torch_const(log_alphas_cumprod_v).cuda()
        self.log_one_minus_alphas_cumprod_v = to_torch_const(log_1_min_a(log_alphas_cumprod_v)).cuda()


        # center pos
        center_pos_mode = config.center_pos_mode  # ['none', 'protein']

        _, protein_coords_predict,_ = center_pos(protein_pos, protein_coords_predict, batch_protein, batch_ligand, mode=center_pos_mode) 

        #rdkit 坐标
        _, rd_pos, _ = center_pos(protein_pos, rd_pos.float(), batch_protein, batch_ligand, mode=center_pos_mode) #质心在蛋白上

        #减蛋白质心，蛋白的质心作为原点，配体pos是真实值，而蛋白pos是加噪的
        protein_pos, ligand_pos, offset = center_pos(protein_pos, ligand_pos, batch_protein, batch_ligand, mode=center_pos_mode) #质心在蛋白上




        origin_ligand_pos = copy.deepcopy(ligand_pos) #这个用于构建KNN图，对于对接，我们需要固定KNN的邻接表
        origin_protein_pos = copy.deepcopy(protein_pos) #这个用于构建KNN图，对于对接，我们需要固定KNN的邻接表


        #这2个别忘了减去质心
        cross_ligand    = cross_ligand - offset[batch_ligand]
        cross_protein   = cross_protein - offset[batch_protein]
        
        
        #这里的self.final_timesteps步长有问题，默认是150，应该改一下，改成25/15之类的，不要太大，num_timesteps的取值在[self.initial_timesteps,self.final_timesteps]之间，即[2,25]
        #因此，改一下：self.final_timesteps = total_training_steps
        #self.final_timesteps = total_training_steps

        #因为我们有上限约束，所以无论输入什么的步长(>=0), 都在[self.initial_timesteps, self.final_timesteps]之间，但是是在采样的时候没有这一步，所以要对step=1的时候特殊处理
        
        if GP.ema_exit:
            num_timesteps = timesteps_schedule(
                current_training_step,
                total_training_steps,
                self.initial_timesteps,
                self.final_timesteps,
            )
        else:
            num_timesteps = improved_timesteps_schedule(
            current_training_step,
            total_training_steps,
            self.initial_timesteps,
            self.final_timesteps,
        )

        #print ('*'*80)
        #print (num_timesteps)  #num_timesteps可以认为是当前的步长
        #下面是一次加噪的过程。因为我们有多个当前步长，因此要执行下面这个过程多次。即相当于DDPM的一次前馈加噪
        #当前步长的方差序列
        ''''
        sigmas(step = 25) = 
        tensor([2.0000e-03, 5.2449e-03, 1.2238e-02, 2.6055e-02, 5.1532e-02, 9.5931e-02,
        1.6975e-01, 2.8773e-01, 4.6998e-01, 7.4338e-01, 1.1431e+00, 1.7146e+00,
        2.5152e+00, 3.6171e+00, 5.1092e+00, 7.1005e+00, 9.7232e+00, 1.3136e+01,
        1.7528e+01, 2.3123e+01, 3.0183e+01, 3.9017e+01, 4.9979e+01, 6.3483e+01,
        8.0000e+01])
        】
        sigmas的取值从0和80之间逐步往中间逼近，0和80是一直存在的

        '''
        num_graphs = max(batch_ligand) + 1
        sigmas = karras_schedule(
            num_timesteps, self.sigma_min, self.sigma_max, self.rho, ligand_pos.device
        )

        #新加内容
        #sigmas[0] += 1e-8 #对于第一步，因为值很小，所以加一个很小的小数，保证模型的稳定性

        #print (sigmas,sigmas.shape) # len(sigmas) == num_timesteps
        noise = torch.randn_like(ligand_pos)
        #print (xyzs.shape[0])
        if GP.ema_exit:
            timesteps = torch.randint(0, num_timesteps - 1, (num_graphs,), device=ligand_pos.device)
        else:
            timesteps = lognormal_timestep_distribution(num_graphs, sigmas, self.lognormal_mean, self.lognormal_std) 
            #和之前代码比，这里变化了,时间t的选取改成对数分布了，之前是均匀分布

        #print (noise.shape,timesteps.shape)
        current_sigmas = sigmas[timesteps]  #当前步的噪音方差, 序列的长度等于原子数量
        next_sigmas = sigmas[timesteps + 1] #下一步的噪声方差
        #print (current_sigmas.shape,next_sigmas.shape)

        #因为图不是等数量原子，所以要更改填充方法，只要当批量大小是1时，才不会报错

        current_pos  = current_sigmas.index_select(0, batch_ligand).view([-1, 1])
        next_pos     = next_sigmas.index_select(0, batch_ligand).view([-1, 1]) 

        next_xyzs = ligand_pos + next_pos * noise #当sigmas很小的时候，pad_dims_like(next_sigmas, xyzs) * noise  == 0, next_xyzs趋近于真实的xyzs
        #print (next_xyzs.shape

        num_classes = ligand_atom_feature_dim
        #原子类型加噪
        log_ligand_v0 = index_to_log_onehot(ligand_v, num_classes) #把原子类别给one-hot,连续的one-hot

        #log_ligand_v0 = F.one_hot(ligand_v, num_classes) #离散的one-hot

        #next_ligand_v_perturbed, next_log_ligand_vt = self.q_v_sample(log_ligand_v0, timesteps + 1, batch_ligand) #给原子类别特征加噪音
        next_ligand_v_perturbed = log_ligand_v0 #不加噪音，但原子类型信息依旧使用，作为节点嵌入的初始部分
        next_log_ligand_vt = 0


        next_preds = model_forward_wrapper(
            online_model,
            None,
            None,
            next_xyzs, #变化参数
            None,
            next_sigmas, #变化参数
            self.sigma_data,
            self.sigma_min,

            protein_pos=protein_pos, #加了噪音
            protein_v=protein_v, 
            batch_protein=batch_protein,

            init_ligand_pos=next_xyzs, #加噪音了 #变化参数
            init_ligand_v=next_ligand_v_perturbed,  #加噪音了
            batch_ligand=batch_ligand,
            time_step=timesteps  + 1 + 1, #多加一个1，是为了让时间从1开始，否则在采样的时候，对全0的时间编码，可能无效 #变化参数
            org_ligand_pos = origin_ligand_pos,
            org_protein_pos = origin_protein_pos,
            ligand_bond_index = ligand_bond_index, ligand_bond_type = ligand_bond_type, ligand_bond_type_batch = ligand_bond_type_batch,
            protein_element = protein_element, ligand_element = ligand_element,
            scale = scale,
            rate = rate,

            ligand_atom_isring  = ligand_atom_isring,
            ligand_atom_isO     = ligand_atom_isO,
            ligand_atom_isN     = ligand_atom_isN,

            protein_atom_isring = protein_atom_isring,
            protein_atom_isO    = protein_atom_isO,
            protein_atom_isN    = protein_atom_isN,


            cross_lig_isring_flag   = cross_lig_isring_flag,
            cross_lig_isO_flag      = cross_lig_isO_flag,
            cross_lig_isN_flag      = cross_lig_isN_flag,

            cross_pro_isring_flag   = cross_pro_isring_flag,
            cross_pro_isO_flag      = cross_pro_isO_flag,
            cross_pro_isN_flag      = cross_pro_isN_flag,

            cross_ligand    = cross_ligand,
            cross_protein   = cross_protein,
            cross_distance  = cross_distance,

            cross_bond_index = cross_bond_index, 
            cross_bond_type  = cross_bond_type, 
            cross_bond_index_reverse = cross_bond_index_reverse, 
            cross_bond_type_reverse  = cross_bond_type_reverse,

            protein_coords_predict = protein_coords_predict,

            complex_mol = complex_mol,

            protein_element_batch = protein_element_batch,
            protein_link_t_batch = protein_link_t_batch,
            protein_link_t_reverse_batch = protein_link_t_reverse_batch,
            ligand_element_batch = ligand_element_batch,

            rd_pos = rd_pos,
            
        )

        if not GP.ema_exit:
            ema_model = online_model

        with torch.no_grad(): #当前步噪音不参与参数更新，否则模型就有了两套参数，没必要
            current_xyzs = ligand_pos + current_pos * noise

            #current_ligand_v_perturbed, current_log_ligand_vt = self.q_v_sample(log_ligand_v0, timesteps, batch_ligand) #给原子类别特征加噪音
            current_ligand_v_perturbed = log_ligand_v0
            current_log_ligand_vt = 0

            current_preds = model_forward_wrapper(
                ema_model,
                None,
                None,
                current_xyzs, #变化参数
                None,
                current_sigmas, #变化参数
                self.sigma_data,
                self.sigma_min,

                protein_pos=protein_pos, #加了噪音
                protein_v=protein_v, 
                batch_protein=batch_protein,

                init_ligand_pos=current_xyzs, #加噪音了 #变化参数
                init_ligand_v=current_ligand_v_perturbed,  #加噪音了
                batch_ligand=batch_ligand,
                time_step=timesteps + 1, #多加一个1，是为了让时间从1开始，否则在采样的时候，对全0的时间编码，可能无效，  #变化参数
                org_ligand_pos = origin_ligand_pos,
                org_protein_pos = origin_protein_pos,
                ligand_bond_index = ligand_bond_index, ligand_bond_type = ligand_bond_type, ligand_bond_type_batch = ligand_bond_type_batch,
                protein_element = protein_element, ligand_element = ligand_element,
                scale = scale,
                rate = rate,

                ligand_atom_isring  = ligand_atom_isring,
                ligand_atom_isO     = ligand_atom_isO,
                ligand_atom_isN     = ligand_atom_isN,

                protein_atom_isring = protein_atom_isring,
                protein_atom_isO    = protein_atom_isO,
                protein_atom_isN    = protein_atom_isN,


                cross_lig_isring_flag   = cross_lig_isring_flag,
                cross_lig_isO_flag      = cross_lig_isO_flag,
                cross_lig_isN_flag      = cross_lig_isN_flag,

                cross_pro_isring_flag   = cross_pro_isring_flag,
                cross_pro_isO_flag      = cross_pro_isO_flag,
                cross_pro_isN_flag      = cross_pro_isN_flag,

                cross_ligand    = cross_ligand,
                cross_protein   = cross_protein,
                cross_distance  = cross_distance,

                cross_bond_index = cross_bond_index, 
                cross_bond_type  = cross_bond_type, 
                cross_bond_index_reverse = cross_bond_index_reverse, 
                cross_bond_type_reverse  = cross_bond_type_reverse,

                protein_coords_predict = protein_coords_predict,

                complex_mol = complex_mol,

                protein_element_batch = protein_element_batch,
                protein_link_t_batch = protein_link_t_batch,
                protein_link_t_reverse_batch = protein_link_t_reverse_batch,
                ligand_element_batch = ligand_element_batch,

                rd_pos = rd_pos,
                
            )
        

        loss_weights = pad_dims_like(improved_loss_weighting(sigmas)[timesteps], next_xyzs)
        
        #st = time.perf_counter()
        
        #求键长，键角度，二面角的损失，以及计算机amr rmsd值

        #rmsd = py_rmsd(next_preds['pred_ligand_pos'], current_preds['pred_ligand_pos'])

        #rmsd = py_rmsd(next_preds['pred_ligand_pos'], origin_ligand_pos)

        n_rmsd = py_rmsd(next_preds['pred_ligand_pos'], origin_ligand_pos) #对接的重要指标就是rmsd，这里我们直接和真实的进行优化。
        c_rmsd = py_rmsd(current_preds['pred_ligand_pos'], origin_ligand_pos) 
        rmsd   = F.mse_loss(n_rmsd, c_rmsd) #和真是的对齐之后，再mse

        #现在有一个问题，我们能否拿预测的直接和真实的键长，键角，rmsd等进行优化了？而不是和ema。
        #也就说，online和ema的结果先和真实的做一下mse对齐，然而再算mse损失呢？这样的好处可以避免，当模型训练不充分或者拟合能力不足时，出现online和ema结果相近，但距离
        #真实的结构，差距甚远。有利于模型更好地训练和收敛


        #与真实结构优化, 当模型趋于稳定时，权重调成1，开始训练时，权重调成很小
        ref_loss   = F.mse_loss(next_preds['pred_ligand_pos'], origin_ligand_pos)

        
        
        '''
        assert ligand_bond_index.shape[0] == 2
        next_pos                = next_preds['pred_ligand_pos']
        next_bond_lengths       = (next_pos[ligand_bond_index[0]] - next_pos[ligand_bond_index[1]]).norm(dim=-1).unsqueeze(-1) # E * 1

        current_pos             = current_preds['pred_ligand_pos']
        current_bond_lengths    = (current_pos[ligand_bond_index[0]] - current_pos[ligand_bond_index[1]]).norm(dim=-1).unsqueeze(-1) # E * 1

        #loss_bond = scatter_mean(((next_bond_lengths - current_bond_lengths) ** 2).sum(-1), ligand_bond_type_batch, dim=0).
        #loss_bond = torch.mean(loss_bond)

        #scatter_mean计算机均值的速度很慢，所以不用,尤其是当数据量很大时候
        #print('next_bond_lengths:', next_bond_lengths.shape)
        #print("next_preds['pred_ligand_pos']:", next_preds['pred_ligand_pos'].shape)
        #next_bond_lengths: torch.Size([888, 1])
        #next_preds['pred_ligand_pos']: torch.Size([422, 3]) 
        loss_bond = F.mse_loss(next_bond_lengths, current_bond_lengths)
        
        
        #计算dismat损失，一个分子内，任意两个原子之间的距离,不要使用list，直接使用GPU tensor
        
        n_dismat_list = []
        c_dismat_list = []
        for ids in range(num_graphs): #遍历每一个图
            n_pos, c_pos = next_preds['pred_ligand_pos'][batch_ligand == ids], current_preds['pred_ligand_pos'][batch_ligand == ids]
            n_dis = distance(n_pos) #得到一个n*n矩阵，这里的我们已经将其展平成向量，方便连接多个图
            c_dis = distance(c_pos)
            n_dismat_list.append(n_dis)
            c_dismat_list.append(c_dis)


        n_dismats = torch.cat(n_dismat_list)
        c_dismats = torch.cat(c_dismat_list)

        loss_dismat = F.mse_loss(n_dismats, c_dismats) #出现nan,是不是rmsd问题
        '''

        '''
        使用 zamts来计算内坐标
        ligand_fill_coords,
        ligand_fill_zmats,
        ligand_fill_masks,
        '''

        #把变长坐标，变成等长坐标
        next_cp_ligand_fill_coords    = ligand_fill_coords.clone()
        current_cp_ligand_fill_coords = ligand_fill_coords.clone()

        #print('next_cp_ligand_fill_coords:', next_cp_ligand_fill_coords.shape) #torch.Size([500, 3])
        #print('ligand_fill_masks:', ligand_fill_masks.shape) #torch.Size([500])
        #print('ligand_fill_zmats:', ligand_fill_zmats.shape)
        #print('ligand_fill_masks:', ligand_fill_masks.shape)
        #print('batch_ligand:', batch_ligand)

        '''
        next_cp_ligand_fill_coords: torch.Size([500, 3])
        ligand_fill_masks: torch.Size([500])
        ligand_fill_zmats: torch.Size([500, 4])
        ligand_fill_masks: torch.Size([500])
        '''
        
        #print('next_cp_ligand_fill_coords[ligand_fill_masks==True]:', next_cp_ligand_fill_coords[ligand_fill_masks==True].shape)
        #print("next_preds['pred_ligand_pos'][ligand_fill_atom_order]:", next_preds['pred_ligand_pos'][ligand_fill_atom_order].shape)
        next_cp_ligand_fill_coords[ligand_fill_masks==True]      = next_preds['pred_ligand_pos'][ligand_fill_atom_order] #按zmats重排原子顺序的方式，重排坐标
        current_cp_ligand_fill_coords[ligand_fill_masks==True]   = current_preds['pred_ligand_pos'][ligand_fill_atom_order]

        #pred,target,zmats,gmasks: torch.Size([900, 9, 3]) torch.Size([900, 9, 3]) torch.Size([900, 9, 5]) torch.Size([900, 9])
        #改变数据的形状
        loss_dismat,loss_bond,loss_angle,loss_dihedral = self.IC_Loss(next_cp_ligand_fill_coords.view(-1,GP.max_atoms,3),\
                                current_cp_ligand_fill_coords.view(-1,GP.max_atoms,3), ligand_fill_zmats.view(-1,GP.max_atoms,5), ligand_fill_masks.view(-1,GP.max_atoms))

        ic_loss=loss_dismat+loss_angle+loss_bond+loss_dihedral
        #exit()
        #end = time.perf_counter()
        #print('a loss_dismat/bond time s:', round(end - st, 4))
        

        #loss_dismat = torch.tensor(0)
        #loss_bond   = torch.tensor(0) #其它损失先不要
        #rmsd = torch.tensor(0) #的确是计算这些值导致了浪费了大量时间

        #st2 = time.perf_counter()
        #这种损失没有出现梯度爆炸的问题，相反使用前面的伪huber损失然而出现了梯度爆炸
        #loss_pos = scatter_mean(((current_preds['pred_ligand_pos'] - next_preds['pred_ligand_pos']) ** 2).sum(-1), batch_ligand, dim=0) 
        #这不同于F.mse_loss， F.mse_loss是对所有元素求均值，而不是对矩阵的行求均值，不过意义差不多
        #分组求均值，同一个图里面的原子求均值。分组求均值和放在一起求均值不一样，仅当每一个组的原子数量都一样时，分组均值等于总体均值，
        #分组均值的目的在于防止个别组的异常值过分影响全体，偏向局部优化
        #loss_pos = torch.mean(loss_pos) #这是对批量求均值

        #loss_pos = F.mse_loss(current_preds['pred_ligand_pos'], next_preds['pred_ligand_pos'])
        #注意F.mse_loss计算的是逐元素之间的差，如果输入的是矩阵，则先把矩阵变成向量，再操作，最后求均值，均值的范围是整个矩阵的元素数量，而不是行数
        #在计算坐标时，mse和scatter_mean，损失所用时间差不多
        #end2 = time.perf_counter()
        #print('a loss_pos time s:', round(end2 - st2, 4))

        #CMV2的损失
        loss_pos = torch.mean(loss_weights.squeeze() * pseudo_huber_loss(current_preds['pred_ligand_pos'], next_preds['pred_ligand_pos'], batch_ligand)) #loss_weights：每一个批量的权重


        #### 添加配体原子到蛋白原子的距离损失 GP.max_protein_atoms
        c_l_pos = current_preds['pred_ligand_pos']
        c_p_pos = current_preds['final_pos'][current_preds['mask_ligand'] == 0]

        n_l_pos = next_preds['pred_ligand_pos']
        n_p_pos = next_preds['final_pos'][next_preds['mask_ligand'] == 0]

        # 填充数据到固定长度
        #配体
        fill_c_l_pos_list = []
        mask_c_l_pos_list = []

        fill_c_p_pos_list = []
        mask_c_p_pos_list = []

        for j in range(max(ligand_element_batch) + 1):
            l_mask = batch_ligand[batch_ligand == j]
            p_mask = batch_protein[batch_protein == j]
            new_c_l_pos = c_l_pos[l_mask]  #IndexError: The shape of the mask [224] at index 0 does not match the shape of the indexed tensor [131, 3] at index 0
            new_c_p_pos = c_p_pos[p_mask]  

            mask_c_l_pos = torch.zeros([GP.max_protein_atoms], dtype=bool).cuda()
            mask_c_l_pos[:new_c_l_pos.shape[0]] = True
            fill_c_l_pos = torch.zeros([GP.max_protein_atoms, 3]).cuda()
            fill_c_l_pos[mask_c_l_pos] = new_c_l_pos

            mask_c_p_pos = torch.zeros([GP.max_protein_atoms], dtype=bool).cuda()
            mask_c_p_pos[:new_c_p_pos.shape[0]] = True
            fill_c_p_pos = torch.zeros([GP.max_protein_atoms, 3]).cuda()
            fill_c_p_pos[mask_c_p_pos] = new_c_p_pos

            fill_c_l_pos_list.append(fill_c_l_pos)
            mask_c_l_pos_list.append(mask_c_l_pos)
            fill_c_p_pos_list.append(fill_c_p_pos)
            mask_c_p_pos_list.append(mask_c_p_pos)
        
    
        fill_c_l_pos_s = torch.cat(fill_c_l_pos_list, dim = 0)
        mask_c_l_pos_s = torch.cat(mask_c_l_pos_list, dim = 0)
        fill_c_p_pos_s = torch.cat(fill_c_p_pos_list, dim = 0)
        mask_c_p_pos_s = torch.cat(mask_c_p_pos_list, dim = 0)
        c_lp_distance = self.calculate_distance_matrix_batch(fill_c_l_pos_s.view(-1, GP.max_protein_atoms, 3), fill_c_p_pos_s.view(-1, GP.max_protein_atoms, 3)) # batch_size * n * m

        
        # next
        fill_n_l_pos_list = []
        mask_n_l_pos_list = []

        fill_n_p_pos_list = []
        mask_n_p_pos_list = []

        for j in range(max(ligand_element_batch) + 1):
            #l_mask = next_preds['mask_ligand'][next_preds['batch_all'] == j] == True
            #p_mask = next_preds['mask_ligand'][next_preds['batch_all'] == j] == False
            l_mask = batch_ligand[batch_ligand == j]
            p_mask = batch_protein[batch_protein == j]

            new_n_l_pos = n_l_pos[l_mask] 
            new_n_p_pos = n_p_pos[p_mask]  

            mask_n_l_pos = torch.zeros([GP.max_protein_atoms], dtype=bool).cuda()
            mask_n_l_pos[:new_n_l_pos.shape[0]] = True
            fill_n_l_pos = torch.zeros([GP.max_protein_atoms, 3]).cuda()
            fill_n_l_pos[mask_n_l_pos] = new_n_l_pos

            mask_n_p_pos = torch.zeros([GP.max_protein_atoms], dtype=bool).cuda()
            mask_n_p_pos[:new_n_p_pos.shape[0]] = True
            fill_n_p_pos = torch.zeros([GP.max_protein_atoms, 3]).cuda()
            fill_n_p_pos[mask_n_p_pos] = new_n_p_pos

            fill_n_l_pos_list.append(fill_n_l_pos)
            mask_n_l_pos_list.append(mask_n_l_pos)
            fill_n_p_pos_list.append(fill_n_p_pos)
            mask_n_p_pos_list.append(mask_n_p_pos)
        
    
        fill_n_l_pos_s = torch.cat(fill_n_l_pos_list, dim = 0)
        mask_n_l_pos_s = torch.cat(mask_n_l_pos_list, dim = 0)
        fill_n_p_pos_s = torch.cat(fill_n_p_pos_list, dim = 0)
        mask_n_p_pos_s = torch.cat(mask_n_p_pos_list, dim = 0)
        n_lp_distance = self.calculate_distance_matrix_batch(fill_n_l_pos_s.view(-1, GP.max_protein_atoms, 3), fill_n_p_pos_s.view(-1, GP.max_protein_atoms, 3)) # batch_size * n * m


        #保证蛋白一样
        #assert c_lp_distance.requires_grad #这个确实没梯度
        assert n_lp_distance.requires_grad
        assert torch.allclose(c_p_pos, n_p_pos, atol=0.02)
        assert torch.allclose(c_p_pos, origin_protein_pos, atol=0.02)

        #依据参考的配体到蛋白的距离，截断8ai，作为掩码
        #true_lp_distance  = self.calculate_distance_matrix(origin_ligand_pos, origin_protein_pos)

        fill_n_origin_ligand_pos_list = []
        mask_n_origin_ligand_pos_list = []

        fill_n_origin_protein_pos_list = []
        mask_n_origin_protein_pos_list = []

        for j in range(max(ligand_element_batch) + 1):
            #l_mask = next_preds['mask_ligand'][next_preds['batch_all'] == j] == True
            #p_mask = next_preds['mask_ligand'][next_preds['batch_all'] == j] == False
            l_mask = batch_ligand[batch_ligand == j]
            p_mask = batch_protein[batch_protein == j]
        
            new_n_origin_ligand_pos = origin_ligand_pos[l_mask] 
            new_n_origin_protein_pos = origin_protein_pos[p_mask]  

            mask_n_origin_ligand_pos = torch.zeros([GP.max_protein_atoms], dtype=bool).cuda()
            mask_n_origin_ligand_pos[:new_n_origin_ligand_pos.shape[0]] = True
            fill_n_origin_ligand_pos = torch.zeros([GP.max_protein_atoms, 3]).cuda()
            fill_n_origin_ligand_pos[mask_n_origin_ligand_pos] = new_n_origin_ligand_pos

            mask_n_origin_protein_pos = torch.zeros([GP.max_protein_atoms], dtype=bool).cuda()
            mask_n_origin_protein_pos[:new_n_origin_protein_pos.shape[0]] = True
            fill_n_origin_protein_pos = torch.zeros([GP.max_protein_atoms, 3]).cuda()
            fill_n_origin_protein_pos[mask_n_origin_protein_pos] = new_n_origin_protein_pos

            fill_n_origin_ligand_pos_list.append(fill_n_origin_ligand_pos)
            mask_n_origin_ligand_pos_list.append(mask_n_origin_ligand_pos)
            fill_n_origin_protein_pos_list.append(fill_n_origin_protein_pos)
            mask_n_origin_protein_pos_list.append(mask_n_origin_protein_pos)
        
    
        fill_n_origin_ligand_pos_s = torch.cat(fill_n_origin_ligand_pos_list, dim = 0)
        mask_n_origin_ligand_pos_s = torch.cat(mask_n_origin_ligand_pos_list, dim = 0)
        fill_n_origin_protein_pos_s = torch.cat(fill_n_origin_protein_pos_list, dim = 0)
        mask_n_origin_protein_pos_s = torch.cat(mask_n_origin_protein_pos_list, dim = 0)
        true_lp_distance = self.calculate_distance_matrix_batch(fill_n_origin_ligand_pos_s.view(-1, GP.max_protein_atoms, 3), fill_n_origin_protein_pos_s.view(-1, GP.max_protein_atoms, 3)) # batch_size * n * m

        mask_cutoff = true_lp_distance < 8
        cross_distance_loss = F.mse_loss(c_lp_distance[mask_cutoff], n_lp_distance[mask_cutoff]) #截断之后，矩阵变成了向量.
        

        #计算mse(参考配体到蛋白的距离, 预测配体到蛋白的距离)
        ref_d  = true_lp_distance
        pred_d = n_lp_distance
        ref_cross_distance_loss = F.mse_loss(ref_d[mask_cutoff], pred_d[mask_cutoff]) #

        #assert ref_d.requires_grad
        assert pred_d.requires_grad
        


        '''
        #计算配体内部的原子距离, 存在inf的情况，说有有上溢出和下溢出，这里可以判断一下，将inf替换成0，但这个0是要求带梯度的，否则报错
        c_ll_distance = self.calculate_distance_matrix(c_l_pos, c_l_pos) # n * n
        n_ll_distance = self.calculate_distance_matrix(n_l_pos, n_l_pos) # n * n

        # 使用 torch.where 将 inf 和 -inf 替换为 0
        c_ll_distance = torch.where(torch.nan(c_ll_distance), torch.tensor(0.0, requires_grad=True), c_ll_distance)
        n_ll_distance = torch.where(torch.nan(n_ll_distance), torch.tensor(0.0, requires_grad=True), n_ll_distance)

        true_ll_distance     = self.calculate_distance_matrix(origin_ligand_pos, origin_ligand_pos)
        mask_cutoff          = true_ll_distance < 8
        ligand_distance_loss = F.mse_loss(c_ll_distance[mask_cutoff], n_ll_distance[mask_cutoff])
        '''
        ligand_distance_loss = 0.0



        #exit()

        '''
        # atom type loss，原子类型分类损失
        log_ligand_v_recon = F.log_softmax(next_preds['pred_ligand_v'], dim=-1) #真实分类
        log_v_model_prob   = self.q_v_posterior(log_ligand_v_recon, next_log_ligand_vt, timesteps, batch_ligand) #预测出来的噪音
        log_v_true_prob    = self.q_v_posterior(log_ligand_v0, next_log_ligand_vt, timesteps, batch_ligand) #带有噪音的真实值
        

        #计算KL损失
        kl_v = self.compute_v_Lt(log_v_model_prob=log_v_model_prob, log_v0=log_ligand_v0,
                                log_v_true_prob=log_v_true_prob, t=timesteps, batch=batch_ligand)
        loss_v = torch.mean(kl_v)
        '''

        loss_v = torch.tensor(0)
        

        #亲和度
        #loss_exp = F.mse_loss(next_preds['final_exp_pred'], affinity) #这是真实的亲和度和预测的亲和度之间损失,亲和度是不加噪音的
        loss_exp = torch.tensor(0)

    

        #loss_exp = torch.tensor(0)
        
        #rmsd损失暂时不用
        #只使用内坐标
        if self.use_classifier_guide:  #The default is True
            #loss = ic_loss * GP.loss_weight['ic'] + loss_pos * GP.loss_weight['xyz']
            loss = ic_loss * GP.loss_weight['ic'] + loss_pos * GP.loss_weight['xyz'] + cross_distance_loss * GP.loss_weight['cross_distance'] + ref_cross_distance_loss * GP.loss_weight['ref_cross'] + ref_loss * GP.loss_weight['ref']
        else:
            #loss = ic_loss * GP.loss_weight['ic'] + loss_pos * GP.loss_weight['xyz']
            loss = ic_loss * GP.loss_weight['ic'] + loss_pos * GP.loss_weight['xyz'] + cross_distance_loss * GP.loss_weight['cross_distance'] + ref_cross_distance_loss * GP.loss_weight['ref_cross'] + ref_loss * GP.loss_weight['ref']

        

        return {
            'loss_pos': loss_pos,
            'loss_v': loss_v,
            'loss_exp': loss_exp,
            'loss': loss,
            'rmsd': rmsd, 
            'loss_dismat': loss_dismat,
            'loss_bond': loss_bond,
            'loss_angle': loss_angle,
            'loss_dihedral': loss_dihedral,
            'x0': ligand_pos,  #减过蛋白质心的真实值
            'pred_ligand_pos': next_preds['pred_ligand_pos'],
            'pred_ligand_v': torch.zeros_like(log_ligand_v0).cuda(),
            'pred_exp': next_preds['final_exp_pred'], #亲和度
            'pred_pos_noise': torch.zeros_like(next_preds['pred_ligand_pos']).cuda(),
            'ligand_v_recon': torch.zeros_like(log_ligand_v0).cuda(), #分类的概率
            'final_ligand_h': next_preds['final_ligand_h']  #原子的嵌入
        }

class ConsistencySamplingAndEditing:
    """Implements the Consistency Sampling and Zero-Shot Editing algorithms.

    Parameters
    ----------
    sigma_min : float, default=0.002
        Minimum standard deviation of the noise.
    sigma_data : float, default=0.5
        Standard deviation of the data.
    """

    def __init__(self, sigma_min: float = 0.002, sigma_data: float = 0.5) -> None:
        self.sigma_min = sigma_min
        self.sigma_data = sigma_data



    def q_v_sample(self, log_v0, t, batch):
        log_qvt_v0 = self.q_v_pred(log_v0, t, batch)
        sample_prob = log_sample_categorical(log_qvt_v0)
        sample_index = sample_prob.argmax(dim=-1) #返回最后一维度上的最大值索引，返回概率最大的索引
        log_sample = index_to_log_onehot(sample_index, self.num_classes) #one-hot化，概率索引
        return sample_index, log_sample

    # atom type generative process
    def q_v_posterior(self, log_v0, log_vt, t, batch):
        # q(vt-1 | vt, v0) = q(vt | vt-1, v0) * q(vt-1 | v0) / q(vt | v0)
        t_minus_1 = t - 1
        # Remove negative values, will not be used anyway for final decoder
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_qvt1_v0 = self.q_v_pred(log_v0, t_minus_1, batch)
        unnormed_logprobs = log_qvt1_v0 + self.q_v_pred_one_timestep(log_vt, t, batch)
        log_vt1_given_vt_v0 = unnormed_logprobs - torch.logsumexp(unnormed_logprobs, dim=-1, keepdim=True)
        return log_vt1_given_vt_v0
    


    # atom type diffusion process
    def q_v_pred_one_timestep(self, log_vt_1, t, batch):
        # q(vt | vt-1)
        log_alpha_t = extract(self.log_alphas_v, t, batch)
        log_1_min_alpha_t = extract(self.log_one_minus_alphas_v, t, batch)

        # alpha_t * vt + (1 - alpha_t) 1 / K
        log_probs = log_add_exp(
            log_vt_1 + log_alpha_t,
            log_1_min_alpha_t - np.log(self.num_classes)
        )
        return log_probs

    def q_v_pred(self, log_v0, t, batch):
        # compute q(vt | v0)
        log_cumprod_alpha_t = extract(self.log_alphas_cumprod_v, t, batch)
        log_1_min_cumprod_alpha = extract(self.log_one_minus_alphas_cumprod_v, t, batch)

        log_probs = log_add_exp(
            log_v0 + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - np.log(self.num_classes)
        )
        return log_probs

    def __call__(
        self,

        model: nn.Module,
        protein_atom_feature_dim,
        ligand_atom_feature_dim,

        config,

        #ground truth
        #protein_pos,
        #protein_v,
        affinity,
        #batch_protein,
        ligand_pos,
        ligand_v,
        org_ligand_pos,
        #batch_ligand,

        #sample params
        guide_mode,
        value_model,
        type_grad_weight,
        pos_grad_weight,

        protein_pos,
        protein_v,
        batch_protein,

        init_ligand_pos,
        init_ligand_v,
        batch_ligand,

        num_steps,
        center_pos_mode,

        ligand_bond_index, ligand_bond_type, ligand_bond_type_batch,


        ligand_atom_isring  =  None,
        ligand_atom_isO     =  None,
        ligand_atom_isN     =  None,

        protein_atom_isring =  None,
        protein_atom_isO    =  None,
        protein_atom_isN    =  None,



        cross_lig_isring_flag = None,
        cross_lig_isO_flag = None,
        cross_lig_isN_flag = None,

        cross_pro_isring_flag = None,
        cross_pro_isO_flag = None,
        cross_pro_isN_flag = None,


        cross_ligand    = None,
        cross_protein   = None,
        cross_distance  = None,


        cross_bond_index = None, 
        cross_bond_type = None, 
        cross_bond_index_reverse = None, 
        cross_bond_type_reverse = None,

        protein_coords_predict = None,

        complex_mol = None,

        protein_element_batch = None,
        protein_link_t_batch = None,
        protein_link_t_reverse_batch = None,

        ligand_element_batch = None,


        protein_element = None,
        ligand_element  = None,

        rd_pos = None,


        batch_center_pos = None,

        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        sigma_data: float = 0.5,
        initial_timesteps: int = 2, #最小是2
        final_timesteps: int = 25,  ##这里的self.final_timesteps步长有问题，默认是150，应该改一下，改成25/15之类的，不要太大，因为在构象生成中，不需要那么多步长):
        total_training_steps: int = 25,


        #not use
        feats: Tensor = None,
        adjs: Tensor = None,
        y: Tensor = None,
        gmasks: Tensor = None,
        sigmas: Iterable[Union[Tensor, float]]  = None,


        mask: Optional[Tensor] = None,
        transform_fn: Callable[[Tensor], Tensor] = lambda x: x,
        inverse_transform_fn: Callable[[Tensor], Tensor] = lambda x: x,
        start_from_y: bool = False,
        add_initial_noise: bool = False, # default True
        clip_denoised: bool = False,
        verbose: bool = False,
        **kwargs: Any,
    ) -> Tensor:
        """Runs the sampling/zero-shot editing loop.

        With the default parameters the function performs consistency sampling.

        Parameters
        ----------
        model : nn.Module
            Model to sample from.
        y : Tensor
            Reference sample e.g: a masked image or noise.
        sigmas : Iterable[Union[Tensor, float]]
            Decreasing standard deviations of the noise.
        mask : Tensor, default=None
            A mask of zeros and ones with ones indicating where to edit. By
            default the whole sample will be edited. This is useful for sampling.
        transform_fn : Callable[[Tensor], Tensor], default=lambda x: x
            An invertible linear transformation. Defaults to the identity function.
        inverse_transform_fn : Callable[[Tensor], Tensor], default=lambda x: x
            Inverse of the linear transformation. Defaults to the identity function.
        start_from_y : bool, default=False
            Whether to use y as an initial sample and add noise to it instead of starting
            from random gaussian noise. This is useful for tasks like style transfer.
        add_initial_noise : bool, default=True
            Whether to add noise at the start of the schedule. Useful for tasks like interpolation
            where noise will alerady be added in advance.
        clip_denoised : bool, default=False
            Whether to clip denoised values to [-1, 1] range.
        verbose : bool, default=False
            Whether to display the progress bar.
        **kwargs : Any
            Additional keyword arguments to be passed to the model.

        Returns
        -------
        Tensor
            Edited/sampled sample.
        """

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.sigma_data = sigma_data
        self.initial_timesteps = initial_timesteps
        self.final_timesteps = final_timesteps
        self.num_classes = ligand_atom_feature_dim
        self.ref_atom_type = copy.deepcopy(ligand_v)


        # atom type diffusion schedule in log space
        alphas_v                = cosine_beta_schedule(self.final_timesteps, 0.01)
        log_alphas_v            = np.log(alphas_v)
        log_alphas_cumprod_v    = np.cumsum(log_alphas_v)
        self.log_alphas_v       = to_torch_const(log_alphas_v).cuda()
        self.log_one_minus_alphas_v         = to_torch_const(log_1_min_a(log_alphas_v)).cuda()
        self.log_alphas_cumprod_v           = to_torch_const(log_alphas_cumprod_v).cuda()
        self.log_one_minus_alphas_cumprod_v = to_torch_const(log_1_min_a(log_alphas_cumprod_v)).cuda()


        num_graphs = max(batch_ligand) + 1


        #采样的时候，sigma和训练过程不同，sigma要保证每一步都一样，从最大值开始,因此sigma序列要固定
        #self.final_timesteps的取值在[1, *]
        sigmas = karras_schedule(
                self.final_timesteps, self.sigma_min, self.sigma_max, self.rho, init_ligand_pos.device
            )

        #print('sigmas1:', sigmas)
        if self.final_timesteps == 1:
            sigmas= reversed(sigmas) #注意从大到小排序，取逆序即可
            sigmas[-1] = 80 #这里不能使用 sigmas[-1] += 1e-8。因为只有1步时，sigmas1: tensor([0.0020]), 此时噪音太小容易，c_cout趋近0, 
            #去噪不起作用，所得结果是正态分布噪音，所以改成最大值80。训练阶段步长是从2开始的，所以没遇到这问题
            #sigmas[-1] += 1e-8
        else:
            #sigmas= reversed(sigmas)[:-1] #第一步不用，这个不能用，直接报错了，索引溢出, 此时需要每一步前加1，保证第一步
            sigmas= reversed(sigmas)
            #print('sigmas2:', sigmas[-1].detach().cpu().numpy())
            sigmas[-1] += 1e-8  #修改了这里,如果不加一个很小的数，会导致c_out变成负数，下溢出，趋近0
            #print('sigmas3:', sigmas[-1].detach().cpu().numpy())
            #sigmas2: 0.0019999996
            #sigmas3: 0.0020000096

        

        time_list = np.array(list(reversed(list(range(len(sigmas)))))) #记录时间步长，最好从1开始。从0开始，会导致编码失效的





        # Set mask to all ones which is useful for sampling and style transfer
        if mask is None:
            mask = torch.ones_like(y)

        # Use y as an initial sample which is useful for tasks like style transfer
        # and interpolation where we want to use content from the reference sample
        x = y if start_from_y else torch.zeros_like(y)

        # Sample at the end of the schedule
        y = self.__mask_transform(x, y, mask, transform_fn, inverse_transform_fn)
        # For tasks like interpolation where noise will already be added in advance we
        # can skip the noising process

        #print('sigmas[0]:', sigmas[0])
        x = y + batch_center_pos + sigmas[0] * torch.randn_like(y)
        #x = y + batch_center_pos + torch.randn_like(y)
        #x = y + torch.randn_like(y)

        #x = y + org_ligand_pos

        #x = y + sigmas[0] * torch.randn_like(y)

        #x = y + batch_center_pos + sigmas[0] * org_ligand_pos #勉强可以

        #x = org_ligand_pos + sigmas[0] * torch.randn_like(y) #勉强可以

        #x = y + batch_center_pos  + org_ligand_pos  

        #print('origin_cross_ligand:', cross_ligand.shape)
        #print('org_ligand_pos:', org_ligand_pos.shape)
        #print('x:', x.shape)

        #print('origin_cross_ligand[:3]:', cross_ligand[:])
        #print('org_ligand_pos[:3]:', org_ligand_pos[:])
        pos_traj, v_traj, exp_traj, exp_atom_traj = [], [], [], []
        
        #pos_traj.append(x.clone().cpu()) 
        
        init_protein_pos, init_ligand_pos, offset = center_pos(protein_pos, x, batch_protein, batch_ligand, mode=center_pos_mode) #配体和蛋白减去了蛋白的质心

        #rdkit 坐标
        _, rd_pos, _ = center_pos(protein_pos, rd_pos.float(), batch_protein, batch_ligand, mode=center_pos_mode) #质心在蛋白上


        org_protein_pos, org_ligand_pos, org_offset = center_pos(protein_pos, org_ligand_pos, batch_protein, batch_ligand, mode=center_pos_mode) #配体和蛋白减去了蛋白的质心
        

        _, protein_coords_predict, _ = center_pos(protein_pos, protein_coords_predict, batch_protein, batch_ligand, mode=center_pos_mode)


        #print('origin_cross_ligand[:3]:', cross_ligand[:3])
        #print('org_ligand_pos[:3]:', org_ligand_pos[:3]) #这个坐标有问题，与实际的坐标不一样，正常，因为这里的x是加噪的坐标，而不是原始的, 这里的org_ligand_pos是rdkit的，也不是参考的

        #这2个别忘了减去质心
        cross_ligand    = cross_ligand - offset[batch_ligand]
        cross_protein   = cross_protein - offset[batch_protein]


        
        v0_pred_traj, vt_pred_traj = [], []
        ligand_pos, ligand_v = init_ligand_pos, init_ligand_v
        protein_pos = init_protein_pos

        '''
        np.set_printoptions(suppress=True, precision=4)
        torch.set_printoptions(sci_mode=False, precision=4)

        print('cross_ligand:', cross_ligand.shape)
        print('org_ligand_pos:', org_ligand_pos.shape)
        print('ligand_pos:', ligand_pos.shape)

        print('cross_ligand[:3]:', cross_ligand[:3])
        print('org_ligand_pos[:3]:', org_ligand_pos[:3])
        print('ligand_pos[:3]:', ligand_pos[:3])

        #蛋白是一样的，但配体坐标不一样，什么原因？因为x是加噪的
        print('cross_protein[:3]:', cross_protein[:])
        print('protein_pos[:3]:', protein_pos[:])

        print('cross_protein.shape:', cross_protein.shape)
        print('protein_pos.shape:', protein_pos.shape)

        raise Exception('stop')
        '''



        sigma = torch.full((num_graphs,), sigmas[0], dtype=x.dtype, device=x.device)
        #固定sigma，即step也固定
        timesteps = torch.full((num_graphs,), time_list[0], dtype=torch.int64, device=init_ligand_pos.device)


        with torch.no_grad():
        #with torch.enable_grad():
            preds, type_grad, pos_grad = self.consistency_pv_joint_guide(
                model,
                None,
                None,
                ligand_pos, #修改
                None,
                sigma, #修改
                self.sigma_data,
                self.sigma_min,

                protein_pos=protein_pos, #
                protein_v=protein_v, 
                batch_protein=batch_protein,

                init_ligand_pos=ligand_pos, #加噪音了 #修改
                init_ligand_v=ligand_v,  
                batch_ligand=batch_ligand,
                time_step=timesteps + 1, #修改
                org_ligand_pos = org_ligand_pos,
                org_protein_pos = org_protein_pos,
                ligand_bond_index = ligand_bond_index, ligand_bond_type = ligand_bond_type, ligand_bond_type_batch = ligand_bond_type_batch,
                args = None,
                protein_element = protein_element,
                ligand_element  = ligand_element,
                sample = True,   #采样效果差，原因找到了，sample应该设置为True, 使用与训练不同的原子类型嵌入方法
                scale = True, #
                
                ligand_atom_isring  = ligand_atom_isring,
                ligand_atom_isO     = ligand_atom_isO,
                ligand_atom_isN     = ligand_atom_isN,

                protein_atom_isring = protein_atom_isring,
                protein_atom_isO    = protein_atom_isO,
                protein_atom_isN    = protein_atom_isN,

                cross_lig_isring_flag   = cross_lig_isring_flag,
                cross_lig_isO_flag      = cross_lig_isO_flag,
                cross_lig_isN_flag      = cross_lig_isN_flag,

                cross_pro_isring_flag   = cross_pro_isring_flag,
                cross_pro_isO_flag      = cross_pro_isO_flag,
                cross_pro_isN_flag      = cross_pro_isN_flag,

                cross_ligand    = cross_ligand,
                cross_protein   = cross_protein,
                cross_distance  = cross_distance,

                cross_bond_index = cross_bond_index, 
                cross_bond_type = cross_bond_type, 
                cross_bond_index_reverse = cross_bond_index_reverse, 
                cross_bond_type_reverse = cross_bond_type_reverse,

                protein_coords_predict = protein_coords_predict,

                complex_mol = complex_mol,

                protein_element_batch = protein_element_batch,
                protein_link_t_batch = protein_link_t_batch,
                protein_link_t_reverse_batch = protein_link_t_reverse_batch,

                ligand_element_batch = ligand_element_batch,

                rd_pos = rd_pos,



            )


        if clip_denoised:
            preds['pred_ligand_pos'] = preds['pred_ligand_pos'].clamp(min=-1.0, max=1.0)
        
        ligand_pos = preds['pred_ligand_pos']

        #谐振子优化
        #print('ligand_pos:', ligand_pos.requires_grad) #True
        #print('protein_pos:', protein_pos.requires_grad)
        #print('cross_distance[0]:', cross_distance[0].requires_grad)
        #ligand_pos = self.force_gradient(ligand_pos.clone().detach(), protein_pos.clone().detach(), cross_distance[0].clone().detach())

        #ligand_pos = ligand_pos + self.Distance_Opt(ligand_pos.clone().detach(), protein_pos.clone().detach(), cross_distance[0].clone().detach())

        batch_all   = preds['batch_all']
        mask_ligand = preds['mask_ligand']
        final_pos   = preds['final_pos']
        p_pos = [final_pos[batch_all == k][mask_ligand[batch_all == k] == 0] for k in range(max(batch_all) + 1)]
        #assert np.allclose(p_pos[0].cpu(), p_pos[-1].cpu(), atol=0.02) #看看蛋白是否改变了
        
        if GP.final_timesteps == 1:
            if GP.with_MMFF_guide:
                #同步优化：优化的坐标和神经网络预测出来的坐标相加
                '''
                x, #配体和蛋白坐标
                rdkit_mols, #配体和蛋白的复合物的ridkt mol,这个要改坐标, 减质心问题
                gmasks.bool(), #标志当前图的编号
                loop=guide_loops, #梯度下降优化次数，默认1次，扩散一次优化一次
                show_state=show_state,
                min_type=min_type,  #选择优化器，如LBFGS
                fix_masks=fix_mask, #固定不动原子，如蛋白
                pocket_masks=pocket_labels.bool(), #哪些是蛋白原子
                ligand_masks=ligand_labels.bool()  #哪些是配体原子
                '''
                preds['final_pos'][preds['mask_ligand']] = preds['pred_ligand_pos']
                complex_pos = preds['final_pos']
                

                with torch.enable_grad():
                    #_,  cross_loss = self.Distance_Opt(ligand_pos.clone().detach(), protein_pos.clone().detach(), cross_distance[0].clone().detach(), min_type = GP.min_type)
                    if GP.guide_type=='synchronous': #如果是同步，那这里的坐标x，应该是在神经网络的输入，而非输出，其它和异步一样
                        if GP.opt_types=="complex":
                            pass
                            #x_moves,energy_min=opt_complex_coords_moves(x_bp,rdkit_mols,gmasks.bool(),loop=1,show_state=Ture,min_type=LBFGS,fix_masks=fix_mask,pocket_masks=pocket_labels.bool(),ligand_masks=ligand_labels.bool())
                        else:
                            pass
                            #x_moves,energy_min=opt_coords_moves(x_bp,rdkit_mols,gmasks.bool(),loop=guide_loops,show_state=show_state,min_type=min_type,fix_masks=fix_mask)
                    
                    #异步优化，先神经网络，后优化。我们使用异步+complex, asynchronous 
                    else:
                        try:
                            if GP.opt_types=="complex":
                                x_moves,energy_min=opt_complex_coords_moves(complex_pos.clone().detach(), copy.deepcopy(complex_mol), preds['batch_all'].clone().detach(), loop=1, show_state=True, min_type=GP.min_type, mask_ligand = preds['mask_ligand'].clone().detach(), cross_loss = None, ligand_pos = ligand_pos.clone().detach(), protein_pos = protein_pos.clone().detach(), cross_distance = copy.deepcopy(cross_distance))
                            else:
                                x_moves,energy_min=opt_coords_moves(complex_pos.clone().detach(), copy.deepcopy(complex_mol), preds['batch_all'].clone().detach(), loop=1, show_state=True, min_type=GP.min_type, mask_ligand = preds['mask_ligand'].clone().detach(), cross_loss = None, ligand_pos = ligand_pos.clone().detach(), protein_pos = protein_pos.clone().detach(), cross_distance = copy.deepcopy(cross_distance))
                        except Exception as e:
                            print(e)
                            x_moves = 0


                ligand_pos = ligand_pos+x_moves #x_moves可以看成是移动量，也可以看成是优化的坐标，这里实际就是神经网络的坐标+优化后的坐标

        


        ligand_pos = self.__mask_transform(ligand_pos, y, mask, transform_fn, inverse_transform_fn)
        exp_pred = preds['final_exp_pred']


        ori_ligand_pos = ligand_pos + offset[batch_ligand]  #加上质心
        pos_traj.append(ori_ligand_pos.clone().cpu())       #存放解析出来的坐标
        v_traj.append(ligand_v.clone().cpu())               #存放解析出来的v
    
    



        #亲和度
        if exp_pred is not None:
            exp_traj.append(exp_pred.clone().cpu())
            exp_atom_traj.append(preds['atom_affinity'].clone().cpu())


        for step, (sigma, t) in enumerate(zip(sigmas[1:], time_list[1:])):
            #print('sigma:', sigma)
            noise = torch.randn_like(ligand_pos)

            #步长序列，类似time_step
            #timesteps = torch.randint(0, num_timesteps - 1, (ligand_pos.shape[0],), device=ligand_pos.device) #取值是[ ),前闭后开
            #timesteps = torch.full((num_graphs,), t, dtype=ligand_pos.dtype, device=ligand_pos.device)
            timesteps = torch.full((num_graphs,), t, dtype=torch.int64, device=ligand_pos.device)
            #print (noise.shape,timesteps.shape)
            next_sigma = torch.full((num_graphs,), sigma, dtype=ligand_pos.dtype, device=ligand_pos.device) #值得注意的是，在训练过程中，sigma是多变的，不同的节点可能取不能的sigma,但是在采样的过程中
            #print (current_sigmas.shape,next_sigmas.shape)
            #print(f'step{step+1}:', next_sigma)

            new_sigma = (next_sigma**2 - self.sigma_min**2) ** 0.5
            #print(f'step{step + 1} new_sigma:', new_sigma)
            next_pos  = new_sigma.index_select(0, batch_ligand).view([-1, 1])

                
            #加噪
            ligand_pos = ligand_pos + next_pos * noise 


            #当sigmas很小的时候，pad_dims_like(next_sigmas, xyzs) * noise  == 0, next_xyzs趋近于真实的xyzs
            #这里是sigma**2 - self.sigma_min**2，意味着当simga很小的时候，两者相减趋近于0，此时神经网络的输出以及加的噪音不起作用，x取自身


            with torch.no_grad():
            #with torch.enable_grad():
                preds, type_grad, pos_grad = self.consistency_pv_joint_guide(
                    model,
                    None,
                    None,
                    ligand_pos, #修改
                    None,
                    next_sigma, #修改
                    self.sigma_data,
                    self.sigma_min,

                    protein_pos=protein_pos, #
                    protein_v=protein_v, 
                    batch_protein=batch_protein,

                    init_ligand_pos=ligand_pos, #加噪音了 #修改
                    init_ligand_v=ligand_v,   #
                    batch_ligand=batch_ligand,
                    time_step=timesteps + 1, #修改
                    org_ligand_pos = org_ligand_pos,
                    org_protein_pos = org_protein_pos,
                    ligand_bond_index = ligand_bond_index, ligand_bond_type = ligand_bond_type, ligand_bond_type_batch = ligand_bond_type_batch,
                    args = None,
                    protein_element = protein_element,
                    ligand_element  = ligand_element,
                    sample = True,  #采样效果差，原因找到了，sample应该设置为True, 使用与训练不同的原子类型嵌入方法
                    scale = True, 

                    ligand_atom_isring  = ligand_atom_isring,
                    ligand_atom_isO     = ligand_atom_isO,
                    ligand_atom_isN     = ligand_atom_isN,

                    protein_atom_isring = protein_atom_isring,
                    protein_atom_isO    = protein_atom_isO,
                    protein_atom_isN    = protein_atom_isN,

                    cross_lig_isring_flag   = cross_lig_isring_flag,
                    cross_lig_isO_flag      = cross_lig_isO_flag,
                    cross_lig_isN_flag      = cross_lig_isN_flag,

                    cross_pro_isring_flag   = cross_pro_isring_flag,
                    cross_pro_isO_flag      = cross_pro_isO_flag,
                    cross_pro_isN_flag      = cross_pro_isN_flag,

                    cross_ligand    = cross_ligand,
                    cross_protein   = cross_protein,
                    cross_distance  = cross_distance,

                    cross_bond_index = cross_bond_index, 
                    cross_bond_type = cross_bond_type, 
                    cross_bond_index_reverse = cross_bond_index_reverse, 
                    cross_bond_type_reverse = cross_bond_type_reverse,

                    protein_coords_predict = protein_coords_predict,

                    complex_mol = complex_mol,

                    protein_element_batch = protein_element_batch,
                    protein_link_t_batch = protein_link_t_batch,
                    protein_link_t_reverse_batch = protein_link_t_reverse_batch,

                    ligand_element_batch = ligand_element_batch,

                    rd_pos = rd_pos,
                )
            
            if clip_denoised:
                preds['pred_ligand_pos'] = preds['pred_ligand_pos'].clamp(min=-1.0, max=1.0)

            ligand_pos = preds['pred_ligand_pos']

            #谐振子优化
            #print('ligand_pos:', ligand_pos.requires_grad)
            #print('protein_pos:', protein_pos.requires_grad)
            #print('cross_distance[0]:', cross_distance[0].requires_grad)
            #ligand_pos = self.force_gradient(ligand_pos.clone().detach(), protein_pos.clone().detach(), cross_distance[0].clone().detach())
            #ligand_pos = ligand_pos + self.Distance_Opt(ligand_pos.clone().detach(), protein_pos.clone().detach(), cross_distance[0].clone().detach())
            #ligand_pos = self.Distance_Opt(ligand_pos.clone().detach(), protein_pos.clone().detach(), cross_distance[0].clone().detach())

            batch_all   = preds['batch_all']
            mask_ligand = preds['mask_ligand']
            final_pos   = preds['final_pos']
            p_pos = [final_pos[batch_all == k][mask_ligand[batch_all == k] == 0] for k in range(max(batch_all) + 1)]
            #assert np.allclose(p_pos[0].cpu(), p_pos[-1].cpu(), atol=0.02) #看看蛋白是否改变了

            if step in list(range(len(sigmas[1:])))[-GP.force_step:]:
                if GP.with_MMFF_guide:
                    #同步优化：优化的坐标和神经网络预测出来的坐标相加
                    '''
                    x, #配体和蛋白坐标
                    rdkit_mols, #配体和蛋白的复合物的ridkt mol,这个要改坐标, 减质心问题
                    gmasks.bool(), #标志当前图的编号
                    loop=guide_loops, #梯度下降优化次数，默认1次，扩散一次优化一次
                    show_state=show_state,
                    min_type=min_type,  #选择优化器，如LBFGS
                    fix_masks=fix_mask, #固定不动原子，如蛋白
                    pocket_masks=pocket_labels.bool(), #哪些是蛋白原子
                    ligand_masks=ligand_labels.bool()  #哪些是配体原子
                    '''
                    preds['final_pos'][preds['mask_ligand']] = preds['pred_ligand_pos']
                    complex_pos = preds['final_pos']
                    
                    with torch.enable_grad():
                        #_,  cross_loss = self.Distance_Opt(ligand_pos.clone().detach(), protein_pos.clone().detach(), cross_distance[0].clone().detach(), min_type = GP.min_type)
                        if GP.guide_type=='synchronous': #如果是同步，那这里的坐标x，应该是在神经网络的输入，而非输出，其它和异步一样
                            if GP.opt_types=="complex":
                                pass
                                #x_moves,energy_min=opt_complex_coords_moves(x_bp,rdkit_mols,gmasks.bool(),loop=1,show_state=Ture,min_type=LBFGS,fix_masks=fix_mask,pocket_masks=pocket_labels.bool(),ligand_masks=ligand_labels.bool())
                            else:
                                pass
                                #x_moves,energy_min=opt_coords_moves(x_bp,rdkit_mols,gmasks.bool(),loop=guide_loops,show_state=show_state,min_type=min_type,fix_masks=fix_mask)
                        
                        #异步优化，先神经网络，后优化。我们使用异步+complex, asynchronous 
                        else:
                            try:
                                if GP.opt_types=="complex":
                                    x_moves,energy_min=opt_complex_coords_moves(complex_pos.clone().detach(), copy.deepcopy(complex_mol), preds['batch_all'].clone().detach(), loop=GP.loop, show_state=True, min_type=GP.min_type, mask_ligand = preds['mask_ligand'].clone().detach(), cross_loss = None, ligand_pos = ligand_pos.clone().detach(), protein_pos = protein_pos.clone().detach(), cross_distance = copy.deepcopy(cross_distance))
                                else:
                                    x_moves,energy_min=opt_coords_moves(complex_pos.clone().detach(), copy.deepcopy(complex_mol), preds['batch_all'].clone().detach(), loop=GP.loop, show_state=True, min_type=GP.min_type, mask_ligand = preds['mask_ligand'].clone().detach(), cross_loss = None, ligand_pos = ligand_pos.clone().detach(), protein_pos = protein_pos.clone().detach(), cross_distance = copy.deepcopy(cross_distance))
                            except Exception as e:
                                print(e)
                                x_moves = 0

                    ligand_pos = ligand_pos+x_moves #x_moves可以看成是移动量，也可以看成是优化的坐标，这里实际就是神经网络的坐标+优化后的坐标# * 0.1, 5step



            ligand_pos = self.__mask_transform(ligand_pos, y, mask, transform_fn, inverse_transform_fn)
            exp_pred = preds['final_exp_pred']



            ori_ligand_pos = ligand_pos + offset[batch_ligand]  #加上质心
            pos_traj.append(ori_ligand_pos.clone().cpu())       #存放解析出来的坐标
            v_traj.append(ligand_v.clone().cpu())               #存放解析出来的v
        
            
            #亲和度
            if exp_pred is not None:
                exp_traj.append(exp_pred.clone().cpu())
                exp_atom_traj.append(preds['atom_affinity'].clone().cpu())
            


        ligand_pos = ligand_pos + offset[batch_ligand]

        #把用于构建连接表的坐标保存成sdf，看看对不对
        coords_predict = protein_coords_predict + offset[batch_ligand]

        assert coords_predict.shape == ligand_pos.shape
                
        #ligand_pos = pos_traj[-1]
        #ligand_v   = v_traj[-1] #指定步长结果
        return {
            'pos': ligand_pos,
            'coords_predict': coords_predict,
            'v': ligand_v,
            'exp': exp_traj[-1] if len(exp_traj) else [],
            'pos_traj': pos_traj,
            'v_traj': v_traj,
            'exp_traj': exp_traj,
            'exp_atom_traj': exp_atom_traj,
            'v0_traj': v0_pred_traj,
            'vt_traj': vt_pred_traj,
        }








    def Distance_Opt_old(self, ligand_pos, protein_pos, cross_distance, iterations=1, min_type='AdamW', early_stoping=10):
        with torch.enable_grad():
            #加谐振子约束优化
            ligand_pos.requires_grad = True
            #coor_pred_detach = coor_pred.detach()
            ligand_pos_Parm = torch.nn.Parameter(ligand_pos.detach().clone()).to(ligand_pos.device)
            #protein_pos.requires_grad = True
            pred_distance = torch.cdist(ligand_pos_Parm,protein_pos,compute_mode='donot_use_mm_for_euclid_dist')
            #print('pred_distance:', pred_distance.requires_grad) # False
            Distance_loss = F.mse_loss(pred_distance, cross_distance)
            print ('before optim distance:', Distance_loss)
            if min_type == 'SGD':
                optimizer = torch.optim.SGD([ligand_pos_Parm], lr=0.02)
            elif min_type =='AdamW':
                optimizer = torch.optim.AdamW([ligand_pos_Parm], lr=0.02)
            else:
                optimizer = torch.optim.LBFGS([ligand_pos_Parm], lr=0.0001) #LBFGS

            for i in range(iterations): 
                def closure():
                    optimizer.zero_grad()
                    pred_distance = torch.cdist(ligand_pos_Parm,protein_pos,compute_mode='donot_use_mm_for_euclid_dist')
                    loss = F.mse_loss(pred_distance, cross_distance)
                    loss.backward()
                    return loss
                close_loss = closure()
                if min_type == 'SGD':
                    optimizer.step()
                elif min_type == 'AdamW':
                    optimizer.step()
                else: 
                    optimizer.step(closure)
                #print('close_loss:', close_loss.item())

            ligand_pos_Min = ligand_pos_Parm.detach().clone().to(ligand_pos.device)
            pred_distance = torch.cdist(ligand_pos_Min, protein_pos, compute_mode='donot_use_mm_for_euclid_dist')
            Distance_loss = F.mse_loss(pred_distance, cross_distance)
            print('after optim ditance:', Distance_loss)

            return ligand_pos_Min, close_loss
        


    def Distance_Opt(self, ligand_pos, protein_pos, cross_distance, iterations=1, min_type='AdamW', early_stoping=10):
        with torch.enable_grad():
            #加谐振子约束优化
            ligand_pos.requires_grad = True
            #coor_pred_detach = coor_pred.detach()
            ligand_pos_Parm = torch.nn.Parameter(ligand_pos.detach().clone()).to(ligand_pos.device)
            #protein_pos.requires_grad = True
            pred_distance = torch.cdist(ligand_pos_Parm,protein_pos,compute_mode='donot_use_mm_for_euclid_dist')
            #print('pred_distance:', pred_distance.requires_grad) # False
            Distance_loss = F.mse_loss(pred_distance, cross_distance)
            print ('before optim distance:', Distance_loss)
            if min_type == 'SGD':
                optimizer = torch.optim.SGD([ligand_pos_Parm], lr=0.02)
            elif min_type =='AdamW':
                optimizer = torch.optim.AdamW([ligand_pos_Parm], lr=0.02)
            else:
                optimizer = torch.optim.LBFGS([ligand_pos_Parm], lr=0.0001) #LBFGS

            for i in range(iterations): 
                def closure():
                    #optimizer.zero_grad()
                    pred_distance = torch.cdist(ligand_pos_Parm,protein_pos,compute_mode='donot_use_mm_for_euclid_dist')
                    loss = F.mse_loss(pred_distance, cross_distance)
                    #loss.backward() #不在这里更新梯度
                    return loss
                close_loss = closure()

                #if min_type == 'SGD':
                    #optimizer.step()
                #elif min_type == 'AdamW':
                    #optimizer.step()
                #else: 
                    #optimizer.step(closure)
                #print('close_loss:', close_loss.item())

            ligand_pos_Min = ligand_pos_Parm.detach().clone().to(ligand_pos.device)
            pred_distance = torch.cdist(ligand_pos_Min, protein_pos, compute_mode='donot_use_mm_for_euclid_dist')
            Distance_loss = F.mse_loss(pred_distance, cross_distance)
            print('after optim ditance:', Distance_loss)

            return ligand_pos_Min, close_loss








    def force_gradient(self, ligand_pos, protein_pos, cross_distance, iterations=1, early_stoping=1):
        gradient_data_dict = {
                    'ligand_pos': ligand_pos,
                    'protein_pos': protein_pos,
                    'cross_distance': cross_distance
                    }
        
        torch.save(gradient_data_dict, 'gradient_data_dict.pt')
        
        #print('ligand_pos:', ligand_pos.requires_grad) # True
        #print('protein_pos:', protein_pos.requires_grad)# False
        #print('cross_distance:', cross_distance.requires_grad)# False
    
        #加谐振子约束优化
        ligand_pos.requires_grad = True
        #coor_pred_detach = coor_pred.detach()
        ligand_pos_Parm = torch.nn.Parameter(ligand_pos.detach().clone()).to(ligand_pos.device)
        #protein_pos.requires_grad = True
        pred_distance = self.calculate_distance_matrix(ligand_pos_Parm, protein_pos) #到这里没梯度了，导致后面ligand_pos没参与
        assert pred_distance.shape == cross_distance.shape
        
        #print('pred_distance:',pred_distance.requires_grad) # False

        distance = F.mse_loss(pred_distance, cross_distance)
        print('befor optim ditance:', distance)
        clone_protein_pos, clone_cross_distance = protein_pos.clone(), cross_distance.clone()

        optimizer = torch.optim.LBFGS([ligand_pos_Parm], lr=1.0)
        bst_loss, times = 10000.0, 0

        pred_distance.requires_grad = True
        cross_distance.requires_grad = True

        #print(tensor_with_grad.requires_grad)
        #print('ligand_pos:', ligand_pos.requires_grad)
        #print('pred_distance_detached:', pred_distance.requires_grad)
        #print('cross_distance_detached:', cross_distance.requires_grad)
            
        for i in range(iterations): 
            def closure():
                optimizer.zero_grad()
                ##注意直接预测的距离和通过预测的坐标求的距离不一样，这里的coords初始值是rdkit坐标，这里通过优化与通过嵌入预测出来距离来，得到新的坐标
                #在前面的GNN中，coords只是以距离的方式参与（尽管这个距离是从rdkit构象获取的，但是作为边注意力的一部分使用是足够的，没问题的，使用rdkit得到的构象，它的相对分子距离是正确的），
                # 并未对坐标直接建模，所以这里是将rdkit的坐标与预测的距离计算mse进行梯度更新，来优化rdkit坐标，之后再送入坐标的神经网络里面
                #loss = F.mse_loss(pred_distance.detach().requires_grad_(True), cross_distance.detach().requires_grad_(True))
                loss = F.mse_loss(pred_distance, cross_distance) * 1 #看看优化后mse是否下降，以及坐标是否发生改变？
                #loss = self.scoring_function(ligand_pos, protein_pos, pred_distance_detached, cross_distance_detached)
                loss.backward(retain_graph=True)
                print('optim loss:', loss.item())
                return loss
            loss = optimizer.step(closure)
            if loss.item() < bst_loss:
                bst_loss = loss.item()
                times = 0 
            else:
                times += 1
                if times > early_stoping:
                    break

        

        assert torch.equal(clone_protein_pos,protein_pos) and torch.equal(clone_cross_distance,cross_distance)
        pred_distance = self.calculate_distance_matrix(ligand_pos_Parm, protein_pos)
        distance = F.mse_loss(pred_distance, cross_distance) #值没有变化，也就是坐标没动
        print('after optim ditance:', distance)
        print('-----------------------------------------------------')

        return ligand_pos_Parm.detach()



    def calculate_distance_matrix(self, A, B):
        """
        计算两个坐标矩阵之间的欧氏距离

        参数:
        A (torch.Tensor): 大小为 (n, 3) 的坐标矩阵
        B (torch.Tensor): 大小为 (m, 3) 的坐标矩阵

        返回:
        torch.Tensor: 大小为 (n, m) 的距离矩阵
        """
        # A 的形状为 (n, 3)，B 的形状为 (m, 3)
        # 计算 A 的每个点与 B 的每个点之间的距离
        print('A:', A.requires_grad) # True
        print('B:', B.requires_grad)# True
        print('A.unsqueeze(1):', A.unsqueeze(1).requires_grad)# True
        print('B.unsqueeze(0):', B.unsqueeze(0).requires_grad)# True
        diff = A.unsqueeze(1) - B.unsqueeze(0)  # diff 的形状为 (n, m, 3),# 相减之后没有梯度了，是什么情况？
        print('diff.shape:', diff.shape)
        #print('diff:', diff)
        print('A + A:', (A + A).requires_grad) # False
        print('diff:', diff.requires_grad) # False
        dist_matrix = torch.sqrt(torch.sum(diff**2, dim=2))  # dist_matrix 的形状为 (n, m)
        print('dist_matrix:', dist_matrix.requires_grad)
        return dist_matrix


    def consistency_pv_joint_guide(
        self,
        model: nn.Module,
        feats: Tensor = None,
        adjs: Tensor = None,
        xyzs: Tensor = None,
        gmasks: Tensor = None,
        sigma: Tensor = None,
        sigma_data: float = 0.5,
        sigma_min: float = 0.002,

        
        protein_pos=None, #
        protein_v=None, 
        batch_protein=None,

        init_ligand_pos=None, #加噪音了
        init_ligand_v=None,  #
        batch_ligand=None,
        time_step=None,
        org_ligand_pos = None,
        org_protein_pos = None,
        ligand_bond_index = None, ligand_bond_type = None, ligand_bond_type_batch = None,
        protein_max_atom_num = None, ligand_max_atom_num  = None,
        args = None,
        protein_element = None,
        ligand_element  = None,
        sample = True,
        scale = True,
        ligand_atom_isring  = None,
        ligand_atom_isO     = None,
        ligand_atom_isN     = None,

        protein_atom_isring = None,
        protein_atom_isO    = None,
        protein_atom_isN    = None,

        cross_lig_isring_flag   = None,
        cross_lig_isO_flag      = None,
        cross_lig_isN_flag      = None,

        cross_pro_isring_flag   = None,
        cross_pro_isO_flag      = None,
        cross_pro_isN_flag      = None,


        cross_ligand    = None,
        cross_protein   = None,
        cross_distance  = None,

        
        cross_bond_index = None, 
        cross_bond_type = None, 
        cross_bond_index_reverse = None, 
        cross_bond_type_reverse = None,

        protein_coords_predict = None,

        complex_mol = None,

        protein_element_batch = None,
        protein_link_t_batch = None,
        protein_link_t_reverse_batch = None,

        ligand_element_batch = None,

        rd_pos = None,


        ) -> Tensor:


        #with torch.no_grad():
        with torch.enable_grad():
            outputs = model_forward_wrapper(
                model,
                None,
                None,
                xyzs, #修改
                None,
                sigma, #修改
                self.sigma_data,
                self.sigma_min,

                protein_pos=protein_pos, #
                protein_v=protein_v, 
                batch_protein=batch_protein,

                init_ligand_pos=init_ligand_pos, #加噪音了 #修改
                init_ligand_v=init_ligand_v,  
                batch_ligand=batch_ligand,
                time_step=time_step, #修改
                

                org_ligand_pos = org_ligand_pos,
                org_protein_pos = org_protein_pos,
                ligand_bond_index = ligand_bond_index, ligand_bond_type = ligand_bond_type, ligand_bond_type_batch = ligand_bond_type_batch,
                protein_max_atom_num = protein_max_atom_num, ligand_max_atom_num  = protein_max_atom_num,

                protein_element = protein_element,
                ligand_element  = ligand_element,
                sample = sample,
                scale = scale,

                ligand_atom_isring  = ligand_atom_isring,
                ligand_atom_isO     = ligand_atom_isO,
                ligand_atom_isN     = ligand_atom_isN,

                protein_atom_isring = protein_atom_isring,
                protein_atom_isO    = protein_atom_isO,
                protein_atom_isN    = protein_atom_isN,

                cross_lig_isring_flag   = cross_lig_isring_flag,
                cross_lig_isO_flag      = cross_lig_isO_flag,
                cross_lig_isN_flag      = cross_lig_isN_flag,

                cross_pro_isring_flag   = cross_pro_isring_flag,
                cross_pro_isO_flag      = cross_pro_isO_flag,
                cross_pro_isN_flag      = cross_pro_isN_flag,

                cross_ligand    = cross_ligand,
                cross_protein   = cross_protein,
                cross_distance  = cross_distance,

                cross_bond_index = cross_bond_index, 
                cross_bond_type = cross_bond_type, 
                cross_bond_index_reverse = cross_bond_index_reverse, 
                cross_bond_type_reverse = cross_bond_type_reverse,

                protein_coords_predict = protein_coords_predict,

                complex_mol = complex_mol,

                protein_element_batch = protein_element_batch,
                protein_link_t_batch = protein_link_t_batch,
                protein_link_t_reverse_batch = protein_link_t_reverse_batch,
                
                ligand_element_batch = ligand_element_batch,

                rd_pos = rd_pos,


                )

            '''
            outputs = {
                'pred_ligand_pos': final_ligand_pos,
                'pred_ligand_v': final_ligand_v,
                'final_pos': final_pos,
                'final_h': final_h, #存放的是配体和蛋白合并在一起的节点嵌入
                'final_ligand_h': final_ligand_h,
                'atom_affinity': atom_affinity, #未分组求均值之前的亲和度，即原子级别的亲和度，每一个原子对应的亲和度
                'final_exp_pred': final_exp_pred, #分组求和的亲和度，用这个，也是个序列
                'batch_all': batch_all, #是蛋白和配体合并一起后的批量，由mask_ligand来标识
                'mask_ligand': mask_ligand,
                'ligand_v': ligand_v,
                'ligand_pos': init_ligand_pos,
                }
            '''

        
            batch_all, mask_ligand           = outputs['batch_all'], outputs['mask_ligand']
            atom_affinity, pred_affinity     = outputs['atom_affinity'], outputs['final_exp_pred']
            final_ligand_pos, final_ligand_h = outputs['pred_ligand_pos'], outputs['final_ligand_h']
            final_h     = outputs['final_h']
            ligand_v    = outputs['ligand_v']
            ligand_pos  = outputs['ligand_pos']
            final_ligand_v = outputs['pred_ligand_v']

            # pred_affinity = scatter_mean(self.expert_pred(final_h).squeeze(-1), batch_all)
            pred_affinity_log = pred_affinity.log()
            
            #这里的梯度，目前是加在原子类型和坐标的均值上的，对于consitency用不上
            #type_grad = torch.autograd.grad(pred_affinity, ligand_v,grad_outputs=torch.ones_like(pred_affinity),retain_graph=True)[0]
            type_grad = 0.0
            #pos_grad = torch.autograd.grad(pred_affinity_log, ligand_pos,grad_outputs=torch.ones_like(pred_affinity),retain_graph=True)[0]
            pos_grad = 0.0

        

        preds = {
            'pred_ligand_pos': final_ligand_pos,
            'pred_ligand_v': final_ligand_v,
            'atom_affinity': atom_affinity,
            'final_h': final_h,
            'final_ligand_h': final_ligand_h,
            'final_exp_pred': pred_affinity,
            'batch_all': batch_all,
            'mask_ligand': mask_ligand,
            'final_pos': outputs['final_pos']
        }
        return preds, type_grad, pos_grad
    

    def interpolate(
        self,
        model: nn.Module,
        a: Tensor,
        b: Tensor,
        ab_ratio: float,
        sigmas: Iterable[Union[Tensor, float]],
        clip_denoised: bool = False,
        verbose: bool = False,
        **kwargs: Any,
    ) -> Tensor:
        """Runs the interpolation  loop.

        Parameters
        ----------
        model : nn.Module
            Model to sample from.
        a : Tensor
            First reference sample.
        b : Tensor
            Second refernce sample.
        ab_ratio : float
            Ratio of the first reference sample to the second reference sample.
        clip_denoised : bool, default=False
            Whether to clip denoised values to [-1, 1] range.
        verbose : bool, default=False
            Whether to display the progress bar.
        **kwargs : Any
            Additional keyword arguments to be passed to the model.

        Returns
        -------
        Tensor
            Intepolated sample.
        """
        # Obtain latent samples from the initial samples
        a = a + sigmas[0] * torch.randn_like(a)
        b = b + sigmas[0] * torch.randn_like(b)

        # Perform spherical linear interpolation of the latents
        omega = torch.arccos(torch.sum((a / a.norm(p=2)) * (b / b.norm(p=2))))
        a = torch.sin(ab_ratio * omega) / torch.sin(omega) * a
        b = torch.sin((1 - ab_ratio) * omega) / torch.sin(omega) * b
        ab = a + b

        # Denoise the interpolated latents
        return self(
            model,
            ab,
            sigmas,
            start_from_y=True,
            add_initial_noise=False,
            clip_denoised=clip_denoised,
            verbose=verbose,
            **kwargs,
        )

    def __mask_transform(
        self,
        x: Tensor,
        y: Tensor,
        mask: Tensor,
        transform_fn: Callable[[Tensor], Tensor] = lambda x: x,
        inverse_transform_fn: Callable[[Tensor], Tensor] = lambda x: x,
    ) -> Tensor:
        return inverse_transform_fn(transform_fn(y) * (1.0 - mask) + x * mask)
