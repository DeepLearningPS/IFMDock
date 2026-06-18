
import math
from typing import Any, Callable, Iterable, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from tqdm.auto import tqdm


from typing import Iterator

from torch import Tensor, nn

import random
import numpy as np
import torch.nn.functional as F
import copy
import random

from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Rotation as R
from rdkit.Chem import AllChem, rdMolTransforms
from rdkit import Geometry
from rdkit.Chem import AllChem, GetPeriodicTable, RemoveHs

from rdkit.Geometry.rdGeometry import Point3D
import time


np.random.seed(2023)
torch.manual_seed(2023)
random.seed(2023)
torch.cuda.manual_seed_all(2023)



def force_gradient(ligand_pos, protein_pos, cross_distance, iterations=1, early_stoping=1):
        data_dict = {
                    'ligand_pos': ligand_pos,
                    'protein_pos': protein_pos,
                    'cross_distance': cross_distance
                    }
        
        torch.save(data_dict, 'gradient_data_dict.pt')
        
        #print('ligand_pos:', ligand_pos.requires_grad) # True
        #print('protein_pos:', protein_pos.requires_grad)# False
        #print('cross_distance:', cross_distance.requires_grad)# False
    
        #加谐振子约束优化
        ligand_pos.requires_grad = True
        #coor_pred_detach = coor_pred.detach()
        ligand_pos_Parm = torch.nn.Parameter(ligand_pos.detach().clone()).to(ligand_pos.device)
        #protein_pos.requires_grad = True
        pred_distance = calculate_distance_matrix(ligand_pos_Parm, protein_pos) #到这里没梯度了，导致后面ligand_pos没参与
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
        pred_distance = calculate_distance_matrix(ligand_pos_Parm, protein_pos)
        distance = F.mse_loss(pred_distance, cross_distance) #值没有变化，也就是坐标没动
        print('after optim ditance:', distance)
        print('-----------------------------------------------------')

        return ligand_pos_Parm.detach()



def calculate_distance_matrix(A, B):
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

if __name__ == '__main__':
    pass
