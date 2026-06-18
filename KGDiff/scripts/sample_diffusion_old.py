import argparse
import os
import shutil
import time
import sys
sys.path.append(os.path.abspath('./'))

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from torch_scatter import scatter_sum, scatter_mean
from tqdm.auto import tqdm

import utils.misc as misc
import utils.transforms as trans
from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH
from models.molopt_score_model import ScorePosNet3D, log_sample_categorical
from utils.evaluation import atom_num
import copy
from rdkit import Chem
from rdkit.Chem import AllChem
import copy
from tqdm import tqdm
from rdkit.Geometry.rdGeometry import Point3D


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
    return molobj

def unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms):
    all_step_v = [[] for _ in range(n_data)]
    for v in ligand_v_traj:  # step_i
        v_array = v.cpu().numpy()
        for k in range(n_data):
            all_step_v[k].append(v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
    all_step_v = [np.stack(step_v) for step_v in all_step_v]  # num_samples * [num_steps, num_atoms_i]
    return all_step_v


def sample_diffusion_ligand(model, data, num_samples, batch_size=16, device='cuda',
                            num_steps=None, center_pos_mode='protein',
                            sample_num_atoms='ref',guide_mode='joint',
                            value_model=None,
        type_grad_weight=1.,pos_grad_weight=1.):
    
    all_pred_pos, all_pred_v, all_pred_exp = [], [], []
    all_pred_pos_traj, all_pred_v_traj, all_pred_exp_traj, all_pred_exp_atom_traj = [], [], [], [] #弄清楚，*traj是什么？采样的每一步结果
    all_pred_v0_traj, all_pred_vt_traj = [], []
    time_list = []
    num_batch = int(np.ceil(num_samples / batch_size))
    current_i = 0
    for i in tqdm(range(num_batch)):
        n_data = batch_size if i < num_batch - 1 else num_samples - batch_size * (num_batch - 1) #不够一个批量，则有多少为多少
        batch = Batch.from_data_list([data.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH).to(device)

        t1 = time.time()
        with torch.no_grad():

            #截断原子的索引范围。模式不同，截断的方法也不同。mod == ref,则表示生成的原子数量等于配体
            batch_protein = batch.protein_element_batch
            if sample_num_atoms == 'prior':
                pocket_size = atom_num.get_space_size(batch.protein_pos.detach().cpu().numpy())
                ligand_num_atoms = [atom_num.sample_atom_num(pocket_size).astype(int) for _ in range(n_data)]
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
            elif sample_num_atoms == 'range':
                ligand_num_atoms = list(range(current_i + 1, current_i + n_data + 1))
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
            elif sample_num_atoms == 'ref':
                batch_ligand = batch.ligand_element_batch
                ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand, dim=0).tolist()
            else:
                raise ValueError
            
            #现在面临一个问题，在训练阶段构建KNN图时，我们用真实的坐标来固定KNN图，这点和使用全连接图来固定连接表是一样的，好处在于减少计算成本。然而
            #在采样过程中，由于不知道真实的坐标，这个时候只能采样随机坐标的方式来构建KNN图，但是这样会不会有问题？或者说，如果想在采样过程中还想固定
            #KNN，那么我们能不能在KNN图和全连接图之间建立映射关系呢？或者更直接的就是蛋白使用KNN图，配体使用全连接图，这样就实现了固定邻接表的要求了
            #主要蛋白的坐标是已知，所以不用管它，且在训练过程中，我们不更新蛋白坐标
            org_ligand_pos = copy.deepcopy(batch.ligand_pos)

            # init ligand pos
            '''
            center_pos = scatter_mean(batch.protein_pos, batch_protein, dim=0)
            batch_center_pos = center_pos[batch_ligand]
            init_ligand_pos = batch_center_pos + torch.randn_like(batch_center_pos)
            '''

            #如果以蛋白为中心作为初始值，则对接生成的配体是有问题的。所有会集中在一点。
            center_pos = scatter_mean(batch.protein_pos, batch_protein, dim=0)
            batch_center_pos = center_pos[batch_ligand]
            init_ligand_pos = batch_center_pos + torch.randn_like(batch_center_pos)  #以蛋白质心 + 正太分布
            #init_ligand_pos = batch.ligand_pos + torch.randn_like(batch_center_pos) #原始坐标 + 正太分布，这种方法不可取，因为采样生成的过程中，坐标未知
            #init_ligand_pos = batch.ligand_pos #无噪音试试
            #init_ligand_pos = torch.randn_like(batch_center_pos) #纯正态分布，有些结构生成不了

            #org_ligand_pos = copy.deepcopy(init_ligand_pos)

            #org_ligand_pos = org_ligand_pos + torch.randn_like(batch_center_pos) #加一个正态噪音试试

            

            # init ligand v
            uniform_logits = torch.zeros(len(batch_ligand), model.num_classes).to(device)
            init_ligand_v_prob = log_sample_categorical(uniform_logits)
            init_ligand_v = init_ligand_v_prob.argmax(dim=-1)
            '''
            r = {
            'pos': ligand_pos,
            'v': 0,
            'exp': exp_traj[-1] if len(exp_traj) else [],
            'pos_traj': pos_traj,
            'v_traj': 0,
            'exp_traj': exp_traj,
            'exp_atom_traj': exp_atom_traj,
            'v0_traj': 0,
            'vt_traj': 0,
            }

            '''
            r = model.sample_diffusion(
                guide_mode=guide_mode,
                value_model=value_model,
                type_grad_weight=type_grad_weight,
                pos_grad_weight=pos_grad_weight,
                protein_pos=batch.protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch_protein,

                init_ligand_pos=init_ligand_pos,
                init_ligand_v=init_ligand_v,
                org_ligand_pos = org_ligand_pos,
                batch_ligand=batch_ligand,
                num_steps=num_steps,
                center_pos_mode=center_pos_mode,

                ligand_bond_index = batch.ligand_bond_index, #[2, 582]
                ligand_bond_type  = batch.ligand_bond_type,
                ligand_bond_type_batch = batch.ligand_bond_type_batch,

                #knn_ligand_pos = batch.knn_ligand_pos
            )

            #值得注意的是，在神经网络中，配体和蛋白分组减去了蛋白质心的，最后对接或保存结构的时候是否需要再加上质心了？目前来看是不需要的
            ligand_pos, ligand_v, ligand_pos_traj, ligand_v_traj = r['pos'], r['v'], r['pos_traj'], r['v_traj']
            ligand_v0_traj, ligand_vt_traj = r['v0_traj'], r['vt_traj']
            exp_traj = r['exp_traj'] #是一个2维度张量
            exp_atom_traj = r['exp_atom_traj']

            # unbatch exp
            if guide_mode == 'joint' or guide_mode == 'pdbbind_random' or guide_mode == 'valuenet' or guide_mode == 'wo':
                all_pred_exp += exp_traj[-1]
                all_pred_exp_traj += exp_traj
            
            # unbatch pos，预测出来的分子ligand_pos坐标形状是2维度，这是把当前批量生成的分子给连接在一起，由原来的3维度变成了2维度。而我们则需要在这个图上截取不同数量的原子
            # 截断原子的索引范围。模式不同，截断的方法也不同。mod == ref,则表示生成的原子数量等于配体
            #print('ligand_pos.shape:', ligand_pos.shape) #ligand_pos.shape: torch.Size([111, 3]).关于对接，我们希望生成配体的坐标形状等于真实的配体坐标. 37 * 3 == 111, ok
            ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms) #截断原子的取值范围，加上开始的下标索引0
            #print('ligand_cum_atoms:', ligand_cum_atoms) #ligand_cum_atoms: [  0  37  74 111]

            ligand_pos_array = ligand_pos.cpu().numpy().astype(np.float64)
            all_pred_pos += [ligand_pos_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in
                             range(n_data)]  # num_samples * [num_atoms_i, 3]


            #弄清楚all_pred_pos_traj是啥，采样的每一步都坐标
            all_step_pos = [[] for _ in range(n_data)]
            ##print('ligand_pos_traj.shape:', ligand_pos_traj) #是一个list, 且形状不一样
            for p in ligand_pos_traj:  # step_i
                p_array = p.cpu().numpy().astype(np.float64)
                for k in range(n_data):
                    all_step_pos[k].append(p_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
            all_step_pos = [np.stack(step_pos) for step_pos in
                            all_step_pos]  # num_samples * [num_steps, num_atoms_i, 3]
            all_pred_pos_traj += [p for p in all_step_pos]

            '''
            # unbatch v
            ligand_v_array = ligand_v.cpu().numpy()
            all_pred_v += [ligand_v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in range(n_data)]

            all_step_v = unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms)
            all_pred_v_traj += [v for v in all_step_v]
            all_step_v0 = unbatch_v_traj(ligand_v0_traj, n_data, ligand_cum_atoms)
            all_pred_v0_traj += [v for v in all_step_v0]
            all_step_vt = unbatch_v_traj(ligand_vt_traj, n_data, ligand_cum_atoms)
            all_pred_vt_traj += [v for v in all_step_vt]
            '''

            all_step_exp_atom = unbatch_v_traj(exp_atom_traj, n_data, ligand_cum_atoms)
            all_pred_exp_atom_traj += [v for v in all_step_exp_atom]
            
            
        t2 = time.time()
        time_list.append(t2 - t1)
        current_i += n_data
        
        
    all_pred_exp = torch.stack(all_pred_exp,dim=0).numpy()
    all_pred_exp_traj = torch.stack(all_pred_exp_traj,dim=0).numpy()
        
    return all_pred_pos, all_pred_v, all_pred_exp, all_pred_pos_traj, all_pred_v_traj, all_pred_exp_traj, all_pred_v0_traj, all_pred_vt_traj, all_pred_exp_atom_traj, time_list



def generate_3d_conformer_from_smiles(smiles):
    # 从 SMILES 字符串创建分子对象
    ##print(smiles)
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    ##print(mol)
    flag = False

    # 生成三维构象
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    try:
        AllChem.UFFOptimizeMolecule(mol)
    except Exception as e:
        ##print(e)
        flag = True

    Chem.SanitizeMol(mol) #标准化
    mol = Chem.RemoveHs(mol) #去除氢原子
    
    return mol


def save_sdf(mol, output_sdf):
    # 创建 SDF 文件写入对象
    writer = Chem.SDWriter(output_sdf)
    
    # 将分子写入 SDF 文件
    writer.write(mol)
    
    # 关闭 SDF 文件写入对象
    writer.close()

    #print(f"SDF 文件已生成：{output_sdf}")


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
    return molobj

def main():
    '''
        CUDA_VISIBLE_DEVICES=0 python scripts/sample_diffusion.py --config ./configs/sampling.yml -i 0 --guide_mode pdbbind_random \
            --type_grad_weight 100 --pos_grad_weight 25 --result_path ./cd2020_pro_0_res
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/sampling.yml')
    parser.add_argument('-i', '--data_id', type=int, default=81) #数据开始的位置
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--guide_mode', type=str, default='pdbbind_random', choices=['joint', 'pdbbind_random', 'vina', 'valuenet', 'wo'])  
    parser.add_argument('--type_grad_weight', type=float, default=0) #注意下，这里的权重如何使用的
    parser.add_argument('--pos_grad_weight', type=float, default=0)
    parser.add_argument('--result_path', type=str, default='./test_package') #分子生成的路径

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        exit()
    args = parser.parse_args()
        
    result_path = args.result_path
    os.makedirs(result_path, exist_ok=True)
    shutil.copyfile(args.config, os.path.join(result_path, 'sample.yml'))
    logger = misc.get_logger('sampling', log_dir=result_path)

    # Load config
    config = misc.load_config(args.config)
    logger.info(config)
    misc.seed_all(config.sample.seed)

    # Load checkpoint,不同的模式，选择不同的模型
    if args.guide_mode == 'joint': #这是从整理好的药物设计数据来生成分子的
        ckpt = torch.load(config.model['joint_ckpt'], map_location=args.device)
        value_ckpt = None
    elif args.guide_mode == 'pdbbind_random': #从pdbbind直接来生成分子，我们用这个做对接
        ckpt = torch.load(config.model['pdbbind_random'], map_location=args.device)
        value_ckpt = None
    elif args.guide_mode == 'vina':
        ckpt = torch.load(config.model['policy_ckpt'], map_location=args.device)
        value_ckpt = None
    elif args.guide_mode == 'valuenet':
        ckpt = torch.load(config.model['policy_ckpt'], map_location=args.device)
        value_ckpt = torch.load(config.model['value_ckpt'], map_location=args.device)
    elif args.guide_mode == 'wo':
        ckpt = torch.load(config.model['policy_ckpt'], map_location=args.device)
        value_ckpt = None
    else:
        raise NotImplementedError
    
    logger.info(f"Training Config: {ckpt['config']}")
    logger.info(f"args: {args}")
    
    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose([
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),  #弄清楚是否在后续使用了键嵌入
    ])

    # Load dataset
    dataset, subsets = get_dataset(  #数据来自于训练模型中保存的配置文件,因此，采样的数据设置和训练的配置文件有关
        config=ckpt['config'].data,
        transform=transform
    )
    if ckpt['config'].data.name == 'pl':
        #test_set = subsets['test'] #这里我们用训练集来测试
        test_set = subsets['train']
    elif ckpt['config'].data.name == 'pdbbind':
        #test_set = subsets['test']
        test_set = subsets['train']
    else:
        raise ValueError
    logger.info(f'Test: {len(test_set)}')


    #print('ok3')
    tr_success = 0
    for tr in test_set:
        try:
            #if tr.ligand_org_smiles:
            if tr.ligand_smiles:
                tr_success += 1
                ##print('exist')
        except Exception as e:
            #print(e)
            ##print('no exist')
            ##print('error:', e)
    #print('test_num:', len(test_set))
    #print('test_seccess:', tr_success)


    # Load model
    model = ScorePosNet3D(
        ckpt['config'].model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)
    model.load_state_dict(ckpt['model'])
    
    if value_ckpt is not None: #默认是None，仅仅valuenet模型有效
        # value model
        value_model = ScorePosNet3D(
            value_ckpt['config'].model,
            protein_atom_feature_dim=protein_featurizer.feature_dim,
            ligand_atom_feature_dim=ligand_featurizer.feature_dim
        ).to(args.device)
        value_model.load_state_dict(value_ckpt['model'])
    else:
        value_model = None

        
    for i in list(range(91))[3:5]:
        #data = test_set[args.data_id] #只取一个蛋白靶点，如果想要更多，则需要写个循环或者多次生成，每次指定不同的复合物
        data = copy.deepcopy(test_set[i])

        target_dir = os.path.join(result_path, f'result_{i}')
        if os.path.exists(target_dir):
            # If it exists, remove it
            shutil.rmtree(target_dir)

        """
        
        #使用rdkit生成3D构象用于构建KNN图
        # 生成三维构象
        mol = generate_3d_conformer_from_smiles(data.ligand_smiles) #mmft优化失败是无所谓的，只要能生成mol对象即可
        #print('smiles:', data.ligand_smiles)
        #continue

        conformer      = mol.GetConformer()
        knn_ligand_pos = conformer.GetPositions() #专门用于构建KNN图的
        knn_ligand_pos = torch.FloatTensor(knn_ligand_pos)

        centor = knn_ligand_pos.mean(dim = 0) - data.ligand_pos.mean(dim = 0)
        knn_ligand_pos = knn_ligand_pos - centor

        if knn_ligand_pos.shape != data.ligand_pos.shape:
            #print(f'{knn_ligand_pos.shape} != {data.ligand_pos.shape}')
            raise Exception('矩阵形状不一样，错误')


        new_mol = Change_Mol_D3coord(mol, knn_ligand_pos)
        dirs  = os.path.join(result_path, f'result_{i}')
        os.makedirs(dirs, exist_ok=True)
        target_sdf = os.path.join(dirs, f'sample_rdkit_ligand_{i}.sdf')
        #print('rdkit ligand sdf path:', target_sdf)
        save_sdf(new_mol, target_sdf)
        
        #print('打开RDkit文件')
        rd_mol = Chem.SDMolSupplier(target_sdf)
        #print('remol:', rd_mol)
        current_path = os.path.join(os.path.abspath(os.getcwd()), target_sdf)
        #print('current_path:', current_path)
        #continue


        data.knn_ligand_pos = knn_ligand_pos
        """

        #print('origin_ligand.shape:', data.ligand_pos.shape) #origin_ligand.shape: torch.Size([37, 3])
        pred_pos, pred_v, pred_exp, pred_pos_traj, pred_v_traj, pred_exp_traj, pred_v0_traj, pred_vt_traj, pred_exp_atom_traj, time_list = sample_diffusion_ligand(
            model, data, config.sample.num_samples,
            batch_size=args.batch_size, device=args.device,
            num_steps=config.sample.num_steps,
            center_pos_mode=config.sample.center_pos_mode,
            sample_num_atoms=config.sample.sample_num_atoms,
            guide_mode=args.guide_mode,
            value_model=value_model,
            type_grad_weight=args.type_grad_weight,
            pos_grad_weight=args.pos_grad_weight
        )
        result = {
            'data': data,
            'pred_ligand_pos': pred_pos,
            'pred_ligand_v': pred_v,
            'pred_exp': pred_exp,
            'pred_ligand_pos_traj': pred_pos_traj,
            'pred_ligand_v_traj': pred_v_traj,
            'pred_exp_traj': pred_exp_traj,
            'pred_exp_atom_traj': pred_exp_atom_traj,
            'time': time_list
        }
        logger.info('Sample done!')

        #print('save_gen_ligand')
        #保存配体和蛋白sdf和pdb
        #protein_filename='BSD_ASPTE_1_130_0/2z3h_A_rec_1wn6_bst_lig_tt_docked_3_pocket10.pdb',
        #ligand_filename='BSD_ASPTE_1_130_0/2z3h_A_rec_1wn6_bst_lig_tt_docked_3.sdf',
        
        
        protein_filename=data.protein_filename
        ligand_filename =data.ligand_filename
        #print('ligand_filename:', ligand_filename)

        s_dir = os.path.dirname(ckpt['config'].data.path)
        if 'v2020-other-PL' in ligand_filename: ##使用pdb2020训练时采用的测试方法
            #print('use v2020-other-PL test')
            dir_name = '/'.join(ligand_filename.split('/')[:-1])
            source_dir = os.path.join(s_dir, 'pdbbind2020_r10', dir_name)
        elif 'refined-set' in ligand_filename: ##使用pdb2020训练时采用的测试方法
            #print('use refined-set test')
            dir_name = '/'.join(ligand_filename.split('/')[:-1])
            source_dir = os.path.join(s_dir, 'pdbbind2020_r10', dir_name)
        else:
            #print('use test_set test')
            dir_name = ligand_filename.split('/')[0]
            source_dir = os.path.join(s_dir, 'test_set', dir_name)

        

        shutil.copytree(source_dir, target_dir, dirs_exist_ok=True) #复制的时候，保证目标目录为空,否则需要加上dirs_exist_ok=True参数

        os.makedirs(target_dir, exist_ok=True)
        torch.save(result, os.path.join(target_dir, f'result_{i}.pt'))

        #save_sdf(new_mol, target_sdf)

        pred_ligand_pos = copy.deepcopy(pred_pos)


        if 'v2020-other-PL' in ligand_filename:
            origin_ligand_file = f'{s_dir}/pdbbind2020_r10/{ligand_filename}'
        elif 'refined-set' in ligand_filename:
            origin_ligand_file = f'{s_dir}/pdbbind2020_r10/{ligand_filename}'
        else:
            origin_ligand_file = f'{s_dir}/test_set/{ligand_filename}'

        dt_mol_list = Chem.SDMolSupplier(origin_ligand_file)
        origin_mol = dt_mol_list[0]


        try:
            new_ligand_file = os.path.join(target_dir, f'origin_ligand_{i}.sdf')
            supp=Chem.SDWriter(new_ligand_file)
            new_mol = Change_Mol_D3coord(origin_mol, data.ligand_pos)
            mol2 = Chem.RemoveHs(new_mol)
            supp.write(mol2)
            supp.close()    #需要手动关闭
        except Exception as e:
            continue


        new_ligand_file = os.path.join(target_dir, f'gen_ligand_{i}.sdf')
        conformer = origin_mol.GetConformer() #conformer.GetAtomPosition(atom_idx)
        origin_pos  = origin_mol.GetConformer().GetPositions() #GetPositions(),'s'表示获取所有分子
        origin_pos2 = data.ligand_pos #这个数据是保真的


        ##print('origin_pos[:2]:', origin_pos[:2])
        ##print('origin_pos2[:2]:', origin_pos2[:2])
        supp=Chem.SDWriter(new_ligand_file)
        for pos_i in pred_ligand_pos: #[sample_n, atom_n, 3]
            if pos_i.shape != origin_pos.shape or origin_pos.shape != origin_pos2.shape:
                raise Exception('pos_i.shape != origin_pos.shape')
            new_mol = Change_Mol_D3coord(origin_mol, pos_i)
            mol2 = Chem.RemoveHs(new_mol)
            try:
                supp.write(mol2)
            except Exception:
                continue
        supp.close()    #需要手动关闭


        # 写入 XYZ 文件
        '''
        3
        Water molecule
        O 0.0 0.0 0.0
        H 0.757 0.586 0.0
        H -0.757 0.586 0.0

        '''

        symbols = [atom.GetSymbol() for atom in origin_mol.GetAtoms()]
        filename = os.path.join(target_dir, f'gen_ligand_{i}.xyz')
        num_atoms = pred_ligand_pos[0].shape[0]
        with open(filename, 'w') as xyz_file:
            

            for pos_i in pred_ligand_pos: #[sample_n, atom_n, 3]
                #pred_atom_type = transforms.get_atomic_number_from_index(atom_types, mode=args.atom_enc_mode)
                xyz_file.write(f"{num_atoms}\n")
                xyz_file.write("\n")
                for pos, id_atom in zip(pos_i, symbols):
                    xyz_file.write(f"{id_atom} {round(pos[0], 4)} {round(pos[1], 4)} {round(pos[2], 4)}\n")
                
                #xyz_file.write('\n')
                #xyz_file.write(f"{num_atoms}\n")
                #xyz_file.write("Generated by RDKit\n")







if __name__ == '__main__':
    main()