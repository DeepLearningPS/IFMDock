import numpy as np
import torch as th
from joblib import Parallel, delayed
import pandas as pd
import argparse
import os, sys
import MDAnalysis as mda
#sys.path.append("/home/shenchao/resdocktest2/rtmscore2")
sys.path.append(os.path.abspath(__file__).replace("rtmscore.py",".."))
from torch.utils.data import DataLoader
from RTMScore.data.data import VSDataset
from RTMScore.model.utils import collate, run_an_eval_epoch
from RTMScore.model.model2 import RTMScore, DGLGraphTransformer #LigandNet, TargetNet
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

#you need to set the babel libdir first if you need to generate the pocket
#os.environ["BABEL_LIBDIR"] = '/mnt_191/fanzhiguang/47/47_anaconda3/new_torch2.1.0/lib/openbabel/3.1.0' # "/data/fan_zg/anaconda3/envs/torch2.1.0/lib/openbabel/3.1.0"


import copy
from rdkit import Chem
from rdkit.Chem import AllChem
import copy
import subprocess
import time
#import multiprocessin
from tqdm import tqdm 
import warnings
warnings.simplefilter("ignore")  # 更彻底的忽略方式

def scoring(prot, lig, modpath,
            cut=10.0,
            gen_pocket=False,
            reflig=None,
            atom_contribution=False,
            res_contribution=False,
            explicit_H=False,
            use_chirality=True,
            parallel=False,
            **kwargs
            ):
    """
    prot: The input protein file ('.pdb')
    lig: The input ligand file ('.sdf|.mol2', multiple ligands are supported)
    modpath: The path to store the pre-trained model
    gen_pocket: whether to generate the pocket from the protein file.
    reflig: The reference ligand to determine the pocket.
    cut: The distance within the reference ligand to determine the pocket.
    atom_contribution: whether the decompose the score at atom level.
    res_contribution: whether the decompose the score at residue level.
    explicit_H: whether to use explicit hydrogen atoms to represent the molecules.
    use_chirality: whether to adopt the information of chirality to represent the molecules.
    parallel: whether to generate the graphs in parallel. (This argument is suitable for the situations when there are lots of ligands/poses)
    kwargs: other arguments related with model
    """
    #try:
    
    #改造这里，我们设置一个模式，训练还是测试，如果训练，则蛋白与配体数量一样，如果测试，则蛋白只有一个。我们需要做的就是传递一组rdkit mol对象
    data = VSDataset(ligs=lig,
                    prot=prot, #这里要改造生成只支持多蛋白的
                    cutoff=cut,
                    gen_pocket=gen_pocket,
                    reflig=reflig,
                    explicit_H=explicit_H,
                    use_chirality=use_chirality,
                    parallel=parallel)


    test_loader = DataLoader(dataset=data,
                            batch_size=kwargs["batch_size"],
                            shuffle=False,
                            num_workers=kwargs["num_workers"],
                            collate_fn=collate)

    ligmodel = DGLGraphTransformer(in_channels=kwargs["num_node_featsl"],
                                    edge_features=kwargs["num_edge_featsl"],
                                    num_hidden_channels=kwargs["hidden_dim0"],
                                    activ_fn=th.nn.SiLU(),
                                    transformer_residual=True,
                                    num_attention_heads=4,
                                    norm_to_apply='batch',
                                    dropout_rate=0.15,
                                    num_layers=6
                                    )

    protmodel = DGLGraphTransformer(in_channels=kwargs["num_node_featsp"],
                                    edge_features=kwargs["num_edge_featsp"],
                                    num_hidden_channels=kwargs["hidden_dim0"],
                                    activ_fn=th.nn.SiLU(),
                                    transformer_residual=True,
                                    num_attention_heads=4,
                                    norm_to_apply='batch',
                                    dropout_rate=0.15,
                                    num_layers=6
                                    )

    model = RTMScore(ligmodel, protmodel,
                    in_channels=kwargs["hidden_dim0"],
                    hidden_dim=kwargs["hidden_dim"],
                    n_gaussians=kwargs["n_gaussians"],
                    dropout_rate=kwargs["dropout_rate"],
                    dist_threhold=kwargs["dist_threhold"]).to(kwargs['device'])

    checkpoint = th.load(modpath, map_location=th.device(kwargs['device']))
    #checkpoint = th.load(modpath)
    model.load_state_dict(checkpoint['model_state_dict'])
    #model = model.cuda()
    model = model.to(kwargs['device'])
    if atom_contribution:
        preds, at_contrs, _ = run_an_eval_epoch(model,
                                                test_loader,
                                                pred=True,
                                                atom_contribution=True,
                                                res_contribution=False,
                                                dist_threhold=kwargs['dist_threhold'], device=kwargs['device'])

        atids = ["%s%s"%(a.GetSymbol(),a.GetIdx()) for a in data.ligs[0].GetAtoms()]
        return data.ids, preds, atids, at_contrs

    elif res_contribution:
        preds, _, res_contrs = run_an_eval_epoch(model,
                                                test_loader,
                                                pred=True,
                                                atom_contribution=False,
                                                res_contribution=True,
                                                dist_threhold=kwargs['dist_threhold'], device=kwargs['device'])
        u = mda.Universe(data.prot)
        resids = ["%s_%s%s"%(x[0],y,z) for x,y,z in zip(u.residues.chainIDs, u.residues.resnames, u.residues.resids)]
        return data.ids, preds, resids, res_contrs
    else:
        preds = run_an_eval_epoch(model, test_loader, pred=True, dist_threhold=kwargs['dist_threhold'], device=kwargs['device'])
        return data.ids, preds


def one_data(p_file, l_file, ref_lig_file, out_file, origin_gen_l_file, rtm_srot_l_file, inargs):
    args={}
    args["batch_size"] = 128
    args["dist_threhold"] = 5
    #args['device'] = 'cuda' if th.cuda.is_available() else 'cpu'
    #print("args['device']:", args['device']) #cdua
    #args['device'] = 'cuda:7'
    args['device'] = 'cpu' #不需要gpu，速度依旧很快
    args["num_workers"] = 20
    args["num_node_featsp"] = 41
    args["num_node_featsl"] = 41
    args["num_edge_featsp"] = 5
    args["num_edge_featsl"] = 10
    args["hidden_dim0"] = 128
    args["hidden_dim"] = 128
    args["n_gaussians"] = 10
    args["dropout_rate"] = 0.10
    
    #值得注意的是inargs.atom_contribution，只是用于标识是否返回原子或残基的贡献度，可以不提供，和计算打分无关
    if inargs.atom_contribution:
        #print('atom_contribution')
        ids, scores, atids, at_contrs = scoring(prot=p_file,
                                            lig=l_file,
                                            modpath=inargs.model,
                                            cut=inargs.cutoff,
                                            gen_pocket=inargs.gen_pocket,
                                            reflig=ref_lig_file,
                                            atom_contribution=True, #默认是原子贡献度
                                            explicit_H=False,
                                            use_chirality=True,
                                            parallel=inargs.parallel,
                                            **args
                                            )
        df = pd.DataFrame(at_contrs).T
        df.columns= ids
        df.index = atids
        df = df[df.apply(np.sum,axis=1)!=0].T
        dfx = pd.DataFrame(zip(*(ids, scores)),columns=["id","score"])
        dfx.index = dfx.id
        df = pd.concat([dfx["score"],df],axis=1)
        df.sort_values("score", ascending=False, inplace=True)
        df.to_csv("%s.csv"%out_file)
    elif inargs.res_contribution:
        #print('res_contribution')
        ids, scores, resids, res_contrs = scoring(prot=p_file,
                                            lig=l_file,
                                            modpath=inargs.model,
                                            cut=inargs.cutoff,
                                            gen_pocket=inargs.gen_pocket,
                                            reflig=ref_lig_file,
                                            res_contribution=True,
                                            explicit_H=False,
                                            use_chirality=True,
                                            parallel=inargs.parallel,
                                            **args
                                            )
        df = pd.DataFrame(res_contrs).T
        df.columns= ids
        df.index = resids
        df = df[df.apply(np.sum,axis=1)!=0].T
        dfx = pd.DataFrame(zip(*(ids, scores)),columns=["id","score"])
        dfx.index = dfx.id
        df = pd.concat([dfx["score"],df],axis=1)
        df.sort_values("score", ascending=False, inplace=True)
        df.to_csv("%s.csv"%out_file)
    else:
        #默认是这个
        #print('不使用原子以及残基打分')
        ids, scores = scoring(prot=p_file,
                            lig=l_file,
                            modpath=inargs.model,
                            cut=inargs.cutoff,
                            gen_pocket=inargs.gen_pocket,
                            reflig=ref_lig_file,
                            explicit_H=False,
                            use_chirality=True,
                            parallel=inargs.parallel,
                            **args
                            )
        df = pd.DataFrame(zip(*(ids, scores)),columns=["id","score"])
        df.sort_values("score", ascending=False, inplace=True)
        #print('df:', df)
        df.to_csv("%s2.csv"%out_file, index=False)
    
    origin_gen_mol       = []
    rtm_score_index_list = []
    
    #origin_gen_l_file, rtm_srot_l_file
    origin_gen_mol = Chem.rdmolfiles.SDMolSupplier(origin_gen_l_file)
    
    new_score = []


    
    with open("%s2.csv"%out_file, 'r') as f:
        f.readline()
        for line in f:
            #print('line:', line)
            tg0 = line.strip().split(',')[0].rsplit('-', 1)[1]
            tg1 = line.strip().split(',')[1]
            rtm_score_index_list.append(int(tg0))
            new_score.append(f'{tg0}\t{tg1}')

    #简化格式后再写入
    with open("%s.csv"%out_file, 'w') as f:
        for line in new_score:
            f.write(line + '\n')
    
    

    
    
    #print('rtm_score_index_list:', rtm_score_index_list) #[15, 15, 15, 15, 15]
    
    #exit()
    new_mol_list = []
    for idx in rtm_score_index_list:
        mol = origin_gen_mol[idx]
        new_mol_list.append(mol)
        
    #写入排序后的分子 
    supp=Chem.SDWriter(rtm_srot_l_file)
    for mol in new_mol_list:
        mol2 = Chem.RemoveHs(mol)
        try:
            supp.write(mol2)
        except Exception as e:
            print(e)
            continue
    supp.close()    #需要手动关闭

            
            
def is_file_exist_and_not_empty(filepath):
    # 检查文件是否存在
    if not os.path.exists(filepath):
        return False
    
    # 检查是否是文件（不是目录）
    if not os.path.isfile(filepath):
        return False
    
    # 检查文件大小是否大于0
    if os.path.getsize(filepath) > 0:
        return True
    else:
        return False

    
if __name__ == '__main__':
    pass




