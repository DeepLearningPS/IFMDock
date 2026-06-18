# RTMScore

RTMScore is a a novel scoring function based on residue-atom distance likelihood potential and graph transformer, for the prediction of protein-ligand interactions. 
<div align=center>
<img src="https://github.com/sc8668/RTMScore/blob/main/121.jpg" width="600px" height="300px">
</div> 

The proteins and ligands were first characterized as 3D residue graphs and 2D molecular graphs, respectively, followed by two groups of independent graph transformer layers to learn the node representations of proteins and ligands. Then all node features were concatenated in a pairwise manner, and input into an MDN to calculate the parameters needed for a mixture density model. Through this model, the probability distribution of the minimum distance between each residue and each ligand atom could be obtained, and aggregated into a statistical potential by summing all independent negative log-likelihood values.

### Requirements
dgl-cuda11.1==0.7.0   
mdanalysis==2.0.0    
pandas==1.0.3   
prody==2.1.0   
python==3.8.11   
pytorch==1.9.0   
rdkit==2021.03.5   
openbabel==3.1.0    
scikit-learn==0.24.2    
scipy==1.6.2   
seaborn==0.11.2   
numpy==1.20.3    
pandas==1.3.2   
matplotlib==3.4.3   
joblib==1.0.1   

```
conda create --prefix xxx --file ./requirements_conda.txt      
pip install -r ./requirements_pip.txt
```
### Datasets
[PDBbind](http://www.pdbbind.org.cn)    
[CASF-2016](http://www.pdbbind.org.cn)    
[PDBbind-CrossDocked-Core](https://zenodo.org/record/5525936)      
[DEKOIS2.0](https://zenodo.org/record/6623202)       
[DUD-E](https://zenodo.org/record/6623202)

### Examples for using the trained model for prediction
```
cd example
```
___# input is protein (need to extract the pocket first)___
```
python rtmscore.py -p ./1qkt_p.pdb -l ./1qkt_decoys.sdf -rl ./1qkt_l.sdf -gen_pocket -c 10.0 -m ../trained_models/rtmscore_model1.pth
```
___# input is pocket___
```
python rtmscore.py -p ./1qkt_p_pocket_10.0.pdb -l ./1qkt_decoys.sdf -m ../trained_models/rtmscore_model1.pth
```
___# calculate the atom contributions of the score___
```
python rtmscore.py -p ./1qkt_p_pocket_10.0.pdb -l ./1qkt_decoys.sdf -ac -m ../trained_models/rtmscore_model1.pth
```
___# calculate the residue contributions of the score___
```
python rtmscore.py -p ./1qkt_p_pocket_10.0.pdb -l ./1qkt_decoys.sdf -rc -m ../trained_models/rtmscore_model1.pth
```

cd /data/fan_zg/MDocking/RTMScore-main
conda activate torch2.1.0
python batch_deal.py




# 代码执行。

## GPU版本有问题

#!/bin/bash

# input is protein (needs to be converted to pocket)
CUDA_VISIBLE_DEVICES=7 python rtmscore.py --refine_flag norefine --data_dir /data/fan_zg/MDocking/VSDS_ECDock_Gen --data_name /data/fan_zg/MDocking/data_name.txt -p ./1qkt_p.pdb -l ./1qkt_decoys.sdf -rl ./1qkt_l.sdf -gen_pocket -c 10.0 -m trained_models/rtmscore_model1.pth




# 原子打分
CUDA_VISIBLE_DEVICES=7 python rtmscore.py --refine_flag norefine --data_dir /data/fan_zg/MDocking/VSDS_ECDock_Gen --data_name /data/fan_zg/MDocking/data_name.txt -gen_pocket -c 10.0 -ac -m trained_models/rtmscore_model1.pth 


# 残基打分
CUDA_VISIBLE_DEVICES=7 python rtmscore.py --refine_flag norefine --data_dir /data/fan_zg/MDocking/VSDS_ECDock_Gen --data_name /data/fan_zg/MDocking/data_name.txt -gen_pocket -c 10.0 -rc -m trained_models/rtmscore_model1.pth

# 两种都不打分
CUDA_VISIBLE_DEVICES=7 python rtmscore.py --refine_flag norefine --data_dir /data/fan_zg/MDocking/VSDS_ECDock_Gen --data_name /data/fan_zg/MDocking/data_name.txt -gen_pocket -c 10.0 -m trained_models/rtmscore_model1.pth



# posebusters v1
CUDA_VISIBLE_DEVICES=7 python rtmscore_posebusters.py --refine_flag refine -gen_pocket -c 10.0 -m trained_models/rtmscore_model1.pth

CUDA_VISIBLE_DEVICES=7 python rtmscore_posebusters.py --refine_flag norefine -gen_pocket -c 10.0 -m trained_models/rtmscore_model1.pth


# input is pocket
python rtmscore.py -p ./1qkt_p_pocket_10.0.pdb -l ./1qkt_decoys.sdf -m ../trained_models/rtmscore_model1.pth


# calculate the atom contributions of the score
python rtmscore.py -p ./1qkt_p_pocket_10.0.pdb -l ./1qkt_decoys.sdf -ac -m ../trained_models/rtmscore_model1.pth


# calculate the residue contributions of the score
python rtmscore.py -p ./1qkt_p_pocket_10.0.pdb -l ./1qkt_decoys.sdf -rc -m ../trained_models/rtmscore_model1.pth







fuser -v /dev/nvidia1 | awk '{print $0}' |  xargs kill -9


ps aux | grep batch_deal.py | grep -v grep | awk '{print $2}' | xargs kill -9

ps aux | grep glide_backend | grep -v grep | awk '{print $2}' | xargs kill -9

ps aux | grep python | grep -v grep | awk '{print $2}' | xargs kill -9



 # 杀死进程
ps aux | glide_docking_only_score_for_ecdock | grep -v grep | awk '{print $2}' | xargs kill -9

kill -9 $(ps aux | grep glide_docking_only_score_for_ecdock.py | grep -v grep | awk '{print $2}')  #这种方法更有效

kill -9 $(ps aux | grep add_h.py | grep -v grep | awk '{print $2}')

kill -9 $(ps aux | grep glide_backend  | grep -v grep | awk '{print $2}')