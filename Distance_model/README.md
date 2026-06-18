Uni-Mol Docking V2
===================================================================
[![arXiv](https://img.shields.io/badge/arXiv-2405.11769-00ff00.svg)](https://arxiv.org/abs/2405.11769) ![Static Badge](https://img.shields.io/badge/Bohrium_Apps-Uni--Mol_Docking_V2-blue?link=https%3A%2F%2Fbohrium.dp.tech%2Fapps%2Funimoldockingv2)

<p align="center"><img src="figure/bohrium_app.png" width=60%></p>
<p align="center"><b>Uni-Mol Docking V2 Bohrium App Interface</b></p>

We update Uni-Mol Docking to Uni-Mol Docking V2, which demonstrates a remarkable improvement in performance, accurately predicting the binding poses of 77+% of ligands in the PoseBusters benchmark with an RMSD value of less than 2.0 Å, and 75+\% passing all quality checks. This represents a significant increase from the 62% achieved by the previous Uni-Mol Docking model. Notably, our Uni-Mol Docking approach generates chemically accurate predictions, circumventing issues such as chirality inversions and steric
clashes that have plagued previous ML models.

Service of Uni-Mol Docking V2 is avaiable at https://bohrium.dp.tech/apps/unimoldockingv2

Dependencies
------------
 - [Uni-Core](https://github.com/dptech-corp/Uni-Core), check its [Installation Documentation](https://github.com/dptech-corp/Uni-Core#installation).
 - rdkit==2022.9.3, install via `pip install rdkit-pypi==2022.9.3 -i https://pypi.tuna.tsinghua.edu.cn/simple/ --trusted-host pypi.tuna.tsinghua.edu.cn`
  - biopandas==0.4.1, install via `pip install biopandas`

Data
----------------------------------
| Data                     | File Size  | Update Date | Download Link                                                                                                             | 
|--------------------------|------------| ----------- |---------------------------------------------------------------------------------------------------------------------------|
| Raw training data       | 4.95GB   | May 14 2024 |https://zenodo.org/records/11191555     |
| Posebusters and Astex   | 8.2MB   | Nov 16 2023 |https://github.com/dptech-corp/Uni-Mol/files/13352676/eval_sets.zip     |


Note that we use the `Posebusters V1` (428 datapoints, released in August 2023). For the latest version, please refer to [Posebusters repo](https://github.com/maabuu/posebusters).


Model weights
----------------------------------

| Model                     | File Size  |Update Date | Download Link                                                | 
|--------------------------|------------| ------------|--------------------------------------------------------------|
| unimol docking v2       | 464MB   | May 17 2024 |https://www.dropbox.com/scl/fi/sfhrtx1tjprce18wbvmdr/unimol_docking_v2_240517.pt?rlkey=5zg7bh150kcinalrqdhzmyyoo&st=n6j0nt6c&dl=0                |


Results
----------------------------------
|< 2.0 Å RMSD(% )      | PoseBusters (N=428) | Astex (N=85) | 
|--------|----|----|   
| DeepDock | 17.8 | 34.12 |
| DiffDock        |  37.9 |71.76  |
| UMol |   45| - | 
| Vina      |  52.3  | 57.65 | 
| Uni-Mol Docking     |  58.9 | 82.35 | 
| AlphaFold latest     |  73.6 | - |
| **Uni-Mol Docking V2**   |  **77.6** | **95.29**|

To reproduce the Posebusters results, we provide a notebook `interface/posebuster_demo` that includes the pipeline from data processing, model inference to metric calculation.

Training
----------------------------------

In the training script, `data_path`, `save_dir`, `finetune_mol_model`, and `finetune_pocket_model` need to be specified. 

The pretrained molecular and pocket model weights can be obtained from [Uni-Mol repo]((https://github.com/maabuu/posebusters)). We use the no_h version weights for molecule.

```
bash train.sh
```

Inference
----------------------------------

We add an interface for model inference in `interface/demo.py`.

About inputs and outpus:

- `--input-protein`: PDB file, abusolute path or raletive path, in batch_one2one mode, list of paths

- `--input-ligand`: SDF file, abusolute path or raletive path; in batch mode, list of paths

- `--input-docking-grid`: JSON file, include center coordinate and box size, abusolute path or raletive path; in batch mode, list of paths

- `--output-ligand-name`: str, the output SDF file name; in batch mode, list of names

- `--output-ligand-dir`: str, abusolute path or raletive path

In batch mode, you can save `input_protein`, `input_ligand`, `input_docking_grid`, and `output_ligand_name` to a CSV file and use `--input-batch-file` to input it.

Other parameters used:

-  `--steric-clash-fix`: The predicted SDF file will be corrected for chemical detail and clash relaxation.

- `--mode`: optional values are `single`, `batch_one2one` and `batch_one2many`. 
  - `single` represents one protein and one ligand as input. 
  - `batch_one2one` represents a batch of proteins and a batch of ligands, where the relationship is one-to-one. 
  - `batch_one2many` represents one protein and a batch of ligands, where the relationship is one-to-many.


## 环境要求：记得安装gpu版本的unicore
下载
git clone https://github.com/dptech-corp/Uni-Core.git
## 要准备配体的质心和以及box框大小

# 6erv, 6d3x 生成非常缓慢

## 对接box size大小很影响模型效果，建议缩小到20,20,20. 默认30,30,30太大，不成结构。对接盒子越大，搜索空间越大，速度越慢，有些模型未必能找到最优解

## glide对接，我们使用的是默认30,30,30，这一点在文章中说清楚

## unimol在处理数据集时，存在卡死的情况，可能和一次读取的数据量有关，要适当减少进程数量, 即修改nthreads参数


Demo:

```
cd interface
bash demo.sh  # demo_batch_one2one.sh for batch mode
```
Or refer to this notebook `interface/posebuster_demo`.


Citation
--------

Please kindly cite this paper if you use the data/code/model.
```
@article{alcaide2024uni,
  title={Uni-Mol Docking V2: Towards Realistic and Accurate Binding Pose Prediction},
  author={Alcaide, Eric and Gao, Zhifeng and Ke, Guolin and Li, Yaqi and Zhang, Linfeng and Zheng, Hang and Zhou, Gengmo},
  journal={arXiv pre#print arXiv:2405.11769},
  year={2024}
}
```

License
-------

This project is licensed under the terms of the MIT license. See [LICENSE](https://github.com/dptech-corp/Uni-Mol/blob/main/LICENSE) for additional details.



# 实验执行记录

## 训练
### 训练的开始文件：/data/fan_zg/anaconda3/envs/cu128/lib/python3.10/site-packages/unicore_cli/train.py

# 先加载验证集数据，之后是训练集，没有测试构象数量的方法，批量大小影响构象数量，仔细研究一下
### 加载数据集文件：/data/fan_zg/MDocking/unimol_docking_v2/unimol/tasks/docking_pose_v2.py

### 数据集定义在：/data/fan_zg/MDocking/Docking_baseline/unimol_docking_v2/unimol/tasks/docking_pose_v2.py

### 预处理的数据已经解决了，目前导致数据少的原因是/data/fan_zg/MDocking/Docking_baseline/unimol_docking_v2/unimol/tasks/docking_pose_v2.py中的去氢， 建议一开始
### 传递的pdb文件时，就是无氢的

### 模型定义：/data/fan_zg/MDocking/Docking_baseline/unimol_docking_v2/unimol/models/docking_pose_v2.py

### 参数设置/data/fan_zg/MDocking/Docking_baseline/unimol_docking_v2/unicore/options.py

valid_step 定义在：/data/fan_zg/MDocking/Docking_baseline/unimol_docking_v2/unicore/tasks/unicore_task.py

build_loss实际上是一个模型，里面含有损失，在/data/fan_zg/MDocking/Docking_baseline/unimol_docking_v2/unimol/losses/docking_pose_v2.py

# 对于处理；加载验证集时没问题，但加载训练集有问题，如果数据是清洗过的，会出问题

先加载测试集，后训练集；之后在训练时，先训练集，后测试集


conda activate torch2.1.0
python demo_batch_one2one_vsds.py

# 如果是训练，则必须将将构象数量设置为1？

# 任务设置：实际上就是根据不同的任务名，来声明不同的模型类对象。这里的task可以看成就是模型所在的类对象
task = tasks.setup_task(args) 

# 有些函数与类搜索不到是因为被封装到了系统包里， /data/fan_zg/anaconda3/envs/torch2.1.0/lib/python3.10/site-packages/unicore

模型的任务注册：unimol/tasks/docking_pose_v2.py
模型的定义：unimol/models/docking_pose_v2.py
损失的定义：unimol/losses/docking_pose_v2.py