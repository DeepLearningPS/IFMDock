from rdkit import Chem
from rdkit.Chem.rdchem import ChiralType
import json 

class GPARMAS:
    def __init__(self):
        self.atom_types=[1,6,7,8,9,15,16,17,35,53]
        self.bond_types=[Chem.BondType.SINGLE,Chem.BondType.DOUBLE,Chem.BondType.TRIPLE,Chem.BondType.AROMATIC]
        self.if_chiral=False
        self.chiral_types=[ ChiralType.CHI_UNSPECIFIED, ChiralType.CHI_TETRAHEDRAL_CW, ChiralType.CHI_TETRAHEDRAL_CCW, ChiralType.CHI_OTHER,
                            ChiralType.CHI_TETRAHEDRAL, ChiralType.CHI_ALLENE, ChiralType.CHI_SQUAREPLANAR, ChiralType.CHI_TRIGONALBIPYRAMIDAL,ChiralType.CHI_OCTAHEDRAL]
        self.max_atoms=250 #需要修改,250这是pdbbind2020，如果是其它数据集则需要重新另外指定,posebusters:64
        self.max_protein_atoms = 300
        self.batchsize=50
        self.device='cuda'
        self.dim=(16,16)
        self.dim_head=(16,16)
        self.heads=(8,4)
        self.num_linear_att_heads=0
        self.num_degrees=2
        self.depth=6
        self.consistency_training_steps=5 #注意修改
        self.sigma_min=0.002
        self.sigma_max=80.0  #默认是80.0
        self.rho=7.0
        self.sigma_data=0.5
        self.initial_timesteps=2  #可修改
        self.lr_patience=100
        self.lr_cooldown=100
        self.n_workers=20
        self.multi_step = 0
        self.final_timesteps=5  #可修改
        self.recover=False
        

        self.ema_exit = False #设置成False，表示使用CMv2
        self.with_MMFF_guide = True
        self.guide_type = 'asynchronous' #synchronous / asynchronous 
        self.opt_types = 'complex'
        self.min_type = 'AdamW' #/SGD/AdamW/LBFGS
        self.cross_loss = 0 #在没有探明谐振子和力场之间的权重前，先使用力场. 目前1最佳,甚至更小
        self.force_loss = 1 #力场权重, 默认1
        self.loop       = 1 #力场优化步数,LBFGS 需要1步
        self.mmf_method_mode = 'karmadock' # 'xu'/karmadock, xu的mmf虽然是批量处理，但速度慢，另外如果一个批量里有一个分子优化失败，则整体失败，不如rdkit的逐个分子优化
        
        self.rdkit_force_mode = 'mmff' #mmff/uff

        self.sample_batch_size = 5 # #扩散的时候，不同的扩散步骤对不同的分子作用不一样，不一样越大越好

        self.force_step = 1 #力场优化参与的扩散的步长

        self.embedding3d = False #是否启动3d嵌入
        self.embedding3d_noise_pos = False #当启动3D嵌入时，是否使用噪音坐标构建配体图

        self.glide_vina = False #是否是基于glide或vina连接表（距离矩阵）的, 在测试或生成glide、vina数据集时，需要启动

        self.cross_distance_num = 'best' #距离矩阵的数量, 取值是best/20/40, Glide and Vina setting is 5, ECDock setting is 20

        self.single_cross_distance_id = None #从0开始，优先级最高，是否仅仅按某一个距离矩阵来生成结构，，默认值是None，表示不开启，如果开启，则指定距离矩阵id，我们可以测试一下前5个距离矩阵

        
        self.data_type = 'VSDS_Glide_refine_fail_dataset' #数据集类型，是CrossDocked2020还是DEKOIS2.0或者VSDS_DTEBV-D(虚拟筛选), VSDS_Glide_refine_fail_dataset 
        #内坐标权重不宜太大
        self.loss_weight={"ic":0.01, "xyz":1.0, "cross_distance":1.0, "ref_cross":1.0, "ref":1.0} #{"ic":0.9,"xyz":0.1}, 目前按{"ic":0.1,"xyz":0.9}，内坐标不够好，可以加大内坐标损失的权重，这是未来要调整的
        self.interaction_stype = 'interaction'  #atom/centor/distance/all/interaction/interaction_all, 模糊相互作用的添加方法，是以口袋质心，还是以每一个配体原子为中心，之后再构建全连接图，默认是否则
        #interaction_all模式：因为unimol可用距离是4.5，很小，所以距离4.5以内的距离矩阵可以完全使用，不再区分O，N，环
        self.interaction_distance = 10  #相互作用距离，这个可以调小，看情况，当self.interaction_stype == 'centor/atom' 时起效果
        self.cross_distance_cutoff = 4.5 #unimol 的cross_distance的截断，目前是4.5比较好, 当self.interaction_stype == 'interaction' 时起效果

        self.min_distance_atom_num = 200 # self.interaction_stype == 'ditance'时起效果
        self.atom2atom_distance = 8  #相互作用距离，默认是8ai，按以参考的配体原子为中心的距离

        self.bond_type_num = 20 #键类型的数量，可以是4,8,20, 改这里的数量后，需要相应地改配体文件中的键嵌入维度对应4/88/340, 但这个维度影响equifrmer

        #self.steps_list=[150]  #指定训练步长, [1,2,5,10,15,25],扩大训练步数
        #self.steps_list=[25]  #指定训练步长, [1,2,5,10,15,25],扩大训练步数
        #self.steps_list=[1,2,5,10,15,25]  #指定训练步长, [1,2,5,10,15,25]
        #self.steps_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 50]
        #self.steps_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25] #在已知模糊相互作用的条件下，试一下动态的相互作用，考虑速度问题，先设置连续的15步+skip25
        #self.steps_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        #self.steps_list=[1,2,3,4,5]
        #有一个问题，指定步长和跳步长训练，应该使用固定间隔，还是前连续后跳, 还是前跳后连续？
        '''
        #sigmas = karras_schedule(num_timesteps, self.sigma_min, self.sigma_max, self.rho, ligand_pos.device)
        sigmas和当前步长有关，其取值从0.002~80.0, 追着步长的增加，sigma的值逐步向[0.002,80.0]中间逼近
        '''
        #self.steps_list=[1, 3, 5, 7, 9, 11, 13, 15]
        #self.steps_list=[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
        #self.steps_list=[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49]
        #self.steps_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
        #self.steps_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]

        #使用连续的步数或者更多指定的步数后，EGNN会出现溢出问题，损失异常
            
def Loaddict2obj(dict,obj):
    objdict=obj.__dict__
    for i in dict.keys():
        if i not in objdict.keys():
            print ("%s not is not a standard setting option!"%i)
        objdict[i]=dict[i]
    obj.__dict__==objdict

def Update_GPARAMS(jsonfile):
    with open(jsonfile,'r') as f:
        jsondict=json.load(f)
        Loaddict2obj(jsondict,GP)
    return 

GP=GPARMAS()
