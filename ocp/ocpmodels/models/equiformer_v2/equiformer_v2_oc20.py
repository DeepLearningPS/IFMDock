import logging
import math
from typing import List, Optional

import torch
import torch.nn as nn

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import conditional_grad
from ocpmodels.models.base import BaseModel
from ocpmodels.models.scn.smearing import GaussianSmearing

try:
    pass
except ImportError:
    pass


from .edge_rot_mat import init_edge_rot_mat
from .gaussian_rbf import GaussianRadialBasisLayer
from .input_block import EdgeDegreeEmbedding
from .layer_norm import (
    EquivariantLayerNormArray,
    EquivariantLayerNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
    get_normalization_layer,
)
from .module_list import ModuleListInfo
from .radial_function import RadialFunction
from .so3 import (
    CoefficientMappingModule,
    SO3_Embedding,
    SO3_Grid,
    SO3_LinearV2,
    SO3_Rotation,
)
from .transformer_block import (
    FeedForwardNetwork,
    SO2EquivariantGraphAttention,
    TransBlockV2,
)

from EcConf.comparm import *


# Statistics of IS2RE 100K
_AVG_NUM_NODES = 77.81317
_AVG_DEGREE = (
    23.395238876342773  # IS2RE: 100k, max_radius = 5, max_neighbors = 100
)


@registry.register_model("equiformer_v2")
class EquiformerV2_OC20(BaseModel):
    """
    Equiformer with graph attention built upon SO(2) convolution and feedforward network built upon S2 activation

    Args:
        use_pbc (bool):         Use periodic boundary conditions #使用周期边界条件
        regress_forces (bool):  Compute forces
        otf_graph (bool):       Compute graph On The Fly (OTF)
        '''
        "Compute graph on the fly" 意味着在运行时动态地生成计算图。在深度学习中，计算图是描述模型中各个层之间数据流和操作的图形结构。
        动态地生成计算图可以提供灵活性和效率，特别是对于具有可变长度输入的模型或需要根据输入数据进行不同操作的模型。这种方法可以在模型
        的每次运行中根据输入数据动态地构建计算图，而不是预先定义固定的计算图
        '''
        max_neighbors (int):    Maximum number of neighbors per atom ##每个原子的最大邻居数量
        max_radius (float):     Maximum distance between nieghboring atoms in Angstroms #两个原子之间的最大距离，用于构建半径图
        max_num_elements (int): Maximum atomic number  #最大原子类型数量，还是最大原子序号？这个显然是用来one-hot的。应该是最大原子的序号。

        num_layers (int):             Number of layers in the GNN #网络层数量

        sphere_channels (int):        Number of spherical channels (one set per resolution) #球形通道的数量（每个分辨率一组），和球谐函数相关
        '''
        在某些上下文中，"Number of spherical channels" 可能指的是在球形坐标系中使用的通道数量。在球形坐标系中，通常使用球谐函数来表示函数。
        这些函数可以被视为在球体表面上的 "通道" 或 "频率"。

        一组球形通道每个分辨率" 意味着对于每个分辨率级别，我们都有一组球形通道。在球形坐标系中，分辨率级别通常与球谐函数的角动量量子数 
        l 相关联。较低的l值表示更低的分辨率, 较高的 l 值表示更高的分辨率。

        球谐函数通常用来解决球对称性势场中的定态薛定谔方程。在原子物理中，原子的电子波函数可以表示为球谐函数的线性组合.
        球谐函数在求解球对称性系统的问题时是非常有用的，特别是在原子物理学和分子物理学中。它们还在地球物理学中用于描述地球的重力场和磁场，
        以及在电磁学中用于描述球对称性电荷分布的电势场
        '''

        attn_hidden_channels (int): Number of hidden channels used during SO(2) graph attention #SO(2)隐藏层的注意力头对应的维度
        num_heads (int):            Number of attention heads #注意力头的数量
        attn_alpha_head (int):      Number of channels for alpha vector in each attention head #每一个注意力头对应的嵌入维度，最终的维度 = 注意力头数量*每一个注意力头对应的维度
        attn_value_head (int):      Number of channels for value vector in each attention head
        ffn_hidden_channels (int):  Number of hidden channels used during feedforward network  #前馈网络的隐藏层维度
        norm_type (str):            Type of normalization layer (['layer_norm', 'layer_norm_sh', 'rms_norm_sh']) #批量归一化的方法, BN不适合输入变长的网络，所以用LN

        lmax_list (int):              List of maximum degree of the spherical harmonics (1 to 10)   #球谐最大度列表，这个影响球面谐波函数的阶数
        mmax_list (int):              List of maximum order of the spherical harmonics (0 to lmax)  #球面谐波的最大阶数列表，不同阶数的球面谐波函数。                                                                                                
        '''
        #球面谐波函数用于表示或模拟其它函数.在球坐标系中，使用球面谐波函数来表示其它函数，它的阶数解决了表示的能力，阶数越大，模拟表示能力越强。这和在笛卡尔坐标系下的泰勒等公式
        去模拟其他函数类似。

        球谐函数的最大阶数，通常表示为l_max，表示用球谐函数展开函数的最高角动量量子数。这个参数决定了球谐函数表示的细节或分辨率水平。

        在各种应用中，例如计算化学、计算机图形学和地球物理学中，选择l_max至关重要，因为它决定了表示的准确性和复杂性。较高的l_max允许更详细地表示函数，但会增加计算成本。

        为什么要引入求坐标系？为了等变。球坐标系本身就是一个球形对称坐标系，而等变操作一般作用于对称系统。所以，为方便实现等变操作，引入了对称系统求坐标，在球坐标系下
        进行等变操作？


        在数学中，SO(2) 和 SO(3) 是特殊正交群（Special Orthogonal Group）的表示，表示在二维和三维欧几里得空间中的旋转变换

        #即整体旋转过程中，保证各个点的距离和角度不变

        SO(2)： 它是二维欧几里得空间中的旋转群。 
        SO(2) 包含了所有的二维旋转变换，它们保持点之间的距离和角度不变。 
        SO(2) 可以用一个单一的实数表示，即角度。例如，
        SO(2) 中的元素可以表示为一个角度 θ，表示绕原点旋转角度为 θ 的旋转变换。

        SO(3)： 它是三维欧几里得空间中的旋转群。 
        SO(3) 包含了所有的三维旋转变换，它们保持点之间的距离和角度不变。 
        SO(3) 可以用一个三维旋转矩阵表示，这个矩阵是一个特殊正交矩阵，它的行和列是单位向量，并且矩阵的行列式为1 (3*3旋转矩阵)

        '''
        grid_resolution (int):        Resolution of SO3_Grid  #SO3_Grid的分辨率，即球谐函数的阶数，默认是None。可以由其它参数提供该值

        num_sphere_samples (int):     Number of samples used to approximate the integration of the sphere in the output blocks
        #用于近似输出块中球体积分的样本数，即为了近似求积分，需要采样多少样本

        edge_channels (int):                Number of channels for the edge invariant features #边不变特征的通道数，即边嵌入维度
        use_atom_edge_embedding (bool):     Whether to use atomic embedding along with relative distance for edge scalar features #默认True
        #是否对边标量特征使用原子嵌入（即边嵌入由其两个原子的嵌入组成）和相对距离（边长度），另外，边特征还可以由边类型组成

        share_atom_edge_embedding (bool):   Whether to share `atom_edge_embedding` across all blocks #是否共享原子边嵌入，默认不共享
        use_m_share_rad (bool):             Whether all m components within a type-L vector of one channel share radial function weights
        #一个通道的L型矢量内的所有m个分量是否共享径向函数权重， 默认是False，其实共享是为了减少参数。理论上，不共享不影响模型的预测能力

        distance_function ("gaussian", "sigmoid", "linearsigmoid", "silu"):  Basis function used for distances #默认gaussian， 用于扩大原子距离，维度未变

        attn_activation (str):      Type of activation function for SO(2) graph attention #so(2)的激活函数'scaled_silu'
        use_s2_act_attn (bool):     Whether to use attention after S2 activation. Otherwise, use the same attention as Equiformer 
        #在so(2)后是否使用注意力，默认False，使用Equiformer注意力

        use_attn_renorm (bool):     Whether to re-normalize attention weights #再次归一化注意力。默认Ture
        ffn_activation (str):       Type of activation function for feedforward network #前馈网络的激活函数'scaled_silu'
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation  #不使用gate激活
        use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs for FFNs. #使用投影到网格并执行FFN的MLP，是否使用MLP，不使用
        use_sep_s2_act (bool):      If `True`, use separable S2 activation when `use_gate_act` is False. #当“use_gate_act”为False时，使用可分离的S2激活。默认True

        alpha_drop (float):         Dropout rate for attention weights #0.1
        drop_path_rate (float):     Drop path rate #0.05
        proj_drop (float):          Dropout rate for outputs of attention and FFN in Transformer blocks #网络层的去除率0.0

        weight_init (str):          ['normal', 'uniform'] initialization of weights of linear layers except those in radial functions #初始化，默认正态分布


        #下面这些参数在新V2中不存在
        enforce_max_neighbors_strictly (bool):      When edges are subselected based on the `max_neighbors` arg, arbitrarily select amongst equidistant / degenerate edges to have exactly the correct number.
        #当根据“max_neighbors”arg对边进行子选择时，在等距/退化边中任意选择，使其具有正确的数量。什么意思呀?
        
        avg_num_nodes (float):      Average number of nodes per graph    #每个图的平均节点数量
        avg_degree (float):         Average degree of nodes in the graph #每个图的平均节点度


        #下面这两项是关于能量的设置，use_energy_lin_ref 和 load_energy_lin_ref 的值要保持一致，要么都是True，要么都是false，在训练和验证阶段是False，在预测阶段是True
        #这两项不用管，这是针对OC22数据集的，看一下怎么用的？？？
        use_energy_lin_ref (bool):  Whether to add the per-atom energy references during prediction.
                                    During training and validation, this should be kept `False` since we use the `lin_ref` parameter in the OC22 dataloader to subtract the per-atom linear references from the energy targets.
                                    During prediction (where we don't have energy targets), this can be set to `True` to add the per-atom linear references to the predicted energies.
        
        是否在预测期间添加每个原子的能量参考。
        在训练和验证过程中，这应该保持为“False”，因为我们使用OC22数据加载器中的“lin_ref”参数从能量目标中减去每个原子的线性参考。
        在预测过程中（我们没有能量目标），可以将其设置为“True”，以将每个原子的线性参考添加到预测的能量中。
        
    
        load_energy_lin_ref (bool): Whether to add nn.Parameters for the per-element energy references.
                                    This additional flag is there to ensure compatibility when strict-loading checkpoints, since the `use_energy_lin_ref` flag can be either True or False even if the model is trained with linear references.
                                    You can't have use_energy_lin_ref = True and load_energy_lin_ref = False, since the model will not have the parameters for the linear references. All other combinations are fine.
    
        是否为每个元素能量参考，添加nn.Parameters。(即对真实的能量值建模)
        此附加标志用于确保在严格加载检查点时的兼容性，因为即使使用线性引用训练模型，“use_energy_lin_ref”标志也可以为True或False。
        不能将use_energy_lin_ref设置为True，将load_energy_lin_ref设置为False，因为模型将没有线性参照的参数。所有其他组合都很好。

    """

    def __init__(
        self,
        num_atoms = None,      # not used
        bond_feat_dim = None,  # not used
        num_targets = None,    # not used
        use_pbc=True,        #使用周期边界条件
        regress_forces=True, #计算力场
        otf_graph=True,      #是否动态生成计算图，在输入变长的情况下，效果好
        max_neighbors=500,   #每个原子的最大邻居数量,默认20                 #500是不是太大了？32可以？
        max_radius=5.0,      #两个原子之间的最大距离，用于构建半径图。默认12.0 #半径图应该是不需要的？
        max_num_elements=90, #用来one-hot的。应该是最大原子的序号。   #这个参数需要调整？要和我们的数据集中原子序号保持一致

        num_layers=12,       #网络层数量
        sphere_channels=128, #球谐函数的嵌入维度
        attn_hidden_channels=128, #SO(2)隐藏层的一个注意力头对应的维度，还是所有注意力头放在一起的总维度？，默认64
        num_heads=8,              #注意力头的数量
        attn_alpha_channels=32,   #每一个注意力头对应的嵌入维度, 256/8 = 32，默认64
        attn_value_channels=16,   #128/8 = 16
        ffn_hidden_channels=512,  #前馈网络的隐藏层维度  512/8 = 64，默认128
        
        norm_type='rms_norm_sh',  #默认，'layer_norm_sh' 。批量归一化的方法, BN不适合输入变长的网络，所以用LN
        
        lmax_list=[6],            #球谐最大度列表，这个影响球面谐波函数的阶数
        mmax_list=[2],            #球面谐波的最大阶数列表，不同阶数的球面谐波函数。阶数越大，表示的越好，但计算机代价也大
        grid_resolution=None,     #默认16，SO3_Grid的分辨率，即球谐函数的阶数，默认是None。可以由其它参数提供该值（如mmax_list提供）

        num_sphere_samples=128,   #用于近似输出块中球体积分的样本数，即为了近似求积分，需要采样多少样本。这个数量的多少影响性能？

        edge_channels=128,        #边不变特征的通道数，即边嵌入维度
        use_atom_edge_embedding=True, 
        #是否对边标量特征使用原子嵌入（即边嵌入由其两个原子的嵌入组成）和相对距离（边长度），另外，边特征还可以由边类型组成
        
        share_atom_edge_embedding=False, #是否共享原子边嵌入，默认不共享
        use_m_share_rad=False,
        #一个通道的L型矢量内的所有m个分量是否共享径向函数权重， 默认是False，其实共享是为了减少参数。理论上，不共享不影响模型的预测能力

        distance_function="gaussian",  #默认gaussian， 用于扩大原子距离，维度未变
        num_distance_basis=512,        #这个参数没有用到

        attn_activation='scaled_silu', #默认'silu'，so(2)的激活函数'scaled_silu'

        use_s2_act_attn=False,        #在so(2)后是否使用注意力，默认False，使用Equiformer注意力
        use_attn_renorm=True,         #再次归一化注意力。默认Ture
        ffn_activation='scaled_silu', #默认'silu'，前馈网络的激活函数'scaled_silu'
        use_gate_act=False,           #不使用gate激活
        use_grid_mlp=False,           #默认True，使用投影到网格并执行FFN的MLP，是否使用MLP，不使用。看看这部分在FFN前面还是后面使用？适用于降维还是什么呢？
        use_sep_s2_act=True,          #当“use_gate_act”为False时，使用可分离的S2激活。默认True

        alpha_drop=0.1,
        drop_path_rate=0.05, 
        proj_drop=0.0,                #网络层的去除率0.0

        weight_init='normal',         #默认'uniform'，初始化
        learn_energy = False,

        distance_resolution: float = 0.02, #Distance between distance basis functions in Angstroms

        #这些参数在新版本中不存在
        enforce_max_neighbors_strictly: bool = False,
        avg_num_nodes: Optional[float] = None,
        avg_degree: Optional[float] = None,

        #这两个和能量相关
        use_energy_lin_ref: Optional[bool] = False,
        load_energy_lin_ref: Optional[bool] = False,
    ):
        super().__init__()

        import sys

        if "e3nn" not in sys.modules:
            logging.error(
                "You need to install e3nn==0.4.4 to use EquiformerV2."
            )
            raise ImportError

        self.use_pbc = use_pbc
        self.regress_forces = regress_forces
        self.otf_graph = otf_graph
        self.max_neighbors = max_neighbors
        self.max_radius = max_radius
        self.cutoff = max_radius
        self.max_num_elements = max_num_elements + 1

        self.num_layers = num_layers
        self.sphere_channels = sphere_channels
        self.attn_hidden_channels = attn_hidden_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.ffn_hidden_channels = ffn_hidden_channels
        self.norm_type = norm_type

        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.grid_resolution = grid_resolution

        self.num_sphere_samples = num_sphere_samples

        self.edge_channels = edge_channels
        self.use_atom_edge_embedding = use_atom_edge_embedding
        self.share_atom_edge_embedding = share_atom_edge_embedding
        if self.share_atom_edge_embedding: #默认不开启
            assert self.use_atom_edge_embedding
            self.block_use_atom_edge_embedding = False
        else:
            self.block_use_atom_edge_embedding = self.use_atom_edge_embedding
        self.use_m_share_rad = use_m_share_rad
        self.distance_function = distance_function
        self.num_distance_basis = num_distance_basis

        self.attn_activation = attn_activation
        self.use_s2_act_attn = use_s2_act_attn
        self.use_attn_renorm = use_attn_renorm
        self.ffn_activation = ffn_activation
        self.use_gate_act = use_gate_act
        self.use_grid_mlp = use_grid_mlp
        self.use_sep_s2_act = use_sep_s2_act

        self.alpha_drop = alpha_drop
        self.drop_path_rate = drop_path_rate
        self.proj_drop = proj_drop
        self.learn_energy = learn_energy

        self.distance_resolution = distance_resolution

        #新本V2没有
        self.avg_num_nodes = avg_num_nodes or _AVG_NUM_NODES
        self.avg_degree = avg_degree or _AVG_DEGREE
        self.use_energy_lin_ref = use_energy_lin_ref
        self.load_energy_lin_ref = load_energy_lin_ref
        self.enforce_max_neighbors_strictly = enforce_max_neighbors_strictly  #默认是True，上面4个参数默认不开启
        assert not (
            self.use_energy_lin_ref and not self.load_energy_lin_ref
        ), "You can't have use_energy_lin_ref = True and load_energy_lin_ref = False, since the model will not have the parameters for the linear references. All other combinations are fine."


        self.weight_init = weight_init
        assert self.weight_init in ["normal", "uniform"]

        self.device = "cuda"  # torch.cuda.current_device()

        self.grad_forces = True  #是否需要力场梯度，这个要看一下,这个没用到？至少在当前文件中没用到
        self.num_resolutions: int = len(self.lmax_list) #球谐函数：数学中使用球坐标解Laplace方程的解。List of maximum degree of the spherical harmonics (1 to 10)，即函数的阶数

        #球面嵌入的维度。思考一个问题：为什么要引入球坐标？为了等变操作？
        self.sphere_channels_all: int = (
            self.num_resolutions * self.sphere_channels #最大维度是函数的阶层 * 隐层维度(128)
        )

        # Weights for message initialization
        #节点特征的初始化，假如有10种原子，则生成10*d的参数矩阵，当前批量的原子则可以根据原子类型获取对应的初始特征
        #这是带参数的初始化。我们也可以通过one-hot对原子类型进行初始嵌入化，再经过mlp压缩降维，然后和这里的随机初始的参数进行相加或连接
        #另外，因为求得是坐标，因此，边长度可以计算，这就意味着我们也要对边进行初始化，以及建模，同理边嵌入也可以采样节点嵌入的方法，one-hot+随机初始化
        #在对接中，边可以分为：节点对在配体内部：无键，单键，双键，三键，芳香键；配体和蛋白之间：节点对在蛋白内，节点对在配体和蛋白，节点对在蛋白和配体。共有5+3=8种
        #另外，考虑到对接任务是其它不变，只预测坐标，这也就意味着，如果预测出的坐标和真实的坐标越相似，则能量差越小。我们在训练过程中是否可以引入其他训练目标来辅助训练？
        #比如除了坐标的mse还有能量的mse的，即多任务学习
        self.sphere_embedding = nn.Embedding(
            self.max_num_elements, self.sphere_channels_all
        )

        # Initialize the function used to measure the distances between atoms #将标量的原子距离扩展到1维张量？还是说因为原子距离是非常小，单位很小，因为计算机mse等时，可能
        #出现溢出问题，因此这里的是为了等比例扩大原子距离的，采用的是GaussianSmearing()
        '''
        GaussianSmearing 是一种常用的方法，用于在计算化学中对电子密度进行平滑处理。它通常用于分子动力学模拟、电子结构计算等领域，以模拟原子核周围的电子分布。在这种方法中，
        原子核周围的电子密度被建模为一系列高斯函数的叠加，这些高斯函数代表了不同能量水平上的电子分布。

        GaussianSmearing 通常用于计算材料的能带结构，其中原子之间的距离被扩展以模拟电子的运动。通过在原子间添加高斯分布的电子密度，可以模拟电子的运动方式，并生成能带结构图。
        '''

        #目前，测试的结果是参数量很小，但是GPU显存占用大，很可能是非梯度的参数多，或者数据填充扩充了过多的维度
        assert self.distance_function in [
            "gaussian",
        ]

        self.num_gaussians = int(max_radius / self.distance_resolution)

        if self.distance_function == "gaussian":
            self.distance_expansion = GaussianSmearing(
                0.0,
                self.cutoff,
                #self.num_gaussians # 12.0 / 0.02 = 600
                int(self.num_gaussians / GP.bond_type_num),  #75, #默认600，为了使用边类型，这里改成75， 75 * 8(边类型数量) = 600,记得把其他地方的self.distance_expansion也改了。想降低GPU显存使用，则需要在这些地方减少填充
                2.0,
            )
            # self.distance_expansion = GaussianRadialBasisLayer(num_basis=self.num_distance_basis, cutoff=self.max_radius)
        else:
            raise ValueError

        # Initialize the sizes of radial functions（径向函数，即映射函数，多为MLP） (input channels and 2 hidden channels) #对距离建模处理mlp
        ##print('int(self.distance_expansion.num_output):', int(self.distance_expansion.num_output * 8))
        #exit()
        self.edge_channels_list = [int(self.distance_expansion.num_output * GP.bond_type_num)] + [
            self.edge_channels
        ] * 2

        # Initialize atom edge embedding #值得注意的是边嵌入和边长度不一样，可以分别初始化和建模，最后合并一起，送入GNN
        #边的初始嵌入可以使用one-hot,以及随机参数化。而边距离在扩充长度后只能通过参数化来初始化了，因为长度是连续的浮点数而不是有限的离散值
        #这里采用的是边嵌入有左右节点的嵌入组合而成
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding: #self.share_atom_edge_embedding默认是False
            self.source_embedding = nn.Embedding(
                self.max_num_elements, self.edge_channels_list[-1]
            )
            self.target_embedding = nn.Embedding(
                self.max_num_elements, self.edge_channels_list[-1]
            )
            self.edge_channels_list[0] = (
                self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
            )
        else:
            self.source_embedding, self.target_embedding = None, None

        # Initialize the module that compute WignerD matrices and other values for spherical harmonic calculations
        '''
        #引入球坐标是为了等变
        Wigner D 矩阵是量子力学中旋转群的表示矩阵，通常用于描述由三个欧拉角参数化的旋转操作。这些矩阵是由 Eugene Wigner
        在20世纪的早期引入的，用于研究原子物理学中的角动量耦合问题。

        Wigner D 矩阵是特殊正交群SO(3) 的表示，它描述了三维空间中的旋转变换。这些矩阵具有非常有用的性质，例如它们是幺正的（unitary）和不可约的（irreducible），
        并且它们可以用于描述由旋转操作产生的态之间的变换关系
        '''
        self.SO3_rotation = nn.ModuleList()
        for i in range(self.num_resolutions): #球谐函数的阶，越大质量越好，但计算代价也大，默认是6阶，这里的list实际上，我们只需要提供一个值即可
            self.SO3_rotation.append(SO3_Rotation(self.lmax_list[i]))

        # Initialize conversion between degree l and order m layouts #初始化l阶和m阶之间的转换，系数映射
        self.mappingReduced = CoefficientMappingModule(
            self.lmax_list, self.mmax_list #默认是（6,2）
        )

        # Initialize the transformations between spherical（球形的） and grid representations
        '''
        Grid representations 在计算化学和机器学习中是一种常见的分子表示方法。它们将分子的结构和性质转化为在网格上的数值表示，
        用于描述分子的电子分布、电荷密度、等电子密度等信息。
        这里的网格表示通常是指分子在3D笛卡尔坐标系下的表示。我们要知道弄清楚笛卡尔到球坐标，和球坐标系到笛卡尔的变换？？？
        '''
        self.SO3_grid = ModuleListInfo(
            "({}, {})".format(max(self.lmax_list), max(self.lmax_list))
        )
        for lval in range(max(self.lmax_list) + 1): #1~6层
            SO3_m_grid = nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(
                        lval,
                        m,
                        resolution=self.grid_resolution,
                        normalization="component",
                    )
                )
            self.SO3_grid.append(SO3_m_grid) #有 6*6个SO3_Grid

        # Edge-degree embedding，边嵌入，使用节点嵌入表示边嵌入，只是这里的节点嵌入是为边嵌入设计的，而不用于实际的节点嵌入
        #最终的边嵌入由这里的嵌入连接距离嵌入，
        #其实，这里因为边类型比较少，原子类型也少，即使使用原子嵌入表示边嵌入，也很容易参数不是很多，这个使用one-hot去表示原子或边嵌入，再共享一个MLP变换得到低维度嵌入一样
        #如果想进一步扩展，则可以选择最终的边嵌入=one-hot+MLP嵌入, 距离嵌入(MLP), 边类型随机初始化（使用一组随机参数变量表示）
        #原子嵌入最终嵌入= one-hot+MLP嵌入, 原子类型随机初始化（使用一组随机参数变量表示）。这些不同信息，可以分别经过处理，最终合并一起，使用加法或连接，由于我们不知道哪种效果好
        #建议连接，当然可以做一个开关，相加或连接后，跟着一个调整维度的mlp
        self.edge_degree_embedding = EdgeDegreeEmbedding(
            self.sphere_channels,
            self.lmax_list,
            self.mmax_list,
            self.SO3_rotation,
            self.mappingReduced,
            self.max_num_elements,
            self.edge_channels_list,
            self.block_use_atom_edge_embedding, #边嵌入在这
            rescale_factor=self.avg_degree, #avg_degree or _AVG_DEGREE，由于avg_degree默认是None，所以最终的值是_AVG_DEGREE
        )

        # Initialize the blocks for each layer of EquiformerV2 #EquiformerV2的节点嵌入模块
        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            block = TransBlockV2(
                self.sphere_channels,
                self.attn_hidden_channels,
                self.num_heads,
                self.attn_alpha_channels,
                self.attn_value_channels,
                self.ffn_hidden_channels,
                self.sphere_channels,
                self.lmax_list,
                self.mmax_list,
                self.SO3_rotation,
                self.mappingReduced,
                self.SO3_grid,
                self.max_num_elements,
                self.edge_channels_list,
                self.block_use_atom_edge_embedding,
                self.use_m_share_rad,
                self.attn_activation,
                self.use_s2_act_attn,
                self.use_attn_renorm,
                self.ffn_activation,
                self.use_gate_act,
                self.use_grid_mlp,
                self.use_sep_s2_act,
                self.norm_type,
                self.alpha_drop,
                self.drop_path_rate,
                self.proj_drop,
            )
            self.blocks.append(block)


        #EquiformerV2的能量块和力场块，这里我们可以做一个多任务学习，保留能量预测，做一个开关，用于消融
        # Output blocks for energy and forces
        self.norm = get_normalization_layer(
            self.norm_type,
            lmax=max(self.lmax_list), # 6
            num_channels=self.sphere_channels, # 128
        )

        # 能量块，有一个问题，真实的能量，怎么获取，直接从真实的构象计算出来？我们要计算损失

        if self.learn_energy:
            self.energy_block = FeedForwardNetwork(
                self.sphere_channels,
                self.ffn_hidden_channels,
                1,
                self.lmax_list,
                self.mmax_list,
                self.SO3_grid,
                self.ffn_activation,
                self.use_gate_act,
                self.use_grid_mlp,
                self.use_sep_s2_act,
            )

        #力场块
        if self.regress_forces:
            self.force_block = SO2EquivariantGraphAttention(
                self.sphere_channels,
                self.attn_hidden_channels,
                self.num_heads,
                self.attn_alpha_channels,
                self.attn_value_channels,
                1,
                self.lmax_list,
                self.mmax_list,
                self.SO3_rotation,
                self.mappingReduced,
                self.SO3_grid,
                self.max_num_elements,
                self.edge_channels_list,
                self.block_use_atom_edge_embedding,
                self.use_m_share_rad,
                self.attn_activation,
                self.use_s2_act_attn,
                self.use_attn_renorm,
                self.use_gate_act,
                self.use_sep_s2_act,
                alpha_drop=0.0,
            )

        if self.load_energy_lin_ref: #是否加载真实的能量，默认false
            self.energy_lin_ref = nn.Parameter(
                torch.zeros(self.max_num_elements),
                requires_grad=False,
            )

        self.apply(self._init_weights) #使用其它的初始化方法初始声明的网络参数，怎么调用的呀？
        self.apply(self._uniform_init_rad_func_linear_weights)

    @conditional_grad(torch.enable_grad())
    def forward(self,         
        h               = None, 
        pos             = None, 
        distance_vec    = None, 
        edge_dist       = None,
        element         = None,
        edge_type       = None, 
        edge_index      = None, 
        mask_ligand     = None, 
        sigmas          = None, 
        mask            = None, 
        batch           = None, 
        protein_max_atom_num = None, 
        ligand_max_atom_num  = None, 
        node_atom            = None,
        e_w                  =None, 
        fix_x                =None,

            ):
        self.batch_size = max(batch) + 1 #图的个数，即批量大小，这个要改，natoms存在每一个图的原子数量
        self.dtype  = pos.dtype   #float32
        self.device = pos.device  #cdua()

        atomic_numbers = element.long() #原子序号
        num_atoms      = len(atomic_numbers) #原子类型的数量

        edge_distance     = edge_dist 
        edge_distance_vec = distance_vec

        ###print('self.batch_size:', self.batch_size)
        ###print('atomic_numbers:',atomic_numbers.shape) #torch.Size([488])
        ####print('atomic_numbers:',atomic_numbers) #存放的是每一个原子的原子序号
        ####print('batch flag:', batch)
        ###print('self.max_num_elements:', self.max_num_elements) #17
        ###print('max(atomic_numbers):', max(atomic_numbers)) #max(atomic_numbers): tensor(17, device='cuda:0')

        if max(atomic_numbers) > self.max_num_elements - 1:
            raise Exception(f'{max(atomic_numbers) > self.max_num_elements - 1}')

        ###print('edge_index:', edge_index.shape)
        ####print('edge_index:', edge_index[:,:50]) #编号从开始的
        ###print('edge_distance:', edge_distance.shape)
        ###print('edge_distance_vec:', edge_distance_vec.shape)

        '''
        atomic_numbers: torch.Size([488])
        self.max_num_elements: 17
        edge_index: torch.Size([2, 15507])
        edge_distance: torch.Size([15507])
        edge_distance_vec: torch.Size([15507, 3])
        x: torch.Size([488, 49, 128])
        offset: 128
        offset_res: 49
        edge_distance befor: torch.Size([15507])
        edge_distance after: torch.Size([15507, 600])
        '''
        
        '''
        #图就不用在这里构建了
        #构建半径图，返回距离和距离向量，这个要改
        (
            edge_index,
            edge_distance,
            edge_distance_vec,
            cell_offsets,
            _,  # cell offset distances
            neighbors,
        ) = self.generate_graph(
            data,
            enforce_max_neighbors_strictly=self.enforce_max_neighbors_strictly, #是否严格执行最大邻居，如果为True，不足则填充。这个我们设置为False即可
        )
        '''

        ###############################################################
        # Initialize data structures
        ###############################################################

        # Compute 3x3 rotation matrix per edge #每一个边的旋转矩阵
        edge_rot_mat = self._init_edge_rot_mat(
            None, None, edge_distance_vec
        )

        # Initialize the WignerD matrices and other values for spherical harmonic calculations #等变旋转矩阵
        for i in range(self.num_resolutions):
            self.SO3_rotation[i].set_wigner(edge_rot_mat)

        ###############################################################
        # Initialize node embeddings
        ###############################################################

        #原子嵌入
        # Init per node representations using an atomic number based embedding, 初始化原子嵌入，随机初始化
        offset = 0
        x = SO3_Embedding(
            num_atoms, #原子数量
            self.lmax_list,  #6
            self.sphere_channels, #8
            self.device,
            self.dtype,
        )

        #print('x1:', x.embedding.shape)#torch.Size([489, 49, 8]) #看看这个49的注意力头怎么来的？
        ##这个参数难以控制，由lmax_list决定，所以是固定的，已经知道的，就是49，而8可以看作是一个注意力头的维度，这个值可以进一步降到4，即使这样，每一个原子的隐藏层嵌入维度
        #也是4*49==196
        #exit()

        '''
        edge_index: torch.Size([2, 17829])                                                                                                                                                                               
        edge_distance: torch.Size([17829])                                                                                                                                                                               
        edge_distance_vec: torch.Size([17829, 3])                                                                                                                                                                        
        x: torch.Size([560, 49, 128])                                                                                                                                                                                    
        /home/bingxing2/wangzw/pengyq/1.12.1/pytorch/aten/src/ATen/native/cuda/Indexing.cu:975: indexSelectLargeIndex: block: [553,0,0], thread: [0,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
        '''

        offset_res = 0
        offset = 0
        # Initialize the l = 0, m = 0 coefficients for each resolution #每一阶层
        for i in range(self.num_resolutions): #1层
            '''
            offset in: 0                                                                                                                                                                                                     
            offset_res in: 0                                                                                                                                                                                                 
            x in befor: torch.Size([560, 49, 128]) 
            '''
            #print('offset in:', offset)
            ###print('offset_res in:', offset_res)
            ###print('self.max_num_elements:', self.max_num_elements) #17
            ###print('max(atomic_numbers):', max(atomic_numbers)) #max(atomic_numbers): tensor(17, device='cuda:0')

            if max(atomic_numbers.detach().cpu().tolist()) > self.max_num_elements - 1:
                raise Exception(f'{max(atomic_numbers) > self.max_num_elements - 1}')
            
            if self.num_resolutions == 1:

                #下面这句报错，维度不对，正常应该是[560, 0, 0]， 但是现在却变成了[553,0,0]
                ###print('x in befor:', x.embedding.shape)
                ###print('self.sphere_embedding:', self.sphere_embedding.weight.shape) #17 * 128 , torch.Size([17, 128]) ,
                #原因找到了，原子序号最大是17，self.sphere_embedding的维度虽然也是17 * 128，但是编号从0开始，所以原子序号17时，数据越界了，
                #因此声明self.sphere_embedding，维度加1即可
                ###print('self.sphere_embedding(atomic_numbers) befor:', self.sphere_embedding(atomic_numbers).shape) #这行出错

                x.embedding[:, offset_res, :] = self.sphere_embedding(
                    atomic_numbers
                )

                ###print('x in after:', x.embedding.shape)
            else:
                x.embedding[:, offset_res, :] = self.sphere_embedding(
                    atomic_numbers
                )[:, offset : offset + self.sphere_channels]
            offset = offset + self.sphere_channels
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)

        #print('x2:', x.embedding.shape) #torch.Size([489, 49, 8])
        ###print('offset:', offset)
        ###print('offset_res:', offset_res)
        #边嵌入
        # Edge encoding (distance and atom edge)
        ###print('edge_distance befor:', edge_distance.shape)
        edge_distance = self.distance_expansion(edge_distance) #扩充边长
        ###print('edge_distance after:', edge_distance.shape)

        #考虑边类型信息
        #print('edge_type, edge_distance:', edge_type.shape, edge_distance.shape)
        #edge_type, edge_distance: torch.Size([15626, 8]) torch.Size([15626, 75])

        #edge_type, edge_distance: torch.Size([9354, 8]) torch.Size([9354, 600])

        edge_distance = self.outer_product(edge_type, edge_distance) #最终得到的嵌入是edge_type, edge_distance的嵌入维度的乘积 即 8 * 600 = 4800，所以要减少edge_distance的维度

        #print('edge_distance1:', edge_distance.shape) #edge_distance1: torch.Size([15626, 600])
        #edge_distance: torch.Size([9354, 4800])
        #exit()
        

        if self.share_atom_edge_embedding and self.use_atom_edge_embedding: #默认不共享参数
            source_element = atomic_numbers[
                edge_index[0]
            ]  # Source atom atomic number
            target_element = atomic_numbers[
                edge_index[1]
            ]  # Target atom atomic number
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            edge_distance = torch.cat(
                (edge_distance, source_embedding, target_embedding), dim=1
            )

        #print('edge_distance2:', edge_distance.shape) #edge_distance1: torch.Size([15626, 600])
        # Edge-degree embedding，度的嵌入。有什么用？用于原子嵌入
        
        #raise Exception('stop')
    
        '''
        atomic_numbers: torch.Size([488])
        self.max_num_elements: 17
        edge_index: torch.Size([2, 15507])
        edge_distance: torch.Size([15507])
        edge_distance_vec: torch.Size([15507, 3])
        x: torch.Size([488, 49, 128])
        offset: 128
        offset_res: 49
        edge_distance befor: torch.Size([15507])
        edge_distance after: torch.Size([15507, 600])
        '''

        ###print('edge_index2:', edge_index.shape)
        ####print('edge_index:', edge_index[:,:50]) #编号从开始的
        ###print('edge_distance2:', edge_distance.shape)
        ###print('edge_distance_vec2:', edge_distance_vec.shape)


        #这里有问题
        edge_degree = self.edge_degree_embedding(
            atomic_numbers, edge_distance, edge_index
        )

        

        ###print('edge_degree:', edge_degree.embedding.shape)
        #raise Exception('stop')
        #x.embedding = x.embedding + edge_degree.embedding #等于原子嵌入+原子出入度

        #加上传递过来的节点嵌入
        try:
            #print('h:', h.shape)
            #print('x.embedding:', x.embedding.shape)
            '''
            h: torch.Size([716, 3136]) , 即 torch.Size([716, 49， 64])                                                                                                                                                            
            x.embedding: torch.Size([716, 49, 128])
            '''
            x.embedding = x.embedding + edge_degree.embedding 
            #x.embedding = x.embedding + h.unsqueeze(dim=1)
            x.embedding = x.embedding + h.view(x.embedding.size())
            #print('h:', h.shape)
            #print('x.embedding:', x.embedding.shape)
            #exit()
            '''
            #在输入方面，我们让原子的嵌入的初始维度等于25*4，这样可以很方便和Equiformer进行维度对齐
            h: torch.Size([969, 100])
            x.embedding: torch.Size([969, 25, 4])
            h: torch.Size([969, 100])
            x.embedding: torch.Size([969, 25, 4])
            '''
        except Exception as e:
            #print('error:', e)
            #print('x.embedding:', x.embedding.shape)
            #print('h:', h.shape)
            raise Exception(e)

        #print('x3:', x.embedding.shape) #torch.Size([489, 49, 8])
        

        ###############################################################
        # Update spherical node embeddings
        # GNN模块，更新节点嵌入
        ###############################################################

        for i in range(self.num_layers):
            x = self.blocks[i](
                x,  # SO3_Embedding
                atomic_numbers,
                edge_distance,
                edge_index,
                batch=batch,  # for GraphDropPath
            )

        
        # Final layer norm
        x.embedding = self.norm(x.embedding)
        ###print('x.embedding output:', x.embedding.shape)

        #outputs = {"emb": x.embedding.sum(dim = 1)}
        outputs = {"emb": x.embedding.view(x.embedding.size(0), -1)} #由3维度，变成2维

        #print('x4:', x.embedding.shape)  #x4: torch.Size([489, 49, 8])
        ###############################################################
        # Energy estimation
        #能量预测
        ###############################################################

        if self.learn_energy:
            #print('predict energy')
            node_energy = self.energy_block(x)
            node_energy = node_energy.embedding.narrow(1, 0, 1)
            energy = torch.zeros(
                self.batch_size,
                device=node_energy.device,
                dtype=node_energy.dtype,
            )
            energy.index_add_(0, batch, node_energy.view(-1))
            energy = energy / self.avg_num_nodes

            # Add the per-atom linear references to the energy.
            if self.use_energy_lin_ref and self.load_energy_lin_ref:
                # During training, target E = (E_DFT - E_ref - E_mean) / E_std, and
                # during inference, \hat{E_DFT} = \hat{E} * E_std + E_ref + E_mean
                # where
                #
                # E_DFT = raw DFT energy,
                # E_ref = reference energy,
                # E_mean = normalizer mean,
                # E_std = normalizer std,
                # \hat{E} = predicted energy,
                # \hat{E_DFT} = predicted DFT energy.
                #
                # We can also write this as
                # \hat{E_DFT} = E_std * (\hat{E} + E_ref / E_std) + E_mean,
                # which is why we save E_ref / E_std as the linear reference.
                with torch.cuda.amp.autocast(False):
                    energy = energy.to(self.energy_lin_ref.dtype).index_add(
                        0,
                        batch,
                        self.energy_lin_ref[atomic_numbers],
                    )

            outputs = {"energy": energy}
        ###############################################################
        # Force estimation,预测力场这部分非常消耗GPU内存，优化一下，看看能不能降低维度，可能是edge_distance的维度是n * 600导致的
        ###############################################################
        if self.regress_forces:
            #print('predict forces')
            forces = self.force_block(
                x, atomic_numbers, edge_distance, edge_index
            )
            forces = forces.embedding.narrow(1, 1, 3)
            forces = forces.view(-1, 3)
        

            #修改配体坐标
            pos[mask_ligand]  = forces[mask_ligand]
            #pos = pos + forces * mask_ligand[:, None]  # only ligand positions will be updated
            outputs["forces"] = pos

            ###print('forces:', pos.shape)
            #raise Exception('stop')
            #exit()
        else:
            outputs["forces"] = None

        
        return outputs["emb"], outputs["forces"]



    def outer_product(self, *vectors):
        '''
            edge_type[n_src & n_dst] = 0   #表示在配体内部
            edge_type[n_src & ~n_dst] = 1  #表示源节点在配体，目标节点在蛋白
            edge_type[~n_src & n_dst] = 2  #表示源节点在蛋白，目标节点在配体
            edge_type[~n_src & ~n_dst] = 3 #表示在蛋白内部
            edge_type = F.one_hot(edge_type, num_classes=4)
        '''
        #vectors = (edge_attr, dist_feat)； edge_attr值表示邻接表的节点是否在配体和蛋白上
        for index, vector in enumerate(vectors): 
            if index == 0: #边类型处理
                out = vector.unsqueeze(-1) #E * D1 * 1
            else: #边长度处理
                out = out * vector.unsqueeze(1) #vector.unsqueeze(1) ,E * 1 * D2
                out = out.view(out.shape[0], -1).unsqueeze(-1) # E * (D1*D2)
        return out.squeeze()


    # Initialize the edge rotation matrics
    def _init_edge_rot_mat(self, data, edge_index, edge_distance_vec):
        return init_edge_rot_mat(edge_distance_vec)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear) or isinstance(m, SO3_LinearV2):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            if self.weight_init == "normal":
                std = 1 / math.sqrt(m.in_features)
                torch.nn.init.normal_(m.weight, 0, std)

        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    def _uniform_init_rad_func_linear_weights(self, m):
        if isinstance(m, RadialFunction):
            m.apply(self._uniform_init_linear_weights)

    def _uniform_init_linear_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            std = 1 / math.sqrt(m.in_features)
            torch.nn.init.uniform_(m.weight, -std, std)

    @torch.jit.ignore
    def no_weight_decay(self) -> set:
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if isinstance(
                module,
                (
                    torch.nn.Linear,
                    SO3_LinearV2,
                    torch.nn.LayerNorm,
                    EquivariantLayerNormArray,
                    EquivariantLayerNormArraySphericalHarmonics,
                    EquivariantRMSNormArraySphericalHarmonics,
                    EquivariantRMSNormArraySphericalHarmonicsV2,
                    GaussianRadialBasisLayer,
                ),
            ):
                for parameter_name, _ in module.named_parameters():
                    if isinstance(module, (torch.nn.Linear, SO3_LinearV2)):
                        if "weight" in parameter_name:
                            continue
                    global_parameter_name = module_name + "." + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)

        return set(no_wd_list)
