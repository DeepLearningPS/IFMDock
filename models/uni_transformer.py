import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import radius_graph, knn_graph
from torch_scatter import scatter_softmax, scatter_sum
from EcConf.comparm import *

from models.common import GaussianSmearing, MLP, batch_hybrid_edge_connection, outer_product
from ocp.ocpmodels.models.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20
from ocp.ocpmodels.models.escn.escn import eSCN
import math
from collections import defaultdict
from ordered_set import OrderedSet

class BaseX2HAttLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, edge_feat_dim, r_feat_dim,
                 act_fn='relu', norm=True, ew_net_type='r', out_fc=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.act_fn = act_fn
        self.edge_feat_dim = edge_feat_dim
        self.r_feat_dim = r_feat_dim
        self.ew_net_type = ew_net_type
        self.out_fc = out_fc

        # attention key func
        ##print('edge_feat_dim 4?:', edge_feat_dim) #4,应该改成8
        #edge_feat_dim = 4
        
        #raise Exception('test')
        kv_input_dim = input_dim * 2 + edge_feat_dim + r_feat_dim #这个维度有问题，需要改一下，否则当我们改变键类型的维度时，导致维度不对而报错
        #print('input_dim, hidden_dim, output_dim, kv_input_dim:', input_dim, hidden_dim, output_dim, kv_input_dim) #128 128 128
        # 128 128 128 424
        self.hk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)

        # attention value func
        self.hv_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)

        # attention query func
        self.hq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        if ew_net_type == 'r':
            self.ew_net = nn.Sequential(nn.Linear(r_feat_dim, 1), nn.Sigmoid())
        elif ew_net_type == 'm':
            self.ew_net = nn.Sequential(nn.Linear(output_dim, 1), nn.Sigmoid())

        if self.out_fc:
            self.node_output = MLP(2 * hidden_dim, hidden_dim, hidden_dim, norm=norm, act_fn=act_fn)

    def forward(self, h, r_feat, edge_feat, edge_index, e_w=None):

        #(h_in, dist_feat, edge_feat, edge_index, e_w=e_w) #edge_feat代表边类型，指示了边的节点在配体和蛋白上的情况
        N = h.size(0)
        src, dst = edge_index
        hi, hj = h[dst], h[src]

        # multi-head attention
        # decide inputs of k_func and v_func
        kv_input = torch.cat([r_feat, hi, hj], -1)
        if edge_feat is not None:
            kv_input = torch.cat([edge_feat, kv_input], -1)

        # compute k
        ##print('kv_input:', kv_input.shape) #kv_input: torch.Size([114023, 424])，但是MLP的in_dim, out_dim, hidden_dim: 344 128 128
        #维度对不上，现在要么改x的维度，要么该初始化MLP时的in_dim的维度，正确的方向应该是改in_dim
        #raise Exception('test')
        k = self.hk_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads) #这个之前，有出现使用MLP的，能通过，而这里则维度不对
        ##print('hk_func:', k.shape)
        # compute v
        v = self.hv_func(kv_input)

        if self.ew_net_type == 'r':
            e_w = self.ew_net(r_feat)
        elif self.ew_net_type == 'm':
            e_w = self.ew_net(v[..., :self.hidden_dim])
        elif e_w is not None:
            e_w = e_w.view(-1, 1)
        else:
            e_w = 1.
        v = v * e_w
        v = v.view(-1, self.n_heads, self.output_dim // self.n_heads)

        # compute q
        q = self.hq_func(h).view(-1, self.n_heads, self.output_dim // self.n_heads)

        # compute attention weights
        alpha = scatter_softmax((q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0,
                                dim_size=N)  # [num_edges, n_heads]

        # perform attention-weighted message-passing
        m = alpha.unsqueeze(-1) * v  # (E, heads, H_per_head)
        output = scatter_sum(m, dst, dim=0, dim_size=N)  # (N, heads, H_per_head)
        output = output.view(-1, self.output_dim)
        if self.out_fc:
            output = self.node_output(torch.cat([output, h], -1))

        output = output + h
        return output


class BaseH2XAttLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, edge_feat_dim, r_feat_dim,
                 act_fn='relu', norm=True, ew_net_type='r'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.r_feat_dim = r_feat_dim
        self.act_fn = act_fn
        self.ew_net_type = ew_net_type

        kv_input_dim = input_dim * 2 + edge_feat_dim + r_feat_dim

        self.xk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        self.xv_func = MLP(kv_input_dim, self.n_heads, hidden_dim, norm=norm, act_fn=act_fn)
        self.xq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        if ew_net_type == 'r':
            self.ew_net = nn.Sequential(nn.Linear(r_feat_dim, 1), nn.Sigmoid())

    def forward(self, h, rel_x, r_feat, edge_feat, edge_index, e_w=None):
        N = h.size(0)
        src, dst = edge_index
        hi, hj = h[dst], h[src]

        # multi-head attention
        # decide inputs of k_func and v_func
        kv_input = torch.cat([r_feat, hi, hj], -1)
        if edge_feat is not None:
            kv_input = torch.cat([edge_feat, kv_input], -1)

        k = self.xk_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads)
        ##print('xk_func:',k.shape)

        v = self.xv_func(kv_input)
        ##print('xv_func:', v.shape)
        if self.ew_net_type == 'r':
            e_w = self.ew_net(r_feat)
        elif self.ew_net_type == 'm':
            e_w = 1.
        elif e_w is not None:
            e_w = e_w.view(-1, 1)
        else:
            e_w = 1.
        v = v * e_w

        v = v.unsqueeze(-1) * rel_x.unsqueeze(1)  # (xi - xj) [n_edges, n_heads, 3]
        q = self.xq_func(h).view(-1, self.n_heads, self.output_dim // self.n_heads)

        # Compute attention weights
        alpha = scatter_softmax((q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0, dim_size=N)  # (E, heads)

        # Perform attention-weighted message-passing
        m = alpha.unsqueeze(-1) * v  # (E, heads, 3)
        output = scatter_sum(m, dst, dim=0, dim_size=N)  # (N, heads, 3)
        return output.mean(1)  # [num_nodes, 3]

#神经网络的一层，看看坐标和嵌入是怎么结合的
class AttentionLayerO2TwoUpdateNodeGeneral(nn.Module):
    def __init__(self, hidden_dim, n_heads, num_r_gaussian, edge_feat_dim, act_fn='relu', norm=True,
                 num_x2h=1, num_h2x=1, r_min=0., r_max=10., num_node_types=8,
                 ew_net_type='r', x2h_out_fc=True, sync_twoup=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.num_r_gaussian = num_r_gaussian
        self.norm = norm
        self.act_fn = act_fn
        self.num_x2h = num_x2h
        self.num_h2x = num_h2x
        self.r_min, self.r_max = r_min, r_max
        self.num_node_types = num_node_types
        self.ew_net_type = ew_net_type
        self.x2h_out_fc = x2h_out_fc
        self.sync_twoup = sync_twoup

        self.distance_expansion = GaussianSmearing(self.r_min, self.r_max, num_gaussians=num_r_gaussian)

        self.x2h_layers = nn.ModuleList()
        for i in range(self.num_x2h):
            self.x2h_layers.append(
                BaseX2HAttLayer(hidden_dim, hidden_dim, hidden_dim, n_heads, edge_feat_dim,
                                r_feat_dim=num_r_gaussian * 4,
                                act_fn=act_fn, norm=norm,
                                ew_net_type=self.ew_net_type, out_fc=self.x2h_out_fc)
            )
        self.h2x_layers = nn.ModuleList()
        for i in range(self.num_h2x):
            self.h2x_layers.append(
                BaseH2XAttLayer(hidden_dim, hidden_dim, hidden_dim, n_heads, edge_feat_dim,
                                r_feat_dim=num_r_gaussian * 4,
                                act_fn=act_fn, norm=norm,
                                ew_net_type=self.ew_net_type)
            )

    def forward(self, h, x, edge_attr, edge_index, mask_ligand, e_w=None, fix_x=False):
        #edge_attr边类型，配体和蛋白的信息合并在了一起处理， 看看edge_attr是怎么处理的，是简单的mlp处理一下，还是另有特殊操作，比如用于判断节点的在蛋白
        #还是在配体上
        src, dst = edge_index
        if self.edge_feat_dim > 0:
            edge_feat = edge_attr  # shape: [#edges_in_batch, #bond_types]
        else:
            edge_feat = None

        rel_x = x[dst] - x[src]
        dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True) #边长度

        h_in = h
        # 4 separate distance embedding for p-p, p-l, l-p, l-l，分别对坐标和原子嵌入进行4种处理，求
        for i in range(self.num_x2h):
            dist_feat = self.distance_expansion(dist) #GaussianSmearing处理边长度，不带参数，算是扩充维度
            #print('edge_attr, dist_feat:', edge_attr.shape, dist_feat.shape)
            #edge_attr, dist_feat: torch.Size([240503, 8]) torch.Size([240503, 20])
            
            dist_feat = outer_product(edge_attr, dist_feat) #边长度嵌入=边类型嵌入+边长度嵌入，不带参数,其中edge_attr提供了节点是否在配体和蛋白的信息
            #print('dist_feat:', dist_feat.shape)
            #dist_feat: torch.Size([240503, 160])
            #exit()
            h_out = self.x2h_layers[i](h_in, dist_feat, edge_feat, edge_index, e_w=e_w) #问题在这，键类型维度的更改，会导致这里出问题
            h_in = h_out
        x2h_out = h_in

        new_h = h if self.sync_twoup else x2h_out #可以将原子嵌入作为坐标的嵌入网络的输入，默认不使用，意味着坐标和原子嵌入没关系？我们可以试一下
        #将节点嵌入的输出作为节点坐标的输入，否则前后神经网络没啥关系
        for i in range(self.num_h2x):
            dist_feat = self.distance_expansion(dist) #distance_expansion无参数的
            dist_feat = outer_product(edge_attr, dist_feat)
            delta_x = self.h2x_layers[i](new_h, rel_x, dist_feat, edge_feat, edge_index, e_w=e_w) 
            #注意这是和节点嵌入网络是不同的，多了rel_x参数，但是基本结果是不变化的
            if not fix_x: #fix_x默认是false，表示不固定，即蛋白不动
                x = x + delta_x * mask_ligand[:, None]  # only ligand positions will be updated
            rel_x = x[dst] - x[src] #注意dist在每一层网络是变化的，这点不同于节点嵌入网络，因为坐标发生改变了
            dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True)

        return x2h_out, x


class UniTransformerO2TwoUpdateGeneral(nn.Module):
    def __init__(self, num_blocks, num_layers, hidden_dim, n_heads=1, k=32,
                num_r_gaussian=50, edge_feat_dim=0, num_node_types=8, act_fn='relu', norm=True,
                cutoff_mode='radius', ew_net_type='r',
                num_init_x2h=1, num_init_h2x=0, num_x2h=1, num_h2x=1, r_max=10., x2h_out_fc=True, sync_twoup=False,
                equiformer_args = None, equiformer = False, escn_args = None, escn = False):
        super().__init__()
        # Build the network
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.num_r_gaussian = num_r_gaussian
        self.edge_feat_dim = edge_feat_dim
        self.act_fn = act_fn
        self.norm = norm
        self.num_node_types = num_node_types
        # radius graph / knn graph ，截断网络
        self.cutoff_mode = cutoff_mode  # [radius, none]
        self.k = k
        self.ew_net_type = ew_net_type  # [r, m, none]，默认是global

        self.num_x2h = num_x2h
        self.num_h2x = num_h2x
        self.num_init_x2h = num_init_x2h
        self.num_init_h2x = num_init_h2x
        self.r_max = r_max
        self.x2h_out_fc = x2h_out_fc
        self.sync_twoup = sync_twoup
        self.distance_expansion = GaussianSmearing(0., r_max, num_gaussians=num_r_gaussian) #高斯平滑
        if self.ew_net_type == 'global':
            self.edge_pred_layer = MLP(num_r_gaussian, 1, hidden_dim)
        
        self.init_h_emb_layer = self._build_init_h_layer()

        self.equiformer_args = equiformer_args
        self.equiformer = equiformer
        self.escn_args = escn_args
        self.escn = escn

        if GP.embedding3d:
            self.linear_transform_dim = nn.Linear(200 + 200, 200)


            self.ligand_block =  EquiformerV2_OC20(
                use_pbc             = self.equiformer_args.use_pbc,
                regress_forces      = False,
                otf_graph           = self.equiformer_args.otf_graph,
                max_neighbors       = self.equiformer_args.max_neighbors,
                max_radius          = self.equiformer_args.max_radius,
                max_num_elements    = self.equiformer_args.max_num_elements,
                num_layers          = 3,
                sphere_channels     = self.equiformer_args.sphere_channels,
                attn_hidden_channels        = self.equiformer_args.attn_hidden_channels,
                num_heads                   = self.equiformer_args.num_heads,
                attn_alpha_channels         = self.equiformer_args.attn_alpha_channels,
                attn_value_channels         = self.equiformer_args.attn_value_channels,
                ffn_hidden_channels         = self.equiformer_args.ffn_hidden_channels,
                norm_type                   = self.equiformer_args.norm_type,
                lmax_list                   = self.equiformer_args.lmax_list,
                mmax_list                   = self.equiformer_args.mmax_list,
                grid_resolution             = self.equiformer_args.grid_resolution,
                num_sphere_samples          = self.equiformer_args.num_sphere_samples,
                edge_channels               = self.equiformer_args.edge_channels,
                use_atom_edge_embedding     = self.equiformer_args.use_atom_edge_embedding,
                share_atom_edge_embedding   = self.equiformer_args.share_atom_edge_embedding,
                use_m_share_rad     = self.equiformer_args.use_m_share_rad, #False
                distance_function   = self.equiformer_args.distance_function,
                num_distance_basis  = self.equiformer_args.num_distance_basis,
                attn_activation     = self.equiformer_args.attn_activation,
                use_s2_act_attn     = self.equiformer_args.use_s2_act_attn,
                use_attn_renorm     = self.equiformer_args.use_attn_renorm,
                ffn_activation      = self.equiformer_args.ffn_activation,
                use_gate_act        = self.equiformer_args.use_gate_act,
                use_grid_mlp        = self.equiformer_args.use_grid_mlp,
                use_sep_s2_act      = self.equiformer_args.use_sep_s2_act,
                alpha_drop          = self.equiformer_args.alpha_drop,
                drop_path_rate      = self.equiformer_args.drop_path_rate,
                proj_drop           = self.equiformer_args.proj_drop,
                weight_init         = self.equiformer_args.weight_init,        
                )


            self.protein_block =  EquiformerV2_OC20(
                use_pbc             = self.equiformer_args.use_pbc,
                regress_forces      = False,
                otf_graph           = self.equiformer_args.otf_graph,
                max_neighbors       = self.equiformer_args.max_neighbors,
                max_radius          = self.equiformer_args.max_radius,
                max_num_elements    = self.equiformer_args.max_num_elements,
                num_layers          = 3,
                sphere_channels     = self.equiformer_args.sphere_channels,
                attn_hidden_channels        = self.equiformer_args.attn_hidden_channels,
                num_heads                   = self.equiformer_args.num_heads,
                attn_alpha_channels         = self.equiformer_args.attn_alpha_channels,
                attn_value_channels         = self.equiformer_args.attn_value_channels,
                ffn_hidden_channels         = self.equiformer_args.ffn_hidden_channels,
                norm_type                   = self.equiformer_args.norm_type,
                lmax_list                   = self.equiformer_args.lmax_list,
                mmax_list                   = self.equiformer_args.mmax_list,
                grid_resolution             = self.equiformer_args.grid_resolution,
                num_sphere_samples          = self.equiformer_args.num_sphere_samples,
                edge_channels               = self.equiformer_args.edge_channels,
                use_atom_edge_embedding     = self.equiformer_args.use_atom_edge_embedding,
                share_atom_edge_embedding   = self.equiformer_args.share_atom_edge_embedding,
                use_m_share_rad     = self.equiformer_args.use_m_share_rad, #False
                distance_function   = self.equiformer_args.distance_function,
                num_distance_basis  = self.equiformer_args.num_distance_basis,
                attn_activation     = self.equiformer_args.attn_activation,
                use_s2_act_attn     = self.equiformer_args.use_s2_act_attn,
                use_attn_renorm     = self.equiformer_args.use_attn_renorm,
                ffn_activation      = self.equiformer_args.ffn_activation,
                use_gate_act        = self.equiformer_args.use_gate_act,
                use_grid_mlp        = self.equiformer_args.use_grid_mlp,
                use_sep_s2_act      = self.equiformer_args.use_sep_s2_act,
                alpha_drop          = self.equiformer_args.alpha_drop,
                drop_path_rate      = self.equiformer_args.drop_path_rate,
                proj_drop           = self.equiformer_args.proj_drop,
                weight_init         = self.equiformer_args.weight_init,        
                )
        







        if equiformer == True: #
            print('使用equiformer')
            self.base_block =  EquiformerV2_OC20(
                use_pbc             = self.equiformer_args.use_pbc,
                regress_forces      = self.equiformer_args.regress_forces,
                otf_graph           = self.equiformer_args.otf_graph,
                max_neighbors       = self.equiformer_args.max_neighbors,
                max_radius          = self.equiformer_args.max_radius,
                max_num_elements    = self.equiformer_args.max_num_elements,
                num_layers          = self.equiformer_args.num_layers,
                sphere_channels     = self.equiformer_args.sphere_channels,
                attn_hidden_channels        = self.equiformer_args.attn_hidden_channels,
                num_heads                   = self.equiformer_args.num_heads,
                attn_alpha_channels         = self.equiformer_args.attn_alpha_channels,
                attn_value_channels         = self.equiformer_args.attn_value_channels,
                ffn_hidden_channels         = self.equiformer_args.ffn_hidden_channels,
                norm_type                   = self.equiformer_args.norm_type,
                lmax_list                   = self.equiformer_args.lmax_list,
                mmax_list                   = self.equiformer_args.mmax_list,
                grid_resolution             = self.equiformer_args.grid_resolution,
                num_sphere_samples          = self.equiformer_args.num_sphere_samples,
                edge_channels               = self.equiformer_args.edge_channels,
                use_atom_edge_embedding     = self.equiformer_args.use_atom_edge_embedding,
                share_atom_edge_embedding   = self.equiformer_args.share_atom_edge_embedding,
                use_m_share_rad     = self.equiformer_args.use_m_share_rad, #False
                distance_function   = self.equiformer_args.distance_function,
                num_distance_basis  = self.equiformer_args.num_distance_basis,
                attn_activation     = self.equiformer_args.attn_activation,
                use_s2_act_attn     = self.equiformer_args.use_s2_act_attn,
                use_attn_renorm     = self.equiformer_args.use_attn_renorm,
                ffn_activation      = self.equiformer_args.ffn_activation,
                use_gate_act        = self.equiformer_args.use_gate_act,
                use_grid_mlp        = self.equiformer_args.use_grid_mlp,
                use_sep_s2_act      = self.equiformer_args.use_sep_s2_act,
                alpha_drop          = self.equiformer_args.alpha_drop,
                drop_path_rate      = self.equiformer_args.drop_path_rate,
                proj_drop           = self.equiformer_args.proj_drop,
                weight_init         = self.equiformer_args.weight_init,        
                )
            
        elif escn == True: #
            print('使用escn')
            self.base_block =  eSCN(
                use_pbc             = self.escn_args.use_pbc,
                regress_forces      = self.escn_args.regress_forces,
                otf_graph           = self.escn_args.otf_graph,
                max_neighbors       = self.escn_args.max_neighbors,
                max_num_elements    = self.escn_args.max_num_elements,
                num_layers          = self.escn_args.num_layers,

                lmax_list           = self.escn_args.lmax_list,
                mmax_list           = self.escn_args.mmax_list,           
                #grid_resolution     = self.escn_args.grid_resolution,

                sphere_channels     = self.escn_args.sphere_channels,
                hidden_channels     = self.escn_args.hidden_channels,
                edge_channels       = self.escn_args.edge_channels,
                use_grid            = self.escn_args.use_grid,
                num_sphere_samples  = self.escn_args.num_sphere_samples,
                distance_function   = self.escn_args.distance_function,
                basis_width_scalar  = self.escn_args.basis_width_scalar,
                distance_resolution = self.escn_args.distance_resolution,
                show_timing_info    = False,
        
                )
        else:
            print('使用egnn')
            self.base_block = self._build_share_blocks()

    def __repr__(self):
        return f'UniTransformerO2(num_blocks={self.num_blocks}, num_layers={self.num_layers}, n_heads={self.n_heads}, ' \
               f'act_fn={self.act_fn}, norm={self.norm}, cutoff_mode={self.cutoff_mode}, ew_net_type={self.ew_net_type}, ' \
               f'init h emb: {self.init_h_emb_layer.__repr__()} \n' \
               f'base block: {self.base_block.__repr__()} \n' \
               f'edge pred layer: {self.edge_pred_layer.__repr__() if hasattr(self, "edge_pred_layer") else "None"}) '

    def _build_init_h_layer(self):
        layer = AttentionLayerO2TwoUpdateNodeGeneral(
            self.hidden_dim, self.n_heads, self.num_r_gaussian, self.edge_feat_dim, act_fn=self.act_fn, norm=self.norm,
            num_x2h=self.num_init_x2h, num_h2x=self.num_init_h2x, r_max=self.r_max, num_node_types=self.num_node_types,
            ew_net_type=self.ew_net_type, x2h_out_fc=self.x2h_out_fc, sync_twoup=self.sync_twoup,
        )
        return layer

    def _build_share_blocks(self):
        # Equivariant layers
        base_block = []
        for l_idx in range(self.num_layers):
            layer = AttentionLayerO2TwoUpdateNodeGeneral(
                self.hidden_dim, self.n_heads, self.num_r_gaussian, self.edge_feat_dim, act_fn=self.act_fn,
                norm=self.norm,
                num_x2h=self.num_x2h, num_h2x=self.num_h2x, r_max=self.r_max, num_node_types=self.num_node_types,
                ew_net_type=self.ew_net_type, x2h_out_fc=self.x2h_out_fc, sync_twoup=self.sync_twoup,
            )
            base_block.append(layer)
        return nn.ModuleList(base_block)

    def _connect_edge(self, x, mask_ligand, batch):
        if self.cutoff_mode == 'radius':
            edge_index = radius_graph(x, r=self.r, batch=batch, flow='source_to_target')
        elif self.cutoff_mode == 'knn': #默认是KNN图，这里是如何去分edge_inde中哪些是配体，哪些是蛋白的？合并在一起了，不用区分哪些是蛋白，哪些是配体
            edge_index = knn_graph(x, k=self.k, batch=batch, flow='source_to_target')
        elif self.cutoff_mode == 'hybrid':
            edge_index = batch_hybrid_edge_connection(
                x, k=self.k, mask_ligand=mask_ligand, batch=batch, add_p_index=True)
        else:
            raise ValueError(f'Not supported cutoff mode: {self.cutoff_mode}')
        return edge_index

    @staticmethod
    def _build_edge_type_8(edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch):
        #可以全部换成GPU tensor运算，加速训练
        #获取配体的内部的原子id
        ligand_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 1])))).numpy()

    
        #获取配体和蛋白放在一起的原子id
        protein_ligand_node_list = list(range(len(mask_ligand)))

        ligand_node_global = torch.LongTensor(protein_ligand_node_list)[mask_ligand.cpu() == 1].numpy() #pytorch2.0要求datas[index]，数据和索引都在一个设备上

        ##print('ligand_node_list:', len(ligand_node_local))
        ##print('protein_ligand_node_list:', len(protein_ligand_node_list))
        ##print('ligand_node_global:', len(ligand_node_global))
        ##print('edge_index:', edge_index.shape)

        '''
        ligand_node_list: 282
        protein_ligand_node_list: 3388
        ligand_node_global: 282
        edge_index: torch.Size([2, 108416])
        '''

        #制作id映射
        ligand_node_local2global_dict = {}
        for k, v in zip(ligand_node_local, ligand_node_global):
            ligand_node_local2global_dict[k] = v
        

        #更新id
        new_ligand_bond_index = torch.zeros(ligand_bond_index.T.shape, dtype = torch.int64).numpy()
        for i, bd in enumerate(ligand_bond_index.T.detach().cpu().numpy()):
            ##print('bd:', bd) #bd: [0 1]
            new_ligand_bond_index[i][0] = ligand_node_local2global_dict[bd[0]]
            new_ligand_bond_index[i][1] = ligand_node_local2global_dict[bd[1]]
        
        new_ligand_bond_index = torch.from_numpy(new_ligand_bond_index.T).cuda()
        ##print('new_ligand_bond_index:', new_ligand_bond_index.shape) #torch.Size([2, 582])


        #raise Exception('test')
        # NxN connectivity matrix where 0 means no connection and 1/2/3/4 means single/double/triple/aromatic bonds.

        #不是键类型，是用来判断蛋白和配体之间的是否有连接
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1 #1表示是配体
        n_dst = mask_ligand[dst] == 1

        #需要扩充配体的键类型，增加单键，双键，三键，芳香键，我们需要知道配体和蛋白的这些键类型有什么作用？？？
        #edge_type在神经网络中只是和边长度嵌入和节点嵌入连接在了一起，并没有什么特殊的判断处理，虽然这里的0,1,2,3的确用来标识节点是蛋白或者配体上
        #但是在神经网络中用没有用来区分配体和蛋白节点
        edge_type[n_src & n_dst]   = 1  #表示在配体内部
        edge_type[n_src & ~n_dst]  = 5  #表示源节点在配体，目标节点在蛋白
        edge_type[~n_src & n_dst]  = 6  #表示源节点在蛋白，目标节点在配体
        edge_type[~n_src & ~n_dst] = 7  #表示在蛋白内部


        indices = (edge_type == 1).nonzero().view(-1) #寻找非0元素下标
        #indices = (edge_type == 1).nonzero().squeeze() #寻找非0元素下标

        # 要删除的行的索引
        rows_to_remove = indices.detach().cpu().tolist()

        # 使用 torch.index_select() 函数选择不删除的行,去掉配体连接
        indices_to_keep = torch.tensor(list(set(range(edge_type.size(0))) - set(rows_to_remove))).cuda()
        new_edge_type   = torch.index_select(edge_type, 0, indices_to_keep)
        new_edge_index  = torch.index_select(edge_index, 1, indices_to_keep)  #2 * N

        #添加配体内部连接
        ##print('ligand_bond_type:', ligand_bond_type)
        ##print('new_ligand_bond_index:', new_ligand_bond_index)

    
        new_edge_type  = torch.cat([new_edge_type, ligand_bond_type], dim = 0)  #扩充配体键类型
        #new_edge_type  = torch.cat([new_edge_type, torch.zeros_like(ligand_bond_type, dtype = torch.int64)], dim = 0)#不扩充配体键类型，依旧使用0
        new_edge_index = torch.cat([new_edge_index, new_ligand_bond_index], dim = 1)

        #

        ##print('new_edge_type:', new_edge_type.shape)
        ##print('new_edge_index:', new_edge_index.shape)
        #new_edge_type: torch.Size([103536])
        #new_edge_index: torch.Size([2, 103536])

        #(edge_type,edge_index)每一行的顺序是无所谓的，不影响卷积，关键在于里面使用的原子id一定要是根据当前批量获得的全局id
        #(edge_type,edge_index)只要配体和蛋白之间的，其余的全部由配体和蛋白的全连接图来取代
        edge_type_dim = F.one_hot(new_edge_type, num_classes=8) #由原来的4种，变成了8种,除了把这里改了成8之外，其它地方也要改吧？否则报错
        #探索一下，哪有哪里在使用键长度
        return edge_type_dim, new_edge_index




    @staticmethod
    def _build_edge_type_8_gpu(edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch):
        #可以全部换成GPU tensor运算，加速训练
        #获取配体的内部的原子id
        ligand_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 1])))).cuda()

    
        #获取配体和蛋白放在一起的原子id
        protein_ligand_node_list = torch.LongTensor(list(range(len(mask_ligand)))).cuda()

        ligand_node_global = protein_ligand_node_list[mask_ligand == 1] #pytorch2.0要求datas[index]，数据和索引都在一个设备上

        ##print('ligand_node_list:', len(ligand_node_local))
        ##print('protein_ligand_node_list:', len(protein_ligand_node_list))
        ##print('ligand_node_global:', len(ligand_node_global))
        ##print('edge_index:', edge_index.shape)

        '''
        ligand_node_list: 282
        protein_ligand_node_list: 3388
        ligand_node_global: 282
        edge_index: torch.Size([2, 108416])
        '''

        #制作id映射
        ligand_node_local2global_dict = {}
        for k, v in zip(ligand_node_local, ligand_node_global):
            ligand_node_local2global_dict[k.item()] = v
        

        #更新id
        new_ligand_bond_index = torch.zeros(ligand_bond_index.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(ligand_bond_index.T):
            #print('bd:', bd) #numpy数据 bd: [0, 9], 如果是torch tensor，这是 tensor([0, 9], device='cuda:0')
            #print('bd[0]:', bd[0]) #bd[0]: tensor(0, device='cuda:0')
            #print('bd[1]:', bd[1]) #bd[0]: tensor(9, device='cuda:0')
            #print('ligand_node_local:', ligand_node_local) #ligand_node_local: tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,18, 19], device='cuda:0')
            #exit()
            #print('ligand_node_local2global_dict:', ligand_node_local2global_dict)
            new_ligand_bond_index[i][0] = ligand_node_local2global_dict[bd[0].item()] #tensor张量不适合作为字典的k，所以要更换，否则报错
            new_ligand_bond_index[i][1] = ligand_node_local2global_dict[bd[1].item()]
        
        new_ligand_bond_index = new_ligand_bond_index.T
        ##print('new_ligand_bond_index:', new_ligand_bond_index.shape) #torch.Size([2, 582])


        #raise Exception('test')
        # NxN connectivity matrix where 0 means no connection and 1/2/3/4 means single/double/triple/aromatic bonds.

        #不是键类型，是用来判断蛋白和配体之间的是否有连接
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1 #1表示是配体
        n_dst = mask_ligand[dst] == 1

        #需要扩充配体的键类型，增加单键，双键，三键，芳香键，我们需要知道配体和蛋白的这些键类型有什么作用？？？
        #edge_type在神经网络中只是和边长度嵌入和节点嵌入连接在了一起，并没有什么特殊的判断处理，虽然这里的0,1,2,3的确用来标识节点是蛋白或者配体上
        #但是在神经网络中用没有用来区分配体和蛋白节点
        edge_type[n_src & n_dst]   = 1  #表示在配体内部
        edge_type[n_src & ~n_dst]  = 5  #表示源节点在配体，目标节点在蛋白
        edge_type[~n_src & n_dst]  = 6  #表示源节点在蛋白，目标节点在配体
        edge_type[~n_src & ~n_dst] = 7  #表示在蛋白内部


        indices = (edge_type == 1).nonzero().view(-1) #寻找非0元素下标
        #indices = (edge_type == 1).nonzero().squeeze() #寻找非0元素下标

        # 要删除的行的索引
        rows_to_remove = indices.detach().cpu().tolist()

        # 使用 torch.index_select() 函数选择不删除的行,去掉配体连接
        indices_to_keep = torch.tensor(list(set(range(edge_type.size(0))) - set(rows_to_remove))).cuda()
        new_edge_type   = torch.index_select(edge_type, 0, indices_to_keep)
        new_edge_index  = torch.index_select(edge_index, 1, indices_to_keep)  #2 * N

        #添加配体内部连接
        ##print('ligand_bond_type:', ligand_bond_type)
        ##print('new_ligand_bond_index:', new_ligand_bond_index)

    
        new_edge_type  = torch.cat([new_edge_type, ligand_bond_type], dim = 0)  #扩充配体键类型
        #new_edge_type  = torch.cat([new_edge_type, torch.zeros_like(ligand_bond_type, dtype = torch.int64)], dim = 0)#不扩充配体键类型，依旧使用0
        new_edge_index = torch.cat([new_edge_index, new_ligand_bond_index], dim = 1)

        #

        ##print('new_edge_type:', new_edge_type.shape)
        ##print('new_edge_index:', new_edge_index.shape)
        #new_edge_type: torch.Size([103536])
        #new_edge_index: torch.Size([2, 103536])

        #(edge_type,edge_index)每一行的顺序是无所谓的，不影响卷积，关键在于里面使用的原子id一定要是根据当前批量获得的全局id
        #(edge_type,edge_index)只要配体和蛋白之间的，其余的全部由配体和蛋白的全连接图来取代
        edge_type_dim = F.one_hot(new_edge_type, num_classes=8) #由原来的4种，变成了8种,除了把这里改了成8之外，其它地方也要改吧？否则报错
        #探索一下，哪有哪里在使用键长度
        return edge_type_dim, new_edge_index







    @staticmethod
    def _build_edge_type_20_gpu(edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch):
        #可以全部换成GPU tensor运算，加速训练
        #获取配体的内部的原子id
        ligand_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 1])))).cuda()

    
        #获取配体和蛋白放在一起的原子id
        protein_ligand_node_list = torch.LongTensor(list(range(len(mask_ligand)))).cuda()

        ligand_node_global = protein_ligand_node_list[mask_ligand == 1] #pytorch2.0要求datas[index]，数据和索引都在一个设备上

        ###print('ligand_node_list:', len(ligand_node_local))
        ###print('protein_ligand_node_list:', len(protein_ligand_node_list))
        ###print('ligand_node_global:', len(ligand_node_global))
        ###print('edge_index:', edge_index.shape)

        '''
        ligand_node_list: 282
        protein_ligand_node_list: 3388
        ligand_node_global: 282
        edge_index: torch.Size([2, 108416])
        '''

        #制作id映射
        ligand_node_local2global_dict = {}
        ligand_node_global2local_dict = {}
        for k, v in zip(ligand_node_local, ligand_node_global):
            ##print('K:', k) #tensor(165, device='cuda:0')
            ##print('v:', v) #tensor(1396, device='cuda:0')
            ligand_node_local2global_dict[k.item()] = v
            ligand_node_global2local_dict[v.item()] = k
        

        #更新id
        new_ligand_bond_index = torch.zeros(ligand_bond_index.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(ligand_bond_index.T):
            ##print('bd:', bd) #numpy数据 bd: [0, 9], 如果是torch tensor，这是 tensor([0, 9], device='cuda:0')
            ##print('bd[0]:', bd[0]) #bd[0]: tensor(0, device='cuda:0')
            ##print('bd[1]:', bd[1]) #bd[0]: tensor(9, device='cuda:0')
            ##print('ligand_node_local:', ligand_node_local) #ligand_node_local: tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,18, 19], device='cuda:0')
            #exit()
            ##print('ligand_node_local2global_dict:', ligand_node_local2global_dict)
            new_ligand_bond_index[i][0] = ligand_node_local2global_dict[bd[0].item()] #tensor张量不适合作为字典的k，所以要更换，否则报错
            new_ligand_bond_index[i][1] = ligand_node_local2global_dict[bd[1].item()]
        
        new_ligand_bond_index = new_ligand_bond_index.T
        ###print('new_ligand_bond_index:', new_ligand_bond_index.shape) #torch.Size([2, 582])


        '''
        #蛋白和配体之间的映射，重点像配体一样获取配体蛋白合并之后的id与未合并之前蛋白的id对应关系
        '''
        protein_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 0])))).cuda()
        #获取配体和蛋白放在一起的原子id
        protein_ligand_node_list = torch.LongTensor(list(range(len(mask_ligand)))).cuda()
        protein_node_global = protein_ligand_node_list[mask_ligand == 0] #pytorch2.0要求datas[index]，数据和索引都在一个设备上

        #制作id映射
        protein_node_local2global_dict = {}
        protein_node_global2local_dict = {}
        for k, v in zip(protein_node_local, protein_node_global):
            protein_node_local2global_dict[k.item()] = v
            protein_node_global2local_dict[v.item()] = k




        #raise Exception('test')
        # NxN connectivity matrix where 0 means no connection and 1/2/3/4 means single/double/triple/aromatic bonds.

        #不是键类型，是用来判断蛋白和配体之间的是否有连接
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1 #1表示是配体
        n_dst = mask_ligand[dst] == 1

        #需要扩充配体的键类型，增加单键，双键，三键，芳香键，我们需要知道配体和蛋白的这些键类型有什么作用？？？
        #edge_type在神经网络中只是和边长度嵌入和节点嵌入连接在了一起，并没有什么特殊的判断处理，虽然这里的0,1,2,3的确用来标识节点是蛋白或者配体上
        #但是在神经网络中用没有用来区分配体和蛋白节点
        edge_type[n_src & n_dst]   = 1  #表示在配体内部
        edge_type[n_src & ~n_dst]  = 5  #表示源节点在配体，目标节点在蛋白
        edge_type[~n_src & n_dst]  = 6  #表示源节点在蛋白，目标节点在配体
        edge_type[~n_src & ~n_dst] = 7  #表示在蛋白内部


        indices = (edge_type == 1).nonzero().view(-1) #寻找非0元素下标
        #indices = (edge_type == 1).nonzero().squeeze() #寻找非0元素下标

        # 要删除的行的索引
        rows_to_remove = indices.detach().cpu().tolist()

        # 使用 torch.index_select() 函数选择不删除的行,去掉配体连接
        indices_to_keep = torch.tensor(list(set(range(edge_type.size(0))) - set(rows_to_remove))).cuda()
        new_edge_type   = torch.index_select(edge_type, 0, indices_to_keep)
        new_edge_index  = torch.index_select(edge_index, 1, indices_to_keep)  #2 * N

        #把配体和蛋白的信息分离出来
        '''配体内部信息'''
        only_ligand_bond_type     = ligand_bond_type.clone()
        only_ligand_bond_index    = new_ligand_bond_index.clone()
        only_ligand_edge_type_dim = F.one_hot(only_ligand_bond_type, num_classes=20)

        #更新id，从全局到局部
        #print('only_ligand_bond_index.T.shape:', only_ligand_bond_index.T.shape) #torch.Size([376, 2])
        new_only_ligand_bond_index = torch.zeros(only_ligand_bond_index.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(only_ligand_bond_index.T):
            try:
                new_only_ligand_bond_index[i][0] = ligand_node_global2local_dict[bd[0].item()] #tensor张量不适合作为字典的k，所以要更换，否则报错
                new_only_ligand_bond_index[i][1] = ligand_node_global2local_dict[bd[1].item()]
            except Exception as e:
                #print('i:', i)
                #print('bd', bd)
                #print('error:', e)
                raise Exception(ligand_node_global2local_dict)
            
        new_only_ligand_bond_index = new_only_ligand_bond_index.T

        '''获取蛋白内部信息'''
        indices1 = (edge_type == 7).nonzero().view(-1) #寻找非0元素下标
        # 使用 torch.index_select() 函数选择不删除的行,去掉配体连接
        indices_to_keep = indices1
        only_protein_bond_type      = torch.index_select(edge_type, 0, indices_to_keep)
        only_protein_bond_index     = torch.index_select(edge_index, 1, indices_to_keep)  #2 * N
        only_protein_edge_type_dim  = F.one_hot(only_protein_bond_type, num_classes=20)

        #更新id，从全局到局部
        new_only_protein_bond_index = torch.zeros(only_protein_bond_index.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(only_protein_bond_index.T):
            try:
                new_only_protein_bond_index[i][0] = protein_node_global2local_dict[bd[0].item()] #tensor张量不适合作为字典的k，所以要更换，否则报错
                new_only_protein_bond_index[i][1] = protein_node_global2local_dict[bd[1].item()]
            except Exception as e:
                #print('i:', i)
                #print('bd', bd)
                #print('error:', e)
                raise Exception(protein_node_global2local_dict)
        
        new_only_protein_bond_index = new_only_protein_bond_index.T


        #添加配体内部连接
        ###print('ligand_bond_type:', ligand_bond_type)
        ###print('new_ligand_bond_index:', new_ligand_bond_index)

    
        new_edge_type  = torch.cat([new_edge_type, ligand_bond_type], dim = 0)  #扩充配体键类型
        #new_edge_type  = torch.cat([new_edge_type, torch.zeros_like(ligand_bond_type, dtype = torch.int64)], dim = 0)#不扩充配体键类型，依旧使用0
        new_edge_index = torch.cat([new_edge_index, new_ligand_bond_index], dim = 1)

        #

        ###print('new_edge_type:', new_edge_type.shape)
        ###print('new_edge_index:', new_edge_index.shape)
        #new_edge_type: torch.Size([103536])
        #new_edge_index: torch.Size([2, 103536])

        #(edge_type,edge_index)每一行的顺序是无所谓的，不影响卷积，关键在于里面使用的原子id一定要是根据当前批量获得的全局id
        #(edge_type,edge_index)只要配体和蛋白之间的，其余的全部由配体和蛋白的全连接图来取代
        edge_type_dim = F.one_hot(new_edge_type, num_classes=20) #由原来的4种，变成了8种,除了把这里改了成8之外，其它地方也要改吧？否则报错
        #探索一下，哪有哪里在使用键长度
        return edge_type_dim, new_edge_index, ligand_node_global2local_dict, protein_node_global2local_dict, only_ligand_edge_type_dim, new_only_ligand_bond_index, only_protein_edge_type_dim, new_only_protein_bond_index






    # @staticmethod #当需要self时，把静态函数声明去掉，否则会把self当作普通参数，而不是类对象
    def _build_edge_type_interaction_20_gpu(self, x, edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch,
        batch, #这个参数要结合mask_liagnd来确定哪些是配体哪些是蛋白上的节点
        atom_isring,
        atom_isO,
        atom_isN
        ):
        #可以全部换成GPU tensor运算，加速训练
        #获取配体的内部的原子id
        #把配体和蛋白的之间的连接替换成可能的相互作用
        ligand_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 1])))).cuda()

    
        #获取配体和蛋白放在一起的原子id
        protein_ligand_node_list = torch.LongTensor(list(range(len(mask_ligand)))).cuda()

        ligand_node_global = protein_ligand_node_list[mask_ligand == 1] #pytorch2.0要求datas[index]，数据和索引都在一个设备上

        ##print('ligand_node_list:', len(ligand_node_local))
        ##print('protein_ligand_node_list:', len(protein_ligand_node_list))
        ##print('ligand_node_global:', len(ligand_node_global))
        ##print('edge_index:', edge_index.shape)

        '''
        ligand_node_list: 282
        protein_ligand_node_list: 3388
        ligand_node_global: 282
        edge_index: torch.Size([2, 108416])
        '''

        #制作id映射
        ligand_node_local2global_dict = {}
        for k, v in zip(ligand_node_local, ligand_node_global):
            ligand_node_local2global_dict[k.item()] = v
        

        #更新id
        new_ligand_bond_index = torch.zeros(ligand_bond_index.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(ligand_bond_index.T):
            #print('bd:', bd) #numpy数据 bd: [0, 9], 如果是torch tensor，这是 tensor([0, 9], device='cuda:0')
            #print('bd[0]:', bd[0]) #bd[0]: tensor(0, device='cuda:0')
            #print('bd[1]:', bd[1]) #bd[0]: tensor(9, device='cuda:0')
            #print('ligand_node_local:', ligand_node_local) #ligand_node_local: tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,18, 19], device='cuda:0')
            #exit()
            #print('ligand_node_local2global_dict:', ligand_node_local2global_dict)
            new_ligand_bond_index[i][0] = ligand_node_local2global_dict[bd[0].item()] #tensor张量不适合作为字典的k，所以要更换，否则报错
            new_ligand_bond_index[i][1] = ligand_node_local2global_dict[bd[1].item()]
        
        new_ligand_bond_index = new_ligand_bond_index.T
        ##print('new_ligand_bond_index:', new_ligand_bond_index.shape) #torch.Size([2, 582])


        '''
        #蛋白和配体之间的映射，重点像配体一样获取配体蛋白合并之后的id与未合并之前蛋白的id对应关系
        '''
        protein_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 0])))).cuda()

    
        #获取配体和蛋白放在一起的原子id
        protein_ligand_node_list = torch.LongTensor(list(range(len(mask_ligand)))).cuda()
        protein_node_global = protein_ligand_node_list[mask_ligand == 0] #pytorch2.0要求datas[index]，数据和索引都在一个设备上


        #制作id映射
        protein_node_local2global_dict = {}
        for k, v in zip(protein_node_local, protein_node_global):
            protein_node_local2global_dict[k.item()] = v
        

        #根据原子掩码构建配体和蛋白之间的连接
        '''
        ligand_atom_isring
        ligand_atom_isO
        ligand_atom_isN

        protein_atom_isring
        protein_atom_isO
        protein_atom_isN
        '''
        l2p = [] #
        p2l = [] #
        #torch.unique(tensor, dim=0)
        #print('max(batch):', max(batch))
        #raise Exception('test0')

        for b in range(max(batch) + 1): #遍历每一个图, 没有循环？
            ligand_atom  = protein_ligand_node_list[batch == b][mask_ligand[batch == b] == 1] #先取每一个图蛋白-配体，然而再根据掩码找哪些是蛋白，哪些是配体
            protein_atom = protein_ligand_node_list[batch == b][mask_ligand[batch == b] == 0]

            #print('ligand_atom:', ligand_atom.shape)
            #print('protein_atom:', protein_atom.shape)
            #print('type(atom_isring):', atom_isring.dtype)
            #print('atom_isring:', atom_isring.shape)
            #print('atom_isO:', atom_isO.shape)
            #print('atom_isN:', atom_isN.shape)

            #找到对应的原子掩码
            ligand_atom_isring = atom_isring[batch == b][mask_ligand[batch == b] == 1]
            ligand_atom_isO    = atom_isO[batch == b][mask_ligand[batch == b] == 1]
            ligand_atom_isN    = atom_isN[batch == b][mask_ligand[batch == b] == 1]

            protein_atom_isring = atom_isring[batch == b][mask_ligand[batch == b] == 0]
            protein_atom_isO = atom_isO[batch == b][mask_ligand[batch == b] == 0]
            protein_atom_isN = atom_isN[batch == b][mask_ligand[batch == b] == 0]

            #sub_x = x[batch == b][mask_ligand[batch == b] == 0] #传递这个参数也行，所有x也行，因为构建的边是基于全局索引的

            #print('ligand_atom_isring:', ligand_atom_isring.shape)
            #print('ligand_atom_isO:', ligand_atom_isO.shape)
            #print('ligand_atom_isN:', ligand_atom_isN.shape)
            #print('protein_atom_isring:', protein_atom_isring.shape)
            #print('protein_atom_isO:', protein_atom_isO.shape)
            #print('protein_atom_isN:', protein_atom_isN.shape)
            #构建[2, n*m,]的连接矩阵
            #配体到蛋白
            l_combinations_isring = self.combinations(x, ligand_atom, protein_atom, ligand_atom_isring, protein_atom_isring).cuda()
            l_combinations_isO    = self.combinations(x, ligand_atom, protein_atom, ligand_atom_isO, protein_atom_isN).cuda()
            l_combinations_isN    = self.combinations(x, ligand_atom, protein_atom, ligand_atom_isN, protein_atom_isO).cuda()

            l2p.append(l_combinations_isring)
            l2p.append(l_combinations_isO)
            l2p.append(l_combinations_isN)

            #print('l_combinations_isring:', l_combinations_isring.shape)
            #print('l_combinations_isO:', l_combinations_isO.shape)
            #print('l_combinations_isN:', l_combinations_isN.shape)

            #蛋白到配体
            p_combinations_isring = self.combinations(x, protein_atom, ligand_atom, protein_atom_isring, ligand_atom_isring).cuda()
            p_combinations_isO    = self.combinations(x, protein_atom, ligand_atom, protein_atom_isN, ligand_atom_isO).cuda()
            p_combinations_isN    = self.combinations(x, protein_atom, ligand_atom, protein_atom_isO, ligand_atom_isN).cuda()

            p2l.append(p_combinations_isring)
            p2l.append(p_combinations_isO)
            p2l.append(p_combinations_isN)

            #print('p_combinations_isring:', p_combinations_isring.shape)
            #print('p_combinations_isO:', p_combinations_isO.shape)
            #print('p_combinations_isN:', p_combinations_isN.shape)

            

        l2p_edge_index = torch.unique(torch.cat(l2p, dim = -1), dim = -1)    
        p2l_edge_index = torch.unique(torch.cat(p2l, dim = -1), dim = -1) 


        #print('l2p[:, :3]:', l2p_edge_index[:, :3]) #[]
        #print('p2l[:, :3]:', p2l_edge_index[:, :3]) #[]

        #print('l2p:', l2p_edge_index.shape) #[]
        #print('p2l:', p2l_edge_index.shape) #[]

        #raise Exception('test')

        '''
        ligand_atom: torch.Size([34])
        protein_atom: torch.Size([400])
        type(atom_isring): torch.bool
        atom_isring: torch.Size([434])
        atom_isO: torch.Size([434])
        atom_isN: torch.Size([434])
        ligand_atom_isring: torch.Size([34])
        ligand_atom_isO: torch.Size([34])
        ligand_atom_isN: torch.Size([34])
        protein_atom_isring: torch.Size([400])
        protein_atom_isO: torch.Size([400])
        protein_atom_isN: torch.Size([400])
        l_combinations_isring: torch.Size([2, 0])
        l_combinations_isO: torch.Size([2, 364])
        l_combinations_isN: torch.Size([2, 324])
        p_combinations_isring: torch.Size([2, 0])
        p_combinations_isO: torch.Size([2, 364])
        p_combinations_isN: torch.Size([2, 324])
        l2p[:, :3]: tensor([[400, 400, 400],
                [  0,   3,   4]], device='cuda:0')
        p2l[:, :3]: tensor([[  0,   0,   0],
                [400, 405, 412]], device='cuda:0')
        l2p: torch.Size([2, 688])
        p2l: torch.Size([2, 688])
        '''


        '''
        #如果不想使用knn来构建蛋白内部的连接,而是使用自定义的(即蛋白自身的原有连接),则可以使用下面的代码像配体一样更新id
        #另外，如果不提供蛋白的键类型,则在更新edge_type时,要把键类型填充设置设置成7
        #更新id
        new_protein_bond_index = torch.zeros(protein_bond_index.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(protein_bond_index.T):
            new_protein_bond_index[i][0] = protein_node_local2global_dict[bd[0].item()] #tensor张量不适合作为字典的k，所以要更换，否则报错
            new_protein_bond_index[i][1] = protein_node_local2global_dict[bd[1].item()]
        
        new_protein_bond_index = new_protein_bond_index.T
        '''
        





        #raise Exception('test')
        # NxN connectivity matrix where 0 means no connection and 1/2/3/4 means single/double/triple/aromatic bonds.

        #不是键类型，是用来判断蛋白和配体之间的是否有连接
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1 #1表示是配体
        n_dst = mask_ligand[dst] == 1

        #需要扩充配体的键类型，增加单键，双键，三键，芳香键，我们需要知道配体和蛋白的这些键类型有什么作用？？？
        #edge_type在神经网络中只是和边长度嵌入和节点嵌入连接在了一起，并没有什么特殊的判断处理，虽然这里的0,1,2,3的确用来标识节点是蛋白或者配体上
        #但是在神经网络中用没有用来区分配体和蛋白节点
        edge_type[n_src & n_dst]   = 1  #表示在配体内部
        edge_type[n_src & ~n_dst]  = 5  #表示源节点在配体，目标节点在蛋白
        edge_type[~n_src & n_dst]  = 6  #表示源节点在蛋白，目标节点在配体
        edge_type[~n_src & ~n_dst] = 7  #表示在蛋白内部


        indices1 = (edge_type == 1).nonzero().view(-1) #寻找非0元素下标
        indices5 = (edge_type == 5).nonzero().view(-1) #寻找非0元素下标
        indices6 = (edge_type == 6).nonzero().view(-1) #寻找非0元素下标
        indices = torch.cat([indices1, indices5, indices6])
        #indices = (edge_type == 1).nonzero().squeeze() #寻找非0元素下标

        # 要删除的行的索引
        rows_to_remove = indices.detach().cpu().tolist()

        # 使用 torch.index_select() 函数选择不删除的行,去掉配体连接
        indices_to_keep = torch.tensor(list(set(range(edge_type.size(0))) - set(rows_to_remove))).cuda()
        new_edge_type   = torch.index_select(edge_type, 0, indices_to_keep)
        new_edge_index  = torch.index_select(edge_index, 1, indices_to_keep)  #2 * N

        #添加配体内部连接
        ##print('ligand_bond_type:', ligand_bond_type)
        ##print('new_ligand_bond_index:', new_ligand_bond_index)

        l2p_edge_type = torch.full([l2p_edge_index.size(1)], 5, dtype = torch.int64).cuda()
        p2l_edge_type = torch.full([p2l_edge_index.size(1)], 6, dtype = torch.int64).cuda()

        new_edge_type  = torch.cat([new_edge_type, ligand_bond_type, l2p_edge_type, p2l_edge_type], dim = 0)  #扩充配体键类型
        #new_edge_type  = torch.cat([new_edge_type, torch.zeros_like(ligand_bond_type, dtype = torch.int64)], dim = 0)#不扩充配体键类型，依旧使用0
        new_edge_index = torch.cat([new_edge_index, new_ligand_bond_index, l2p_edge_index, p2l_edge_index], dim = 1)

        #

        ##print('new_edge_type:', new_edge_type.shape)
        ##print('new_edge_index:', new_edge_index.shape)
        #new_edge_type: torch.Size([103536])
        #new_edge_index: torch.Size([2, 103536])

        #(edge_type,edge_index)每一行的顺序是无所谓的，不影响卷积，关键在于里面使用的原子id一定要是根据当前批量获得的全局id
        #(edge_type,edge_index)只要配体和蛋白之间的，其余的全部由配体和蛋白的全连接图来取代
        edge_type_dim = F.one_hot(new_edge_type, num_classes=20) #由原来的4种，变成了8种,除了把这里改了成8之外，其它地方也要改吧？否则报错
        #探索一下，哪有哪里在使用键长度
        return edge_type_dim, new_edge_index






    # @staticmethod #当需要self时，把静态函数声明去掉，否则会把self当作普通参数，而不是类对象
    def _build_edge_type_interaction_8_gpu(self, x, edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch,
        batch, #这个参数要结合mask_liagnd来确定哪些是配体哪些是蛋白上的节点
        atom_isring,
        atom_isO,
        atom_isN
        ):
        #可以全部换成GPU tensor运算，加速训练
        #获取配体的内部的原子id
        #把配体和蛋白的之间的连接替换成可能的相互作用
        ligand_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 1])))).cuda()

    
        #获取配体和蛋白放在一起的原子id
        protein_ligand_node_list = torch.LongTensor(list(range(len(mask_ligand)))).cuda()

        ligand_node_global = protein_ligand_node_list[mask_ligand == 1] #pytorch2.0要求datas[index]，数据和索引都在一个设备上

        ##print('ligand_node_list:', len(ligand_node_local))
        ##print('protein_ligand_node_list:', len(protein_ligand_node_list))
        ##print('ligand_node_global:', len(ligand_node_global))
        ##print('edge_index:', edge_index.shape)

        '''
        ligand_node_list: 282
        protein_ligand_node_list: 3388
        ligand_node_global: 282
        edge_index: torch.Size([2, 108416])
        '''

        #制作id映射
        ligand_node_local2global_dict = {}
        for k, v in zip(ligand_node_local, ligand_node_global):
            ligand_node_local2global_dict[k.item()] = v
        

        #更新id
        new_ligand_bond_index = torch.zeros(ligand_bond_index.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(ligand_bond_index.T):
            #print('bd:', bd) #numpy数据 bd: [0, 9], 如果是torch tensor，这是 tensor([0, 9], device='cuda:0')
            #print('bd[0]:', bd[0]) #bd[0]: tensor(0, device='cuda:0')
            #print('bd[1]:', bd[1]) #bd[0]: tensor(9, device='cuda:0')
            #print('ligand_node_local:', ligand_node_local) #ligand_node_local: tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,18, 19], device='cuda:0')
            #exit()
            #print('ligand_node_local2global_dict:', ligand_node_local2global_dict)
            new_ligand_bond_index[i][0] = ligand_node_local2global_dict[bd[0].item()] #tensor张量不适合作为字典的k，所以要更换，否则报错
            new_ligand_bond_index[i][1] = ligand_node_local2global_dict[bd[1].item()]
        
        new_ligand_bond_index = new_ligand_bond_index.T
        ##print('new_ligand_bond_index:', new_ligand_bond_index.shape) #torch.Size([2, 582])


        '''
        #蛋白和配体之间的映射，重点像配体一样获取配体蛋白合并之后的id与未合并之前蛋白的id对应关系
        '''
        protein_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 0])))).cuda()

    
        #获取配体和蛋白放在一起的原子id
        protein_ligand_node_list = torch.LongTensor(list(range(len(mask_ligand)))).cuda()
        protein_node_global = protein_ligand_node_list[mask_ligand == 0] #pytorch2.0要求datas[index]，数据和索引都在一个设备上


        #制作id映射
        protein_node_local2global_dict = {}
        for k, v in zip(protein_node_local, protein_node_global):
            protein_node_local2global_dict[k.item()] = v
        

        #根据原子掩码构建配体和蛋白之间的连接
        '''
        ligand_atom_isring
        ligand_atom_isO
        ligand_atom_isN

        protein_atom_isring
        protein_atom_isO
        protein_atom_isN
        '''
        l2p = [] #
        p2l = [] #
        #torch.unique(tensor, dim=0)
        #print('max(batch):', max(batch))
        #raise Exception('test0')

        for b in range(max(batch) + 1): #遍历每一个图, 没有循环？
            ligand_atom  = protein_ligand_node_list[batch == b][mask_ligand[batch == b] == 1] #先取每一个图蛋白-配体，然而再根据掩码找哪些是蛋白，哪些是配体
            protein_atom = protein_ligand_node_list[batch == b][mask_ligand[batch == b] == 0]

            #print('ligand_atom:', ligand_atom.shape)
            #print('protein_atom:', protein_atom.shape)
            #print('type(atom_isring):', atom_isring.dtype)
            #print('atom_isring:', atom_isring.shape)
            #print('atom_isO:', atom_isO.shape)
            #print('atom_isN:', atom_isN.shape)

            #找到对应的原子掩码
            ligand_atom_isring = atom_isring[batch == b][mask_ligand[batch == b] == 1]
            ligand_atom_isO    = atom_isO[batch == b][mask_ligand[batch == b] == 1]
            ligand_atom_isN    = atom_isN[batch == b][mask_ligand[batch == b] == 1]

            protein_atom_isring = atom_isring[batch == b][mask_ligand[batch == b] == 0]
            protein_atom_isO = atom_isO[batch == b][mask_ligand[batch == b] == 0]
            protein_atom_isN = atom_isN[batch == b][mask_ligand[batch == b] == 0]

            #sub_x = x[batch == b][mask_ligand[batch == b] == 0] #传递这个参数也行，所有x也行，因为构建的边是基于全局索引的

            #print('ligand_atom_isring:', ligand_atom_isring.shape)
            #print('ligand_atom_isO:', ligand_atom_isO.shape)
            #print('ligand_atom_isN:', ligand_atom_isN.shape)
            #print('protein_atom_isring:', protein_atom_isring.shape)
            #print('protein_atom_isO:', protein_atom_isO.shape)
            #print('protein_atom_isN:', protein_atom_isN.shape)
            #构建[2, n*m,]的连接矩阵
            #配体到蛋白
            l_combinations_isring = self.combinations(x, ligand_atom, protein_atom, ligand_atom_isring, protein_atom_isring).cuda()
            l_combinations_isO    = self.combinations(x, ligand_atom, protein_atom, ligand_atom_isO, protein_atom_isN).cuda()
            l_combinations_isN    = self.combinations(x, ligand_atom, protein_atom, ligand_atom_isN, protein_atom_isO).cuda()

            l2p.append(l_combinations_isring)
            l2p.append(l_combinations_isO)
            l2p.append(l_combinations_isN)

            #print('l_combinations_isring:', l_combinations_isring.shape)
            #print('l_combinations_isO:', l_combinations_isO.shape)
            #print('l_combinations_isN:', l_combinations_isN.shape)

            #蛋白到配体
            p_combinations_isring = self.combinations(x, protein_atom, ligand_atom, protein_atom_isring, ligand_atom_isring).cuda()
            p_combinations_isO    = self.combinations(x, protein_atom, ligand_atom, protein_atom_isN, ligand_atom_isO).cuda()
            p_combinations_isN    = self.combinations(x, protein_atom, ligand_atom, protein_atom_isO, ligand_atom_isN).cuda()

            p2l.append(p_combinations_isring)
            p2l.append(p_combinations_isO)
            p2l.append(p_combinations_isN)

            #print('p_combinations_isring:', p_combinations_isring.shape)
            #print('p_combinations_isO:', p_combinations_isO.shape)
            #print('p_combinations_isN:', p_combinations_isN.shape)

            

        l2p_edge_index = torch.unique(torch.cat(l2p, dim = -1), dim = -1)    
        p2l_edge_index = torch.unique(torch.cat(p2l, dim = -1), dim = -1) 


        #print('l2p[:, :3]:', l2p_edge_index[:, :3]) #[]
        #print('p2l[:, :3]:', p2l_edge_index[:, :3]) #[]

        #print('l2p:', l2p_edge_index.shape) #[]
        #print('p2l:', p2l_edge_index.shape) #[]

        #raise Exception('test')

        '''
        ligand_atom: torch.Size([34])
        protein_atom: torch.Size([400])
        type(atom_isring): torch.bool
        atom_isring: torch.Size([434])
        atom_isO: torch.Size([434])
        atom_isN: torch.Size([434])
        ligand_atom_isring: torch.Size([34])
        ligand_atom_isO: torch.Size([34])
        ligand_atom_isN: torch.Size([34])
        protein_atom_isring: torch.Size([400])
        protein_atom_isO: torch.Size([400])
        protein_atom_isN: torch.Size([400])
        l_combinations_isring: torch.Size([2, 0])
        l_combinations_isO: torch.Size([2, 364])
        l_combinations_isN: torch.Size([2, 324])
        p_combinations_isring: torch.Size([2, 0])
        p_combinations_isO: torch.Size([2, 364])
        p_combinations_isN: torch.Size([2, 324])
        l2p[:, :3]: tensor([[400, 400, 400],
                [  0,   3,   4]], device='cuda:0')
        p2l[:, :3]: tensor([[  0,   0,   0],
                [400, 405, 412]], device='cuda:0')
        l2p: torch.Size([2, 688])
        p2l: torch.Size([2, 688])
        '''


        '''
        #如果不想使用knn来构建蛋白内部的连接,而是使用自定义的(即蛋白自身的原有连接),则可以使用下面的代码像配体一样更新id
        #另外，如果不提供蛋白的键类型,则在更新edge_type时,要把键类型填充设置设置成7
        #更新id
        new_protein_bond_index = torch.zeros(protein_bond_index.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(protein_bond_index.T):
            new_protein_bond_index[i][0] = protein_node_local2global_dict[bd[0].item()] #tensor张量不适合作为字典的k，所以要更换，否则报错
            new_protein_bond_index[i][1] = protein_node_local2global_dict[bd[1].item()]
        
        new_protein_bond_index = new_protein_bond_index.T
        '''
        





        #raise Exception('test')
        # NxN connectivity matrix where 0 means no connection and 1/2/3/4 means single/double/triple/aromatic bonds.

        #不是键类型，是用来判断蛋白和配体之间的是否有连接
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1 #1表示是配体
        n_dst = mask_ligand[dst] == 1

        #需要扩充配体的键类型，增加单键，双键，三键，芳香键，我们需要知道配体和蛋白的这些键类型有什么作用？？？
        #edge_type在神经网络中只是和边长度嵌入和节点嵌入连接在了一起，并没有什么特殊的判断处理，虽然这里的0,1,2,3的确用来标识节点是蛋白或者配体上
        #但是在神经网络中用没有用来区分配体和蛋白节点
        edge_type[n_src & n_dst]   = 1  #表示在配体内部
        edge_type[n_src & ~n_dst]  = 5  #表示源节点在配体，目标节点在蛋白
        edge_type[~n_src & n_dst]  = 6  #表示源节点在蛋白，目标节点在配体
        edge_type[~n_src & ~n_dst] = 7  #表示在蛋白内部


        indices1 = (edge_type == 1).nonzero().view(-1) #寻找非0元素下标
        indices5 = (edge_type == 5).nonzero().view(-1) #寻找非0元素下标
        indices6 = (edge_type == 6).nonzero().view(-1) #寻找非0元素下标
        indices = torch.cat([indices1, indices5, indices6])
        #indices = (edge_type == 1).nonzero().squeeze() #寻找非0元素下标

        # 要删除的行的索引
        rows_to_remove = indices.detach().cpu().tolist()

        # 使用 torch.index_select() 函数选择不删除的行,去掉配体连接
        indices_to_keep = torch.tensor(list(set(range(edge_type.size(0))) - set(rows_to_remove))).cuda()
        new_edge_type   = torch.index_select(edge_type, 0, indices_to_keep)
        new_edge_index  = torch.index_select(edge_index, 1, indices_to_keep)  #2 * N

        #添加配体内部连接
        ##print('ligand_bond_type:', ligand_bond_type)
        ##print('new_ligand_bond_index:', new_ligand_bond_index)

        l2p_edge_type = torch.full([l2p_edge_index.size(1)], 5, dtype = torch.int64).cuda()
        p2l_edge_type = torch.full([p2l_edge_index.size(1)], 6, dtype = torch.int64).cuda()

        new_edge_type  = torch.cat([new_edge_type, ligand_bond_type, l2p_edge_type, p2l_edge_type], dim = 0)  #扩充配体键类型
        #new_edge_type  = torch.cat([new_edge_type, torch.zeros_like(ligand_bond_type, dtype = torch.int64)], dim = 0)#不扩充配体键类型，依旧使用0
        new_edge_index = torch.cat([new_edge_index, new_ligand_bond_index, l2p_edge_index, p2l_edge_index], dim = 1)

        #

        ##print('new_edge_type:', new_edge_type.shape)
        ##print('new_edge_index:', new_edge_index.shape)
        #new_edge_type: torch.Size([103536])
        #new_edge_index: torch.Size([2, 103536])

        #(edge_type,edge_index)每一行的顺序是无所谓的，不影响卷积，关键在于里面使用的原子id一定要是根据当前批量获得的全局id
        #(edge_type,edge_index)只要配体和蛋白之间的，其余的全部由配体和蛋白的全连接图来取代
        edge_type_dim = F.one_hot(new_edge_type, num_classes=8) #由原来的4种，变成了8种,除了把这里改了成8之外，其它地方也要改吧？否则报错
        #探索一下，哪有哪里在使用键长度
        return edge_type_dim, new_edge_index



    def combinations(self, x, atom1, atom2, atom_index1, atom_index2):
            #将两个向量，两两组合一起
            vec1 = atom1[atom_index1]
            vec2 = atom2[atom_index2]
            # 使用 torch.meshgrid 构建两个向量的两两组合
            grid_x, grid_y = torch.meshgrid(vec1, vec2, indexing='ij')
            # 将组合的结果转换为两列的二维张量
            combination = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0) #shape = [2, n*m]
            copy_combination = combination.clone()
            
            #小于6ai的边, 约束太苛刻了，难有满足条件的,换大点的
            dis_limit = GP.atom2atom_distance

            dis = torch.norm(x[combination[0]] - x[combination[1]], p = 2, dim = -1) #把整数变成浮点数
            dis_index = dis <= dis_limit #找满足条件的

            combination = combination.t()[dis_index]

            '''
            if len(combination) == 0:
                #print('使用更大范围的距离限制12')
                dis_limit = 12.0 #如果是空，则使用更大范围的信息
                combination = copy_combination.clone()
                dis = torch.norm(x[combination[0]] - x[combination[1]], p = 2, dim = -1) #把整数变成浮点数
                dis_index = dis <= dis_limit #找满足条件的

                combination = combination.t()[dis_index]
            '''
            
            if len(combination) == 0:
                #print('放开距离限制')
                dis_limit = 100000000.0 #如果还是空，则放开限制
                combination = copy_combination.clone()
                dis = torch.norm(x[combination[0]] - x[combination[1]], p = 2, dim = -1) #把整数变成浮点数
                dis_index = dis <= dis_limit #找满足条件的

                combination = combination.t()[dis_index]

            '''
            #只要小于6ai的边, 约束太苛刻了，难有满足条件的。如果我们直接取邻近的60个原子，但现在知道边，有没法直接矩阵运算，实现不了
            nun_limit = 60

            dis = torch.norm(combination.to(torch.float32), p = 2, dim = 1) #把整数变成浮点数
            dis_index = dis <= dis_limit #找满足添加的

            combination = combination[dis_index].t()
            '''

            return combination.t()




    # @staticmethod #当需要self时，把静态函数声明去掉，否则会把self当作普通参数，而不是类对象
    def _build_edge_type_interaction_8_gpu_optim(self, x, org_x, edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch,
        batch, #这个参数要结合mask_liagnd来确定哪些是配体哪些是蛋白上的节点
        atom_isring,
        atom_isO,
        atom_isN,

        cross_isring_flag, 
        cross_isO_flag, 
        cross_isN_flag, 
        cross_lp_pos,
        cross_distance,
        ):
        #可以全部换成GPU tensor运算，加速训练
        #获取配体的内部的原子id
        #把配体和蛋白的之间的连接替换成可能的相互作用
        ligand_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 1])))).cuda()

    
        #获取配体和蛋白放在一起的原子id
        protein_ligand_node_list = torch.LongTensor(list(range(len(mask_ligand)))).cuda()

        ligand_node_global = protein_ligand_node_list[mask_ligand == 1] #pytorch2.0要求datas[index]，数据和索引都在一个设备上

        ##print('ligand_node_list:', len(ligand_node_local))
        ##print('protein_ligand_node_list:', len(protein_ligand_node_list))
        ##print('ligand_node_global:', len(ligand_node_global))
        ##print('edge_index:', edge_index.shape)

        '''
        ligand_node_list: 282
        protein_ligand_node_list: 3388
        ligand_node_global: 282
        edge_index: torch.Size([2, 108416])
        '''

        #制作id映射
        ligand_node_local2global_dict = {}
        for k, v in zip(ligand_node_local, ligand_node_global):
            ligand_node_local2global_dict[k.item()] = v
        

        #更新id
        new_ligand_bond_index = torch.zeros(ligand_bond_index.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(ligand_bond_index.T):
            #print('bd:', bd) #numpy数据 bd: [0, 9], 如果是torch tensor，这是 tensor([0, 9], device='cuda:0')
            #print('bd[0]:', bd[0]) #bd[0]: tensor(0, device='cuda:0')
            #print('bd[1]:', bd[1]) #bd[0]: tensor(9, device='cuda:0')
            #print('ligand_node_local:', ligand_node_local) #ligand_node_local: tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,18, 19], device='cuda:0')
            #exit()
            #print('ligand_node_local2global_dict:', ligand_node_local2global_dict)
            new_ligand_bond_index[i][0] = ligand_node_local2global_dict[bd[0].item()] #tensor张量不适合作为字典的k，所以要更换，否则报错
            new_ligand_bond_index[i][1] = ligand_node_local2global_dict[bd[1].item()]
        
        new_ligand_bond_index = new_ligand_bond_index.T
        ##print('new_ligand_bond_index:', new_ligand_bond_index.shape) #torch.Size([2, 582])


        '''
        #蛋白和配体之间的映射，重点像配体一样获取配体蛋白合并之后的id与未合并之前蛋白的id对应关系
        '''
        protein_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 0])))).cuda()

    
        #获取配体和蛋白放在一起的原子id
        protein_ligand_node_list = torch.LongTensor(list(range(len(mask_ligand)))).cuda()
        protein_node_global = protein_ligand_node_list[mask_ligand == 0] #pytorch2.0要求datas[index]，数据和索引都在一个设备上


        #制作id映射
        protein_node_local2global_dict = {}
        for k, v in zip(protein_node_local, protein_node_global):
            protein_node_local2global_dict[k.item()] = v
        

        #根据原子掩码构建配体和蛋白之间的连接
        '''
        ligand_atom_isring
        ligand_atom_isO
        ligand_atom_isN

        protein_atom_isring
        protein_atom_isO
        protein_atom_isN
        '''
        l2p = [] #
        p2l = [] #
        #torch.unique(tensor, dim=0)
        #print('max(batch):', max(batch))
        #raise Exception('test0')

        for b in range(max(batch) + 1): #遍历每一个图, 没有循环？
            ligand_atom  = protein_ligand_node_list[batch == b][mask_ligand[batch == b] == 1] #先取每一个图蛋白-配体，然而再根据掩码找哪些是蛋白，哪些是配体
            protein_atom = protein_ligand_node_list[batch == b][mask_ligand[batch == b] == 0]

            #print('ligand_atom:', ligand_atom.shape)
            #print('protein_atom:', protein_atom.shape)
            #print('type(atom_isring):', atom_isring.dtype)
            #print('atom_isring:', atom_isring.shape)
            #print('atom_isO:', atom_isO.shape)
            #print('atom_isN:', atom_isN.shape)

            #找到对应的原子掩码
            ligand_atom_isring = atom_isring[batch == b][mask_ligand[batch == b] == 1]
            ligand_atom_isO    = atom_isO[batch == b][mask_ligand[batch == b] == 1]
            ligand_atom_isN    = atom_isN[batch == b][mask_ligand[batch == b] == 1]

            protein_atom_isring = atom_isring[batch == b][mask_ligand[batch == b] == 0]
            protein_atom_isO = atom_isO[batch == b][mask_ligand[batch == b] == 0]
            protein_atom_isN = atom_isN[batch == b][mask_ligand[batch == b] == 0]


            ligand_cross_isring_flag = cross_isring_flag[batch == b][mask_ligand[batch == b] == 1]
            ligand_cross_isO_flag    = cross_isO_flag[batch == b][mask_ligand[batch == b] == 1]
            ligand_cross_isN_flag    = cross_isN_flag[batch == b][mask_ligand[batch == b] == 1]
            ligand_cross_lp_pos      = cross_lp_pos[batch == b][mask_ligand[batch == b] == 1]

            protein_cross_isring_flag = cross_isring_flag[batch == b][mask_ligand[batch == b] == 0]
            protein_cross_isO_flag    = cross_isO_flag[batch == b][mask_ligand[batch == b] == 0] 
            protein_cross_isN_flag    = cross_isN_flag[batch == b][mask_ligand[batch == b] == 0] 
            protein_cross_lp_pos      = cross_lp_pos[batch == b][mask_ligand[batch == b] == 0]

            #print('protein_cross_lp_pos:', protein_cross_lp_pos.shape)
            #print('cross_distance:', cross_distance.shape)
            #print('batch == b:', (batch == b).shape)
            #protein_cross_lp_pos: torch.Size([125, 3])                                                                                                                                           | 0/40 [00:00<?, ?it/s]
            #cross_distance: torch.Size([13, 125])
            #batch == b: torch.Size([138])       
            cross_distance_matrix = cross_distance[b] #cross_distance应该是一个list，因为cross_distance每一个分子的矩阵形状都不一样
            #print('cross_distance.shape:', cross_distance.shape)
            #cross_distance_matrix = cross_distance[batch == b]



            #sub_x = x[batch == b][mask_ligand[batch == b] == 0] #传递这个参数也行，所有x也行，因为构建的边是基于全局索引的

            #print('ligand_atom_isring:', ligand_atom_isring.shape)
            #print('ligand_atom_isO:', ligand_atom_isO.shape)
            #print('ligand_atom_isN:', ligand_atom_isN.shape)
            #print('protein_atom_isring:', protein_atom_isring.shape)
            #print('protein_atom_isO:', protein_atom_isO.shape)
            #print('protein_atom_isN:', protein_atom_isN.shape)
            #构建[2, n*m,]的连接矩阵
            #配体到蛋白
            centor = x[ligand_atom].mean(dim = 0)
            l_combinations_isring = self.combinations_optim(x, org_x, ligand_atom, protein_atom, ligand_atom_isring, protein_atom_isring, centor, 
                    ligand_cross_isring_flag, protein_cross_isring_flag, cross_distance_matrix, ligand_cross_lp_pos, protein_cross_lp_pos, flag = 'ligand')
            l_combinations_isO    = self.combinations_optim(x, org_x, ligand_atom, protein_atom, ligand_atom_isO, protein_atom_isN, centor, 
                    ligand_cross_isO_flag, protein_cross_isN_flag, cross_distance_matrix, ligand_cross_lp_pos, protein_cross_lp_pos, flag = 'ligand')
            l_combinations_isN    = self.combinations_optim(x, org_x, ligand_atom, protein_atom, ligand_atom_isN, protein_atom_isO, centor, 
                    ligand_cross_isN_flag, protein_cross_isO_flag, cross_distance_matrix, ligand_cross_lp_pos, protein_cross_lp_pos, flag = 'ligand')

            if l_combinations_isring.size(0) != 0:
                l2p.append(l_combinations_isring.cuda())
            if l_combinations_isO.size(0) != 0:
                l2p.append(l_combinations_isO.cuda())
            if l_combinations_isN.size(0) != 0:
                l2p.append(l_combinations_isN.cuda())

            #print('l_combinations_isring:', l_combinations_isring.shape)
            #print('l_combinations_isO:', l_combinations_isO.shape)
            #print('l_combinations_isN:', l_combinations_isN.shape)

            #蛋白到配体
            p_combinations_isring = self.combinations_optim(x, org_x, protein_atom, ligand_atom, protein_atom_isring, ligand_atom_isring, centor, 
                    protein_cross_isring_flag, ligand_cross_isring_flag, cross_distance_matrix, ligand_cross_lp_pos, protein_cross_lp_pos, flag = 'protein')
            p_combinations_isO    = self.combinations_optim(x, org_x, protein_atom, ligand_atom, protein_atom_isN, ligand_atom_isO, centor, 
                    protein_cross_isN_flag, ligand_cross_isO_flag, cross_distance_matrix, ligand_cross_lp_pos, protein_cross_lp_pos, flag = 'protein')
            p_combinations_isN    = self.combinations_optim(x, org_x, protein_atom, ligand_atom, protein_atom_isO, ligand_atom_isN, centor, 
                    protein_cross_isO_flag, ligand_cross_isN_flag, cross_distance_matrix, ligand_cross_lp_pos, protein_cross_lp_pos, flag = 'protein')
            
            if p_combinations_isring.size(0) != 0:
                p2l.append(p_combinations_isring.cuda())
            if p_combinations_isO.size(0) != 0:
                p2l.append(p_combinations_isO.cuda())
            if p_combinations_isN.size(0) != 0:
                p2l.append(p_combinations_isN.cuda())

            #print('p_combinations_isring:', p_combinations_isring.shape)
            #print('p_combinations_isO:', p_combinations_isO.shape)
            #print('p_combinations_isN:', p_combinations_isN.shape)

        
        if len(l2p) != 0:
            l2p_edge_index = torch.unique(torch.cat(l2p, dim = -1), dim = -1)
        else:
            l2p_edge_index = torch.empty(0, 0).cuda()
        
        if len(p2l) != 0:
            p2l_edge_index = torch.unique(torch.cat(p2l, dim = -1), dim = -1) 
        else:
            p2l_edge_index = torch.empty(0, 0).cuda()



        #print('l2p[:, :3]:', l2p_edge_index[:, :3]) #[]
        #print('p2l[:, :3]:', p2l_edge_index[:, :3]) #[]

        #print('l2p:', l2p_edge_index.shape) #[]
        #print('p2l:', p2l_edge_index.shape) #[]

        #raise Exception('test')

        '''
        ligand_atom: torch.Size([34])
        protein_atom: torch.Size([400])
        type(atom_isring): torch.bool
        atom_isring: torch.Size([434])
        atom_isO: torch.Size([434])
        atom_isN: torch.Size([434])
        ligand_atom_isring: torch.Size([34])
        ligand_atom_isO: torch.Size([34])
        ligand_atom_isN: torch.Size([34])
        protein_atom_isring: torch.Size([400])
        protein_atom_isO: torch.Size([400])
        protein_atom_isN: torch.Size([400])
        l_combinations_isring: torch.Size([2, 0])
        l_combinations_isO: torch.Size([2, 364])
        l_combinations_isN: torch.Size([2, 324])
        p_combinations_isring: torch.Size([2, 0])
        p_combinations_isO: torch.Size([2, 364])
        p_combinations_isN: torch.Size([2, 324])
        l2p[:, :3]: tensor([[400, 400, 400],
                [  0,   3,   4]], device='cuda:0')
        p2l[:, :3]: tensor([[  0,   0,   0],
                [400, 405, 412]], device='cuda:0')
        l2p: torch.Size([2, 688])
        p2l: torch.Size([2, 688])
        '''


        '''
        #如果不想使用knn来构建蛋白内部的连接,而是使用自定义的(即蛋白自身的原有连接),则可以使用下面的代码像配体一样更新id
        #另外，如果不提供蛋白的键类型,则在更新edge_type时,要把键类型填充设置设置成7
        #更新id
        new_protein_bond_index = torch.zeros(protein_bond_index.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(protein_bond_index.T):
            new_protein_bond_index[i][0] = protein_node_local2global_dict[bd[0].item()] #tensor张量不适合作为字典的k，所以要更换，否则报错
            new_protein_bond_index[i][1] = protein_node_local2global_dict[bd[1].item()]
        
        new_protein_bond_index = new_protein_bond_index.T
        '''
        





        #raise Exception('test')
        # NxN connectivity matrix where 0 means no connection and 1/2/3/4 means single/double/triple/aromatic bonds.

        #不是键类型，是用来判断蛋白和配体之间的是否有连接
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1 #1表示是配体
        n_dst = mask_ligand[dst] == 1

        #需要扩充配体的键类型，增加单键，双键，三键，芳香键，我们需要知道配体和蛋白的这些键类型有什么作用？？？
        #edge_type在神经网络中只是和边长度嵌入和节点嵌入连接在了一起，并没有什么特殊的判断处理，虽然这里的0,1,2,3的确用来标识节点是蛋白或者配体上
        #但是在神经网络中用没有用来区分配体和蛋白节点
        edge_type[n_src & n_dst]   = 1  #表示在配体内部
        edge_type[n_src & ~n_dst]  = 5  #表示源节点在配体，目标节点在蛋白
        edge_type[~n_src & n_dst]  = 6  #表示源节点在蛋白，目标节点在配体
        edge_type[~n_src & ~n_dst] = 7  #表示在蛋白内部


        indices1 = (edge_type == 1).nonzero().view(-1) #寻找非0元素下标
        indices5 = (edge_type == 5).nonzero().view(-1) #寻找非0元素下标
        indices6 = (edge_type == 6).nonzero().view(-1) #寻找非0元素下标
        indices = torch.cat([indices1, indices5, indices6])
        #indices = (edge_type == 1).nonzero().squeeze() #寻找非0元素下标

        # 要删除的行的索引
        rows_to_remove = indices.detach().cpu().tolist()

        # 使用 torch.index_select() 函数选择不删除的行,去掉配体连接
        indices_to_keep = torch.tensor(list(set(range(edge_type.size(0))) - set(rows_to_remove))).cuda()
        new_edge_type   = torch.index_select(edge_type, 0, indices_to_keep)
        new_edge_index  = torch.index_select(edge_index, 1, indices_to_keep)  #2 * N

        #添加配体内部连接
        ##print('ligand_bond_type:', ligand_bond_type)
        ##print('new_ligand_bond_index:', new_ligand_bond_index)

        l2p_edge_type = torch.full([l2p_edge_index.size(1)], 5, dtype = torch.int64).cuda()
        p2l_edge_type = torch.full([p2l_edge_index.size(1)], 6, dtype = torch.int64).cuda()

        new_edge_type  = torch.cat([new_edge_type, ligand_bond_type], dim = 0)  #扩充配体键类型
        new_edge_index = torch.cat([new_edge_index, new_ligand_bond_index], dim = 1)

        if l2p_edge_index.size(0) != 0:
            new_edge_type  = torch.cat([new_edge_type, l2p_edge_type], dim = 0)  #扩充配体键类型
            new_edge_index = torch.cat([new_edge_index, l2p_edge_index], dim = 1)

        if p2l_edge_index.size(0) != 0:
            new_edge_type  = torch.cat([new_edge_type, p2l_edge_type], dim = 0)
            new_edge_index = torch.cat([new_edge_index, p2l_edge_index], dim = 1)


        #

        ##print('new_edge_type:', new_edge_type.shape)
        ##print('new_edge_index:', new_edge_index.shape)
        #new_edge_type: torch.Size([103536])
        #new_edge_index: torch.Size([2, 103536])

        #(edge_type,edge_index)每一行的顺序是无所谓的，不影响卷积，关键在于里面使用的原子id一定要是根据当前批量获得的全局id
        #(edge_type,edge_index)只要配体和蛋白之间的，其余的全部由配体和蛋白的全连接图来取代
        edge_type_dim = F.one_hot(new_edge_type, num_classes=8) #由原来的4种，变成了8种,除了把这里改了成8之外，其它地方也要改吧？否则报错
        #探索一下，哪有哪里在使用键长度
        return edge_type_dim, new_edge_index








    # @staticmethod #当需要self时，把静态函数声明去掉，否则会把self当作普通参数，而不是类对象
    def _build_edge_type_interaction_8_gpu_optim_v2(self, x, org_x, edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch,
        batch, #这个参数要结合mask_liagnd来确定哪些是配体哪些是蛋白上的节点
        atom_isring,
        atom_isO,
        atom_isN,

        cross_isring_flag, 
        cross_isO_flag, 
        cross_isN_flag, 
        cross_lp_pos,
        cross_distance,

        cross_bond_index, cross_bond_type, cross_bond_index_reverse, cross_bond_type_reverse
        ):
        #相互作用连接表不使用自定义键，可以兼容之前的方法
        #cross_bond_index_reverse, cross_bond_type_reverse,是表示配体到蛋白的连接表的逆，即反向
        #可以全部换成GPU tensor运算，加速训练
        #获取配体的内部的原子id
        #把配体和蛋白的之间的连接替换成可能的相互作用
        ligand_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 1])))).cuda()
        #获取配体和蛋白放在一起的原子id
        protein_ligand_node_list = torch.LongTensor(list(range(len(mask_ligand)))).cuda()
        ligand_node_global = protein_ligand_node_list[mask_ligand == 1] #pytorch2.0要求datas[index]，数据和索引都在一个设备上

        #制作id映射
        ligand_node_local2global_dict = {}
        for k, v in zip(ligand_node_local, ligand_node_global):
            ligand_node_local2global_dict[k.item()] = v
        

        #更新id
        new_ligand_bond_index = torch.zeros(ligand_bond_index.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(ligand_bond_index.T):
            new_ligand_bond_index[i][0] = ligand_node_local2global_dict[bd[0].item()] #tensor张量不适合作为字典的k，所以要更换，否则报错
            new_ligand_bond_index[i][1] = ligand_node_local2global_dict[bd[1].item()]
        new_ligand_bond_index = new_ligand_bond_index.T



        '''
        #蛋白和配体之间的映射，重点像配体一样获取配体蛋白合并之后的id与未合并之前蛋白的id对应关系
        '''
        protein_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 0])))).cuda()
        #获取配体和蛋白放在一起的原子id
        protein_ligand_node_list = torch.LongTensor(list(range(len(mask_ligand)))).cuda()
        protein_node_global = protein_ligand_node_list[mask_ligand == 0] #pytorch2.0要求datas[index]，数据和索引都在一个设备上

        #制作id映射
        protein_node_local2global_dict = {}
        for k, v in zip(protein_node_local, protein_node_global):
            protein_node_local2global_dict[k.item()] = v


        #根据蛋白和配体的局部和全局id的映射，模仿配体的键id的更新，我们处理cross_bond_index, cross_bond_type， cross_bond_index_reverse, cross_bond_type_reverse
        #cross_bond_index_reverse, cross_bond_type_reverse是配体到蛋白的逆连接表, 键类型[5,6,7],逆连接的键类型[8,9,10]
        #更新id
        new_cross_bond_index = torch.zeros(cross_bond_index.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(cross_bond_index.T): #N * 2
            new_cross_bond_index[i][0] = ligand_node_local2global_dict[bd[0].item()] #第一列是配体原子
            new_cross_bond_index[i][1] = protein_node_local2global_dict[bd[1].item()] #第二列是蛋白原子
        new_cross_bond_index = new_cross_bond_index.T


        #更新id_reverse
        new_cross_bond_index_reverse = torch.zeros(cross_bond_index_reverse.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(cross_bond_index_reverse.T): #N * 2
            new_cross_bond_index_reverse[i][0] = protein_node_local2global_dict[bd[0].item()] #第一列是蛋白原子
            new_cross_bond_index_reverse[i][1] = ligand_node_local2global_dict[bd[1].item()] #第二列是配体原子
        new_cross_bond_index_reverse = new_cross_bond_index_reverse.T

        #去一下重复的配体和蛋白的连接
        new_cross_bond_index = torch.unique(new_cross_bond_index, dim = -1)
        new_cross_bond_index_reverse = torch.unique(new_cross_bond_index_reverse, dim = -1)

        #不是键类型，是用来判断蛋白和配体之间的是否有连接
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1 #1表示是配体
        n_dst = mask_ligand[dst] == 1

        #需要扩充配体的键类型，增加单键，双键，三键，芳香键，我们需要知道配体和蛋白的这些键类型有什么作用？？？
        #edge_type在神经网络中只是和边长度嵌入和节点嵌入连接在了一起，并没有什么特殊的判断处理，虽然这里的0,1,2,3的确用来标识节点是蛋白或者配体上
        #但是在神经网络中用没有用来区分配体和蛋白节点
        edge_type[n_src & n_dst]   = 1  #表示在配体内部
        edge_type[n_src & ~n_dst]  = 5  #表示源节点在配体，目标节点在蛋白
        edge_type[~n_src & n_dst]  = 6  #表示源节点在蛋白，目标节点在配体
        edge_type[~n_src & ~n_dst] = 7  #表示在蛋白内部


        indices1 = (edge_type == 1).nonzero().view(-1) #寻找非0元素下标
        indices5 = (edge_type == 5).nonzero().view(-1) #寻找非0元素下标
        indices6 = (edge_type == 6).nonzero().view(-1) #寻找非0元素下标
        indices = torch.cat([indices1, indices5, indices6])
        #indices = (edge_type == 1).nonzero().squeeze() #寻找非0元素下标

        # 要删除的行的索引
        rows_to_remove = indices.detach().cpu().tolist()

        # 使用 torch.index_select() 函数选择不删除的行,去掉配体连接
        indices_to_keep = torch.tensor(list(set(range(edge_type.size(0))) - set(rows_to_remove))).cuda()
        new_edge_type   = torch.index_select(edge_type, 0, indices_to_keep)
        new_edge_index  = torch.index_select(edge_index, 1, indices_to_keep)  #2 * N

        #添加自定义连接
        #相互作用连接表不使用自定义键，可以兼容之前的方法
        l2p_edge_type = torch.full([new_cross_bond_index.size(1)], 5, dtype = torch.int64).cuda()
        p2l_edge_type = torch.full([new_cross_bond_index_reverse.size(1)], 6, dtype = torch.int64).cuda()

        new_edge_type  = torch.cat([new_edge_type, ligand_bond_type, l2p_edge_type, p2l_edge_type], dim = 0)  #扩充配体键类型
        try:
            new_edge_index = torch.cat([new_edge_index, new_ligand_bond_index, new_cross_bond_index, new_cross_bond_index_reverse], dim = 1)
        except Exception as e:
            print("error:", e)
            print('new_edge_index, new_ligand_bond_index, new_cross_bond_index, new_cross_bond_index_reverse:', new_edge_index.shape, new_ligand_bond_index.shape, new_cross_bond_index.shape, new_cross_bond_index_reverse.shape)
            #torch.Size([2, 8000]) torch.Size([2, 56]) torch.Size([4, 24]) torch.Size([4, 24]), 当批量大于1时，应该按列连接，而非行
        edge_type_dim = F.one_hot(new_edge_type, num_classes=8) #由原来的4种，变成了8种,除了把这里改了成8之外，其它地方也要改吧？否则报错
        #探索一下，哪有哪里在使用键长度
        return edge_type_dim, new_edge_index






    # @staticmethod #当需要self时，把静态函数声明去掉，否则会把self当作普通参数，而不是类对象
    def _build_edge_type_interaction_20_gpu_optim(self, x, org_x, edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch,
        batch, #这个参数要结合mask_liagnd来确定哪些是配体哪些是蛋白上的节点
        atom_isring,
        atom_isO,
        atom_isN,

        cross_isring_flag, 
        cross_isO_flag, 
        cross_isN_flag, 
        cross_lp_pos,
        cross_distance,

        cross_bond_index, cross_bond_type, cross_bond_index_reverse, cross_bond_type_reverse,

        protein_element_batch,
        protein_link_t_batch,
        protein_link_t_reverse_batch,
        ligand_element_batch,
        protein_element,
        ligand_element,
    ):

        #assert x.shape[0] == len(protein_element) + len(ligand_element) #数量没问题。批次和批次之间不影响
        #更改自定义的连接表的原子id，由原来的局部i映射到全局id

        ligand_atom_num_list  = []
        protein_atom_num_list = []
        ligand_atom_num_list.append(0)
        protein_atom_num_list.append(0)
        g_num = max(ligand_element_batch) + 1
        #print('g_num:', g_num)
        #print('ligand_element.shape:', ligand_element.shape) #torch.Size([26])
        #print('ligand_element_batch.shape:', ligand_element_batch.shape)
        #print('ligand_element_batch:', ligand_element_batch)

        #print('protein_element.shape:', protein_element.shape) #torch.Size([250])
        #print('protein_element_batch.shape:', protein_element_batch.shape)
        #print('protein_element_batch:', protein_element_batch)
        lg_nums = 0
        pr_nums = 0
        #print('ligand_element_batch:', ligand_element_batch)
        #print('g_num:', g_num)
        for i in range(g_num):
            nm1 = len(ligand_element[ligand_element_batch == i])
            lg_nums = lg_nums + nm1
            ligand_atom_num_list.append(lg_nums)

            nm2 = len(protein_element[protein_element_batch == i])
            pr_nums = pr_nums + nm2
            protein_atom_num_list.append(pr_nums)
        
        #print('ligand_atom_num_list:', ligand_atom_num_list) #ligand_atom_num_list: [0, 13, 13]
        #print('protein_atom_num_list:', protein_atom_num_list) #protein_atom_num_list: [0, 125, 125]
        
        #print('max(ligand_bond_index) befor:', torch.max(ligand_bond_index))

        #更改配体,pyg在连接多个子图时，会自动编号，所以配体是没问题的，问题在于配体和蛋白之间的相互作用信息，这个可能很难做到正确的自动编号.而当批量为1时，不会出错是因为此时就一个图，不用重新编号
        #而关于蛋白，多批量时，id编号是错误的，因此解决时，我们先减去增量标号，再分别映射配体和蛋白
        '''
        #print('ligand_bond_index.shape:', ligand_bond_index.shape) #tensor(189, device='cuda:0')
        new_ligand_bond_index = torch.zeros(ligand_bond_index.T.shape, dtype = torch.int64).cuda()
        for i in range(g_num):
            mask = ligand_bond_type_batch == i
            #new_ligand_bond_index[mask] = ligand_bond_index.T[mask] + ligand_atom_num_list[i]

            print('i:', i)
            print('ligand_atom_num_list[i]:', ligand_atom_num_list[i])
            print('max(ligand_bond_index.T[mask]):', torch.max(ligand_bond_index.T[mask]))


        ligand_bond_index = new_ligand_bond_index.T
        #exit()
        '''

        
        #更改相互作用，配体到蛋白
        #print('cross_bond_index.shape:', cross_bond_index.shape) #torch.Size([2, 582])
        #print('protein_link_t_batch.shape:', protein_link_t_batch.shape) #torch.Size([582])
        #print('protein_link_t_batch:', protein_link_t_batch) #
        
        new_cross_bond_index = torch.zeros(cross_bond_index.T.shape, dtype = torch.int64).cuda()
        #print('new_cross_bond_index.shape:', new_cross_bond_index.shape) #torch.Size([582, 2])

        #仅仅采样时成立
        #assert torch.allclose(cross_bond_index.T[protein_link_t_batch == 0], cross_bond_index.T[protein_link_t_batch == g_num - 1], atol=0.02)

        
        #print('max(cross_bond_index.T[:, 0]):', torch.max(cross_bond_index.T[:, 0]))
        #print('max(cross_bond_index.T[:, 1]):', torch.max(cross_bond_index.T[:, 1]))
        #print('cross_bond_index.T, before:', cross_bond_index.T)
        for i in range(g_num): #
            #print('i:', i)
            #print('ligand_atom_num_list[i]:', ligand_atom_num_list[i])
            #print('protein_atom_num_list[i]:', protein_atom_num_list[i])
            mask = protein_link_t_batch == i
            #print('mask:', mask)
            #print('cross_bond_index.T[mask][:, 0]:', cross_bond_index.T[mask][:, 0])
            #print('protein_atom_num_list[i]:', ligand_atom_num_list[i])
            #print('cross_bond_index.T[mask][:, 0] + protein_atom_num_list[i]:', cross_bond_index.T[:, 0][mask] + ligand_atom_num_list[i])
            #print('cross_bond_index.T[mask][:, 0].shape:', cross_bond_index.T[mask][:, 0].shape)
            #print('new_cross_bond_index[mask][:, 0].shape:', new_cross_bond_index[mask][:, 0].shape)

            tmp = new_cross_bond_index[mask]
            new_cross_bond_index[:, 0][mask] = (cross_bond_index.T[:, 0][mask] + ligand_atom_num_list[i])

            #print('new_cross_bond_index[mask][:, 0]:', new_cross_bond_index[mask][:, 0]) #0, 没改变值, 原因在于不支持先mask后切片索引的方法,可以先切片再掩码。

    

            new_cross_bond_index[:, 1][mask] = cross_bond_index.T[:, 1][mask] + protein_atom_num_list[i]
            #print('new_cross_bond_index[mask]:', new_cross_bond_index[mask]) #0?
        cross_bond_index = new_cross_bond_index.T 

        #print('max(cross_bond_index.T[:, 0]):', torch.max(cross_bond_index.T[:, 0]))
        #print('max(cross_bond_index.T[:, 1]):', torch.max(cross_bond_index.T[:, 1]))

        #print('cross_bond_index.T, after:', cross_bond_index.T) ##全0？

        #exit()

        #更改相互作用，蛋白到配体
        new_cross_bond_index_reverse = torch.zeros(cross_bond_index_reverse.T.shape, dtype = torch.int64).cuda()
        for i in range(g_num): #
            mask = protein_link_t_reverse_batch == i
            new_cross_bond_index_reverse[:, 0][mask] = cross_bond_index_reverse.T[:, 0][mask] + protein_atom_num_list[i]
            new_cross_bond_index_reverse[:, 1][mask] = cross_bond_index_reverse.T[:, 1][mask] + ligand_atom_num_list[i]

        cross_bond_index_reverse = new_cross_bond_index_reverse.T
        


        #cross_bond_index_reverse, cross_bond_type_reverse,是表示配体到蛋白的连接表的逆，即反向
        #可以全部换成GPU tensor运算，加速训练
        #获取配体的内部的原子id
        #把配体和蛋白的之间的连接替换成可能的相互作用

        #当批量大于1时，会出现出第一个歪连接表异常问题，原因是这里的映射不是全局的，而是局部的，因为数据准备是从预处理来的，而非实时构建的? 这个问题必须要搞清楚是哪里的问题，可能训练也受影响了
        ligand_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 1])))).cuda()
        #获取配体和蛋白放在一起的原子id
        protein_ligand_node_list = torch.LongTensor(list(range(len(mask_ligand)))).cuda()
        ligand_node_global = protein_ligand_node_list[mask_ligand == 1] #pytorch2.0要求datas[index]，数据和索引都在一个设备上

        #制作id映射
        ligand_node_local2global_dict = {}
        for k, v in zip(ligand_node_local, ligand_node_global):
            ligand_node_local2global_dict[k.item()] = v
        
        #print('ligand_node_local2global_dict:', ligand_node_local2global_dict)
        #print('ligand_node_local:', ligand_node_local)
        #print('len(ligand_node_local):', len(ligand_node_local))
        #print('len(ligand_element):', len(ligand_element))
        #print('max(ligand_bond_index):', torch.max(ligand_bond_index)) #tensor(227, device='cuda:0')
        #print('ligand_bond_index.T:', ligand_bond_index.T) 
        

        #更新id
        new_ligand_bond_index = torch.zeros(ligand_bond_index.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(ligand_bond_index.T):
            new_ligand_bond_index[i][0] = ligand_node_local2global_dict[bd[0].item()] #tensor张量不适合作为字典的k，所以要更换，否则报错
            new_ligand_bond_index[i][1] = ligand_node_local2global_dict[bd[1].item()]
        new_ligand_bond_index = new_ligand_bond_index.T



        '''
        #蛋白和配体之间的映射，重点像配体一样获取配体蛋白合并之后的id与未合并之前蛋白的id对应关系
        '''
        protein_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 0])))).cuda()
        #获取配体和蛋白放在一起的原子id
        protein_ligand_node_list = torch.LongTensor(list(range(len(mask_ligand)))).cuda()
        protein_node_global = protein_ligand_node_list[mask_ligand == 0] #pytorch2.0要求datas[index]，数据和索引都在一个设备上

        #制作id映射
        protein_node_local2global_dict = {}
        for k, v in zip(protein_node_local, protein_node_global):
            protein_node_local2global_dict[k.item()] = v


        #根据蛋白和配体的局部和全局id的映射，模仿配体的键id的更新，我们处理cross_bond_index, cross_bond_type， cross_bond_index_reverse, cross_bond_type_reverse
        #cross_bond_index_reverse, cross_bond_type_reverse是配体到蛋白的逆连接表, 键类型[5,6,7],逆连接的键类型[8,9,10]
        #更新id
        new_cross_bond_index = torch.zeros(cross_bond_index.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(cross_bond_index.T): #N * 2
            new_cross_bond_index[i][0] = ligand_node_local2global_dict[bd[0].item()] #第一列是配体原子
            new_cross_bond_index[i][1] = protein_node_local2global_dict[bd[1].item()] #第二列是蛋白原子
        new_cross_bond_index = new_cross_bond_index.T


        #更新id_reverse
        new_cross_bond_index_reverse = torch.zeros(cross_bond_index_reverse.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(cross_bond_index_reverse.T): #N * 2
            new_cross_bond_index_reverse[i][0] = protein_node_local2global_dict[bd[0].item()] #第一列是蛋白原子
            new_cross_bond_index_reverse[i][1] = ligand_node_local2global_dict[bd[1].item()] #第二列是配体原子
        new_cross_bond_index_reverse = new_cross_bond_index_reverse.T


        #不是键类型，是用来判断蛋白和配体之间的是否有连接
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1 #1表示是配体
        n_dst = mask_ligand[dst] == 1

        #需要扩充配体的键类型，增加单键，双键，三键，芳香键，我们需要知道配体和蛋白的这些键类型有什么作用？？？
        #edge_type在神经网络中只是和边长度嵌入和节点嵌入连接在了一起，并没有什么特殊的判断处理，虽然这里的0,1,2,3的确用来标识节点是蛋白或者配体上
        #但是在神经网络中用没有用来区分配体和蛋白节点
        edge_type[n_src & n_dst]   = 1  #表示在配体内部
        edge_type[n_src & ~n_dst]  = 5  #表示源节点在配体，目标节点在蛋白
        edge_type[~n_src & n_dst]  = 6  #表示源节点在蛋白，目标节点在配体
        edge_type[~n_src & ~n_dst] = 11  #表示在蛋白内部


        indices1 = (edge_type == 1).nonzero().view(-1) #寻找非0元素下标
        indices5 = (edge_type == 5).nonzero().view(-1) #寻找非0元素下标
        indices6 = (edge_type == 6).nonzero().view(-1) #寻找非0元素下标
        indices = torch.cat([indices1, indices5, indices6])
        #indices = (edge_type == 1).nonzero().squeeze() #寻找非0元素下标

        # 要删除的行的索引
        rows_to_remove = indices.detach().cpu().tolist()

        # 使用 torch.index_select() 函数选择不删除的行,去掉配体连接
        indices_to_keep = torch.tensor(list(set(range(edge_type.size(0))) - set(rows_to_remove))).cuda()
        new_edge_type   = torch.index_select(edge_type, 0, indices_to_keep)
        new_edge_index  = torch.index_select(edge_index, 1, indices_to_keep)  #2 * N

        #添加自定义连接
        new_edge_type  = torch.cat([new_edge_type, ligand_bond_type, cross_bond_type, cross_bond_type_reverse], dim = 0)  #扩充配体键类型
        new_edge_index = torch.cat([new_edge_index, new_ligand_bond_index, new_cross_bond_index, new_cross_bond_index_reverse], dim = 1)


        edge_type_dim = F.one_hot(new_edge_type, num_classes=20) #由原来的4种，变成了8种,除了把这里改了成8之外，其它地方也要改吧？否则报错
        #探索一下，哪有哪里在使用键长度
        return edge_type_dim, new_edge_index



    def _build_edge_type_interaction_20_gpu_optim_no_interactive_gpu(self, x, org_x, edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch,
        batch, #这个参数要结合mask_liagnd来确定哪些是配体哪些是蛋白上的节点
        atom_isring,
        atom_isO,
        atom_isN,

        cross_isring_flag, 
        cross_isO_flag, 
        cross_isN_flag, 
        cross_lp_pos,
        cross_distance,

        cross_bond_index, cross_bond_type, cross_bond_index_reverse, cross_bond_type_reverse,

        protein_element_batch,
        protein_link_t_batch,
        protein_link_t_reverse_batch,
        ligand_element_batch,
        protein_element,
        ligand_element,
    ):

        #assert x.shape[0] == len(protein_element) + len(ligand_element) #数量没问题。批次和批次之间不影响
        #更改自定义的连接表的原子id，由原来的局部i映射到全局id

        ligand_atom_num_list  = []
        protein_atom_num_list = []
        ligand_atom_num_list.append(0)
        protein_atom_num_list.append(0)
        g_num = max(ligand_element_batch) + 1
        #print('g_num:', g_num)
        #print('ligand_element.shape:', ligand_element.shape) #torch.Size([26])
        #print('ligand_element_batch.shape:', ligand_element_batch.shape)
        #print('ligand_element_batch:', ligand_element_batch)

        #print('protein_element.shape:', protein_element.shape) #torch.Size([250])
        #print('protein_element_batch.shape:', protein_element_batch.shape)
        #print('protein_element_batch:', protein_element_batch)
        lg_nums = 0
        pr_nums = 0
        #print('ligand_element_batch:', ligand_element_batch)
        #print('g_num:', g_num)
        for i in range(g_num):
            nm1 = len(ligand_element[ligand_element_batch == i])
            lg_nums = lg_nums + nm1
            ligand_atom_num_list.append(lg_nums)

            nm2 = len(protein_element[protein_element_batch == i])
            pr_nums = pr_nums + nm2
            protein_atom_num_list.append(pr_nums)
        
        #print('ligand_atom_num_list:', ligand_atom_num_list) #ligand_atom_num_list: [0, 13, 13]
        #print('protein_atom_num_list:', protein_atom_num_list) #protein_atom_num_list: [0, 125, 125]
        
        #print('max(ligand_bond_index) befor:', torch.max(ligand_bond_index))

        #更改配体,pyg在连接多个子图时，会自动编号，所以配体是没问题的，问题在于配体和蛋白之间的相互作用信息，这个可能很难做到正确的自动编号.而当批量为1时，不会出错是因为此时就一个图，不用重新编号
        #而关于蛋白，多批量时，id编号是错误的，因此解决时，我们先减去增量标号，再分别映射配体和蛋白
        '''
        #print('ligand_bond_index.shape:', ligand_bond_index.shape) #tensor(189, device='cuda:0')
        new_ligand_bond_index = torch.zeros(ligand_bond_index.T.shape, dtype = torch.int64).cuda()
        for i in range(g_num):
            mask = ligand_bond_type_batch == i
            #new_ligand_bond_index[mask] = ligand_bond_index.T[mask] + ligand_atom_num_list[i]

            print('i:', i)
            print('ligand_atom_num_list[i]:', ligand_atom_num_list[i])
            print('max(ligand_bond_index.T[mask]):', torch.max(ligand_bond_index.T[mask]))


        ligand_bond_index = new_ligand_bond_index.T
        #exit()
        '''

        
        #更改相互作用，配体到蛋白
        #print('cross_bond_index.shape:', cross_bond_index.shape) #torch.Size([2, 582])
        #print('protein_link_t_batch.shape:', protein_link_t_batch.shape) #torch.Size([582])
        #print('protein_link_t_batch:', protein_link_t_batch) #
        
        new_cross_bond_index = torch.zeros(cross_bond_index.T.shape, dtype = torch.int64).cuda()
        #print('new_cross_bond_index.shape:', new_cross_bond_index.shape) #torch.Size([582, 2])

        #仅仅采样时成立
        #assert torch.allclose(cross_bond_index.T[protein_link_t_batch == 0], cross_bond_index.T[protein_link_t_batch == g_num - 1], atol=0.02)

        
        #print('max(cross_bond_index.T[:, 0]):', torch.max(cross_bond_index.T[:, 0]))
        #print('max(cross_bond_index.T[:, 1]):', torch.max(cross_bond_index.T[:, 1]))
        #print('cross_bond_index.T, before:', cross_bond_index.T)
        for i in range(g_num): #
            #print('i:', i)
            #print('ligand_atom_num_list[i]:', ligand_atom_num_list[i])
            #print('protein_atom_num_list[i]:', protein_atom_num_list[i])
            mask = protein_link_t_batch == i
            #print('mask:', mask)
            #print('cross_bond_index.T[mask][:, 0]:', cross_bond_index.T[mask][:, 0])
            #print('protein_atom_num_list[i]:', ligand_atom_num_list[i])
            #print('cross_bond_index.T[mask][:, 0] + protein_atom_num_list[i]:', cross_bond_index.T[:, 0][mask] + ligand_atom_num_list[i])
            #print('cross_bond_index.T[mask][:, 0].shape:', cross_bond_index.T[mask][:, 0].shape)
            #print('new_cross_bond_index[mask][:, 0].shape:', new_cross_bond_index[mask][:, 0].shape)

            tmp = new_cross_bond_index[mask]
            new_cross_bond_index[:, 0][mask] = (cross_bond_index.T[:, 0][mask] + ligand_atom_num_list[i])

            #print('new_cross_bond_index[mask][:, 0]:', new_cross_bond_index[mask][:, 0]) #0, 没改变值, 原因在于不支持先mask后切片索引的方法,可以先切片再掩码。

    

            new_cross_bond_index[:, 1][mask] = cross_bond_index.T[:, 1][mask] + protein_atom_num_list[i]
            #print('new_cross_bond_index[mask]:', new_cross_bond_index[mask]) #0?
        cross_bond_index = new_cross_bond_index.T 

        #print('max(cross_bond_index.T[:, 0]):', torch.max(cross_bond_index.T[:, 0]))
        #print('max(cross_bond_index.T[:, 1]):', torch.max(cross_bond_index.T[:, 1]))

        #print('cross_bond_index.T, after:', cross_bond_index.T) ##全0？

        #exit()

        #更改相互作用，蛋白到配体
        new_cross_bond_index_reverse = torch.zeros(cross_bond_index_reverse.T.shape, dtype = torch.int64).cuda()
        for i in range(g_num): #
            mask = protein_link_t_reverse_batch == i
            new_cross_bond_index_reverse[:, 0][mask] = cross_bond_index_reverse.T[:, 0][mask] + protein_atom_num_list[i]
            new_cross_bond_index_reverse[:, 1][mask] = cross_bond_index_reverse.T[:, 1][mask] + ligand_atom_num_list[i]

        cross_bond_index_reverse = new_cross_bond_index_reverse.T
        


        #cross_bond_index_reverse, cross_bond_type_reverse,是表示配体到蛋白的连接表的逆，即反向
        #可以全部换成GPU tensor运算，加速训练
        #获取配体的内部的原子id
        #把配体和蛋白的之间的连接替换成可能的相互作用

        #当批量大于1时，会出现出第一个歪连接表异常问题，原因是这里的映射不是全局的，而是局部的，因为数据准备是从预处理来的，而非实时构建的? 这个问题必须要搞清楚是哪里的问题，可能训练也受影响了
        ligand_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 1])))).cuda()
        #获取配体和蛋白放在一起的原子id
        protein_ligand_node_list = torch.LongTensor(list(range(len(mask_ligand)))).cuda()
        ligand_node_global = protein_ligand_node_list[mask_ligand == 1] #pytorch2.0要求datas[index]，数据和索引都在一个设备上

        #制作id映射
        ligand_node_local2global_dict = {}
        for k, v in zip(ligand_node_local, ligand_node_global):
            ligand_node_local2global_dict[k.cpu().numpy().tobytes()] = v  # 使用张量的哈希值作为键,， 避免cpu和gpu数据传输
        
        #print('ligand_node_local2global_dict:', ligand_node_local2global_dict)
        #print('ligand_node_local:', ligand_node_local)
        #print('len(ligand_node_local):', len(ligand_node_local))
        #print('len(ligand_element):', len(ligand_element))
        #print('max(ligand_bond_index):', torch.max(ligand_bond_index)) #tensor(227, device='cuda:0')
        #print('ligand_bond_index.T:', ligand_bond_index.T) 
        

        #更新id
        new_ligand_bond_index = torch.zeros(ligand_bond_index.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(ligand_bond_index.T):
            new_ligand_bond_index[i][0] = ligand_node_local2global_dict[bd[0].cpu().numpy().tobytes()] #tensor张量不适合作为字典的k，所以要更换，否则报错
            new_ligand_bond_index[i][1] = ligand_node_local2global_dict[bd[1].cpu().numpy().tobytes()]
        new_ligand_bond_index = new_ligand_bond_index.T



        '''
        #蛋白和配体之间的映射，重点像配体一样获取配体蛋白合并之后的id与未合并之前蛋白的id对应关系
        '''
        protein_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 0])))).cuda()
        #获取配体和蛋白放在一起的原子id
        protein_ligand_node_list = torch.LongTensor(list(range(len(mask_ligand)))).cuda()
        protein_node_global = protein_ligand_node_list[mask_ligand == 0] #pytorch2.0要求datas[index]，数据和索引都在一个设备上

        #制作id映射
        protein_node_local2global_dict = {}
        for k, v in zip(protein_node_local, protein_node_global):
            protein_node_local2global_dict[k.cpu().numpy().tobytes()] = v


        #根据蛋白和配体的局部和全局id的映射，模仿配体的键id的更新，我们处理cross_bond_index, cross_bond_type， cross_bond_index_reverse, cross_bond_type_reverse
        #cross_bond_index_reverse, cross_bond_type_reverse是配体到蛋白的逆连接表, 键类型[5,6,7],逆连接的键类型[8,9,10]
        #更新id
        new_cross_bond_index = torch.zeros(cross_bond_index.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(cross_bond_index.T): #N * 2
            new_cross_bond_index[i][0] = ligand_node_local2global_dict[bd[0].cpu().numpy().tobytes()] #第一列是配体原子
            new_cross_bond_index[i][1] = protein_node_local2global_dict[bd[1].cpu().numpy().tobytes()] #第二列是蛋白原子
        new_cross_bond_index = new_cross_bond_index.T


        #更新id_reverse
        new_cross_bond_index_reverse = torch.zeros(cross_bond_index_reverse.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(cross_bond_index_reverse.T): #N * 2
            new_cross_bond_index_reverse[i][0] = protein_node_local2global_dict[bd[0].cpu().numpy().tobytes()] #第一列是蛋白原子
            new_cross_bond_index_reverse[i][1] = ligand_node_local2global_dict[bd[1].cpu().numpy().tobytes()] #第二列是配体原子
        new_cross_bond_index_reverse = new_cross_bond_index_reverse.T


        #不是键类型，是用来判断蛋白和配体之间的是否有连接
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1 #1表示是配体
        n_dst = mask_ligand[dst] == 1

        #需要扩充配体的键类型，增加单键，双键，三键，芳香键，我们需要知道配体和蛋白的这些键类型有什么作用？？？
        #edge_type在神经网络中只是和边长度嵌入和节点嵌入连接在了一起，并没有什么特殊的判断处理，虽然这里的0,1,2,3的确用来标识节点是蛋白或者配体上
        #但是在神经网络中用没有用来区分配体和蛋白节点
        edge_type[n_src & n_dst]   = 1  #表示在配体内部
        edge_type[n_src & ~n_dst]  = 5  #表示源节点在配体，目标节点在蛋白
        edge_type[~n_src & n_dst]  = 6  #表示源节点在蛋白，目标节点在配体
        edge_type[~n_src & ~n_dst] = 11  #表示在蛋白内部


        indices1 = (edge_type == 1).nonzero().view(-1) #寻找非0元素下标
        indices5 = (edge_type == 5).nonzero().view(-1) #寻找非0元素下标
        indices6 = (edge_type == 6).nonzero().view(-1) #寻找非0元素下标
        #indices = torch.cat([indices1, indices5, indices6])
        indices = torch.cat([indices1])
        #indices = (edge_type == 1).nonzero().squeeze() #寻找非0元素下标

        # 要删除的行的索引
        #rows_to_remove = indices.detach().cpu().tolist()
        # 使用 torch.index_select() 函数选择不删除的行,去掉配体连接
        #indices_to_keep = torch.tensor(list(set(range(edge_type.size(0))) - set(rows_to_remove))).cuda()
        
        
        #更高效方法
        '''
        mask = ~torch.isin(a, b)
        result = a[mask]
        '''
        all_index = torch.tensor(list(range(edge_type.size(0)))).cuda()
        mask = ~torch.isin(all_index, indices)
        indices_to_keep = all_index[mask]
        
        new_edge_type   = torch.index_select(edge_type, 0, indices_to_keep)
        new_edge_index  = torch.index_select(edge_index, 1, indices_to_keep)  #2 * N

        #添加自定义连接
        new_edge_type  = torch.cat([new_edge_type, ligand_bond_type], dim = 0)  #扩充配体键类型
        new_edge_index = torch.cat([new_edge_index, new_ligand_bond_index], dim = 1)


        edge_type_dim = F.one_hot(new_edge_type, num_classes=20) #由原来的4种，变成了8种,除了把这里改了成8之外，其它地方也要改吧？否则报错
        #探索一下，哪有哪里在使用键长度
        return edge_type_dim, new_edge_index




    def _build_edge_type_interaction_20_gpu_optim_no_interactive(self, x, org_x, edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch,
        batch, #这个参数要结合mask_liagnd来确定哪些是配体哪些是蛋白上的节点
        atom_isring,
        atom_isO,
        atom_isN,

        cross_isring_flag, 
        cross_isO_flag, 
        cross_isN_flag, 
        cross_lp_pos,
        cross_distance,

        cross_bond_index, cross_bond_type, cross_bond_index_reverse, cross_bond_type_reverse,

        protein_element_batch,
        protein_link_t_batch,
        protein_link_t_reverse_batch,
        ligand_element_batch,
        protein_element,
        ligand_element,
    ):

        #assert x.shape[0] == len(protein_element) + len(ligand_element) #数量没问题。批次和批次之间不影响
        #更改自定义的连接表的原子id，由原来的局部i映射到全局id

        ligand_atom_num_list  = []
        protein_atom_num_list = []
        ligand_atom_num_list.append(0)
        protein_atom_num_list.append(0)
        g_num = max(ligand_element_batch) + 1
        #print('g_num:', g_num)
        #print('ligand_element.shape:', ligand_element.shape) #torch.Size([26])
        #print('ligand_element_batch.shape:', ligand_element_batch.shape)
        #print('ligand_element_batch:', ligand_element_batch)

        #print('protein_element.shape:', protein_element.shape) #torch.Size([250])
        #print('protein_element_batch.shape:', protein_element_batch.shape)
        #print('protein_element_batch:', protein_element_batch)
        lg_nums = 0
        pr_nums = 0
        #print('ligand_element_batch:', ligand_element_batch)
        #print('g_num:', g_num)
        for i in range(g_num):
            nm1 = len(ligand_element[ligand_element_batch == i])
            lg_nums = lg_nums + nm1
            ligand_atom_num_list.append(lg_nums)

            nm2 = len(protein_element[protein_element_batch == i])
            pr_nums = pr_nums + nm2
            protein_atom_num_list.append(pr_nums)
        
        #print('ligand_atom_num_list:', ligand_atom_num_list) #ligand_atom_num_list: [0, 13, 13]
        #print('protein_atom_num_list:', protein_atom_num_list) #protein_atom_num_list: [0, 125, 125]
        
        #print('max(ligand_bond_index) befor:', torch.max(ligand_bond_index))

        #更改配体,pyg在连接多个子图时，会自动编号，所以配体是没问题的，问题在于配体和蛋白之间的相互作用信息，这个可能很难做到正确的自动编号.而当批量为1时，不会出错是因为此时就一个图，不用重新编号
        #而关于蛋白，多批量时，id编号是错误的，因此解决时，我们先减去增量标号，再分别映射配体和蛋白
        '''
        #print('ligand_bond_index.shape:', ligand_bond_index.shape) #tensor(189, device='cuda:0')
        new_ligand_bond_index = torch.zeros(ligand_bond_index.T.shape, dtype = torch.int64).cuda()
        for i in range(g_num):
            mask = ligand_bond_type_batch == i
            #new_ligand_bond_index[mask] = ligand_bond_index.T[mask] + ligand_atom_num_list[i]

            print('i:', i)
            print('ligand_atom_num_list[i]:', ligand_atom_num_list[i])
            print('max(ligand_bond_index.T[mask]):', torch.max(ligand_bond_index.T[mask]))


        ligand_bond_index = new_ligand_bond_index.T
        #exit()
        '''

        
        #更改相互作用，配体到蛋白
        #print('cross_bond_index.shape:', cross_bond_index.shape) #torch.Size([2, 582])
        #print('protein_link_t_batch.shape:', protein_link_t_batch.shape) #torch.Size([582])
        #print('protein_link_t_batch:', protein_link_t_batch) #
        
        new_cross_bond_index = torch.zeros(cross_bond_index.T.shape, dtype = torch.int64).cuda()
        #print('new_cross_bond_index.shape:', new_cross_bond_index.shape) #torch.Size([582, 2])

        #仅仅采样时成立
        #assert torch.allclose(cross_bond_index.T[protein_link_t_batch == 0], cross_bond_index.T[protein_link_t_batch == g_num - 1], atol=0.02)

        
        #print('max(cross_bond_index.T[:, 0]):', torch.max(cross_bond_index.T[:, 0]))
        #print('max(cross_bond_index.T[:, 1]):', torch.max(cross_bond_index.T[:, 1]))
        #print('cross_bond_index.T, before:', cross_bond_index.T)
        for i in range(g_num): #
            #print('i:', i)
            #print('ligand_atom_num_list[i]:', ligand_atom_num_list[i])
            #print('protein_atom_num_list[i]:', protein_atom_num_list[i])
            mask = protein_link_t_batch == i
            #print('mask:', mask)
            #print('cross_bond_index.T[mask][:, 0]:', cross_bond_index.T[mask][:, 0])
            #print('protein_atom_num_list[i]:', ligand_atom_num_list[i])
            #print('cross_bond_index.T[mask][:, 0] + protein_atom_num_list[i]:', cross_bond_index.T[:, 0][mask] + ligand_atom_num_list[i])
            #print('cross_bond_index.T[mask][:, 0].shape:', cross_bond_index.T[mask][:, 0].shape)
            #print('new_cross_bond_index[mask][:, 0].shape:', new_cross_bond_index[mask][:, 0].shape)

            tmp = new_cross_bond_index[mask]
            new_cross_bond_index[:, 0][mask] = (cross_bond_index.T[:, 0][mask] + ligand_atom_num_list[i])

            #print('new_cross_bond_index[mask][:, 0]:', new_cross_bond_index[mask][:, 0]) #0, 没改变值, 原因在于不支持先mask后切片索引的方法,可以先切片再掩码。

    

            new_cross_bond_index[:, 1][mask] = cross_bond_index.T[:, 1][mask] + protein_atom_num_list[i]
            #print('new_cross_bond_index[mask]:', new_cross_bond_index[mask]) #0?
        cross_bond_index = new_cross_bond_index.T 

        #print('max(cross_bond_index.T[:, 0]):', torch.max(cross_bond_index.T[:, 0]))
        #print('max(cross_bond_index.T[:, 1]):', torch.max(cross_bond_index.T[:, 1]))

        #print('cross_bond_index.T, after:', cross_bond_index.T) ##全0？

        #exit()

        #更改相互作用，蛋白到配体
        new_cross_bond_index_reverse = torch.zeros(cross_bond_index_reverse.T.shape, dtype = torch.int64).cuda()
        for i in range(g_num): #
            mask = protein_link_t_reverse_batch == i
            new_cross_bond_index_reverse[:, 0][mask] = cross_bond_index_reverse.T[:, 0][mask] + protein_atom_num_list[i]
            new_cross_bond_index_reverse[:, 1][mask] = cross_bond_index_reverse.T[:, 1][mask] + ligand_atom_num_list[i]

        cross_bond_index_reverse = new_cross_bond_index_reverse.T
        


        #cross_bond_index_reverse, cross_bond_type_reverse,是表示配体到蛋白的连接表的逆，即反向
        #可以全部换成GPU tensor运算，加速训练
        #获取配体的内部的原子id
        #把配体和蛋白的之间的连接替换成可能的相互作用

        #当批量大于1时，会出现出第一个歪连接表异常问题，原因是这里的映射不是全局的，而是局部的，因为数据准备是从预处理来的，而非实时构建的? 这个问题必须要搞清楚是哪里的问题，可能训练也受影响了
        ligand_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 1])))).cuda()
        #获取配体和蛋白放在一起的原子id
        protein_ligand_node_list = torch.LongTensor(list(range(len(mask_ligand)))).cuda()
        ligand_node_global = protein_ligand_node_list[mask_ligand == 1] #pytorch2.0要求datas[index]，数据和索引都在一个设备上

        #制作id映射
        ligand_node_local2global_dict = {}
        for k, v in zip(ligand_node_local, ligand_node_global):
            ligand_node_local2global_dict[k.item()] = v
        
        #print('ligand_node_local2global_dict:', ligand_node_local2global_dict)
        #print('ligand_node_local:', ligand_node_local)
        #print('len(ligand_node_local):', len(ligand_node_local))
        #print('len(ligand_element):', len(ligand_element))
        #print('max(ligand_bond_index):', torch.max(ligand_bond_index)) #tensor(227, device='cuda:0')
        #print('ligand_bond_index.T:', ligand_bond_index.T) 
        

        #更新id
        new_ligand_bond_index = torch.zeros(ligand_bond_index.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(ligand_bond_index.T):
            new_ligand_bond_index[i][0] = ligand_node_local2global_dict[bd[0].item()] #tensor张量不适合作为字典的k，所以要更换，否则报错
            new_ligand_bond_index[i][1] = ligand_node_local2global_dict[bd[1].item()]
        new_ligand_bond_index = new_ligand_bond_index.T



        '''
        #蛋白和配体之间的映射，重点像配体一样获取配体蛋白合并之后的id与未合并之前蛋白的id对应关系
        '''
        protein_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 0])))).cuda()
        #获取配体和蛋白放在一起的原子id
        protein_ligand_node_list = torch.LongTensor(list(range(len(mask_ligand)))).cuda()
        protein_node_global = protein_ligand_node_list[mask_ligand == 0] #pytorch2.0要求datas[index]，数据和索引都在一个设备上

        #制作id映射
        protein_node_local2global_dict = {}
        for k, v in zip(protein_node_local, protein_node_global):
            protein_node_local2global_dict[k.item()] = v


        #根据蛋白和配体的局部和全局id的映射，模仿配体的键id的更新，我们处理cross_bond_index, cross_bond_type， cross_bond_index_reverse, cross_bond_type_reverse
        #cross_bond_index_reverse, cross_bond_type_reverse是配体到蛋白的逆连接表, 键类型[5,6,7],逆连接的键类型[8,9,10]
        #更新id
        new_cross_bond_index = torch.zeros(cross_bond_index.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(cross_bond_index.T): #N * 2
            new_cross_bond_index[i][0] = ligand_node_local2global_dict[bd[0].item()] #第一列是配体原子
            new_cross_bond_index[i][1] = protein_node_local2global_dict[bd[1].item()] #第二列是蛋白原子
        new_cross_bond_index = new_cross_bond_index.T


        #更新id_reverse
        new_cross_bond_index_reverse = torch.zeros(cross_bond_index_reverse.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(cross_bond_index_reverse.T): #N * 2
            new_cross_bond_index_reverse[i][0] = protein_node_local2global_dict[bd[0].item()] #第一列是蛋白原子
            new_cross_bond_index_reverse[i][1] = ligand_node_local2global_dict[bd[1].item()] #第二列是配体原子
        new_cross_bond_index_reverse = new_cross_bond_index_reverse.T


        #不是键类型，是用来判断蛋白和配体之间的是否有连接
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1 #1表示是配体
        n_dst = mask_ligand[dst] == 1

        #需要扩充配体的键类型，增加单键，双键，三键，芳香键，我们需要知道配体和蛋白的这些键类型有什么作用？？？
        #edge_type在神经网络中只是和边长度嵌入和节点嵌入连接在了一起，并没有什么特殊的判断处理，虽然这里的0,1,2,3的确用来标识节点是蛋白或者配体上
        #但是在神经网络中用没有用来区分配体和蛋白节点
        edge_type[n_src & n_dst]   = 1  #表示在配体内部
        edge_type[n_src & ~n_dst]  = 5  #表示源节点在配体，目标节点在蛋白
        edge_type[~n_src & n_dst]  = 6  #表示源节点在蛋白，目标节点在配体
        edge_type[~n_src & ~n_dst] = 11  #表示在蛋白内部


        indices1 = (edge_type == 1).nonzero().view(-1) #寻找非0元素下标
        indices5 = (edge_type == 5).nonzero().view(-1) #寻找非0元素下标
        indices6 = (edge_type == 6).nonzero().view(-1) #寻找非0元素下标
        #indices = torch.cat([indices1, indices5, indices6])
        indices = torch.cat([indices1])
        #indices = (edge_type == 1).nonzero().squeeze() #寻找非0元素下标

        # 要删除的行的索引
        rows_to_remove = indices.detach().cpu().tolist()

        # 使用 torch.index_select() 函数选择不删除的行,去掉配体连接
        indices_to_keep = torch.tensor(list(set(range(edge_type.size(0))) - set(rows_to_remove))).cuda()
        new_edge_type   = torch.index_select(edge_type, 0, indices_to_keep)
        new_edge_index  = torch.index_select(edge_index, 1, indices_to_keep)  #2 * N

        #添加自定义连接
        new_edge_type  = torch.cat([new_edge_type, ligand_bond_type], dim = 0)  #扩充配体键类型
        new_edge_index = torch.cat([new_edge_index, new_ligand_bond_index], dim = 1)


        edge_type_dim = F.one_hot(new_edge_type, num_classes=20) #由原来的4种，变成了8种,除了把这里改了成8之外，其它地方也要改吧？否则报错
        #探索一下，哪有哪里在使用键长度
        return edge_type_dim, new_edge_index




    # @staticmethod #当需要self时，把静态函数声明去掉，否则会把self当作普通参数，而不是类对象
    def _build_edge_type_interaction_20_gpu_optim_distance(self, x, org_x, edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch,
        batch, #这个参数要结合mask_liagnd来确定哪些是配体哪些是蛋白上的节点
        atom_isring,
        atom_isO,
        atom_isN,

        cross_isring_flag, 
        cross_isO_flag, 
        cross_isN_flag, 
        cross_lp_pos,
        cross_distance,

        cross_bond_index, cross_bond_type, cross_bond_index_reverse, cross_bond_type_reverse
        ):
        #更cross_distance列表，获取对应配体到蛋白连接的距离，把这一部分也固定，否则就变成了带有噪音的配体坐标到蛋白的距离了
        #这一部分的实现，最好在数据生成时，不要在这里改多两个参数，cross_bond_distance，和 cross_bond_distance_reverse

        #cross_bond_index_reverse, cross_bond_type_reverse,是表示配体到蛋白的连接表的逆，即反向
        #可以全部换成GPU tensor运算，加速训练
        #获取配体的内部的原子id
        #把配体和蛋白的之间的连接替换成可能的相互作用
        ligand_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 1])))).cuda()
        #获取配体和蛋白放在一起的原子id
        protein_ligand_node_list = torch.LongTensor(list(range(len(mask_ligand)))).cuda()
        ligand_node_global = protein_ligand_node_list[mask_ligand == 1] #pytorch2.0要求datas[index]，数据和索引都在一个设备上

        #制作id映射
        ligand_node_local2global_dict = {}
        for k, v in zip(ligand_node_local, ligand_node_global):
            ligand_node_local2global_dict[k.item()] = v
        

        #更新id
        new_ligand_bond_index = torch.zeros(ligand_bond_index.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(ligand_bond_index.T):
            new_ligand_bond_index[i][0] = ligand_node_local2global_dict[bd[0].item()] #tensor张量不适合作为字典的k，所以要更换，否则报错
            new_ligand_bond_index[i][1] = ligand_node_local2global_dict[bd[1].item()]
        new_ligand_bond_index = new_ligand_bond_index.T



        '''
        #蛋白和配体之间的映射，重点像配体一样获取配体蛋白合并之后的id与未合并之前蛋白的id对应关系
        '''
        protein_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 0])))).cuda()
        #获取配体和蛋白放在一起的原子id
        protein_ligand_node_list = torch.LongTensor(list(range(len(mask_ligand)))).cuda()
        protein_node_global = protein_ligand_node_list[mask_ligand == 0] #pytorch2.0要求datas[index]，数据和索引都在一个设备上

        #制作id映射
        protein_node_local2global_dict = {}
        for k, v in zip(protein_node_local, protein_node_global):
            protein_node_local2global_dict[k.item()] = v


        #根据蛋白和配体的局部和全局id的映射，模仿配体的键id的更新，我们处理cross_bond_index, cross_bond_type， cross_bond_index_reverse, cross_bond_type_reverse
        #cross_bond_index_reverse, cross_bond_type_reverse是配体到蛋白的逆连接表, 键类型[5,6,7],逆连接的键类型[8,9,10]
        #更新id
        new_cross_bond_index = torch.zeros(cross_bond_index.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(cross_bond_index.T): #N * 2
            new_cross_bond_index[i][0] = ligand_node_local2global_dict[bd[0].item()] #第一列是配体原子
            new_cross_bond_index[i][1] = protein_node_local2global_dict[bd[1].item()] #第二列是蛋白原子
        new_cross_bond_index = new_cross_bond_index.T


        #更新id_reverse
        new_cross_bond_index_reverse = torch.zeros(cross_bond_index_reverse.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(cross_bond_index_reverse.T): #N * 2
            new_cross_bond_index_reverse[i][0] = protein_node_local2global_dict[bd[0].item()] #第一列是蛋白原子
            new_cross_bond_index_reverse[i][1] = ligand_node_local2global_dict[bd[1].item()] #第二列是配体原子
        new_cross_bond_index_reverse = new_cross_bond_index_reverse.T


        #不是键类型，是用来判断蛋白和配体之间的是否有连接
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1 #1表示是配体
        n_dst = mask_ligand[dst] == 1

        #需要扩充配体的键类型，增加单键，双键，三键，芳香键，我们需要知道配体和蛋白的这些键类型有什么作用？？？
        #edge_type在神经网络中只是和边长度嵌入和节点嵌入连接在了一起，并没有什么特殊的判断处理，虽然这里的0,1,2,3的确用来标识节点是蛋白或者配体上
        #但是在神经网络中用没有用来区分配体和蛋白节点
        edge_type[n_src & n_dst]   = 1  #表示在配体内部
        edge_type[n_src & ~n_dst]  = 5  #表示源节点在配体，目标节点在蛋白
        edge_type[~n_src & n_dst]  = 6  #表示源节点在蛋白，目标节点在配体
        edge_type[~n_src & ~n_dst] = 11  #表示在蛋白内部


        indices1 = (edge_type == 1).nonzero().view(-1) #寻找非0元素下标
        indices5 = (edge_type == 5).nonzero().view(-1) #寻找非0元素下标
        indices6 = (edge_type == 6).nonzero().view(-1) #寻找非0元素下标
        indices = torch.cat([indices1, indices5, indices6])
        #indices = (edge_type == 1).nonzero().squeeze() #寻找非0元素下标

        # 要删除的行的索引
        rows_to_remove = indices.detach().cpu().tolist()

        # 使用 torch.index_select() 函数选择不删除的行,去掉配体连接
        indices_to_keep = torch.tensor(list(set(range(edge_type.size(0))) - set(rows_to_remove))).cuda()
        new_edge_type   = torch.index_select(edge_type, 0, indices_to_keep)
        new_edge_index  = torch.index_select(edge_index, 1, indices_to_keep)  #2 * N

        #添加自定义连接
        new_edge_type  = torch.cat([new_edge_type, ligand_bond_type, cross_bond_type, cross_bond_type_reverse], dim = 0)  #扩充配体键类型
        new_edge_index = torch.cat([new_edge_index, new_ligand_bond_index, new_cross_bond_index, new_cross_bond_index_reverse], dim = 1)


        edge_type_dim = F.one_hot(new_edge_type, num_classes=20) #由原来的4种，变成了8种,除了把这里改了成8之外，其它地方也要改吧？否则报错
        #探索一下，哪有哪里在使用键长度
        return edge_type_dim, new_edge_index





    # @staticmethod #当需要self时，把静态函数声明去掉，否则会把self当作普通参数，而不是类对象
    def _build_edge_type_interaction_20_gpu_optim_distance_extend(self, x, org_x, edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch,
        batch, #这个参数要结合mask_liagnd来确定哪些是配体哪些是蛋白上的节点
        atom_isring,
        atom_isO,
        atom_isN,

        cross_isring_flag, 
        cross_isO_flag, 
        cross_isN_flag, 
        cross_lp_pos,
        cross_distance,

        cross_bond_index, cross_bond_type, cross_bond_index_reverse, cross_bond_type_reverse
        ):
        #更cross_distance列表，获取对应配体到蛋白连接的距离，把这一部分也固定，否则就变成了带有噪音的配体坐标到蛋白的距离了
        #这一部分的实现，最好在数据生成时，不要在这里改多两个参数，cross_bond_distance，和 cross_bond_distance_reverse

        #cross_bond_index_reverse, cross_bond_type_reverse,是表示配体到蛋白的连接表的逆，即反向
        #可以全部换成GPU tensor运算，加速训练
        #获取配体的内部的原子id
        #把配体和蛋白的之间的连接替换成可能的相互作用
        ligand_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 1])))).cuda()
        #获取配体和蛋白放在一起的原子id
        protein_ligand_node_list = torch.LongTensor(list(range(len(mask_ligand)))).cuda()
        ligand_node_global = protein_ligand_node_list[mask_ligand == 1] #pytorch2.0要求datas[index]，数据和索引都在一个设备上

        #制作id映射
        ligand_node_local2global_dict = {}
        for k, v in zip(ligand_node_local, ligand_node_global):
            ligand_node_local2global_dict[k.item()] = v
        

        #更新id
        new_ligand_bond_index = torch.zeros(ligand_bond_index.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(ligand_bond_index.T):
            new_ligand_bond_index[i][0] = ligand_node_local2global_dict[bd[0].item()] #tensor张量不适合作为字典的k，所以要更换，否则报错
            new_ligand_bond_index[i][1] = ligand_node_local2global_dict[bd[1].item()]
        new_ligand_bond_index = new_ligand_bond_index.T



        '''
        #蛋白和配体之间的映射，重点像配体一样获取配体蛋白合并之后的id与未合并之前蛋白的id对应关系
        '''
        protein_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 0])))).cuda()
        #获取配体和蛋白放在一起的原子id
        protein_ligand_node_list = torch.LongTensor(list(range(len(mask_ligand)))).cuda()
        protein_node_global = protein_ligand_node_list[mask_ligand == 0] #pytorch2.0要求datas[index]，数据和索引都在一个设备上

        #制作id映射
        protein_node_local2global_dict = {}
        for k, v in zip(protein_node_local, protein_node_global):
            protein_node_local2global_dict[k.item()] = v


        #根据蛋白和配体的局部和全局id的映射，模仿配体的键id的更新，我们处理cross_bond_index, cross_bond_type， cross_bond_index_reverse, cross_bond_type_reverse
        #cross_bond_index_reverse, cross_bond_type_reverse是配体到蛋白的逆连接表, 键类型[5,6,7],逆连接的键类型[8,9,10]
        #更新id
        new_cross_bond_index = torch.zeros(cross_bond_index.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(cross_bond_index.T): #N * 2
            new_cross_bond_index[i][0] = ligand_node_local2global_dict[bd[0].item()] #第一列是配体原子
            new_cross_bond_index[i][1] = protein_node_local2global_dict[bd[1].item()] #第二列是蛋白原子
        new_cross_bond_index = new_cross_bond_index.T


        #更新id_reverse
        new_cross_bond_index_reverse = torch.zeros(cross_bond_index_reverse.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(cross_bond_index_reverse.T): #N * 2
            new_cross_bond_index_reverse[i][0] = protein_node_local2global_dict[bd[0].item()] #第一列是蛋白原子
            new_cross_bond_index_reverse[i][1] = ligand_node_local2global_dict[bd[1].item()] #第二列是配体原子
        new_cross_bond_index_reverse = new_cross_bond_index_reverse.T


        #不是键类型，是用来判断蛋白和配体之间的是否有连接
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1 #1表示是配体
        n_dst = mask_ligand[dst] == 1

        #需要扩充配体的键类型，增加单键，双键，三键，芳香键，我们需要知道配体和蛋白的这些键类型有什么作用？？？
        #edge_type在神经网络中只是和边长度嵌入和节点嵌入连接在了一起，并没有什么特殊的判断处理，虽然这里的0,1,2,3的确用来标识节点是蛋白或者配体上
        #但是在神经网络中用没有用来区分配体和蛋白节点
        edge_type[n_src & n_dst]   = 1  #表示在配体内部
        edge_type[n_src & ~n_dst]  = 5  #表示源节点在配体，目标节点在蛋白
        edge_type[~n_src & n_dst]  = 6  #表示源节点在蛋白，目标节点在配体
        edge_type[~n_src & ~n_dst] = 11  #表示在蛋白内部


        indices1 = (edge_type == 1).nonzero().view(-1) #寻找非0元素下标
        indices5 = (edge_type == 5).nonzero().view(-1) #寻找非0元素下标
        indices6 = (edge_type == 6).nonzero().view(-1) #寻找非0元素下标
        indices = torch.cat([indices1, indices5, indices6])
        #indices = (edge_type == 1).nonzero().squeeze() #寻找非0元素下标

        # 要删除的行的索引
        rows_to_remove = indices.detach().cpu().tolist()

        # 使用 torch.index_select() 函数选择不删除的行,去掉配体连接
        indices_to_keep = torch.tensor(list(set(range(edge_type.size(0))) - set(rows_to_remove))).cuda()
        new_edge_type   = torch.index_select(edge_type, 0, indices_to_keep)
        new_edge_index  = torch.index_select(edge_index, 1, indices_to_keep)  #2 * N

        #添加自定义连接
        new_edge_type  = torch.cat([new_edge_type, ligand_bond_type, cross_bond_type, cross_bond_type_reverse], dim = 0)  #扩充配体键类型
        new_edge_index = torch.cat([new_edge_index, new_ligand_bond_index, new_cross_bond_index, new_cross_bond_index_reverse], dim = 1)

        #进一步扩充链接数量，之前已经用掉了12,13,14,15键类型，目前还要添加2类，16和17，用于标识配体和蛋白的扩充的键类型
        #第一步获取蛋白的所有坐标，以及对应的索引
        protein_index = protein_node_global
        #protein_pos   = x[protein_index]

        #根据new_cross_bond_index，找3.5~4.5范围内的蛋白原子，目前是<4.5以内的原子混合在一起了，所以这里先使用全部。计算这些蛋白原子的2ai范围的蛋白原子，并获取对应的索引
        #注意，关于连接的顺序是无所谓的，只要保证对接对一个的原子id是全局的即可
        #制作配体到蛋白的字典映射
        ligand_to_protein_index_dict = defaultdict(set)
        for ids_i, ids_j in new_cross_bond_index.T: #N*2
            ligand_to_protein_index_dict[ids_i].add(ids_j)

        extend_cross_bond_index = []
        # 遍历每一个配体原子, 东西太多，速度很慢
        for l_atom_index in ligand_to_protein_index_dict:
            p_atom_set = ligand_to_protein_index_dict[l_atom_index]

            #已有的蛋白索引
            exit_protein_index = p_atom_set #已有的蛋白原子
            #exit_protein_pos   = x[exit_protein_index]
            
            #剔除自环，即去掉原source_protein_index
            target_protein_index = set(protein_index) - set(exit_protein_index)
            #target_protein_pos   = x[target_protein_index]

            #计算exit_protein_index到target_protein_index的距离，得到小于2ai的蛋白原子索引

            #将两个向量，两两组合一起
            vec1 = torch.LongTensor(list(exit_protein_index)).cuda()
            vec2 = torch.LongTensor(list(target_protein_index)).cuda()
            # 使用 torch.meshgrid 构建两个向量的两两组合
            grid_x, grid_y = torch.meshgrid(vec1, vec2, indexing='ij')
            # 将组合的结果转换为两列的二维张量
            combination = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0) #shape = [2, n*m]
            
            #小于6ai的边, 约束太苛刻了，难有满足条件的,换大点的
            dis_limit = 2.0

            dis = torch.norm(x[combination[0]] - x[combination[1]], p = 2, dim = -1) #把整数变成浮点数
            dis_index = dis <= dis_limit #找满足条件的

            combination = combination.t()[dis_index] #k * 2
            combination = combination.t() # 2 * k
            extend_pro_atom_index = torch.unique(combination[1])


            #将配体[l_atom_index]和新得到的蛋白extend_pro_atom_index组合
            vec1 = torch.LongTensor([l_atom_index]).cuda()
            vec2 = extend_pro_atom_index.cuda()
            # 使用 torch.meshgrid 构建两个向量的两两组合
            grid_x, grid_y = torch.meshgrid(vec1, vec2, indexing='ij')
            # 将组合的结果转换为两列的二维张量
            combination = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0) #shape = [2, n*m]
            extend_cross_bond_index.append(combination)

        #之前已经用掉了12,13,14,15键类型，目前还要添加2类，16和17，用于标识配体和蛋白的扩充的键类型
        extend_cross_bond_index = torch.cat(extend_cross_bond_index, dim=-1)
        extend_cross_bond_type  = torch.full([extend_cross_bond_index.size(1)], 16, dtype = torch.int64).cuda()

        # 交换第一行和第二行
        extend_cross_bond_index_reverse = extend_cross_bond_index[[1, 0], :] 
        extend_cross_bond_type_reverse  = torch.full([extend_cross_bond_index_reverse.size(1)], 17, dtype = torch.int64).cuda()

        #添加自定义连接
        new_edge_type  = torch.cat([new_edge_type, extend_cross_bond_type, extend_cross_bond_type_reverse], dim = 0)  #扩充配体键类型
        new_edge_index = torch.cat([new_edge_index, extend_cross_bond_index, extend_cross_bond_index_reverse], dim = 1)

        edge_type_dim = F.one_hot(new_edge_type, num_classes=20) #由原来的4种，变成了8种,除了把这里改了成8之外，其它地方也要改吧？否则报错
        #探索一下，哪有哪里在使用键长度
        return edge_type_dim, new_edge_index





    def truncate(self, arr, decimals):
        factor = 10.0 ** decimals
        #return np.floor(arr * factor) / factor
        return int(arr * factor) #直接取整
        #return math.ceil(arr * factor) #向上取整
        #return np.round(arr, decimals)

    def truncate2(self, arr, decimals = 2):
        factor = 10.0 ** decimals
        #return np.floor(arr * factor) / factor
        #return int(arr * factor) #直接取整
        #return math.ceil(arr * factor) #向上取整
        return np.round(arr, decimals)

    def combinations_optim(self, x, org_x, atom1, atom2, atom_index1, atom_index2, centor, 
            cross_atom_flag1, cross_atom_flag2, cross_distance, cross_ligand, cross_protein, 
            flag
            ):
            #cross_ligand_atom_flag，cross_protein_atom_flag是一个整数张量，用于标识，跨距离配体和蛋白的哪些是O，N，环原子, 1是O, 2是N, 3是环, 0是其余原子
            #将两个向量，两两组合一起。实际上atom1, atom2是配体和蛋白原子的id，而atom_index1, atom_index2是他们对应的特殊原子的标志，是bool值
            #cross_ligand, cross_protein在处理数据的时候，别忘了和x一起减去质心

            if GP.interaction_stype == 'interaction':
                #如果提供基于距离的相互作用信息，则执行这一步
                vec1 = atom1[atom_index1]
                vec2 = atom2[atom_index2]
                x1 = org_x[vec1]
                x2 = org_x[vec2]
                if flag == 'ligand':
                    l_x = x1
                    p_x = x2
                    l_index = vec1
                    p_index = vec2
                    cross_ligand_atom_flag  = cross_atom_flag1
                    cross_protein_atom_flag = cross_atom_flag2
                elif flag == 'protein':
                    l_x = x2
                    p_x = x1
                    l_index = vec2
                    p_index = vec1
                    cross_ligand_atom_flag  = cross_atom_flag2
                    cross_protein_atom_flag = cross_atom_flag1

                #将坐标作为key，index为value
                l_x_index_dict = {}
                p_x_index_dict = {}
                
                l_x_index_dict2 = {}
                p_x_index_dict2 = {}

                for coord, index in zip(l_x, l_index):
                    #k = coord.sum()
                    #k = torch.round(k * 10000) / 10000 #取3位小数，torch.round只支持整数，所以要缩放

                    tg = ''
                    for i in coord:
                        #tg += str(round(i.item(), 3)) + '_'
                        tg += str(self.truncate(i.item(), 3)) + '_' #直接截断，保留2位小数，不进行四舍五入
                    k = str(tg)


                    v = index
                    l_x_index_dict[k] = v
                    
                    
                    tg = ''
                    for i in coord:
                        #tg += str(round(i.item(), 3)) + '_'
                        tg += str(self.truncate2(i.item(), 3)) + '_' #直接截断，保留2位小数，不进行四舍五入
                    k = str(tg)


                    v = index
                    l_x_index_dict2[k] = v

                
                assert len(l_x_index_dict) == len(l_x)
                assert len(l_x_index_dict2) == len(l_x)

                for coord, index in zip(p_x, p_index):
                    #k = coord.sum()
                    #k = torch.round(k * 10000) / 10000 #取3位小数，torch.round只支持整数，所以要缩放

                    tg = ''
                    for i in coord:
                        #tg += str(round(i.item(), 3)) + '_'
                        tg += str(self.truncate(i.item(), 3)) + '_' #直接截断，保留2位小数，不进行四舍五入
                    k = str(tg)

                    v = index
                    p_x_index_dict[k] = v
                    
                    
                    tg = ''
                    for i in coord:
                        #tg += str(round(i.item(), 3)) + '_'
                        tg += str(self.truncate2(i.item(), 3)) + '_' #直接截断，保留2位小数，不进行四舍五入
                    k = str(tg)

                    v = index
                    p_x_index_dict2[k] = v

                assert len(p_x_index_dict) == len(p_x)
                assert len(p_x_index_dict2) == len(p_x)

                #根据标志位，只保留特定类型的原子，进而得到对应的small_cross_distance, 缩小范围
                #注意cross_ligand_atom_flag, cross_protein_atom_flag要根据原子是O，N，环而发生改变，所以在传递参数的时候，以O~N为例，则应该这样
                # cross_ligand_atom_flag = cross_ligand_atom_flag[cross_ligand_atom_flag == 1], cross_protein_atom_flag = cross_protein_atom_flag[cross_protein_atom_flag == 2]
                #print('cross_distance:', cross_distance.shape) #cross_distance: torch.Size([13, 125])
                #print('cross_ligand_atom_flag, cross_protein_atom_flag:', cross_ligand_atom_flag.shape, cross_protein_atom_flag.shape) #torch.Size([13]) torch.Size([125])
                small_cross_distance = cross_distance[cross_ligand_atom_flag][:,cross_protein_atom_flag]
                small_cross_ligand   = cross_ligand[cross_ligand_atom_flag]
                small_cross_protein  = cross_protein[cross_protein_atom_flag]


                #根据坐标，获取small_cross_ligand和small_cross_protein在x中的下标位置
                ligand_index  = []
                protein_index = []

                for coord in small_cross_ligand:
                    #k = coord.sum()
                    #k = torch.round(k * 10000) / 10000 #取3位小数，torch.round只支持整数，所以要缩放

                    tg = ''
                    for i in coord:
                        #tg += str(round(i.item(), 3)) + '_'
                        tg += str(self.truncate(i.item(), 3)) + '_' #直接截断，保留2位小数，不进行四舍五入
                    k = str(tg)

                    try:
                        v = l_x_index_dict[k] #如果找不到，则报错，说明有问题，坐标对不上，理论上是一定可以找到，如果报错，则输出坐标值
                    except KeyError as e:
                        try:
                            tg = ''
                            for i in coord:
                                #tg += str(round(i.item(), 3)) + '_'
                                tg += str(self.truncate2(i.item(), 3)) + '_' #直接截断，保留2位小数，不进行四舍五入
                            k = str(tg)
                            v = l_x_index_dict2[k]
                        except KeyError as e:
                            print('error:', e)
                            print('l_x_index_dict.keys:', list(l_x_index_dict.keys()))
                            raise Exception('error')

                    ligand_index.append(v)

                for coord in small_cross_protein:
                    #k = coord.sum()
                    #k = torch.round(k * 10000) / 10000 #取3位小数，torch.round只支持整数，所以要缩放

                    tg = ''
                    for i in coord:
                        #tg += str(round(i.item(), 3)) + '_'
                        tg += str(self.truncate(i.item(), 3)) + '_' #直接截断，保留2位小数，不进行四舍五入
                    k = str(tg)
                    try:
                        v = p_x_index_dict[k] #如果找不到，则报错，说明有问题，坐标对不上，理论上是一定可以找到，如果报错，则输出坐标值
                    except KeyError as e:
                        try:
                            tg = ''
                            for i in coord:
                                #tg += str(round(i.item(), 3)) + '_'
                                tg += str(self.truncate2(i.item(), 3)) + '_' #直接截断，保留2位小数，不进行四舍五入
                            k = str(tg)
                            v = p_x_index_dict2[k]
                        except KeyError as e:
                            print('error:', e)
                            print('p_x_index_dict.keys:', list(p_x_index_dict.keys()))
                            raise Exception('error')
                
                    protein_index.append(v)

                #print('ligand_index:', ligand_index)  #张量组成的list
                #print('protein_index:', protein_index)#张量组成的list

                small_cross_distance_flag = (2.0 < small_cross_distance) & (small_cross_distance < GP.cross_distance_cutoff)  #对原子距离进一步约束，只要满足一定距离的蛋白原子，是不是不应该加以限制8ai？shape = [n, m]
                
                #print('small_cross_distance_flag.shape[0]*[1]:', small_cross_distance_flag.shape[0] * small_cross_distance_flag.shape[1]) #torch.Size([9, 37]), 37*9 = 333, 
                #print('small_cross_distance_flag.sum():', small_cross_distance_flag.sum()) # tensor(327, device='cuda:0') 这是8ai约束的，如果是6ai，则tensor(99, device='cuda:0')

                new_protein_index = []
                new_ligand_index  = []
                for k in range(small_cross_distance_flag.size(0)):
                    #print('protein_index:', protein_index)
                    if protein_index:
                        tg = torch.stack(protein_index, dim = 0)[small_cross_distance_flag[k]]
                        new_protein_index.append(tg) #tg是一个向量
                        new_ligand_index.append(ligand_index[k].view(-1)) #以向量的形式添加，所以变成向量
                    else:
                        #print('protein_index is [] ?:', protein_index)
                        pass
                
                #print('new_ligand_index:', new_ligand_index)  #向量组成的list
                #print('new_protein_index:', new_protein_index)#向量组成的list

                assert len(new_ligand_index) == len(new_protein_index)

                #raise Exception('test')

                #这一种不对，相当配体的O,N，环原子和蛋白的一对一了。
                #new_protein_index = protein_index
                #new_ligand_index  = ligand_index
                
                
                if flag == 'ligand':
                    combination_list = []
                    for l_i, p_i in zip(new_ligand_index, new_protein_index):
                        grid_x, grid_y = torch.meshgrid(l_i, p_i, indexing='ij') #grid_x, grid_y这是两两元素组合的结果
                        # 将组合的结果转换为两列的二维张量
                        combination = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0) #shape = [2, n*m]
                        combination_list.append(combination)

                elif flag == 'protein':
                    combination_list = []
                    for l_i, p_i in zip(new_ligand_index, new_protein_index):
                        grid_x, grid_y = torch.meshgrid(p_i, l_i, indexing='ij') #grid_x, grid_y这是两两元素组合的结果
                        # 将组合的结果转换为两列的二维张量
                        combination = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0) #shape = [2, n*m]
                        combination_list.append(combination)

                #print('combination_list:', combination_list)
                if combination_list:
                    combination = torch.cat(combination_list, dim = 1)
                else:
                    combination = torch.empty(0, 0) #为空，则返回一个空张量
                    #print('None') #好多为None的情况？
                    #combination = []


            elif GP.interaction_stype == 'interaction_all':
                #如果提供基于距离的相互作用信息，则执行这一步，
                #因为只使用4.5范围内的距离，所以4.5范围原子全部使用，不再区分o,n,环，另外这里可以放心使用，最后会有去重复的操作，不用担心重复问题
                #vec1 = atom1[atom_index1]
                #vec2 = atom2[atom_index2]
                vec1 = atom1
                vec2 = atom2

                x1 = org_x[vec1]
                x2 = org_x[vec2]
                if flag == 'ligand':
                    l_x = x1
                    p_x = x2
                    l_index = vec1
                    p_index = vec2
                    cross_ligand_atom_flag  = cross_atom_flag1
                    cross_protein_atom_flag = cross_atom_flag2
                elif flag == 'protein':
                    l_x = x2
                    p_x = x1
                    l_index = vec2
                    p_index = vec1
                    cross_ligand_atom_flag  = cross_atom_flag2
                    cross_protein_atom_flag = cross_atom_flag1

                #将坐标作为key，index为value
                l_x_index_dict = {}
                p_x_index_dict = {}
                
                l_x_index_dict2 = {}
                p_x_index_dict2 = {}

                for coord, index in zip(l_x, l_index):
                    #k = coord.sum()
                    #k = torch.round(k * 10000) / 10000 #取3位小数，torch.round只支持整数，所以要缩放

                    tg = ''
                    for i in coord:
                        #tg += str(round(i.item(), 3)) + '_'
                        tg += str(self.truncate(i.item(), 3)) + '_' #直接截断，保留2位小数，不进行四舍五入
                    k = str(tg)


                    v = index
                    l_x_index_dict[k] = v
                    
                    
                    tg = ''
                    for i in coord:
                        #tg += str(round(i.item(), 3)) + '_'
                        tg += str(self.truncate2(i.item(), 3)) + '_' #直接截断，保留2位小数，不进行四舍五入
                    k = str(tg)


                    v = index
                    l_x_index_dict2[k] = v

                
                assert len(l_x_index_dict) == len(l_x)
                assert len(l_x_index_dict2) == len(l_x)

                for coord, index in zip(p_x, p_index):
                    #k = coord.sum()
                    #k = torch.round(k * 10000) / 10000 #取3位小数，torch.round只支持整数，所以要缩放

                    tg = ''
                    for i in coord:
                        #tg += str(round(i.item(), 3)) + '_'
                        tg += str(self.truncate(i.item(), 3)) + '_' #直接截断，保留2位小数，不进行四舍五入
                    k = str(tg)

                    v = index
                    p_x_index_dict[k] = v
                    
                    
                    tg = ''
                    for i in coord:
                        #tg += str(round(i.item(), 3)) + '_'
                        tg += str(self.truncate2(i.item(), 3)) + '_' #直接截断，保留2位小数，不进行四舍五入
                    k = str(tg)

                    v = index
                    p_x_index_dict2[k] = v

                assert len(p_x_index_dict) == len(p_x)
                assert len(p_x_index_dict2) == len(p_x)

                #根据标志位，只保留特定类型的原子，进而得到对应的small_cross_distance, 缩小范围
                #注意cross_ligand_atom_flag, cross_protein_atom_flag要根据原子是O，N，环而发生改变，所以在传递参数的时候，以O~N为例，则应该这样
                # cross_ligand_atom_flag = cross_ligand_atom_flag[cross_ligand_atom_flag == 1], cross_protein_atom_flag = cross_protein_atom_flag[cross_protein_atom_flag == 2]
                #print('cross_distance:', cross_distance.shape) #cross_distance: torch.Size([13, 125])
                #print('cross_ligand_atom_flag, cross_protein_atom_flag:', cross_ligand_atom_flag.shape, cross_protein_atom_flag.shape) #torch.Size([13]) torch.Size([125])
                #small_cross_distance = cross_distance[cross_ligand_atom_flag][:,cross_protein_atom_flag]
                #small_cross_ligand   = cross_ligand[cross_ligand_atom_flag]
                #small_cross_protein  = cross_protein[cross_protein_atom_flag]

                #全部使用
                small_cross_distance = cross_distance
                small_cross_ligand   = cross_ligand
                small_cross_protein  = cross_protein


                #根据坐标，获取small_cross_ligand和small_cross_protein在x中的下标位置
                ligand_index  = []
                protein_index = []

                for coord in small_cross_ligand:
                    #k = coord.sum()
                    #k = torch.round(k * 10000) / 10000 #取3位小数，torch.round只支持整数，所以要缩放

                    tg = ''
                    for i in coord:
                        #tg += str(round(i.item(), 3)) + '_'
                        tg += str(self.truncate(i.item(), 3)) + '_' #直接截断，保留2位小数，不进行四舍五入
                    k = str(tg)

                    try:
                        v = l_x_index_dict[k] #如果找不到，则报错，说明有问题，坐标对不上，理论上是一定可以找到，如果报错，则输出坐标值
                    except KeyError as e:
                        try:
                            tg = ''
                            for i in coord:
                                #tg += str(round(i.item(), 3)) + '_'
                                tg += str(self.truncate2(i.item(), 3)) + '_' #直接截断，保留2位小数，不进行四舍五入
                            k = str(tg)
                            v = l_x_index_dict2[k]
                        except KeyError as e:
                            print('error:', e)
                            print('l_x_index_dict.keys:', list(l_x_index_dict.keys()))
                            raise Exception('error')

                    ligand_index.append(v)

                for coord in small_cross_protein:
                    #k = coord.sum()
                    #k = torch.round(k * 10000) / 10000 #取3位小数，torch.round只支持整数，所以要缩放

                    tg = ''
                    for i in coord:
                        #tg += str(round(i.item(), 3)) + '_'
                        tg += str(self.truncate(i.item(), 3)) + '_' #直接截断，保留2位小数，不进行四舍五入
                    k = str(tg)
                    try:
                        v = p_x_index_dict[k] #如果找不到，则报错，说明有问题，坐标对不上，理论上是一定可以找到，如果报错，则输出坐标值
                    except KeyError as e:
                        try:
                            tg = ''
                            for i in coord:
                                #tg += str(round(i.item(), 3)) + '_'
                                tg += str(self.truncate2(i.item(), 3)) + '_' #直接截断，保留2位小数，不进行四舍五入
                            k = str(tg)
                            v = p_x_index_dict2[k]
                        except KeyError as e:
                            print('error:', e)
                            print('p_x_index_dict.keys:', list(p_x_index_dict.keys()))
                            raise Exception('error')
                
                    protein_index.append(v)

                #print('ligand_index:', ligand_index)  #张量组成的list
                #print('protein_index:', protein_index)#张量组成的list

                small_cross_distance_flag = (2.0 < small_cross_distance) & (small_cross_distance < GP.cross_distance_cutoff)  #对原子距离进一步约束，只要满足一定距离的蛋白原子，是不是不应该加以限制8ai？shape = [n, m]
                
                #print('small_cross_distance_flag.shape[0]*[1]:', small_cross_distance_flag.shape[0] * small_cross_distance_flag.shape[1]) #torch.Size([9, 37]), 37*9 = 333, 
                #print('small_cross_distance_flag.sum():', small_cross_distance_flag.sum()) # tensor(327, device='cuda:0') 这是8ai约束的，如果是6ai，则tensor(99, device='cuda:0')

                new_protein_index = []
                new_ligand_index  = []
                for k in range(small_cross_distance_flag.size(0)):
                    #print('protein_index:', protein_index)
                    if protein_index:
                        tg = torch.stack(protein_index, dim = 0)[small_cross_distance_flag[k]]
                        new_protein_index.append(tg) #tg是一个向量
                        new_ligand_index.append(ligand_index[k].view(-1)) #以向量的形式添加，所以变成向量
                    else:
                        #print('protein_index is [] ?:', protein_index)
                        pass
                
                #print('new_ligand_index:', new_ligand_index)  #向量组成的list
                #print('new_protein_index:', new_protein_index)#向量组成的list

                assert len(new_ligand_index) == len(new_protein_index)

                #raise Exception('test')

                #这一种不对，相当配体的O,N，环原子和蛋白的一对一了。
                #new_protein_index = protein_index
                #new_ligand_index  = ligand_index
                
                
                if flag == 'ligand':
                    combination_list = []
                    for l_i, p_i in zip(new_ligand_index, new_protein_index):
                        grid_x, grid_y = torch.meshgrid(l_i, p_i, indexing='ij') #grid_x, grid_y这是两两元素组合的结果
                        # 将组合的结果转换为两列的二维张量
                        combination = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0) #shape = [2, n*m]
                        combination_list.append(combination)

                elif flag == 'protein':
                    combination_list = []
                    for l_i, p_i in zip(new_ligand_index, new_protein_index):
                        grid_x, grid_y = torch.meshgrid(p_i, l_i, indexing='ij') #grid_x, grid_y这是两两元素组合的结果
                        # 将组合的结果转换为两列的二维张量
                        combination = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0) #shape = [2, n*m]
                        combination_list.append(combination)

                #print('combination_list:', combination_list)
                if combination_list:
                    combination = torch.cat(combination_list, dim = 1)
                else:
                    combination = torch.empty(0, 0) #为空，则返回一个空张量
                    #print('None') #好多为None的情况？
                    #combination = []




            elif GP.interaction_stype == 'centor': # 
                vec1 = atom1[atom_index1]
                vec2 = atom2[atom_index2]

                #小于6ai的边, 约束太苛刻了，难有满足条件的,换大点的
                dis_limit = GP.interaction_distance  #默认是8，可以调整

                if flag == 'ligand':
                    dis = torch.norm(centor - x[vec2], p = 2, dim = -1) #把整数变成浮点数
                    dis_index = dis <= dis_limit #找满足条件的
                    new_vec2 = vec2[dis_index]
                    grid_x, grid_y = torch.meshgrid(vec1, new_vec2, indexing='ij') #grid_x, grid_y这是两两元素组合的结果
                    # 将组合的结果转换为两列的二维张量
                    combination = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0) #shape = [2, n*m] 

                elif flag == 'protein':
                    dis = torch.norm(centor - x[vec1], p = 2, dim = -1) #把整数变成浮点数
                    dis_index = dis <= dis_limit #找满足条件的
                    new_vec1 = vec1[dis_index]
                    grid_x, grid_y = torch.meshgrid(new_vec1, vec2, indexing='ij') #grid_x, grid_y这是两两元素组合的结果
                    # 将组合的结果转换为两列的二维张量
                    combination = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0) #shape = [2, n*m]

                # 根据帅选后两两的连接表，我们将combination.t()的分开，去重复，然后再两两组合，即达到配体的每一个O, N, 环原子和蛋白中的所有O, N, 环原子存在连接，即全连接
                vec1 = torch.unique(combination[0]) #先去除重复 unique_consecutive只是去除相邻元素的重复，而不是全部
                vec2 = torch.unique(combination[1])
                # 使用 torch.meshgrid 构建两个向量的两两组合， vec1, vec2实际就是特殊原子的id
                grid_x, grid_y = torch.meshgrid(vec1, vec2, indexing='ij')
                combination = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0) #shape = [2, n*m] 

            elif GP.interaction_stype == 'all':
                #最终将两个向量，两两组合一起
                vec1 = atom1[atom_index1]
                vec2 = atom2[atom_index2]
                # 使用 torch.meshgrid 构建两个向量的两两组合
                grid_x, grid_y = torch.meshgrid(vec1, vec2, indexing='ij')
                # 将组合的结果转换为两列的二维张量
                combination = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0) #shape = [2, n*m]

            elif GP.interaction_stype == 'distance':  #这种方法还有个问题，就是需要知道参考配体，显然不合适
                raise Exception('Unsupport combinations_optim, please change')
                '''
                grid_x, grid_y = torch.meshgrid(atom1, atom2, indexing='ij') #grid_x, grid_y这是两两元素组合的结果
                # 将组合的结果转换为两列的二维张量
                combination = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0) #shape = [2, n*m]

                #以每一个原子为中心构建相互作用，之后建立全连接
                copy_combination = combination.clone()
                
                #对配体和蛋白原子距离进行排序，找前200个
                dis = torch.norm(x[combination[0]] - x[combination[1]], p = 2, dim = -1) #把整数变成浮点数
                sorted_dis, sorted_indices = torch.sort(dis) # 排序，并返回对应下标

                atom_num  = GP.min_distance_atom_num  #取最近距离前num个
                dis_index = sorted_indices[:atom_num]  #找满足条件的

                combination = combination.t()[dis_index] ##shape = [2, n*m] -> [n*m, 2]

                #对atom1, atom2形成新的标识，即满足距离条件的，存在combination中的原子，标识成True，然后再相互作用标识atom_index1, atom_index2求交集
                new_atom_index1, new_atom_index2 = torch.zeros_like(atom_index1, dtype=torch.bool), torch.zeros_like(atom_index2, dtype=torch.bool)

                vec1 = torch.unique(combination.t()[0]) #先去除重复，保持顺序，再两两组合
                vec2 = torch.unique(combination.t()[1])

                for i in vec1:
                    new_atom_index1[i] = True
                
                for i in vec2:
                    new_atom_index2[i] = True
                
                final_atom_index1, final_atom_index2 = new_atom_index1 & atom_index1, new_atom_index2 & atom_index2


                #最终将两个向量，两两组合一起
                vec1 = atom1[final_atom_index1]
                vec2 = atom2[final_atom_index2]
                # 使用 torch.meshgrid 构建两个向量的两两组合
                grid_x, grid_y = torch.meshgrid(vec1, vec2, indexing='ij')
                # 将组合的结果转换为两列的二维张量
                combination = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0) #shape = [2, n*m]
                '''


            else:
                raise Exception('Unsupport combinations_optim, please change')
                """
                vec1 = atom1[atom_index1]
                vec2 = atom2[atom_index2]

                # 使用 torch.meshgrid 构建两个向量的两两组合， vec1, vec2实际就是特殊原子的id
                grid_x, grid_y = torch.meshgrid(vec1, vec2, indexing='ij') #grid_x, grid_y这是两两元素组合的结果
                # 将组合的结果转换为两列的二维张量
                combination = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0) #shape = [2, n*m] 

                #以每一个原子为中心构建相互作用，之后建立全连接
                copy_combination = combination.clone()
                
                #小于6ai的边, 约束太苛刻了，难有满足条件的,换大点的
                dis_limit = GP.interaction_distance  #默认是8，可以调整

                dis = torch.norm(x[combination[0]] - x[combination[1]], p = 2, dim = -1) #把整数变成浮点数
                dis_index = dis <= dis_limit #找满足条件的

                combination = combination.t()[dis_index]

                '''
                if len(combination) == 0:
                    #print('使用更大范围的距离限制12')
                    dis_limit = 12.0 #如果是空，则使用更大范围的信息
                    combination = copy_combination.clone()
                    dis = torch.norm(x[combination[0]] - x[combination[1]], p = 2, dim = -1) #把整数变成浮点数
                    dis_index = dis <= dis_limit #找满足条件的

                    combination = combination.t()[dis_index]
                '''
                
                if len(combination) == 0:
                    #print('放开距离限制')
                    dis_limit = 100000000.0 #如果还是空，则放开限制
                    combination = copy_combination.clone()
                    dis = torch.norm(x[combination[0]] - x[combination[1]], p = 2, dim = -1) #把整数变成浮点数
                    dis_index = dis <= dis_limit #找满足条件的

                    combination = combination.t()[dis_index]

                '''
                #只要小于6ai的边, 约束太苛刻了，难有满足条件的。如果我们直接取邻近的60个原子，但现在知道边，有没法直接矩阵运算，实现不了
                nun_limit = 60

                dis = torch.norm(combination.to(torch.float32), p = 2, dim = 1) #把整数变成浮点数
                dis_index = dis <= dis_limit #找满足添加的

                combination = combination[dis_index].t()
                '''

                # 根据帅选后两两的连接表，我们将combination.t()的分开，去重复，然后再两两组合，即达到配体的每一个O, N, 环原子和蛋白中的所有O, N, 环原子存在连接，即全连接
                vec1 = torch.unique(combination.t()[0]) #先去除重复，保持顺序，再两两组合
                vec2 = torch.unique(combination.t()[1])
                # 使用 torch.meshgrid 构建两个向量的两两组合， vec1, vec2实际就是特殊原子的id
                grid_x, grid_y = torch.meshgrid(vec1, vec2, indexing='ij')
                combination = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0) #shape = [2, n*m] 
                """
            #不去重复了
            #if combination.shape[0] != 0:
                #combination = torch.unique(combination, dim = -1)
            return combination






    @staticmethod
    def _build_edge_type_82(edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch):
        #获取配体的内部的原子id
        ligand_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 1])))).numpy()

    
        #获取配体和蛋白放在一起的原子id
        protein_ligand_node_list = list(range(len(mask_ligand)))

        ligand_node_global = torch.LongTensor(protein_ligand_node_list)[mask_ligand == 1].numpy()

        ##print('ligand_node_list:', len(ligand_node_local))
        ##print('protein_ligand_node_list:', len(protein_ligand_node_list))
        ##print('ligand_node_global:', len(ligand_node_global))
        ##print('edge_index:', edge_index.shape)

        '''
        ligand_node_list: 282
        protein_ligand_node_list: 3388
        ligand_node_global: 282
        edge_index: torch.Size([2, 108416])
        '''

        #制作id映射
        ligand_node_local2global_dict = {}
        for k, v in zip(ligand_node_local, ligand_node_global):
            ligand_node_local2global_dict[k] = v
        

        #更新id
        new_ligand_bond_index = torch.zeros(ligand_bond_index.T.shape, dtype = torch.int64).numpy()
        for i, bd in enumerate(ligand_bond_index.T.detach().cpu().numpy()):
            ##print('bd:', bd) #bd: [0 1]
            new_ligand_bond_index[i][0] = ligand_node_local2global_dict[bd[0]]
            new_ligand_bond_index[i][1] = ligand_node_local2global_dict[bd[1]]
        
        new_ligand_bond_index = torch.from_numpy(new_ligand_bond_index.T).cuda()
        ##print('new_ligand_bond_index:', new_ligand_bond_index.shape) #torch.Size([2, 582])


        #raise Exception('test')
        # NxN connectivity matrix where 0 means no connection and 1/2/3/4 means single/double/triple/aromatic bonds.

        #不是键类型，是用来判断蛋白和配体之间的是否有连接
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1 #1表示是配体
        n_dst = mask_ligand[dst] == 1

        #需要扩充配体的键类型，增加单键，双键，三键，芳香键，我们需要知道配体和蛋白的这些键类型有什么作用？？？
        #edge_type在神经网络中只是和边长度嵌入和节点嵌入连接在了一起，并没有什么特殊的判断处理，虽然这里的0,1,2,3的确用来标识节点是蛋白或者配体上
        #但是在神经网络中用没有用来区分配体和蛋白节点
        edge_type[n_src & n_dst]   = 1  #表示在配体内部
        edge_type[n_src & ~n_dst]  = 5  #表示源节点在配体，目标节点在蛋白
        edge_type[~n_src & n_dst]  = 6  #表示源节点在蛋白，目标节点在配体
        edge_type[~n_src & ~n_dst] = 7  #表示在蛋白内部


        indices = (edge_type == 1).nonzero().squeeze() #寻找非0元素下标

        # 要删除的行的索引
        rows_to_remove = indices.detach().cpu().tolist()
        
        '''
        # 使用 torch.index_select() 函数选择不删除的行,去掉配体连接
        indices_to_keep = torch.tensor(list(set(range(edge_type.size(0))) - set(rows_to_remove))).cuda()
        new_edge_type   = torch.index_select(edge_type, 0, indices_to_keep)
        new_edge_index  = torch.index_select(edge_index, 1, indices_to_keep)  #2 * N
        '''

        #添加配体内部连接
        ##print('ligand_bond_type:', ligand_bond_type)
        ##print('new_ligand_bond_index:', new_ligand_bond_index)

        #在KNN图的基础上，我们添加了配体的内部的原有邻接表，这部分是正确，已知的
        #new_edge_type  = torch.cat([edge_type, ligand_bond_type], dim = 0)  #扩充配体键类型
        new_edge_type = edge_type #配体连接表动态变化，我们难以获取对应的键类型，因为此时ligand_bond_type与使用KNN构建的配体链接不对应
        #new_edge_type  = torch.cat([new_edge_type, torch.zeros_like(ligand_bond_type, dtype = torch.int64)], dim = 0)#不扩充配体键类型，依旧使用0
        #new_edge_index = torch.cat([edge_index, new_ligand_bond_index], dim = 1)
        new_edge_index = edge_index

        #

        ##print('new_edge_type:', new_edge_type.shape)
        ##print('new_edge_index:', new_edge_index.shape)
        #new_edge_type: torch.Size([103536])
        #new_edge_index: torch.Size([2, 103536])

        #(edge_type,edge_index)每一行的顺序是无所谓的，不影响卷积，关键在于里面使用的原子id一定要是根据当前批量获得的全局id
        #(edge_type,edge_index)只要配体和蛋白之间的，其余的全部由配体和蛋白的全连接图来取代
        edge_type_dim = F.one_hot(new_edge_type, num_classes=8) #由原来的4种，变成了8种,除了把这里改了成8之外，其它地方也要改吧？否则报错
        #探索一下，哪有哪里在使用键长度
        return edge_type_dim, new_edge_index


        #new_edge_type  = torch.cat([new_edge_type, ligand_bond_type], dim = 0)  #扩充配体键类型
        new_edge_type = edge_type
        #new_edge_type  = torch.cat([new_edge_type, torch.zeros_like(ligand_bond_type, dtype = torch.int64)], dim = 0)#不扩充配体键类型，依旧使用0
        #new_edge_index = torch.cat([new_edge_index, new_ligand_bond_index], dim = 1)
        new_edge_index = edge_index


    @staticmethod
    def _build_edge_type_82_gpu(edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch):
        #可以全部换成GPU tensor运算，加速训练
        #获取配体的内部的原子id
        ligand_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 1])))).cuda()

    
        #获取配体和蛋白放在一起的原子id
        protein_ligand_node_list = torch.LongTensor(list(range(len(mask_ligand)))).cuda()

        ligand_node_global = protein_ligand_node_list[mask_ligand == 1] #pytorch2.0要求datas[index]，数据和索引都在一个设备上

        ##print('ligand_node_list:', len(ligand_node_local))
        ##print('protein_ligand_node_list:', len(protein_ligand_node_list))
        ##print('ligand_node_global:', len(ligand_node_global))
        ##print('edge_index:', edge_index.shape)

        '''
        ligand_node_list: 282
        protein_ligand_node_list: 3388
        ligand_node_global: 282
        edge_index: torch.Size([2, 108416])
        '''

        #制作id映射
        ligand_node_local2global_dict = {}
        for k, v in zip(ligand_node_local, ligand_node_global):
            ligand_node_local2global_dict[k.item()] = v
        

        #更新id
        new_ligand_bond_index = torch.zeros(ligand_bond_index.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(ligand_bond_index.T):
            #print('bd:', bd) #numpy数据 bd: [0, 9], 如果是torch tensor，这是 tensor([0, 9], device='cuda:0')
            #print('bd[0]:', bd[0]) #bd[0]: tensor(0, device='cuda:0')
            #print('bd[1]:', bd[1]) #bd[0]: tensor(9, device='cuda:0')
            #print('ligand_node_local:', ligand_node_local) #ligand_node_local: tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,18, 19], device='cuda:0')
            #exit()
            #print('ligand_node_local2global_dict:', ligand_node_local2global_dict)
            new_ligand_bond_index[i][0] = ligand_node_local2global_dict[bd[0].item()] #tensor张量不适合作为字典的k，所以要更换，否则报错
            new_ligand_bond_index[i][1] = ligand_node_local2global_dict[bd[1].item()]
        
        new_ligand_bond_index = new_ligand_bond_index.T
        ##print('new_ligand_bond_index:', new_ligand_bond_index.shape) #torch.Size([2, 582])


        #raise Exception('test')
        # NxN connectivity matrix where 0 means no connection and 1/2/3/4 means single/double/triple/aromatic bonds.

        #不是键类型，是用来判断蛋白和配体之间的是否有连接
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1 #1表示是配体
        n_dst = mask_ligand[dst] == 1

        #需要扩充配体的键类型，增加单键，双键，三键，芳香键，我们需要知道配体和蛋白的这些键类型有什么作用？？？
        #edge_type在神经网络中只是和边长度嵌入和节点嵌入连接在了一起，并没有什么特殊的判断处理，虽然这里的0,1,2,3的确用来标识节点是蛋白或者配体上
        #但是在神经网络中用没有用来区分配体和蛋白节点
        edge_type[n_src & n_dst]   = 1  #表示在配体内部
        edge_type[n_src & ~n_dst]  = 5  #表示源节点在配体，目标节点在蛋白
        edge_type[~n_src & n_dst]  = 6  #表示源节点在蛋白，目标节点在配体
        edge_type[~n_src & ~n_dst] = 7  #表示在蛋白内部


        indices = (edge_type == 1).nonzero().view(-1) #寻找非0元素下标
        #indices = (edge_type == 1).nonzero().squeeze() #寻找非0元素下标

        # 要删除的行的索引
        rows_to_remove = indices.detach().cpu().tolist()
        
        '''
        # 使用 torch.index_select() 函数选择不删除的行,去掉配体连接
        indices_to_keep = torch.tensor(list(set(range(edge_type.size(0))) - set(rows_to_remove))).cuda()
        new_edge_type   = torch.index_select(edge_type, 0, indices_to_keep)
        new_edge_index  = torch.index_select(edge_index, 1, indices_to_keep)  #2 * N
        '''
        
        #添加配体内部连接
        ##print('ligand_bond_type:', ligand_bond_type)
        ##print('new_ligand_bond_index:', new_ligand_bond_index)

    
        #new_edge_type  = torch.cat([new_edge_type, ligand_bond_type], dim = 0)  #扩充配体键类型
        new_edge_type = edge_type
        #new_edge_type  = torch.cat([new_edge_type, torch.zeros_like(ligand_bond_type, dtype = torch.int64)], dim = 0)#不扩充配体键类型，依旧使用0
        #new_edge_index = torch.cat([new_edge_index, new_ligand_bond_index], dim = 1)
        new_edge_index = edge_index

        #

        ##print('new_edge_type:', new_edge_type.shape)
        ##print('new_edge_index:', new_edge_index.shape)
        #new_edge_type: torch.Size([103536])
        #new_edge_index: torch.Size([2, 103536])

        #(edge_type,edge_index)每一行的顺序是无所谓的，不影响卷积，关键在于里面使用的原子id一定要是根据当前批量获得的全局id
        #(edge_type,edge_index)只要配体和蛋白之间的，其余的全部由配体和蛋白的全连接图来取代
        edge_type_dim = F.one_hot(new_edge_type, num_classes=8) #由原来的4种，变成了8种,除了把这里改了成8之外，其它地方也要改吧？否则报错
        #探索一下，哪有哪里在使用键长度
        return edge_type_dim, new_edge_index

    @staticmethod
    def _build_edge_type_4(edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch):
        #键长为4
        #获取配体的内部的原子id
        ligand_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 1])))).numpy()

    
        #获取配体和蛋白放在一起的原子id
        protein_ligand_node_list = list(range(len(mask_ligand)))

        ligand_node_global = torch.LongTensor(protein_ligand_node_list)[mask_ligand == 1].numpy()

        ##print('ligand_node_list:', len(ligand_node_local))
        ##print('protein_ligand_node_list:', len(protein_ligand_node_list))
        ##print('ligand_node_global:', len(ligand_node_global))
        ##print('edge_index:', edge_index.shape)

        '''
        ligand_node_list: 282
        protein_ligand_node_list: 3388
        ligand_node_global: 282
        edge_index: torch.Size([2, 108416])
        '''

        #制作id映射
        ligand_node_local2global_dict = {}
        for k, v in zip(ligand_node_local, ligand_node_global):
            ligand_node_local2global_dict[k] = v
        

        #更新id
        new_ligand_bond_index = torch.zeros(ligand_bond_index.T.shape, dtype = torch.int64).numpy()
        for i, bd in enumerate(ligand_bond_index.T.detach().cpu().numpy()):
            ##print('bd:', bd) #bd: [0 1]
            new_ligand_bond_index[i][0] = ligand_node_local2global_dict[bd[0]]
            new_ligand_bond_index[i][1] = ligand_node_local2global_dict[bd[1]]
        
        new_ligand_bond_index = torch.from_numpy(new_ligand_bond_index.T).cuda()
        ##print('new_ligand_bond_index:', new_ligand_bond_index.shape) #torch.Size([2, 582])


        #raise Exception('test')
        # NxN connectivity matrix where 0 means no connection and 1/2/3/4 means single/double/triple/aromatic bonds.

        #不是键类型，是用来判断蛋白和配体之间的是否有连接
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1 #1表示是配体
        n_dst = mask_ligand[dst] == 1

        #需要扩充配体的键类型，增加单键，双键，三键，芳香键，我们需要知道配体和蛋白的这些键类型有什么作用？？？
        #edge_type在神经网络中只是和边长度嵌入和节点嵌入连接在了一起，并没有什么特殊的判断处理，虽然这里的0,1,2,3的确用来标识节点是蛋白或者配体上
        #但是在神经网络中用没有用来区分配体和蛋白节点
        edge_type[n_src & n_dst]   = 0  #表示在配体内部
        edge_type[n_src & ~n_dst]  = 1  #表示源节点在配体，目标节点在蛋白
        edge_type[~n_src & n_dst]  = 2  #表示源节点在蛋白，目标节点在配体
        edge_type[~n_src & ~n_dst] = 3  #表示在蛋白内部


        indices = (edge_type == 0).nonzero().squeeze() #寻找非0元素下标，这里要改成edge_type == 0

        # 要删除的行的索引
        rows_to_remove = indices.detach().cpu().tolist()

        # 使用 torch.index_select() 函数选择不删除的行,去掉配体连接
        indices_to_keep = torch.tensor(list(set(range(edge_type.size(0))) - set(rows_to_remove))).cuda()
        new_edge_type   = torch.index_select(edge_type, 0, indices_to_keep)
        new_edge_index  = torch.index_select(edge_index, 1, indices_to_keep)  #2 * N

        #添加配体内部连接
        ##print('ligand_bond_type:', ligand_bond_type)
        ##print('new_ligand_bond_index:', new_ligand_bond_index)

    
        #new_edge_type  = torch.cat([new_edge_type, ligand_bond_type], dim = 0)  #扩充配体键类型
        new_edge_type  = torch.cat([new_edge_type, torch.zeros_like(ligand_bond_type, dtype = torch.int64)], dim = 0)#不扩充配体键类型，依旧使用0
        new_edge_index = torch.cat([new_edge_index, new_ligand_bond_index], dim = 1)

        #

        ##print('new_edge_type:', new_edge_type.shape)
        ##print('new_edge_index:', new_edge_index.shape)
        #new_edge_type: torch.Size([103536])
        #new_edge_index: torch.Size([2, 103536])

        #(edge_type,edge_index)每一行的顺序是无所谓的，不影响卷积，关键在于里面使用的原子id一定要是根据当前批量获得的全局id
        #(edge_type,edge_index)只要配体和蛋白之间的，其余的全部由配体和蛋白的全连接图来取代
        edge_type_dim = F.one_hot(new_edge_type, num_classes=4) #由原来的4种，变成了8种,除了把这里改了成8之外，其它地方也要改吧？否则报错
        #探索一下，哪有哪里在使用键长度
        return edge_type_dim, new_edge_index
    





    @staticmethod
    def _build_edge_type_42(edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch):
        #键长为4
        #获取配体的内部的原子id
        ligand_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 1])))).numpy()

    
        #获取配体和蛋白放在一起的原子id
        protein_ligand_node_list = list(range(len(mask_ligand)))

        ligand_node_global = torch.LongTensor(protein_ligand_node_list)[mask_ligand == 1].numpy()

        ##print('ligand_node_list:', len(ligand_node_local))
        ##print('protein_ligand_node_list:', len(protein_ligand_node_list))
        ##print('ligand_node_global:', len(ligand_node_global))
        ##print('edge_index:', edge_index.shape)

        '''
        ligand_node_list: 282
        protein_ligand_node_list: 3388
        ligand_node_global: 282
        edge_index: torch.Size([2, 108416])
        '''

        #制作id映射
        ligand_node_local2global_dict = {}
        for k, v in zip(ligand_node_local, ligand_node_global):
            ligand_node_local2global_dict[k] = v
        

        #更新id
        new_ligand_bond_index = torch.zeros(ligand_bond_index.T.shape, dtype = torch.int64).numpy()
        for i, bd in enumerate(ligand_bond_index.T.detach().cpu().numpy()):
            ##print('bd:', bd) #bd: [0 1]
            new_ligand_bond_index[i][0] = ligand_node_local2global_dict[bd[0]]
            new_ligand_bond_index[i][1] = ligand_node_local2global_dict[bd[1]]
        
        new_ligand_bond_index = torch.from_numpy(new_ligand_bond_index.T).cuda()
        ##print('new_ligand_bond_index:', new_ligand_bond_index.shape) #torch.Size([2, 582])


        #raise Exception('test')
        # NxN connectivity matrix where 0 means no connection and 1/2/3/4 means single/double/triple/aromatic bonds.

        #不是键类型，是用来判断蛋白和配体之间的是否有连接
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1 #1表示是配体
        n_dst = mask_ligand[dst] == 1

        #需要扩充配体的键类型，增加单键，双键，三键，芳香键，我们需要知道配体和蛋白的这些键类型有什么作用？？？
        #edge_type在神经网络中只是和边长度嵌入和节点嵌入连接在了一起，并没有什么特殊的判断处理，虽然这里的0,1,2,3的确用来标识节点是蛋白或者配体上
        #但是在神经网络中用没有用来区分配体和蛋白节点
        edge_type[n_src & n_dst]   = 0  #表示在配体内部
        edge_type[n_src & ~n_dst]  = 1  #表示源节点在配体，目标节点在蛋白
        edge_type[~n_src & n_dst]  = 2  #表示源节点在蛋白，目标节点在配体
        edge_type[~n_src & ~n_dst] = 3  #表示在蛋白内部


        indices = (edge_type == 0).nonzero().squeeze() #寻找非0元素下标，这里要改成edge_type == 0

        # 要删除的行的索引
        rows_to_remove = indices.detach().cpu().tolist()

        '''
        # 使用 torch.index_select() 函数选择不删除的行,去掉配体连接
        indices_to_keep = torch.tensor(list(set(range(edge_type.size(0))) - set(rows_to_remove))).cuda()
        new_edge_type   = torch.index_select(edge_type, 0, indices_to_keep)
        new_edge_index  = torch.index_select(edge_index, 1, indices_to_keep)  #2 * N
        '''

        #添加配体内部连接
        ##print('ligand_bond_type:', ligand_bond_type)
        ##print('new_ligand_bond_index:', new_ligand_bond_index)

    
        #在KNN图的基础上，我们添加了配体的内部的原有邻接表，这部分是正确，已知的
        #new_edge_type  = torch.cat([edge_type, torch.zeros_like(ligand_bond_type, dtype = torch.int64)], dim = 0)#不扩充配体键类型，依旧使用0
        new_edge_type = edge_type #配体连接表动态变化，我们难以获取对应的键类型，因为此时ligand_bond_type与使用KNN构建的配体链接不对应
        #new_edge_index = torch.cat([edge_index, new_ligand_bond_index], dim = 1)
        new_edge_index = edge_index

        #

        ##print('new_edge_type:', new_edge_type.shape)
        ##print('new_edge_index:', new_edge_index.shape)
        #new_edge_type: torch.Size([103536])
        #new_edge_index: torch.Size([2, 103536])

        #(edge_type,edge_index)每一行的顺序是无所谓的，不影响卷积，关键在于里面使用的原子id一定要是根据当前批量获得的全局id
        #(edge_type,edge_index)只要配体和蛋白之间的，其余的全部由配体和蛋白的全连接图来取代
        edge_type_dim = F.one_hot(new_edge_type, num_classes=4) #由原来的4种，变成了8种,除了把这里改了成8之外，其它地方也要改吧？否则报错
        #探索一下，哪有哪里在使用键长度
        return edge_type_dim, new_edge_index






    @staticmethod
    def _build_edge_type(edge_index, mask_ligand):
        #不是键类型，是用来判断蛋白和配体之间的是否有连接
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1 #1表示是配体
        n_dst = mask_ligand[dst] == 1

        #需要扩充配体的键类型，增加单键，双键，三键，芳香键，我们需要知道配体和蛋白的这些键类型有什么作用？？？
        #edge_type在神经网络中只是和边长度嵌入和节点嵌入连接在了一起，并没有什么特殊的判断处理，虽然这里的0,1,2,3的确用来标识节点是蛋白或者配体上
        #但是在神经网络中用没有用来区分配体和蛋白节点
        edge_type[n_src & n_dst] = 0   #表示在配体内部
        edge_type[n_src & ~n_dst] = 1  #表示源节点在配体，目标节点在蛋白
        edge_type[~n_src & n_dst] = 2  #表示源节点在蛋白，目标节点在配体
        edge_type[~n_src & ~n_dst] = 3 #表示在蛋白内部

        
        edge_type = F.one_hot(edge_type, num_classes=20)
        return edge_type, edge_index


    @staticmethod
    def _build_edge_type_20(edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch):
        #不是键类型，是用来判断蛋白和配体之间的是否有连接
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1 #1表示是配体
        n_dst = mask_ligand[dst] == 1

        #需要扩充配体的键类型，增加单键，双键，三键，芳香键，我们需要知道配体和蛋白的这些键类型有什么作用？？？
        #edge_type在神经网络中只是和边长度嵌入和节点嵌入连接在了一起，并没有什么特殊的判断处理，虽然这里的0,1,2,3的确用来标识节点是蛋白或者配体上
        #但是在神经网络中用没有用来区分配体和蛋白节点
        edge_type[n_src & n_dst] = 0   #表示在配体内部
        edge_type[n_src & ~n_dst] = 1  #表示源节点在配体，目标节点在蛋白
        edge_type[~n_src & n_dst] = 2  #表示源节点在蛋白，目标节点在配体
        edge_type[~n_src & ~n_dst] = 3 #表示在蛋白内部

        
        edge_type = F.one_hot(edge_type, num_classes=20)
        return edge_type, edge_index




    def forward(self, h, x, org_x, element_all, mask_ligand, batch, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch, atom_isring, atom_isO, atom_isN, 
                cross_isring_flag, cross_isO_flag, cross_isN_flag, cross_lp_pos, cross_distance,
                cross_bond_index, cross_bond_type, cross_bond_index_reverse, cross_bond_type_reverse,
                coords_predict,
                complex_mol,

                protein_element_batch = None,
                protein_link_t_batch = None,
                protein_link_t_reverse_batch = None,
                ligand_element_batch = None,
                protein_element = None,
                ligand_element  = None,

                rd_x = None,

                sigmas = None, 
                protein_max_atom_num = None, ligand_max_atom_num  = None, args = None, return_all=False, fix_x=False, equiformer = False, escn = False):

        all_x = [x]
        all_h = [h]

        #print('org_h.shape:', h.shape) #torch.Size([874, 3136])
        
        '''
        #构建KNN图，batch指示了每个节点所在的图id,KNN根据batch,在每一个图上，构建一个小KNN图，最终把所有图合并成一个大图,注意KNN图中，
        #配体和蛋白之间是存在连接信息，虽然配体和蛋白之间没有连接，但是我们需要知道邻接表的两个节点在配体还是在蛋白上，因此需要标识一下这些信息。
        edge_index = self._connect_edge(org_x, mask_ligand, batch) #获取边对。这种动态构图不是和对接任务，因为对接是已经知道了分子结构，唯一不知道是原子坐标。因此需要固定连接表
        #KNN构图不应该在神经网络内部根据新的x之间距离实时变化，取32个最近的点
        #src, dst = edge_index

        # edge type (dim: 4)，获取边权
        edge_type, edge_index = self._build_edge_type(edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch) #边类型，根据两个原子可以判断他们的边类型。返回的是一个边嵌入向量。问题是这样构建边类型对？#这样构建边类型对？
        #边类型可能需要考虑一下是否通过该方式获取？？？这种方式是否正确？这里获取的是真实的边类型？是如何判断单双三键？根据edge_index可以获取任意两个节点之间有多少条链接
        src, dst = edge_index
        ##print('edge_index:', edge_index)
        ''' 
        atom_num_list = torch.tensor([600, 700, 800, 1000, 1200], dtype = torch.int64).cuda()

        for b_idx in range(self.num_blocks):
            #print('self.num_blocks:', self.num_blocks)
            
            '''
            edge_index = self._connect_edge(x, mask_ligand, batch) #获取边对。这种动态构图不是和对接任务，因为对接是已经知道了分子结构，唯一不知道是原子坐标。因此需要固定连接表
            #KNN构图不应该在神经网络内部根据新的x之间距离实时变化，取32个最近的点
            src, dst = edge_index

            # edge type (dim: 4)，获取边权
            edge_type = self._build_edge_type(edge_index, mask_ligand) #边类型，根据两个原子可以判断他们的边类型。返回的是一个边嵌入向量
            '''

            #保证配体和蛋白之间的连接是动态的，因此要部分改一下，所以这里使用动态的x,而非静态的org_x
            #为了减少连接，可以将蛋白内部的KNN图给减少？？？

            if GP.embedding3d:
                #edge_index = self._connect_edge(rd_x, mask_ligand, batch) #获取边对。这种动态构图不是和对接任务，因为对接是已经知道了分子结构，唯一不知道是原子坐标。因此需要固定连接表
                edge_index = self._connect_edge(rd_x, mask_ligand, batch)
            else:
                edge_index = self._connect_edge(x, mask_ligand, batch)

            if GP.embedding3d:
                '''配体和蛋白3D嵌入'''
                '''因为是rdkit生成的结构，且这里只是获取配体和蛋白内部的键和原子信息，不需要配体和蛋白相互作用信息，所以使用直接knn简单处理即可'''
                edge_type_, edge_index_, ligand_node_global2local_dict, protein_node_global2local_dict, only_ligand_edge_type_dim, only_ligand_bond_index, only_protein_edge_type_dim, only_protein_bond_index = self._build_edge_type_20_gpu(edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch)


                #配体和蛋白的信息不能直接从合并之后的数据获取，id对不上，需要全局和局部的映射，改变edge_index, 其它不用动
                #配体
                #print('\n配体层')
                '''存在一个想法，这里未来可以使用带有噪音的x, 可以视为配体与蛋白交互，即先配体和蛋白交互，然后再送入坐标网络，所以务必验证一下'''
                l_h             = h[mask_ligand == True]
                
                if GP.embedding3d_noise_pos:
                    l_x             = x[mask_ligand == True] 
                else:
                    l_x             = rd_x[mask_ligand == True] 
                

                #print('l_x.shape:', l_x.shape) #torch.Size([176, 3])

                #only_ligand_edge_type_dim, only_ligand_bond_index, only_protein_edge_type_dim, only_protein_bond_index
                l_element       = element_all[mask_ligand == True]
                #print('l_element.shape:', l_element.shape) #torch.Size([176])
                #print('edge_type.shape:', edge_type.shape) #torch.Size([42526, 20]) 
                l_edge_type     = only_ligand_edge_type_dim
                #print('l_edge_type.shape:', l_edge_type.shape) #torch.Size([376, 20]), 没变化？
                l_edge_index    = only_ligand_bond_index
                #print('l_edge_index.shape:', l_edge_index.shape) #torch.Size([2, 376])
                ##print('l_edge_index:', l_edge_index) #必须保证边索引是从0开始编号的
                l_mask_ligand   = None #不计算坐标，用不着
                l_batch         = batch[mask_ligand == True]
                #print('l_batch:', l_batch.shape) # torch.Size([176])
                l_fix_x         = None

                l_dst, l_src = l_edge_index

                
                

                if self.ew_net_type == 'global': #默认是global，使用边权重
                    l_dist = torch.norm(l_x[l_dst] - l_x[l_src], p=2, dim=-1, keepdim=True) #第二范式，归一化，应用于边
                    #print('l_dist.shape:', l_dist.shape)
                    l_dist_feat = self.distance_expansion(l_dist) #处理边长度
                    #print('l_dist_feat.shape:', l_dist_feat.shape)
                    l_logits = self.edge_pred_layer(l_dist_feat) #mlp，对边分类，得到边权重
                    #print('l_logits.shape:', l_logits.shape)
                    l_e_w = torch.sigmoid(l_logits)
                else:
                    l_e_w = None
                
                #l_dist.shape: torch.Size([376, 1])
                #l_dist_feat.shape: torch.Size([376, 20])
                #l_logits.sjhape: torch.Size([376, 1])
                



                l_distance_vec = l_x[l_src] - l_x[l_dst]
                l_edge_dist    = l_distance_vec.norm(dim=-1)  #这里，我们可以优化一下，将unimol预测出来的边距离代替这里的配体到蛋白的距离
                #print('l_edge_dist.shape:', l_edge_dist.shape) #l_edge_dist.shape: torch.Size([376])


                l_h, _ = self.ligand_block(h = l_h, pos = l_x, distance_vec = l_distance_vec, edge_dist = l_edge_dist, element = l_element,
                                        edge_type = l_edge_type, edge_index = l_edge_index, mask_ligand = l_mask_ligand, 
                                        sigmas = sigmas, mask = None, batch = l_batch, 
                                        protein_max_atom_num = protein_max_atom_num, ligand_max_atom_num  = protein_max_atom_num, 
                                        node_atom = None,
                                        e_w=l_e_w, fix_x=l_fix_x)
                
                #print('l_h.shape:', l_h.shape) #torch.Size([176, 200])
                

                #蛋白
                #print('\n蛋白层')
                p_h             = h[mask_ligand == False]
                
                if GP.embedding3d_noise_pos:
                    p_x             = x[mask_ligand == False]
                else:
                    p_x             = rd_x[mask_ligand == False]


                #print('p_x.shape:', p_x.shape) #torch.Size([176, 3])

                #only_protein_edge_type_dim, only_protein_bond_index, only_protein_edge_type_dim, only_protein_bond_index
                p_element       = element_all[mask_ligand == False]
                #print('p_element.shape:', p_element.shape) #torch.Size([176])
                #print('edge_type.shape:', edge_type.shape) #torch.Size([42526, 20]) 
                p_edge_type     = only_protein_edge_type_dim
                #print('p_edge_type.shape:', p_edge_type.shape) #torch.Size([376, 20]), 没变化？
                p_edge_index    = only_protein_bond_index
                #print('p_edge_index.shape:', p_edge_index.shape) #torch.Size([2, 376])
                ##print('p_edge_index:', p_edge_index) #必须保证边索引是从0开始编号的
                p_mask_ligand   = None #不计算坐标，用不着
                p_batch         = batch[mask_ligand == False]
                #print('p_batch:', p_batch.shape) # torch.Size([176])
                p_fix_x         = None

                p_dst, p_src = p_edge_index

                
                

                if self.ew_net_type == 'global': #默认是global，使用边权重
                    p_dist = torch.norm(p_x[p_dst] - p_x[p_src], p=2, dim=-1, keepdim=True) #第二范式，归一化，应用于边
                    #print('p_dist.shape:', p_dist.shape)
                    p_dist_feat = self.distance_expansion(p_dist) #处理边长度
                    #print('p_dist_feat.shape:', p_dist_feat.shape)
                    p_logits = self.edge_pred_layer(p_dist_feat) #mlp，对边分类，得到边权重
                    #print('p_logits.shape:', p_logits.shape)
                    p_e_w = torch.sigmoid(p_logits)
                else:
                    p_e_w = None


                p_distance_vec = p_x[p_src] - p_x[p_dst]
                p_edge_dist    = p_distance_vec.norm(dim=-1)  #这里，我们可以优化一下，将unimol预测出来的边距离代替这里的配体到蛋白的距离


                p_h, _ = self.protein_block(h = p_h, pos = p_x, distance_vec = p_distance_vec, edge_dist = p_edge_dist, element = p_element,
                                        edge_type = p_edge_type, edge_index = p_edge_index, mask_ligand = p_mask_ligand, 
                                        sigmas = sigmas, mask = None, batch = p_batch, 
                                        protein_max_atom_num = protein_max_atom_num, ligand_max_atom_num  = protein_max_atom_num, 
                                        node_atom = None,
                                        e_w=p_e_w, fix_x=p_fix_x)



                #连接初始的输入嵌入和3d结构的嵌入
                new_h = torch.empty(h.shape[0], l_h.shape[1] * 2).cuda()
                #print('h:', h.shape)
                #print('l_h:', l_h.shape)
                #print('h[mask_ligand == True]:', h[mask_ligand == True].shape)
                #print('mask_ligand:', mask_ligand.shape)
                '''
                h: torch.Size([1413, 200])                                                                                                                                                                             
                l_h: torch.Size([163, 200])                                                                                                                                                                            
                h[mask_ligand == True]: torch.Size([163, 200])                                                                                                                                                         
                mask_ligand: torch.Size([1413]) 
                '''
                new_h[mask_ligand == True]  = torch.cat([h[mask_ligand == True], l_h], dim = -1) #
                #RuntimeError: shape mismatch: value tensor of shape [163, 400] cannot be broadcast to indexing result of shape [163, 326]
                new_h[mask_ligand == False] = torch.cat([h[mask_ligand == False], p_h], dim = -1)


                
                #维度变换由400 -> 200，将拼接的嵌入降维度
                h = self.linear_transform_dim(new_h)



            '''配体和蛋白相互作用'''
            # # # 注意应该不存在信息泄露问题，因为edge_index（包含配体，蛋白，配体-蛋白）是用带有噪声的配体坐标构建的，之后我们再将配体的噪声连接表换成参考的，以及把使用噪音坐标生成的互相作用连接表给替换成模糊相互作用表
            # # #也就说，根据不可能泄露相互作用信息呀
                
            #edge_type, edge_index = self._build_edge_type_8_gpu(edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch) #边类型，根据两个原子可以判断他们的边类型。返回的是一个边嵌入向量。问题是这样构建边类型对？#这样构建边类型对？
            
            #有信息泄露，这里使用了org_x构建了连接表
            #edge_type, edge_index = self._build_edge_type_interaction_8_gpu(org_x, edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch, batch, atom_isring, atom_isO, atom_isN)
            
            ##coords_predict取代org_x构建了连接表，相当于在已有的结构上，优化
            #edge_type, edge_index = self._build_edge_type_interaction_8_gpu(coords_predict, edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch, batch, atom_isring, atom_isO, atom_isN)
            #简化连接表，其连接信息从数据处理部分实现，这里只是将配体和蛋白局部和全局id映射一下
            #edge_type, edge_index = self._build_edge_type_interaction_8_gpu_optim_v2(x, coords_predict, edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch, 
                    #batch, atom_isring, atom_isO, atom_isN, cross_isring_flag, cross_isO_flag, cross_isN_flag, cross_lp_pos, cross_distance,
                    #cross_bond_index, cross_bond_type, cross_bond_index_reverse, cross_bond_type_reverse)
                    

            #带有unimol距离矩阵的，无信息泄露
            #edge_type, edge_index = self._build_edge_type_interaction_8_gpu_optim(x, org_x, edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch, 
                    #batch, atom_isring, atom_isO, atom_isN, cross_isring_flag, cross_isO_flag, cross_isN_flag, cross_lp_pos, cross_distance)



            #简化连接表，其连接信息从数据处理部分实现，这里只是将配体和蛋白局部和全局id映射一下
            #edge_type, edge_index = self._build_edge_type_interaction_8_gpu_optim_v2(x, org_x, edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch, 
                    #batch, atom_isring, atom_isO, atom_isN, cross_isring_flag, cross_isO_flag, cross_isN_flag, cross_lp_pos, cross_distance,
                    #cross_bond_index, cross_bond_type, cross_bond_index_reverse, cross_bond_type_reverse)


            #区分ON环键类型, 总之预留更多的键类型位置，为了将来扩充键类型做准备。常用的是这个
            
            if GP.use_distance:
                edge_type, edge_index = self._build_edge_type_interaction_20_gpu_optim(x, org_x, edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch, 
                        batch, atom_isring, atom_isO, atom_isN, cross_isring_flag, cross_isO_flag, cross_isN_flag, cross_lp_pos, cross_distance,
                        cross_bond_index, cross_bond_type, cross_bond_index_reverse, cross_bond_type_reverse,  
                        
                        protein_element_batch,
                        protein_link_t_batch,
                        protein_link_t_reverse_batch,
                        ligand_element_batch,
                        protein_element,
                        ligand_element,

                        )
            else:
                #不使用距离矩阵
                edge_type, edge_index = self._build_edge_type_interaction_20_gpu_optim_no_interactive_gpu(x, org_x, edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch, 
                        batch, atom_isring, atom_isO, atom_isN, cross_isring_flag, cross_isO_flag, cross_isN_flag, cross_lp_pos, cross_distance,
                        cross_bond_index, cross_bond_type, cross_bond_index_reverse, cross_bond_type_reverse,  
                        
                        protein_element_batch,
                        protein_link_t_batch,
                        protein_link_t_reverse_batch,
                        ligand_element_batch,
                        protein_element,
                        ligand_element,

                        )
                
            
            
            
            
            #把unimol的距离矩阵拿过来用于equiformer的边距离编码的一部分
            #edge_type, edge_index = self._build_edge_type_interaction_20_gpu_optim_distance(x, org_x, edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch, 
                    #batch, atom_isring, atom_isO, atom_isN, cross_isring_flag, cross_isO_flag, cross_isN_flag, cross_lp_pos, cross_distance,
                    #cross_bond_index, cross_bond_type, cross_bond_index_reverse, cross_bond_type_reverse)
            

            #扩大数据范围，在3.5~4.5范围内，找对应蛋白的原子，然后再取这些蛋白原子附近的2ai范围内的原子，形成一个扩充的蛋白集合，记得去重复，排除unimol距离已有的原子，这个在神经实现比较好
            #如果在这里实现，那速度太慢了，最好在数据预处理时操作
            #edge_type, edge_index = self._build_edge_type_interaction_20_gpu_optim_distance_extend(x, org_x, edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch, 
                    #batch, atom_isring, atom_isO, atom_isN, cross_isring_flag, cross_isO_flag, cross_isN_flag, cross_lp_pos, cross_distance,
                    #cross_bond_index, cross_bond_type, cross_bond_index_reverse, cross_bond_type_reverse)
            
            #build_edge_type_interaction_8_gpu应该使用org_x来约束配体和蛋白之间可能存在的连接，或者也可以试一下使用带有加噪点坐标去约束，这样训练可能存在困难，但这样做也应该比全32KNN图约束
            #要好，因为只关注可能存在的连接.目前来看使用x + 模糊的相互作用是不行的
            
            src, dst = edge_index

            '''
            def _build_edge_type_interaction_8_gpu(self, edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch,
            batch, #这个参数要结合mask_liagnd来确定哪些是配体哪些是蛋白上的节点
            atom_isring,
            atom_isO,
            atom_isN,
            ):
            '''


            distance_vec = x[src] - x[dst]
            edge_dist    = distance_vec.norm(dim=-1)  #这里，我们可以优化一下，将unimol预测出来的边距离代替这里的配体到蛋白的距离
            unimol_dist  = []


            if self.ew_net_type == 'global': #默认是global，使用边权重
                dist = torch.norm(x[dst] - x[src], p=2, dim=-1, keepdim=True) #第二范式，归一化，应用于边
                dist_feat = self.distance_expansion(dist) #处理边长度
                #print('dist_feat.shape:', dist_feat.shape)
                #print('dist_feat.dtype:', dist_feat.dtype) #logits.dtype: torch.float32
                logits = self.edge_pred_layer(dist_feat) #mlp，对边分类，得到边权重
                #print('logits.shape:', logits.shape)
                #print('logits.dtype:', logits.dtype) #logits.dtype: torch.float32
                #exit()
                e_w = torch.sigmoid(logits)
            else:
                e_w = None
            
            #sigma_emb = self.sigma_emb_layer(sigmas.view(-1,1)).unsqueeze(1)
            #h = h + sigma_emb

            #print('h:', h.shape) #h: torch.Size([716, 3136]) , 即 torch.Size([716, 49， 64]) 
            
            if self.equiformer == True:
                h, x = self.base_block(h = h, pos = x, distance_vec = distance_vec, edge_dist = edge_dist, element = element_all,
                                    edge_type = edge_type, edge_index = edge_index, mask_ligand = mask_ligand, 
                                    sigmas = sigmas, mask = None, batch = batch, 
                                    protein_max_atom_num = protein_max_atom_num, ligand_max_atom_num  = protein_max_atom_num, 
                                    node_atom = None,
                                    e_w=e_w, fix_x=fix_x)
                
                #print('h, x:', h.shape, x.shape)
            
            elif self.escn == True:
                h, x = self.base_block(h = h, pos = x, distance_vec = distance_vec, edge_dist = edge_dist, element = element_all,
                                    edge_type = edge_type, edge_index = edge_index, mask_ligand = mask_ligand, 
                                    sigmas = sigmas, mask = None, batch = batch, 
                                    protein_max_atom_num = protein_max_atom_num, ligand_max_atom_num  = protein_max_atom_num, 
                                    node_atom = None,
                                    e_w=e_w, fix_x=fix_x)
                
            else:
                #真正的神经网络在这
                for l_idx, layer in enumerate(self.base_block):
                    h, x = layer(h, x, edge_type, edge_index, mask_ligand, e_w=e_w, fix_x=fix_x) #是否固定坐标，默认不固定。固定的是谁的坐标？固定蛋白坐标
                    

            all_x.append(x)
            all_h.append(h)

        outputs = {'x': x, 'h': h}
        if return_all:
            outputs.update({'all_x': all_x, 'all_h': all_h})
        return outputs