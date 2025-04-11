#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch.nn as nn
from .compound_structure_analysis import CompoundKit
import numpy as np
import torch 
import torch.nn.init as init
from ...config import config
device = config().device


class BondEmbedding(nn.Module):
    """
    Bond Encoder
    """
    def __init__(self, bond_names, embed_dim):
        super(BondEmbedding, self).__init__()
        self.bond_names = bond_names
        
        self.embed_list = nn.ModuleList()
        for name in self.bond_names:
            embed = nn.Embedding(
                    CompoundKit.get_bond_feature_size(name) + 5,
                    embed_dim)
            init.xavier_uniform_(embed.weight)
            self.embed_list.append(embed)

    def forward(self, edge_features):
        """
        Args: 
            edge_features(dict of tensor): edge features.
        """
        out_embed = 0
        for i, name in enumerate(self.bond_names):
            out_embed += self.embed_list[i](edge_features[name])
        return out_embed




class RBF(nn.Module):
    """
    Radial Basis Function
    """
    def __init__(self, centers, gamma, dtype= torch.float32):
        super(RBF, self).__init__()
        self.centers = torch.tensor(centers, dtype=dtype, device=device).reshape(1, -1)
        self.gamma = gamma
    
    def forward(self, x):
        """
        Args:
            x(tensor): (-1, 1).
        Returns:
            y(tensor): (-1, n_centers)
        """
        x = x.reshape(-1, 1)
        return torch.exp(-self.gamma * torch.square(x - self.centers))


class BondFloatRBF(nn.Module):
    """
    Bond Float Encoder using Radial Basis Functions
    """
    def __init__(self, bond_float_names, embed_dim, rbf_params=None):
        super(BondFloatRBF, self).__init__()
        self.bond_float_names = bond_float_names

        if rbf_params is None:
            self.rbf_params = {
                'bond_length': (np.arange(0, 2, 0.1), 10.0),   # (centers, gamma)
            }
        else:
            self.rbf_params = rbf_params
        
        self.linear_list = nn.ModuleList()
        self.rbf_list = nn.ModuleList()
        for name in self.bond_float_names:
            centers, gamma = self.rbf_params[name]
            rbf = RBF(centers, gamma)
            self.rbf_list.append(rbf)
            linear = nn.Linear(len(centers), embed_dim)
            self.linear_list.append(linear)

    def forward(self, bond_float_features):
        """
        Args: 
            bond_float_features(dict of tensor): bond float features.
        """
        out_embed = 0
        for i, name in enumerate(self.bond_float_names):
            x = bond_float_features[name]
            rbf_x = self.rbf_list[i](x)
            out_embed += self.linear_list[i](rbf_x)
        return out_embed



class BondAngleFloatRBF(nn.Module):
    """
    Bond Angle Float Encoder using Radial Basis Functions
    """
    def __init__(self, bond_angle_float_names, embed_dim, rbf_params=None):
        super(BondAngleFloatRBF, self).__init__()
        self.bond_angle_float_names = bond_angle_float_names

        if rbf_params is None:
            self.rbf_params = {
                'bond_angle': (np.arange(0, np.pi, 0.1), 10.0),   # (centers, gamma)
            }
        else:
            self.rbf_params = rbf_params
        
        self.linear_list = nn.ModuleList()
        self.rbf_list = nn.ModuleList()
        for name in self.bond_angle_float_names:
            centers, gamma = self.rbf_params[name]
            rbf = RBF(centers, gamma)
            self.rbf_list.append(rbf)
            linear = nn.Linear(len(centers), embed_dim)
            self.linear_list.append(linear)

    def forward(self, bond_angle_float_features):
        """
        Args: 
            bond_angle_float_features(dict of tensor): bond angle float features.
        """
        out_embed = 0
        for i, name in enumerate(self.bond_angle_float_names):
            x = bond_angle_float_features[name]
            rbf_x = self.rbf_list[i](x)
            out_embed += self.linear_list[i](rbf_x)
        return out_embed


if __name__ == "__main__":
    import pickle as pkl
    import dgl
    import json
    data_list = pkl.load(open(f'work/train_data_list.pkl', 'rb'))  
    graph = data_list[0]['graph']
    atom_names = ["atomic_num", "formal_charge", "degree", "chiral_tag", "total_numHs", "is_aromatic", "hybridization"]
    bond_names = ["bond_dir", "bond_type", "is_in_ring"]
    bond_float_names = ["bond_length"]
    bond_angle_float_names = ["bond_angle"]
    # 创建 DGL 图 ab_g
    edges = graph['edges']  # 边的连接，假设是一个 (num_edges, 2) 的 numpy 数组或 tensor
    node_feats = {name: torch.tensor(graph[name].reshape(-1), dtype=torch.int64) for name in atom_names}
    edge_feats = {name: torch.tensor(graph[name].reshape(-1), dtype=torch.int64) for name in bond_names + bond_float_names}

    # 创建 DGL 图 ab_g，边特征和节点特征在之后可以通过 .ndata 或 .edata 访问
    ab_g = dgl.graph((edges[:, 0], edges[:, 1]), num_nodes=len(graph[atom_names[0]]))

    # 为每个节点添加特征
    for name, feat in node_feats.items():
        ab_g.ndata[name] = feat

    # 为每个边添加特征
    for name, feat in edge_feats.items():
        ab_g.edata[name] = feat

    # 创建 DGL 图 ba_g
    bond_angle_edges = graph['BondAngleGraph_edges']  # 边的连接
    edge_feats_ba = {name: torch.tensor(graph[name].reshape(-1), dtype=torch.int64) for name in bond_angle_float_names}

    # 创建 DGL 图 ba_g，边特征在之后可以通过 .edata 访问
    ba_g = dgl.graph((bond_angle_edges[:, 0], bond_angle_edges[:, 1]), num_nodes=len(graph['edges']))

    # 为每个边添加特征
    for name, feat in edge_feats_ba.items():
        ba_g.edata[name] = feat
    
    model_config = json.load(open('GEM/model_configs/geognn_l8.json', 'r'))
    bond_names = model_config['bond_names']
    bond_float_names = model_config['bond_float_names']
    bond_angle_float_names = model_config['bond_angle_float_names']
    embed_dim = model_config.get('embed_dim', 32)
    eb1= BondEmbedding(bond_names, embed_dim)
    eb2 = BondFloatRBF(bond_float_names, embed_dim)
    eb3 = BondAngleFloatRBF(bond_angle_float_names, embed_dim)
    eb1(ab_g.edata)
    eb2(ab_g.edata)
    eb3(ba_g.edata)
