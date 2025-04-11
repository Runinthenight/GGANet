import torch
import torch.nn as nn
from .utils.gin import * 
from .utils.compound_embedding import *

import torch
import dgl 
from dgl.nn import AvgPooling
  

class GeoGNNBlock(nn.Module):
    """
    GeoGNN Block
    """
    def __init__(self, embed_dim, dropout_rate, activation):
        super(GeoGNNBlock, self).__init__()

        self.embed_dim = embed_dim
        self.activation = activation
        self.gnn = GIN(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        if self.activation:
            self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
    
    def forward(self, graph, node_hidden, edge_hidden):
        """tbd"""
        out = self.gnn(graph, node_hidden, edge_hidden)
        out = self.norm(out)
        out = graph_normal(graph, out)
        if self.activation:
            out = self.act(out)
        out = self.dropout(out)
        out = out + node_hidden
        return out

class GEMConv(nn.Module):

    def __init__(self, hidden_dim, compound_config, dropout_rate=0.2, activation=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        bond_names = compound_config['bond_names']
        bond_float_names = compound_config['bond_float_names']
        bond_angle_float_names = compound_config['bond_angle_float_names']
        self.bond_embedding = BondEmbedding(bond_names, hidden_dim)
        self.bond_float_rbf = BondFloatRBF(bond_float_names, hidden_dim)
        self.bond_angle_float_rbf = BondAngleFloatRBF(bond_angle_float_names, hidden_dim)
        self.atom_bond_gnn = GeoGNNBlock(hidden_dim, dropout_rate, activation=activation)
        self.bond_angle_gnn = GeoGNNBlock(hidden_dim, dropout_rate, activation=activation)
        self.graph_pool = AvgPooling()

    def forward(self, atom_bond_graph, bond_angle_graph,
                node_hidden, edge_hidden):
        

        cur_angle_hidden = self.bond_angle_float_rbf(bond_angle_graph.edata)
        cur_angle_hidden = self.bond_angle_float_rbf(bond_angle_graph.edata)
        
        edge_hidden = self.bond_angle_gnn(bond_angle_graph,
                                          edge_hidden,cur_angle_hidden)
        
        node_hidden = self.atom_bond_gnn(atom_bond_graph,
                                         node_hidden,edge_hidden)
        graph_repr = self.graph_pool(atom_bond_graph, node_hidden)
        
        return node_hidden, edge_hidden, graph_repr

def graph_process(graph):
    atom_names = ["atomic_num", "formal_charge", "degree", "chiral_tag", "total_numHs", "is_aromatic", "hybridization"]
    bond_names = ["bond_dir", "bond_type", "is_in_ring"]
    bond_float_names = ["bond_length"]
    bond_angle_float_names = ["bond_angle"]
    
    edges = graph['edges']  # 边的连接，假设是一个 (num_edges, 2) 的 numpy 数组或 tensor
    node_feats = {name: torch.tensor(graph[name].reshape(-1), dtype=torch.int64) for name in atom_names}
    edge_feats = {name: torch.tensor(graph[name].reshape(-1), dtype=torch.int64) for name in bond_names + bond_float_names}

    atom_bond_graph = dgl.graph((edges[:, 0], edges[:, 1]), num_nodes=len(graph[atom_names[0]]))

    for name, feat in node_feats.items():
        atom_bond_graph.ndata[name] = feat

    for name, feat in edge_feats.items():
        atom_bond_graph.edata[name] = feat

    bond_angle_edges = graph['BondAngleGraph_edges']  # 边的连接
    edge_feats_ba = {name: torch.tensor(graph[name].reshape(-1), dtype=torch.int64) for name in bond_angle_float_names}

    bond_angle_graph = dgl.graph((bond_angle_edges[:, 0], bond_angle_edges[:, 1]), num_nodes=len(graph['edges']))

    for name, feat in edge_feats_ba.items():
        bond_angle_graph.edata[name] = feat

    return atom_bond_graph, bond_angle_graph

if __name__ == "__main__":
    import pickle as pkl
    import dgl
    import json
    # data_list = pkl.load(open(f'work/train_data_list.pkl', 'rb'))  
    # graph = data_list[0]['graph']
    drug = pkl.load(open("../dti_data/gem_process_drug/DrugBank/DB02669.pkl", "rb"))
    graph = drug["graph"]
    node_hidden = drug["node_hidden"]
    edge_hidden = drug["edge_hidden"]
    atom_bond_graph, bond_angle_graph = graph_process(graph)
    
    model_config = json.load(open('exp_GEM_proextracter/model_configs/geognn_l8.json', 'r'))

    # nodes_hidden = torch.rand(size=(atom_bond_graph.num_nodes(), 32))
    # edges_hidden = torch.rand(size=(bond_angle_graph.num_nodes(), 32))
    print(atom_bond_graph.num_nodes(),bond_angle_graph.num_nodes())
    model = GEMConv(32, model_config, activation=True)
    a = model(atom_bond_graph, bond_angle_graph, torch.tensor(node_hidden), torch.tensor(edge_hidden))
    for i in a:
        print(i.shape)

   