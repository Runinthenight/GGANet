import torch
from torch.utils.data import Dataset
import numpy as np
import dgl
import os
import pickle as pkl
from ..config import config
device = config().device

class CustomDataSet(Dataset):
    def __init__(self, pairs, drug_embeding_path, protein_embedding_path, drug_cache_num=999999, protein_cache_num=999999):
        self.pairs = pairs
        self.drug_cache = {}
        self.drug_cache_num = drug_cache_num
        self.drug_cur_len = 0
        self.drug_cache_full = False
        
        self.drug_embeding_path = drug_embeding_path
        self.pro_embedding_dir = protein_embedding_path
        self.protein_cache_num = protein_cache_num
        self.protein_cur_len = 0
        self.protein_cache_full = False
        self.protein_cache = dict()
    def __getitem__(self, item):
        pair = self.pairs[item]
        pair = pair.strip().split(",")
        protein_max = 1000
        drug_id, protein_id, compoundstr, proteinstr, label = pair[-5], pair[-4], pair[-3], pair[-2], pair[-1]
       
        if drug_id not in self.drug_cache:
            drug_path = os.path.join(self.drug_embeding_path, f"{drug_id}.pkl")
            drug_dict = pkl.load(open(drug_path, "rb"))
            node_hidden = torch.tensor(drug_dict["node_hidden"])
            edge_hidden = torch.tensor(drug_dict["edge_hidden"])
            graph = drug_dict["graph"]
            atom_bond_graph, bond_angle_graph = self.graph_process(graph)
            if not self.drug_cache_full:
                self.drug_cache[drug_id] = (atom_bond_graph, bond_angle_graph, node_hidden, edge_hidden)
                self.drug_cur_len += 1
                self.drug_cache_full = (self.drug_cur_len >= self.drug_cache_num)
        else:
            atom_bond_graph, bond_angle_graph, node_hidden, edge_hidden = self.drug_cache[drug_id]


        if protein_id not in self.protein_cache:
            protein_path = os.path.join(self.pro_embedding_dir, protein_id+".pkl")
            protein_data = pkl.load(open(protein_path, "rb"))
            protein = torch.tensor(protein_data["embeding"], dtype=torch.float32)
            split_index = protein_data["split_index"]
            u, v = protein_data["graph"]
            protein_graph = dgl.graph((u, v), num_nodes=protein_max)
            if protein.shape[0]<1000:
                protein = torch.cat([protein, torch.zeros(size=(1000-protein.shape[0], protein.shape[1]))], dim=0)
            if not self.protein_cache_full:
                self.protein_cache[protein_id] = (protein, protein_graph)
                self.protein_cur_len += 1
                self.protein_cache_full = (self.protein_cur_len >= self.protein_cache_num)
        else:
            (protein, protein_graph) = self.protein_cache[protein_id]

        return (atom_bond_graph.to(device), bond_angle_graph.to(device), node_hidden.to(device) ,edge_hidden.to(device),  
                protein.to(device), protein_graph.to(device), int(label) )
    
    def graph_process(self, graph):
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
    

    def __len__(self):
        return len(self.pairs)


def collate_fn(batch_data):
    atom_bond_graph_list, bond_angle_graph_list, node_hidden_list , edge_hidden_list,  protein_list, protein_graph_list, label_list = zip(*batch_data)
    atom_bond_graph = dgl.batch(atom_bond_graph_list)
    bond_angle_graph = dgl.batch(bond_angle_graph_list)
    node_hidden = torch.cat(node_hidden_list, dim=0)
    edge_hidden = torch.cat(edge_hidden_list, dim=0)
    protein = torch.stack(protein_list, dim=0)
    protein_graph = dgl.batch(protein_graph_list)
    label = torch.tensor(label_list, dtype=torch.int64, device=device)
    return (atom_bond_graph, bond_angle_graph, node_hidden, edge_hidden), (protein, protein_graph), label
