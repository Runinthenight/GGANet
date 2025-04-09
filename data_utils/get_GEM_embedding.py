from rdkit import RDLogger
from pahelix.utils.compound_tools import mol_to_geognn_graph_data_MMFF3d
from rdkit.Chem import AllChem
import pgl
import json
from pahelix.model_zoo.gem_model import GeoGNNModel
import paddle as pdl
RDLogger.DisableLog('rdApp.*') # 屏蔽RDKit的warning
import pickle as pkl
import os
import pandas as pd
from tqdm import tqdm 

import sys



def calculate_3D_structure_(smiles):
    while True:
        try:
            molecule = AllChem.MolFromSmiles(smiles)
            molecule_graph = mol_to_geognn_graph_data_MMFF3d(molecule)  # 根据分子力场生成3d分子图
            return molecule_graph
        except:
            return False

def data_transform(graph):
    # 将三维图结构转换为GEM能够处理的tensor
    atom_names = ["atomic_num", "formal_charge", "degree", "chiral_tag", "total_numHs", "is_aromatic", "hybridization"]
    bond_names = ["bond_dir", "bond_type", "is_in_ring"]
    bond_float_names = ["bond_length"]
    bond_angle_float_names = ["bond_angle"]
    
    ab_g = pgl.Graph(
            num_nodes=len(graph[atom_names[0]]),
            edges=graph['edges'],
            node_feat={name: graph[name].reshape([-1, 1]) for name in atom_names},
            edge_feat={
                name: graph[name].reshape([-1, 1]) for name in bond_names + bond_float_names})
    ba_g = pgl.Graph(
            num_nodes=len(graph['edges']),
            edges=graph['BondAngleGraph_edges'],
            node_feat={},
            edge_feat={name: graph[name].reshape([-1, 1]) for name in bond_angle_float_names})
    def _flat_shapes(d):
        """TODO: reshape due to pgl limitations on the shape"""
        for name in d:
            d[name] = d[name].reshape([-1])
    _flat_shapes(ab_g.node_feat)
    _flat_shapes(ab_g.edge_feat)
    _flat_shapes(ba_g.node_feat)
    _flat_shapes(ba_g.edge_feat)
    return ab_g, ba_g

def compute_embeding(model, drug_id, smiles, save_dir, log_path):
    
    molecule_graph = calculate_3D_structure_(smiles)
    if molecule_graph:
        try:
            ab_g, ba_g = data_transform(molecule_graph)
            
            node_repr, edge_repr, graph_repr = model(ab_g.tensor(), ba_g.tensor())
            save_dict = {"graph": molecule_graph, "node_hidden": node_repr.detach().numpy(), 
                        "edge_hidden": edge_repr.detach().numpy(), "graph_repr":graph_repr.detach().numpy()}
            save_path = os.path.join(save_dir, drug_id+".pkl")
            pkl.dump(save_dict, open(save_path, "wb"))
        except Exception as e:
            with open(os.path.join(log_path, "error.txt"), "a", encoding="utf-8") as f:
                f.write(f"{os.path.basename(save_dir)}_{drug_id}在提取期间发生错误：{e}\n")
    else: 
        with open(os.path.join(log_path, "error.txt"), "a", encoding="utf-8") as f:
                f.write(f"{os.path.basename(save_dir)}_{drug_id}无法提取结构\n")


def get_gem_embeding(data_path, save_dir, log_path, initial_datasets):
    compound_encoder_config = json.load(open('./GEM/model_configs/geognn_l8.json', 'r'))  
    model = GeoGNNModel(compound_encoder_config)
    model.set_state_dict(pdl.load("./GEM/weight/class.pdparams")) 
    for dataset in os.listdir(initial_datasets):
        dataset = dataset[:-4]
        data = pd.read_csv(os.path.join(data_path, f"{dataset}.txt"), sep=" ", header=None)
        data_save_path = os.path.join(save_dir, dataset)
        if not os.path.exists(data_save_path):
            os.makedirs(data_save_path)
        id_smiles = set(zip(data.iloc[:, 0], data.iloc[:, 2]))
        id_list, smiles_list = zip(* id_smiles)
        for i in tqdm(range(len(id_list)), total=len(id_list)) :
            compute_embeding(model, id_list[i], smiles_list[i], data_save_path, log_path)
    del sys.modules['paddle']
    del sys.modules['pgl']

if __name__ == "__main__":
    get_gem_embeding("./InitialDataSets", "/hy-tmp/processed_drug")

