import os.path
import sys

from Bio import PDB
from multiprocessing import Pool
import numpy as np
from functools import partial
import torch
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning

# 忽略特定的警告
warnings.simplefilter('ignore', PDBConstructionWarning)

Abb2char = {
    "ALA":  "A",
    "CYS":  "C",
    "ASP":  "D",
    "GLU":  "E",
    "PHE":  "F",
    "GLY":  "G",
    "HIS":  "H",
    "ILE":  "I",
    "LYS":  "K",
    "LEU":  "L",
    "MET":  "M",
    "ASN":  "N",
    "PRO":  "P",
    "GLN":  "Q",
    "ARG":  "R",
    "SER":  "S",
    "THR":  "T",
    "SEC":  "U",
    "VAL":  "V",
    "TRP":  "W",
    "TYR":  "Y"
}



def extract_chain(pdb_file):
    """
    获得残基对象列表
    获取每条链的残基字符串，获得链列表
    如果有相同的就去掉取唯一值，怎么知道去掉的是谁？
        从头到尾遍历，如果发现已存在就返回当前索引，该索引与残基对象的索引一致
        在残基对象列表中弹出所有重复的对象
    通过残基对象获取每个残基的CA坐标
    :param pdb_id:
    :return: (氨基酸字符串， 氨基酸坐标)
    """
    pdb_parser = PDB.PDBParser()
    cif_parser = PDB.MMCIFParser()
    pdb_id = os.path.basename(pdb_file)[:-4]
    if pdb_file[-3:] == "pdb":
        structure = pdb_parser.get_structure(pdb_id, pdb_file)
    else:
        structure = cif_parser.get_structure(pdb_id, pdb_file)
    chains_obj = list(structure[0])
    chains_str = []
    for chain in chains_obj:
        chain_str = ''
        for residue in chain:
            if residue.id[0] == ' ' and len(residue.get_resname()) == 3:
                # try:
                    chain_str += Abb2char[residue.get_resname()]
                # except:
                #     with open("error.txt", "a") as f:
                #         f.write(residue.get_resname()+f":  "+pdb_id + " " + chain_str+"\n", )
        chains_str.append(chain_str)

    coord_list = []
    for chain in chains_obj:
        ca_coords = []
        for residue in chain:
            if residue.id[0] == ' ' and len(residue.get_resname()) == 3:
                # if "CA" not in residue:
                #     print(1)
                ca = residue['CA']
                ca_coords.append(ca.coord)
        coord_list.append(ca_coords)
    total_strs = "0".join(chains_str)
    total_coord = []
    for i, coord in enumerate(coord_list):
        if i == 0:
            total_coord = coord
            continue
        total_coord = total_coord + [0] + coord
    return total_strs, total_coord


def get_edge(index, coord_list, contact_cutoff, skip):
    current_coord = coord_list[index]
    result = []
    if type(current_coord) is int:  # 0为多条链的连接符
        return list(0 for _ in range(len(coord_list)))
    for i, coord in enumerate(coord_list):
        if type(coord) is int:     # 0为多条链的连接符
            result.append(0)
        elif abs(i-index) <= skip:
            result.append(0)
        else:
            dis = np.linalg.norm(current_coord-coord_list[i])
            if dis < contact_cutoff:
                result.append(1)
            else:
                result.append(0)
    return result


def get_protein_graph(coord_list, pool, contact_cutoff=8.0, skip=3):

    """
    :param coord_list:  坐标列表，
    :param pool: 进程池
    :param dis:
    :param np:
    :return: 稀疏矩阵
    """
    dot_index_list = [(i,) for i in range(len(coord_list))]
    func = partial(get_edge, coord_list=coord_list,
                   contact_cutoff=contact_cutoff, skip=skip)
    result = pool.starmap(func, dot_index_list)
    result = torch.tensor(result, dtype=torch.float)
    return result


def handle_pdb(pdb_file, pool):
    AAs, coord_list = extract_chain(pdb_file)
    graph = get_protein_graph(coord_list, pool, contact_cutoff=8.0, skip=3)
    coord = torch.nonzero(graph).T
    return AAs, coord




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import json
    import tqdm
    p = Pool(16)
    datasets = ["DrugBank", "Enzyme", "GPCRs", "ion_channel", "KIBA"]
    # extract_chain("2uzh1.pdb")
    # sys.exit()
    dir_path = "../pdb/"
    res_dict = dict()
    # for dataset in datasets:
    #     max_length = 0
    #     lengths = []
    #     avail_pdb = []
    #     dataset_path = os.path.join(dir_path, dataset)
    #     for pdb in tqdm.tqdm(os.listdir(dataset_path), total=len(os.listdir(dataset_path))):
    #         try:
    #             path = os.path.join(dataset_path, pdb)
    #             temp = handle_pdb(path, p)
    #             lengths.append(temp)
    #             if temp > max_length:
    #                 max_length = temp
    #             if temp <= 2000:
    #                 avail_pdb.append(pdb[:-4])
    #         except Exception as e:
    #             with open("error.txt", "a") as f:
    #                 f.write(pdb+f": {e}" + "\n", )
    # for dataset in datasets:
    #     max_length = 0
    #     lengths = []
    #     avail_pdb = []
    #     dataset_path = os.path.join(dir_path, dataset)
    #     for pdb in tqdm.tqdm(os.listdir(dataset_path), total=len(os.listdir(dataset_path))):
    #         try:
    #             path = os.path.join(dataset_path, pdb)
    #             temp = handle_pdb(path, p)
    #             lengths.append(temp)
    #             if temp > max_length:
    #                 max_length = temp
    #             if temp <= 2000:
    #                 avail_pdb.append(pdb[:-4])
    #         except Exception as e:
    #             with open("error.txt", "a") as f:
    #                 f.write(pdb+f": {e}" + "\n", )
    #     res_dict[f"{dataset}_avail"] = avail_pdb
    #     res_dict[f"{dataset}_max"] = max_length
    #     plt.figure()
    #     plt.hist(lengths, bins=10, edgecolor='black', density=False)
    
    #     # 添加标题和标签
    #     plt.title('Frequency Distribution Histogram')
    #     plt.xlabel('Value')
    #     plt.ylabel('Frequency')
    
    #     # 显示图表
    #     plt.savefig(f"{dataset}_hist.png")
    # json.dump(res_dict, open("avail_pdb.json", "w"), indent=4)


    # sys.exit(0)
    # 使用示例
    pdb_id = '../pdb/DrugBank/O14561.pdb'  # 替换为您的PDB ID
    AAs, graph = handle_pdb(pdb_id, p)
    r = extract_chain(pdb_id)
    print(len(r[1]))
    e = get_edge(8, r[1], 9, 3)
    print(e)
    print(len(e))
    g = get_protein_graph(r[1], p, 8, 5)
    print(g)
    x = torch.rand((48, 128))
    print(x)
    print(g @ x)
