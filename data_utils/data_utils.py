import os.path
import json
import pandas as pd
import shutil
from .handle_pdb import handle_pdb
from multiprocessing import Pool
import pickle as pkl
from tqdm import tqdm
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
import matplotlib.pyplot as plt
import shutil
import torch
from tqdm import tqdm


def del_protein(file, pdb_dir, save_path, sep=","):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = os.path.basename(file)
    df = pd.read_csv(file, sep=sep, header=None)
    pdb_file_list = os.listdir(pdb_dir)
    pdb_list = []
    for pdb_file in pdb_file_list:
        if pdb_file[-3: ] == "pkl":
            pdb_list.append(pdb_file[:-4])
    del_index_list = []
    for i in tqdm(range(len(df)), total=len(df)):
        if df.iloc[i, 1] not in pdb_list:
            del_index_list.append(i)
           
    df = df.drop(del_index_list)
    df.to_csv(os.path.join(save_path, file_name), header=None, index=False)


def cp_pdb(pdb_dir, save_path, aval_json):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    avail_pdb = json.load(open(aval_json, "r"))
    datasets = ["DrugBank", "Enzyme", "GPCRs", "ion_channel", "KIBA"]
    for dataset in datasets:
        dataset_path = os.path.join(pdb_dir, dataset)
        pdb_save_path = os.path.join(save_path, dataset)
        if not os.path.exists(pdb_save_path):
            os.makedirs(pdb_save_path)
        for pdb in avail_pdb[f"{dataset}_avail"]:
            shutil.copy2(os.path.join(dataset_path, pdb+".pdb"), pdb_save_path)


def transform_pdb(pdb_dir, dataset, save_path, pool, logpath):
    save_path = os.path.join(save_path, dataset)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for pdb in tqdm(os.listdir(pdb_dir), total=len(os.listdir(pdb_dir))):
        res_dict = dict()
        save_file = os.path.join(save_path, f"{pdb[:-4]}.pkl")
        pdb_path = os.path.join(pdb_dir, pdb)
        try:
            data = handle_pdb(pdb_path, pool)
            res_dict["sequence"] = data[0]
            res_dict["graph"] = data[1]
            with open(save_file, 'wb') as f:
                pkl.dump(res_dict, f)
        except:
            with open(os.path.join(logpath, "error.txt"), "a", encoding="utf-8") as f:
                f.write(f"{dataset}：{pdb}\n")
                f.close()

def select_protein(data_path, save_path, max_length):
    for dataset in os.listdir(data_path):
        dataset_path = os.path.join(data_path, dataset)
       
        for protein in tqdm(os.listdir(dataset_path), total=len(os.listdir(dataset_path))):
            data_save_path = os.path.join(save_path, dataset, protein)
            if not os.path.exists(os.path.join(save_path, dataset)):
                os.makedirs(os.path.join(save_path, dataset))
            protein_path = os.path.join(dataset_path, protein)

            f = open(protein_path, "rb")
            data = pkl.load(f)
            f.close()
            protein_length = len(data["sequence"])
            if protein_length <= max_length:
                    f = open(data_save_path, "wb")
                    pkl.dump(data, f)
                    f.close()




def draw_drug_nodes(path_file, dataset):
    df = pd.read_csv(path_file)
    drugs = df["SMILES"]
    num_nodes = []
    for drug in drugs:
        atom_featurizer = CanonicalAtomFeaturizer()
        bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        drug_graph = smiles_to_bigraph(drug, add_self_loop=True,
                                       node_featurizer=atom_featurizer,
                                       edge_featurizer=bond_featurizer)
        actual_node_feats = drug_graph.ndata.pop('h')
        num_actual_nodes = actual_node_feats.shape[0]
        num_nodes.append(num_actual_nodes)
    plt.hist(num_nodes, bins=10, edgecolor='black', density=False)
    plt.savefig(f"./temp/{dataset}_drug.png")


def del_drug(file,  save_path, max_length=150):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = os.path.basename(file)
    df = pd.read_csv(file, header=None)

    del_index_list = []
    for i in tqdm(range(len(df)), total=len(df)):
        SMILES = df.iloc[i, 2]
        atom_featurizer = CanonicalAtomFeaturizer()
        bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        drug_graph = smiles_to_bigraph(SMILES, add_self_loop=True,
                                       node_featurizer=atom_featurizer,
                                       edge_featurizer=bond_featurizer)
        actual_node_feats = drug_graph.ndata.pop('h')
        num_actual_nodes = actual_node_feats.shape[0]
        if num_actual_nodes > max_length:
            del_index_list.append(i)
    df = df.drop(del_index_list)
    df.to_csv(os.path.join(save_path, file_name), header=None, index=False)

def del_no_structure_drug(initia_dti_path, drugfile_dir, save_dti_path,sep=","):
    if not os.path.exists(save_dti_path):
        os.makedirs(save_dti_path)
    file_name = os.path.basename(initia_dti_path)
    df = pd.read_csv(initia_dti_path, sep=sep, header=None)
    pdb_file_list = os.listdir(drugfile_dir)
    pdb_list = []
    for pdb_file in pdb_file_list:
        if pdb_file[-3: ] == "pkl":
            pdb_list.append(pdb_file[:-4])
    del_index_list = []
    for i in tqdm(range(len(df)), total=len(df)):
        if df.iloc[i, 0] not in pdb_list:
            del_index_list.append(i)
    df = df.drop(del_index_list)
    df.to_csv(os.path.join(save_dti_path, file_name), header=None, index=False)


def fileter_data(initial_dti_path, processed_protein_path, protein_length, processed_drug_path, save_path, initial_datasets):
    if not os.path.exists("./filtered_protein"):
            os.mkdir("./filtered_protein")
    select_protein(processed_protein_path, "./filtered_protein", protein_length) 
    for dataset in os.listdir(initial_datasets):
        dataset = dataset[:-4]
        if not os.path.exists("./temp"):
            os.mkdir("./temp")
        del_protein(os.path.join(initial_dti_path, dataset+ ".txt"),
                    f"./filtered_protein/{dataset}/",
                    "./temp", sep=" ")
        del_no_structure_drug(f"./temp/{dataset}.txt",
                              os.path.join(processed_drug_path, dataset),
                             save_path)
        shutil.rmtree("./temp")
        shutil.rmtree(f"./filtered_protein/{dataset}")

if __name__ == "__main__":
    select_protein("../dti_data/all_transformed_pdb_string", "../dti_data/plen_1000_new/", 1000)
    
    # 收集上述蛋白质中的dti
    for dataset in ["DrugBank", "Enzyme", "GPCRs", "ion_channel", "KIBA"]:
        del_protein(f"../dti_data/initial_dti/{dataset}.txt",
                    f"../dti_data/plen_1000_new/{dataset}/",
                    "../dti_data/p1000_new_dti", sep=" ")
        
    for dataset in ["DrugBank", "Enzyme", "GPCRs", "ion_channel", "KIBA"]:
        del_no_structure_drug(f"../dti_data/p1000_new_dti/{dataset}.txt",
                              f"../dti_data/gem_processed_drug_32/{dataset}/",
                              f"../dti_data/p1000_dti_gem_filtered_new/"
        )

     # 处理全部pdb
    # pool = Pool(processes=8)
    # for dataset in ["DrugBank", "Enzyme", "GPCRs", "ion_channel", "KIBA"]:
    #     transform_pdb( f"../dti_data/pdb/{dataset}", dataset,
    #                   "../dti_data/all_transformed_pdb/",pool)
        
    # 选择不同长度的蛋白质
    # select_protein("../dti_data/all_transformed_pdb", "../dti_data/plen_1000/", 1000)

    # 收集上述蛋白质中的dti
    # for dataset in ["DrugBank", "Enzyme", "GPCRs", "ion_channel", "KIBA"]:
    #     del_protein(f"../dti_data/initial_dti/{dataset}.txt",
    #                 f"../dti_data/plen_1000/{dataset}/",
    #                 "../dti_data/p1000_dti", sep=" ")
        
    # for dataset in ["DrugBank", "Enzyme", "GPCRs", "ion_channel", "KIBA"]:
    #     del_drug(f"../dti_data/avail_dti/{dataset}.txt",f"../dti_data/final_dti/")





