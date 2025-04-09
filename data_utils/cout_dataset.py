
import pandas as pd
import os

path = "/private/luohong/Projects/dti_data/gem_processed_drug_32/ion_channel"
path = "/private/luohong/Projects/dti_data/esm2_t12_embedding/ion_channel"

print(len(os.listdir(path)))
for dataset in ["DrugBank", "KIBA",  "Enzyme", "GPCRs", "ion_channel"]:
    dataset_dir =  "../dti_data/p1000_dti_gem_filtered/"
    dataset_path = dataset_dir + f'{dataset}.txt'
    dti_data = pd.read_csv(dataset_path, header=None)
    drug = set(dti_data.iloc[:, 0])
    protein = set(dti_data.iloc[:, 1])
    dti = dti_data.iloc[:, -1]
    pos_num = len(dti[dti==1])
    neg_num = len(dti[dti==0])


    # dataset_dir =  "../dti_data/initial_dti/"
    dataset_dir =  "../dti_data/p1000_dti_gem_filtered_new/"
    dataset_path = dataset_dir + f'{dataset}.txt'
    dti_data = pd.read_csv(dataset_path, header=None, sep=",")
    drug_all = set(dti_data.iloc[:, 0])
    protein_all = set(dti_data.iloc[:, 1])
    dti = dti_data.iloc[:, -1]
    pos_num_all = len(dti[dti==1])
    neg_num_all = len(dti[dti==0])

    print(f"数据集:{dataset}, 药物数:{len(drug)}/{len(drug_all)}, 蛋白质数: {len(protein)}/{len(protein_all)}, 正样本数: {pos_num}/{pos_num_all}, 负样本数: {neg_num}/{neg_num_all}")



for dataset in ["DrugBank", "KIBA",  "Enzyme", "GPCRs", "ion_channel"]:
    dataset_dir =  "../dti_data/p1000_dti_gem_filtered/"
    dataset_path = dataset_dir + f'{dataset}.txt'
    dti_data = pd.read_csv(dataset_path, header=None)
    drug = set(dti_data.iloc[:, 0])
    protein = set(dti_data.iloc[:, 1])
    dti = dti_data.iloc[:, -1]
    pos_num = len(dti[dti==1])
    neg_num = len(dti[dti==0])

    print(f"数据集:{dataset}, 药物数:{len(drug)}, 蛋白质数: {len(protein)}, 正样本数: {pos_num}, 负样本数: {neg_num}")