import os
import subprocess
initial_datasets = "./InitialDataSets"
pdb_save_path = "./pdb_files"
processed_pdb_path = "../GGANet_data/processed_pdb"
processed_pdb_log = "./data_log/process_pdb_log"
pdb_log_path =  "./data_log/get_pdb_log"
drug_embedding_path = "../GGANet_data/drug_embedding"
drug_embedding_log = "./data_log/drug_embedding_log"
protein_embedding_path = "../GGANet_data/protein_embedding"
protein_embedding_log = "./data_log/protein_embedding_log"

protein_max_length = 1000
filtered_dataset_path = "./FilteredDataset"



if __name__ == "__main__":

    for path in [pdb_save_path, processed_pdb_path, drug_embedding_path, 
                 drug_embedding_log, protein_embedding_path, protein_embedding_log,filtered_dataset_path]:
        if not os.path.exists(path):
               os.makedirs(path)
    # get drug embeddings
    subprocess.run( ["python", "get_drug_embeddings.py"]) 

    from data_utils.data_utils import transform_pdb, fileter_data
    num_workers = 16

    # process pdb files
    print("process pdb files...")
    from multiprocessing import Pool
    pool = Pool(processes=num_workers)
    for dataset in os.listdir(initial_datasets):
            dataset = dataset[:-4]
            transform_pdb(os.path.join(pdb_save_path, dataset), dataset,
                        processed_pdb_path,pool,processed_pdb_log)

    # filter data to create new datasets
    print("filter datasets...")
    fileter_data(initial_datasets, processed_pdb_path, protein_max_length, 
                 drug_embedding_path, filtered_dataset_path,initial_datasets)

    # get protein embeddings
    print("get protein embeddings...")
    from data_utils.get_esm2_embedding import get_esm2_embeding
    for dataset in os.listdir(initial_datasets):
            dataset = dataset[:-4]
            dataset_path = os.path.join(processed_pdb_path, dataset)
            save_path = os.path.join(protein_embedding_path, dataset)
            get_esm2_embeding(dataset_path, save_path, log_path=protein_embedding_log)




