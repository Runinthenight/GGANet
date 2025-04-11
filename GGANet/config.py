
import torch
import json
from process_data import *

class config:
    def __init__(self):
        self.protein_embedding_dim = 1280
        self.drug_embedding_dim = 32
        self.hidden_dim = 128
        self.protein_filters = [128, 128, 128]
        self.protein_kernels = [3, 7, 11]
        self.protein_max_length = 1000
        self.sim_threshold = 1.0
        self.batch_size = 64
        self.weight_decay = 1e-4
        self.patience = 50
        self.epoch = 200
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = "cuda"
        self.seed = 114514
        self.lr = 3e-4
        self.fold_nums = 5
        self.dropout = 0.0
        self.compound_config =  json.load(open('GEM/model_configs/geognn_l8.json', 'r'))
        self.protein_embedding_path = protein_embedding_path
        self.drug_embedding_path = drug_embedding_path
        self.dti_path = filtered_dataset_path
