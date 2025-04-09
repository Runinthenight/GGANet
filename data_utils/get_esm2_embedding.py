import numpy as np
import esm 
import torch
import os
import pickle as pkl
from datetime import datetime
from tqdm import tqdm
from esm.pretrained import *

def split_string(aas):
    # 该列表中最后一个数表示字符串长度
    split_positions = [index for index, char in enumerate(aas) if char == '0']
    split_positions.append(len(aas))
    return split_positions

def compute_embedding(model, slice, batch_converter, device):
    data = [["protein", slice]]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    with torch.no_grad():
        results = model(batch_tokens.to(device), repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33]
        return token_representations.squeeze()


def get_single_sequence_embeddings(single_sequence, model, batch_converter, max_length=1024, step=512, device="cuda"):

    if len(single_sequence)>max_length:
        i = 0 
        while 1:
            protein_slice = single_sequence[i*step: min(i*step+max_length, len(single_sequence))]
            embedding = compute_embedding(model, protein_slice, batch_converter, device)
            embedding = embedding.cpu().numpy()
            embed = embedding[1:embedding.shape[0] - 1]
            if i==0:
                fullembed = embed
            else:
                fullembed[-step:]=(embed[0:step]+fullembed[-step:])/2
                fullembed = np.append(fullembed,embed[step:],axis=0)
            if i*step+max_length>=len(single_sequence):
                return fullembed
            i += 1
    else:
        embedding = compute_embedding(model, single_sequence, batch_converter, device)
        embedding = embedding.cpu().numpy()
        embed = embedding[1:embedding.shape[0] - 1]
        return embed

def get_protein_embedding(protein_sequence, model, batch_converter, device="cuda"):
    multi_sequence_index = split_string(protein_sequence)
    # 列表中至少有一个值表示蛋白质的长度
    if len(multi_sequence_index)>1:
        embedding_list = []
        begin_index = 0
        for i in multi_sequence_index[:-1]:
            single_seq = protein_sequence[begin_index: i]
            begin_index = i + 1
            embeding = get_single_sequence_embeddings(single_seq, model, batch_converter, device=device)
            padding = np.zeros(shape=(1, embeding.shape[1]))
            embedding_list.append(embeding)
            embedding_list.append(padding)
        single_seq = protein_sequence[begin_index: ]
        embeding = get_single_sequence_embeddings(single_seq, model, batch_converter, device=device)
        embedding_list.append(embeding)
        total_embeding=np.concatenate(embedding_list, axis=0)
    else:
        total_embeding = get_single_sequence_embeddings(protein_sequence, model, batch_converter, device=device)
    assert total_embeding.shape[0] == len(protein_sequence)

    return {"split_index":multi_sequence_index, "embeding": total_embeding}



def get_esm2_embeding(dataset_path, save_dir, device="cuda", log_path=None):
    
    torch.hub.set_dir("./esm2_weights")
    model, alphabet = esm2_t33_650M_UR50D()
    model = model.to("cuda")
    batch_converter = alphabet.get_batch_converter()
    model.eval() 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for protein_file in tqdm(os.listdir(dataset_path), total=len(os.listdir(dataset_path))):
        if protein_file[-3:] == "pkl":
                if protein_file == "A0A0M3KKX1.pkl":
                    print(1)
            # try:
                protein_path = os.path.join(dataset_path, protein_file)
                save_path = os.path.join(save_dir, protein_file)
                protein_data = pkl.load(open(protein_path, "rb"))
                protein_seq = protein_data["sequence"]
                embedding_dict = get_protein_embedding(protein_seq, model, batch_converter, device)
                protein_data.update(embedding_dict)
                pkl.dump(protein_data,open(save_path, "wb"))
            # except Exception as e:
            #     t = datetime.now()
            #     fmt = '%Y-%m-%d %H:%M:%S'
            #     with open(os.path.join(log_path, "error.txt"), "a") as f:
            #         f.write(f"{t.strftime(fmt)}:  {protein_path} 发生错误{e}\n")
            



