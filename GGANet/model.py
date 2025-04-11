import torch
import torch.nn as nn
import torch.nn.functional as F
from dgllife.model.gnn import GCN
import dgl 
from dgl.nn.pytorch.conv.ginconv import GINConv
from .gem_torch.gemconv import GEMConv
import time


class GeoDTI(nn.Module):
    def __init__(self, hp):
        super(GeoDTI, self).__init__()
        self.drug_encoder = DrugEncoder(hp.drug_embedding_dim, hp.hidden_dim, hp.compound_config, hp.dropout)

        self.protein_encoder = ProteinEncoder(hp.protein_embedding_dim, hp.hidden_dim, hp.protein_filters, hp.protein_kernels, hp.dropout, hp.device)
        
        self.Protein_max_pool = nn.MaxPool1d(hp.protein_max_length)
        self.gate_att = GateAtt(hidden_dim=hp.hidden_dim)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.leaky_relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(hp.hidden_dim*2, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 2)
        self.sim_loss = SimLoss(hp.sim_threshold, hp.hidden_dim)

    def forward(self, drug, protein):
        t1 = time.time()
        drugConv = self.drug_encoder(*drug)
        t2 = time.time()
        # print(f"药物特征提取时间： {t2-t1:.2f}s")
        protein_hidden = self.protein_encoder(protein)[0]
        t3 = time.time()
        # print(f"蛋白质特征提取时间： {t3-t2:.2f}s")
        sim_loss = self.sim_loss(protein_hidden.transpose(-1, -2).contiguous(), protein[1])
        t4 = time.time()
        # print(f"相似性损失计算时间： {t4-t3:.2f}s")
        protein_hidden = self.gate_att(drugConv, protein_hidden, False)
        protein_feature = self.Protein_max_pool(protein_hidden).squeeze()
        pair = torch.cat([drugConv, protein_feature], dim=1)
        pair = self.dropout1(pair)
        fully1 = self.leaky_relu(self.fc1(pair))
        fully1 = self.dropout2(fully1)
        fully2 = self.leaky_relu(self.fc2(fully1))
        fully2 = self.dropout3(fully2)
        fully3 = self.leaky_relu(self.fc3(fully2))
        predict = self.out(fully3)

        return predict, sim_loss

class ProteinEncoderLayer(nn.Module):
    # 并行
    def __init__(self, in_channel, out_channel, kernel_size, dropout, device):
        super().__init__()
        assert kernel_size % 2 == 1 
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.scale = torch.sqrt(torch.FloatTensor([1/2])).to(device)
        self.conv =nn.Conv1d(in_channel, out_channel*2, kernel_size, padding=kernel_size//2)
        self.gin = ProteinGIN(out_channel, out_channel)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        protein_input, graph = input    
        protein_cnn = self.conv(protein_input)
        protein_cnn = F.glu(protein_cnn, dim=1)
        protein_cnn = (protein_cnn + protein_input)* self.scale
        protein_gnn = protein_cnn.permute(0, 2, 1).contiguous().view(-1, self.out_channel)
        
        protein_gnn = self.gin(protein_gnn, graph)
        protein_gnn = protein_gnn.permute(0, 2, 1).contiguous()
        protein_out = (protein_gnn + protein_cnn) * self.scale

        return protein_out, graph

class ProteinGIN(nn.Module):
    def __init__(self, in_feats,  hidden_feats=None, activation=None):
        super(ProteinGIN, self).__init__()
        self.lin = nn.Sequential(nn.Linear(in_feats, hidden_feats),
                                 nn.ReLU(),
                                 nn.Linear(hidden_feats, hidden_feats))
        self.gnn = GINConv(self.lin, 'sum', activation=nn.functional.relu)
        self.output_feats = hidden_feats

    def forward(self, node_feats, batch_graph):
        # 分割batch
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats

class ProteinEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, filters, kernels, dropout=0.0, device="cuda:0"):
        super().__init__()
        self.map = nn.Linear(embedding_dim, hidden_dim)
        encod_layers = []
        filters = [hidden_dim] + filters
        for i in range(len(filters)-1):
            encod_layers.append(ProteinEncoderLayer(filters[i], filters[i+1], kernels[i], dropout, device))
        self.encoder = nn.Sequential(*encod_layers)
        

    def forward(self, input):
        protein, graph = input
        protein = self.map(protein).transpose(-1, -2)
        protein, graph = self.encoder((protein, graph))
    
            
        return protein, graph

class DrugEncoder(nn.Module):
    def __init__(self,in_dim, hidden_dim, compound_config, dropout_rate=0.2):
        super().__init__()
        self.atom_proj = nn.Linear(in_dim, hidden_dim)
        self.edge_proj = nn.Linear(in_dim, hidden_dim)
        self.conv1 = GEMConv(hidden_dim, compound_config, dropout_rate=dropout_rate, activation=True)

        # self.conv2 = GEMConv(hidden_dim, compound_config, dropout_rate=dropout_rate, activation=True)

    def forward(self, atom_bond_graph, bond_angle_graph,
                node_hidden, edge_hidden):
        node_hidden = self.atom_proj(node_hidden)
        edge_hidden = self.atom_proj(edge_hidden)

        node_hidden, edge_hidden, graph_repr = self.conv1(atom_bond_graph, bond_angle_graph, node_hidden, edge_hidden)

        # node_hidden, edge_hidden, graph_repr = self.conv2(atom_bond_graph, bond_angle_graph, node_hidden, edge_hidden)

        return graph_repr




class SimLoss(nn.Module):
    def __init__(self, threshold, hidden_dim):
        super().__init__()
        self.threshold = threshold
        self.map = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim//2, hidden_dim)
        )
    def compute_sim(self, g):
        def message_func(edges):
        
            src_feat = edges.src['feat']
            dst_feat = edges.dst['feat']
            # 计算余弦相似性
            cos_sim = F.cosine_similarity(src_feat, dst_feat, dim=-1)
            return {'similarity': cos_sim}
        g.apply_edges(message_func)
        similarities = g.edata['similarity']
        return similarities
    
    def forward(self, x, graph):
        graph.ndata['feat'] =  x.contiguous().view(-1, x.shape[-1])
        mask_sim = self.compute_sim(graph)
        
        mask_sim[mask_sim > self.threshold] = 0
        loss = torch.sum(1 / torch.exp(mask_sim[mask_sim > 0])) / (len(mask_sim[mask_sim > 0]) + 1e-6)
        
        return loss
    
class Glu(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return F.glu(x, dim=self.dim)
    
class GateAtt(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.share_dim = hidden_dim//2

        self.map = nn.Sequential(nn.Conv1d(self.share_dim, self.share_dim, 1),
                                 nn.LeakyReLU(),
                                 nn.Dropout(0.1),
                                 nn.Conv1d(self.share_dim, self.share_dim, 1))
        
        self.att = nn.Sequential(nn.Conv1d(hidden_dim*2, hidden_dim*2, 1),
                                 Glu(1),
                                 nn.Conv1d(hidden_dim, 1, 1),
                                 nn.Sigmoid())
    def forward(self,drug, protein, return_weight=False):
        assert drug.dim() == 2
        assert protein.dim() == 3
        drug = drug.unsqueeze(2).repeat(1,1,protein.shape[-1])
        # drug_special, drug_share = drug[:, :self.share_dim, :], drug[:, self.share_dim:, :]
        # protein_special, protein_share = protein[:, :self.share_dim, :], protein[:, self.share_dim:, :]
        # drug_share = self.map(drug_share)
        # protein_share = self.map(protein_share)
        # drug = torch.cat([drug_special, drug_share], dim=1)
        # protein = torch.cat([protein_special, protein_share], dim=1)
        changed_protein = torch.cat([drug, protein], dim=1)
        att_weight = self.att(changed_protein)
        protein = att_weight * protein * 0.5 + protein*0.5
       
        return protein





if __name__ == "__main__":
    from config import config
    from scipy.sparse import csr_matrix
    import random
    import numpy as np
    import dgl
    seed = 42  # 可以根据需要更改种子值
    torch.manual_seed(seed)  # 为CPU设置种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置种子（如果有多个GPU）
    random.seed(seed)
    np.random.seed(seed)

    config = config()
    model = GeoDTI(config)

    N = 100
    L = 1000
    Protein = torch.randint(low=0, high=26, size=(N, L), dtype=torch.int64)
    P_graph = torch.randint(0, 2, (N, L, L), dtype=torch.float)
    D_N = 100
    ds = []
    for i in range(D_N):
        Edge = np.random.randint(low=0, high=2, size=(D_N, D_N))
        Edge = csr_matrix(Edge)
        graph = dgl.from_scipy(Edge)
        graph.ndata["h"] = torch.rand(D_N, 75)
        graph = dgl.add_self_loop(graph)
        ds.append(graph)
    d_graph = dgl.batch(ds)

    res, sim_loss = model(d_graph, Protein, P_graph)
    loss = res.sum() + sim_loss
    loss.backward()
    print(res.shape)
    # P_E = ProteinEncoderLayer(16, 128, 3, 1)
    # R_graph = torch.randint(0, 2, (100, 1000, 1000), dtype=torch.float)
    # R_P = torch.rand(100, 16, 1000)
    # P_E((R_P, R_graph))
    # rand_protein = torch.randint(low=0, high=1000, size=(100, 1000, 128), dtype=torch.float)
    # MLP = nn.Sequential(nn.Linear(128, 128),
    #                     nn.ReLU(),
    #                     nn.Linear(128, 128))
    # rand_graph = torch.randint(0, 2, (100, 1000, 1000))
    # loss_fn = SimLoss(0.99)
    # adam = torch.optim.Adam(MLP.parameters(), 1e-3)
    # for i in range(100):
    #     F = MLP(rand_protein)
    #     loss = loss_fn(F, rand_graph)
    #     loss.backward()
    #     adam.step()
    #     adam.zero_grad()
    #     print(loss)
    # print(rand_protein.grad)



