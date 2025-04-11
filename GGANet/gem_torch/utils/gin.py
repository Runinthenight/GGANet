import torch
import torch.nn as nn
import dgl
import numpy as np
import torch
import torch.nn as nn
import dgl.function as fn
import math

class GIN(nn.Module):
    """
    Implementation of Graph Isomorphism Network (GIN) layer with edge features in DGL
    """

    def __init__(self, hidden_size):
        super(GIN, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(self, graph, node_feat, edge_feat):
        """
        Args:
            graph (DGLGraph): The input graph.
            node_feat (torch.Tensor): Node features with shape (num_nodes, feature_size).
            edge_feat (torch.Tensor): Edge features with shape (num_edges, feature_size).

        Returns:
            torch.Tensor: Updated node features.
        """
        with graph.local_scope():
            # Assign node and edge features
            graph.srcdata["h"] = node_feat
            graph.edata["h"] = edge_feat

            # Define message passing: source node feature + edge feature
            graph.update_all(
                message_func=fn.u_add_e("h", "h", "m"),
                reduce_func=fn.sum("m", "h_neigh"),
            )

            # Combine self node features with aggregated features
            node_feat = graph.dstdata["h_neigh"]

            # Apply the MLP to the updated node features
            # node_feat = self.mlp(node_feat)

            return node_feat

def graph_normal(graph, feature):

    assert feature.dim() == 2

    norm_coef = torch.ones_like(feature, device=feature.device)
    begin_index = 0
    for node_num in graph.batch_num_nodes():
        norm_coef[begin_index:node_num+begin_index] = norm_coef[begin_index:node_num+begin_index] * math.sqrt(node_num)
        begin_index = node_num
    return feature/norm_coef
    
if __name__ == "__main__":
    edges = [(0, 1), (1, 2)]  # Edge connections: node 0 -> node 1, node 1 -> node 2
    node_feat = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)  # Node features
    edge_feat = torch.tensor([[0.1, 0.5], [0.2, 0.8]], dtype=torch.float32)  # Edge features

    # Create the DGL graph from the edge list
    graph = dgl.graph(edges)

    # Define the GIN layer and perform forward pass
    gin_layer = GIN(hidden_size=2)  # Assume hidden_size is 2
    output = gin_layer(graph, node_feat, edge_feat)

    # Print the output node features
    print("Output Node Features:\n", output.detach().numpy())