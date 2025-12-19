import torch
import torch.nn as nn
import random

"""
Set of modules for aggregating embeddings of neighbors.
"""

class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, features, cuda=False, gcn=False): 
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()

        self.features = features
        self.cuda = cuda
        self.gcn = gcn
        
    def forward(self, nodes, to_neighs, num_sample=10):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, 
                            num_sample,
                            )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs

        if self.gcn:
            samp_neighs = [samp_neigh | {nodes[i]} for i, samp_neigh in enumerate(samp_neighs)]

        unique_nodes_set = set()
        for neigh in samp_neighs:
            unique_nodes_set.update(neigh)
        if not unique_nodes_set:
            unique_nodes_set.update(nodes)
        unique_nodes_list = sorted(unique_nodes_set)
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}

        # Try to infer device from features.weight; if not available, fallback to CUDA flag
        if hasattr(self.features, "weight"):
            device = self.features.weight.device
        else:
            device = torch.device("cuda" if self.cuda and torch.cuda.is_available() else "cpu")

        mask = torch.zeros(len(samp_neighs), len(unique_nodes_list), device=device)
        row_idx = []
        col_idx = []
        for row, neigh in enumerate(samp_neighs):
            for n in neigh:
                col_idx.append(unique_nodes[n])
                row_idx.append(row)
        if row_idx:
            mask[row_idx, col_idx] = 1.0
        num_neigh = mask.sum(1, keepdim=True).clamp_min(1.0)
        mask = mask / num_neigh

        embed_idx = torch.tensor(unique_nodes_list, dtype=torch.long, device=device)
        embed_matrix = self.features(embed_idx)
        to_feats = mask.mm(embed_matrix)
        return to_feats
