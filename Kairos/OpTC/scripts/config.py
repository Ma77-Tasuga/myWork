
"""
    train param
"""
import torch
time_dim = 100
edge_embedding_dimension=100
embedding_dim = edge_embedding_dimension
neighbor_size=20
memory_dim = 100
BATCH=1024
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

max_node_num = maxnode_num+2
min_dst_idx, max_dst_idx = 0, max_node_num
neighbor_loader = LastNeighborLoader(max_node_num, size=neighbor_size, device=device)