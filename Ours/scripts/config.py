import torch


# device = torch.device('cuda')
device  = torch.device('cpu')
time_dim = 100
memory_dim = 100

edge_embedding_dimension=100
embedding_dim = edge_embedding_dimension
neighbor_size=20

edge_type_num = 24
node_type_num = 12

# min_dst_idx, max_dst_idx = 0, max_node_num
BATCH=1024
# BATCH = 10