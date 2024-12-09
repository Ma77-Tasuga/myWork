import torch


device = torch.device('cuda:2')
# device  = torch.device('cpu')
time_dim = 100
memory_dim = 100

edge_embedding_dimension=100
embedding_dim = edge_embedding_dimension

neighbor_size=20

# DARPA T3 trace
# edge_type_num = 24
# node_type_num = 12

# DARPA OPTC
edge_type_num = 8
node_type_num = 3

# min_dst_idx, max_dst_idx = 0, max_node_num

# BATCH=1024
BATCH = 8192

num_epoch = 25
lr = 0.00001
weight_decay = 0
# gamma = 1

gate = 2.5
# gate = 2
# gate = 1.992

# 用于FeatureHasher的维度
encode_len = 16
