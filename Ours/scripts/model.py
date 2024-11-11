from torch.nn.modules.module import T

# from config import *
from .config import device
import torch
from torch_geometric.nn import TransformerConv, TGNMemory
from torch.nn import Linear
import torch.nn.functional as F
from torch import nn

class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super(GraphAttentionEmbedding, self).__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels, heads=8,
                                    dropout=0.0, edge_dim=edge_dim)
        self.conv2 = TransformerConv(out_channels*8, out_channels,heads=1, concat=False,
                             dropout=0.0, edge_dim=edge_dim)

    # def forward(self, x, last_update, edge_index, t, msg):
    #     last_update.to(device)
    #     x = x.to(device)
    #     t = t.to(device)
    #     rel_t = last_update[edge_index[0]] - t
    #     rel_t_enc = self.time_enc(rel_t.to(x.dtype))
    #     edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
    #     x = F.relu(self.conv(x, edge_index, edge_attr))
    #     x = F.relu(self.conv2(x, edge_index, edge_attr))
    #     return x
    def forward(self, x, last_update,edge_index, t, msg):
        last_update = last_update.to(device)
        x=x.to(device)
        msg = msg.to(device)
        t =t.to(device)
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)

        x = F.relu(self.conv(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))

        return x

class NodeClassifier(torch.nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(NodeClassifier, self).__init__()
        self.linear = torch.nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


"""该工作将边分类建模成了边连接预测任务"""
# class LinkPredictor(torch.nn.Module):
#     def __init__(self, in_channels):
#         super(LinkPredictor, self).__init__()
#         self.lin_src = Linear(in_channels, in_channels*2)
#         self.lin_dst = Linear(in_channels, in_channels*2)
#         self.lin_seq = nn.Sequential(
#             Linear(in_channels * 4, in_channels * 8),
#             torch.nn.BatchNorm1d(in_channels * 8),
#             torch.nn.Dropout(0.5),
#             nn.Tanh(),
#             Linear(in_channels * 8, in_channels * 2),
#             torch.nn.BatchNorm1d(in_channels * 2),
#             torch.nn.Dropout(0.5),
#             nn.Tanh(),
#             Linear(in_channels * 2, int(in_channels // 2)),
#             torch.nn.BatchNorm1d(int(in_channels // 2)),
#             torch.nn.Dropout(0.5),
#             nn.Tanh(),
#             Linear(int(in_channels // 2), train_data.msg.shape[1] - 32)
#         )
#
#     def forward(self, z_src, z_dst):
#         h = torch.cat([self.lin_src(z_src) , self.lin_dst(z_dst)],dim=-1)
#         h = self.lin_seq (h)
#         return h

