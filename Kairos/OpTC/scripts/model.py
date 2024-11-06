from config import *
import torch
from torch_geometric.nn import TransformerConv, TGNMemory
from torch.nn import Linear
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn.models.tgn import (LastNeighborLoader, IdentityMessage, MeanAggregator,
                                           LastAggregator)
class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super(GraphAttentionEmbedding, self).__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels, heads=8,
                                    dropout=0.0, edge_dim=edge_dim)
        self.conv2 = TransformerConv(out_channels*8, out_channels,heads=1, concat=False,
                             dropout=0.0, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        last_update.to(device)
        x = x.to(device)
        t = t.to(device)
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        x = F.relu(self.conv(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        return x


"""该工作将边分类建模成了边连接预测任务"""
class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super(LinkPredictor, self).__init__()
        self.lin_src = Linear(in_channels, in_channels*2)
        self.lin_dst = Linear(in_channels, in_channels*2)
        self.lin_seq = nn.Sequential(
            Linear(in_channels * 4, in_channels * 8),
            torch.nn.BatchNorm1d(in_channels * 8),
            torch.nn.Dropout(0.5),
            nn.Tanh(),
            Linear(in_channels * 8, in_channels * 2),
            torch.nn.BatchNorm1d(in_channels * 2),
            torch.nn.Dropout(0.5),
            nn.Tanh(),
            Linear(in_channels * 2, int(in_channels // 2)),
            torch.nn.BatchNorm1d(int(in_channels // 2)),
            torch.nn.Dropout(0.5),
            nn.Tanh(),
            Linear(int(in_channels // 2), train_data.msg.shape[1] - 32)
        )

    def forward(self, z_src, z_dst):
        h = torch.cat([self.lin_src(z_src) , self.lin_dst(z_dst)],dim=-1)
        h = self.lin_seq (h)
        return h

memory = TGNMemory(
    max_node_num,
    train_data.msg.size(-1),
    memory_dim,
    time_dim,
    message_module=IdentityMessage(train_data.msg.size(-1), memory_dim, time_dim),
    aggregator_module=LastAggregator(),
).to(device)

gnn = GraphAttentionEmbedding(
    in_channels=memory_dim,
    out_channels=embedding_dim,
    msg_dim=train_data.msg.size(-1),
    time_enc=memory.time_enc,
).to(device)

link_pred = LinkPredictor(in_channels=embedding_dim).to(device)

optimizer = torch.optim.Adam(
    set(memory.parameters()) | set(gnn.parameters())
    | set(link_pred.parameters()), lr=0.00005, eps=1e-08,weight_decay=0.01)

criterion = nn.CrossEntropyLoss()


assoc = torch.empty(max_node_num, dtype=torch.long, device=device)


saved_nodes=set()