import torch
import os.path as osp
from scripts.utils import *
from scripts.config import *
from tqdm import tqdm
from scripts.model import *
from torch_geometric.nn.models.tgn import (LastNeighborLoader, IdentityMessage, MeanAggregator,
                                           LastAggregator)
import random
import numpy as np

# random.seed(42)
# np.random.seed(42)
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)

"""
    step1
    假设数据已下载并解压
"""

data_dir = './data/DARPA_T3/using_data/train'





class Trainer():
    def __init__(self, data, max_node_num):
        self.data = data
        self.memory = TGNMemory(
            max_node_num,
            data.msg.size(-1),
            memory_dim,
            time_dim,
            message_module=IdentityMessage(data.msg.size(-1), memory_dim, time_dim),
            aggregator_module=LastAggregator(),
            ).to(device)
        self.gnn = GraphAttentionEmbedding(
            in_channels=memory_dim,
            out_channels=embedding_dim,
            msg_dim=data.msg.size(-1),
            time_enc=self.memory.time_enc,
            ).to(device)
        self.neighbor_loader = LastNeighborLoader(max_node_num, size=neighbor_size, device=device)
        self.link_pred = LinkPredictor(in_channels=embedding_dim).to(device)
        self.optimizer = torch.optim.Adam(
            set(self.memory.parameters()) | set(self.gnn.parameters())
            | set(self.link_pred.parameters()), lr=0.00005, eps=1e-08, weight_decay=0.01)
        self.criterion = nn.CrossEntropyLoss()
        self.assoc = torch.empty(max_node_num, dtype=torch.long, device=device)

    def train(self):
        self.memory.train()
        self.gnn.train()
        self.link_pred.train()

        self.memory.reset_state()  # Start with a fresh memory.
        self.neighbor_loader.reset_state()  # Start with an empty graph.
        saved_nodes = set()

        total_loss = 0

        for batch in self.data.seq_batches(batch_size=BATCH):
            self.optimizer.zero_grad()

            src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

            n_id = torch.cat([src, pos_dst]).unique()

            n_id, edge_index, e_id = self.neighbor_loader(n_id)
            self.assoc[n_id] = torch.arange(n_id.size(0), device=device)

            # Get updated memory of all nodes involved in the computation.
            z, last_update = self.memory(n_id)

            z = self.gnn(z, last_update, edge_index, self.data.t[e_id], self.data.msg[e_id])

            pos_out = self.link_pred(z[self.assoc[src]], z[self.assoc[pos_dst]])

            y_pred = torch.cat([pos_out], dim=0)

            y_true = []
            for m in msg:
                l = tensor_find(m[16:-16], 1) - 1
                y_true.append(l)

            y_true = torch.tensor(y_true)
            y_true = y_true.reshape(-1).to(torch.long)

            loss = self.criterion(y_pred, y_true)

            # Update memory and neighbor loader with ground-truth state.
            self.memory.update_state(src, pos_dst, t, msg)
            self.neighbor_loader.insert(src, pos_dst)

            loss.backward()
            self.optimizer.step()
            self.memory.detach()
            total_loss += float(loss) * batch.num_events
        return total_loss / self.data.num_events

    def save(self, save_path):
        self.memory.reset_state()  # Start with a fresxh memory.
        self.neighbor_loader.reset_state()
        model = [self.memory, self.gnn, self.link_pred, self.neighbor_loader]
        torch.save(model, save_path)  # 保存权重

if __name__ == '__main__':
    train_data = torch.load(osp.join(data_dir, 'trace.TemporalData'))

    node_uuid2index = torch.load(osp.join(data_dir, 'trace_uuid2index'))
    max_node_num = len(node_uuid2index)//2 + 1
    print(f'{max_node_num=}')
    # 控制一下内存，训练阶段这个东西应该确实也没有什么用
    del node_uuid2index


    saved_nodes = set()
    trainer = Trainer(train_data, max_node_num+2)
    for epoch in tqdm(range(1, 11)):
            loss = trainer.train()
            print(f'  Epoch: {epoch:02d}, Loss: {loss:.4f}')

    save_path = "./weights/model_saved_trace.pt"

    trainer.save(save_path)
