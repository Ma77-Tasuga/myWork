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


class Trainer():
    def __init__(self, data, max_node_num):
        self.data = data
        self.memory = TGNMemory(
            max_node_num, # +3
            self.data.msg.size(-1),
            memory_dim, # 100
            time_dim, # 100
            message_module=IdentityMessage(self.data.msg.size(-1), memory_dim, time_dim),
            aggregator_module=LastAggregator(), # 使用上一个聚合的特征
        ).to(device)
        self.gnn = GraphAttentionEmbedding(
            in_channels=memory_dim, # 100
            out_channels=embedding_dim, # 100
            msg_dim=self.data.msg.size(-1),
            time_enc=self.memory.time_enc,  # 100
        ).to(device)
        self.neighbor_loader = LastNeighborLoader(max_node_num, size=neighbor_size, device=device) # 20
        self.link_pred = LinkPredictor(in_channels=embedding_dim, out_channels=edge_type_num).to(device)
        # self.classifier = NodeClassifier(embedding_dim=embedding_dim, num_classes=node_type_num).to(device)
        # | 是取并集操作
        # self.optimizer = torch.optim.Adam(
        #     set(self.memory.parameters()) | set(self.gnn.parameters())
        #     | set(self.classifier.parameters()), lr=0.00005, eps=1e-08, weight_decay=0.01)
        self.optimizer = torch.optim.Adam(
            set(self.memory.parameters()) | set(self.gnn.parameters())
            | set(self.link_pred.parameters()), lr=lr, eps=1e-08, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        # 将某个批次内的节点映射到全局节点索引
        self.assoc = torch.empty(max_node_num, dtype=torch.long, device=device)

    # 暂时不需要这个函数
    # def data2vec(self):
    #     # 这里的one-hot操作可能跟util里的冲突
    #     edge2vec = torch.nn.functional.one_hot(torch.arange(0, edge_type_num), num_classes=edge_type_num)
    #     node2vec = torch.nn.functional.one_hot(torch.arange(0, node_type_num), num_classes=node_type_num)
    #     encoded_msg = []
    #     for msg in self.data.msg:
    #         encoded_msg.append(edge2vec[msg])
    #
    #     self.data.msg = torch.vstack(encoded_msg)

    def train(self):
        self.memory.train()
        self.gnn.train()
        # self.classifier.train()
        self.link_pred.train()

        self.memory.reset_state()  # Start with a fresh memory.
        self.neighbor_loader.reset_state()  # Start with an empty graph.
        saved_nodes = set()

        total_loss = 0
        # cnt = 0
        for batch in self.data.seq_batches(batch_size=BATCH):
            # cnt +=1
            # if cnt ==20:
            #     print(f'stop here')
            # now_data = cnt * BATCH
            # print(f'Now finish {now_data}')
            self.optimizer.zero_grad()

            src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
            src = src.to(device)
            pos_dst = pos_dst.to(device)
            t = t.to(device)
            msg = msg.to(device)

            # 这个是全局id
            batch_n_id = torch.cat([src, pos_dst]).unique()  # 将源节点和目的节点合并去重

            # n_id是全局id，index是局部id，e_id是全局id
            n_id, edge_index, e_id = self.neighbor_loader(batch_n_id)
            self.assoc[n_id] = torch.arange(n_id.size(0), device=device)

            # if cnt ==15:
            #     print('stop here')
            # Get updated memory of all nodes involved in the computation.
            z, last_update = self.memory(n_id)

            z = self.gnn(z, last_update, edge_index, self.data.t[e_id.to(torch.device('cpu'))],
                         self.data.msg[e_id.to(torch.device('cpu'))])

            # 重新获取批次内节点
            # gnn编码节点嵌入
            pos_out = self.link_pred(z[self.assoc[src]], z[self.assoc[pos_dst]])
            # pos_out = self.classifier(z[self.assoc[batch_n_id]])

            y_pred = torch.cat([pos_out], dim=0)



            # y_true = self.data.y[self.assoc[batch_n_id].to(torch.device('cpu'))]
            # y_true = self.data.y[batch_n_id.to(torch.device('cpu'))]
            y_true = batch.y
            # for m in msg:
            #     l = tensor_find(m[16:-16], 1) - 1
            #     y_true.append(l)

            # y_true = torch.tensor(y_true)
            # y_true = y_true.reshape(-1).to(torch.long)

            loss = self.criterion(y_pred, y_true.to(device))

            # Update memory and neighbor loader with ground-truth state.
            self.memory.update_state(src, pos_dst, t, msg.to(torch.float32))
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
    """读取和保存路径"""

    data_dir = './data/OPTC/using_data'
    map_dir = './data/OPTC/map'

    """数据加载"""
    train_data = torch.load(osp.join(data_dir, 'optc.TemporalData'))

    node_uuid2index = torch.load(osp.join(map_dir, 'uuid2index'))
    # 检查uuid2index表是否是偶数
    assert len(node_uuid2index) % 2 == 0, f'error in num of uuid2index map, num of map is {len(node_uuid2index)}'

    max_num_node = len(node_uuid2index) // 2 + 1

    print(f'{max_num_node=}')
    # 控制一下内存，训练阶段这个东西应该确实也没有什么用
    del node_uuid2index
    print(f"num train data {len(train_data.msg)}")
    saved_nodes = set()
    print(f'Initiating class....')
    trainer = Trainer(train_data, max_num_node + 2)
    # 暂时不需要转换数据类型
    # print(f'Transfer data....')
    # trainer.data2vec()
    print(f'Start training...')
    for epoch in tqdm(range(1, num_epoch + 1)):
        loss = trainer.train()
        print(f'  Epoch: {epoch:02d}, Loss: {loss:.4f}')

    print(f'Saving model....')
    save_path = f"./weights/model_optc_epoch={num_epoch}_lr={lr}_weight_decay={weight_decay}_loss={loss:.4f}.pt"
    trainer.save(save_path)
