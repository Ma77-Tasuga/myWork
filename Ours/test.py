import os

import torch
from scripts.config import *
from scripts.utils import *
import os.path as osp
from tqdm import tqdm
from scripts.model import *
from torch_geometric.nn.models.tgn import (LastNeighborLoader, IdentityMessage, MeanAggregator,
                                           LastAggregator)

"""
    step2
    测试，会保存一个包含节点id和loss的dic列表文件，后续可以直接读取该文件进行测试
"""
class Test:
    def __init__(self, data, max_node_num):
        self.data = data
        self.memory = TGNMemory(
            max_node_num,
            self.data.msg.size(-1),
            memory_dim,
            time_dim,
            message_module=IdentityMessage(self.data.msg.size(-1), memory_dim, time_dim),
            aggregator_module=LastAggregator(),
        ).to(device)
        self.gnn = GraphAttentionEmbedding(
            in_channels=memory_dim,
            out_channels=embedding_dim,
            msg_dim=self.data.msg.size(-1),
            time_enc=self.memory.time_enc,
        ).to(device)
        self.neighbor_loader = LastNeighborLoader(max_node_num, size=neighbor_size, device=device)
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
        # 将某个批次内的节点创建索引
        # 将全局索引映射到局部索引，即节点id到内存索引的关系
        self.assoc = torch.empty(max_node_num, dtype=torch.long, device=device)

    # def data2vec(self):
    #     edge2vec = torch.nn.functional.one_hot(torch.arange(0, edge_type_num), num_classes=edge_type_num)
    #     encoded_msg = []
    #
    #     for msg in self.data.msg:
    #         encoded_msg.append(edge2vec[msg])
    #     self.data.msg = torch.vstack(encoded_msg)

    @torch.no_grad()
    def test(self):
        self.memory.eval()
        self.gnn.eval()
        self.classifier.eval()

        self.memory.reset_state()
        self.neighbor_loader.reset_state()

        # time_with_loss = {}
        total_loss = 0
        node_list = []
        # edge_list = []
        #
        # unique_nodes = torch.tensor([])
        # total_edges = 0
        #
        # start_time = int(inference_data.t[0])
        # event_count = 0
        #
        # pos_o = []
        #
        # loss_list = []
        for batch in self.data.seq_batches(batch_size=BATCH):
            batch = batch.to(device)
            src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
            batch_n_id = torch.cat([src, pos_dst]).unique()

            # n_id是全局id，index是局部id(即n_id的索引)，e_id是全局id
            n_id, edge_index, e_id = self.neighbor_loader(batch_n_id)
            self.assoc[n_id] = torch.arange(n_id.size(0), device=device)

            # if cnt ==15:
            #     print('stop here')
            # Get updated memory of all nodes involved in the computation.
            z, last_update = self.memory(n_id)

            z = self.gnn(z, last_update, edge_index, self.data.t[e_id.to(torch.device('cpu'))],
                         self.data.msg[e_id.to(torch.device('cpu'))])

            # pos_out = self.classifier(z[self.assoc[batch_n_id]])
            pos_out = self.link_pred(z[self.assoc[src]], z[self.assoc[pos_dst]])
            # 把pos_out按照第0维拼起来，但是只有一个tensor的时候应该是没有区别
            y_pred = torch.cat([pos_out], dim=0)

            # 这里应该就是传全局索引
            # y_true = self.data.y[self.assoc[batch_n_id].to(torch.device('cpu'))]
            # y_true = self.data.y[batch_n_id.to(torch.device('cpu'))]
            y_true = batch.y
            loss = self.criterion(y_pred, y_true.to(device)) # 返回该批次的平均损失
            total_loss += float(loss) * batch.num_events

            # Update memory and neighbor loader with ground-truth state.
            self.memory.update_state(src, pos_dst, t, msg.to(torch.float32))
            self.neighbor_loader.insert(src, pos_dst) # 因为是时间流图，neighbor_loader应该感知不到之后的节点

            each_edge_loss = cal_pos_edge_loss_multiclass(pos_out, y_true.to(device))

            for i in range(len(pos_out)):
                # node_src = int(batch_n_id[i])
                # node_dst = int()
                loss = each_edge_loss[i]

                temp_dic = {
                    'loss': float(loss),
                    'nodeId': src[i],
                }
                node_list.append(temp_dic)
                temp_dic = {
                    'loss': float(loss),
                    'nodeId': pos_dst[i],
                }
                node_list.append(temp_dic)
        return node_list


    def load_model(self, model_path):
        loaded_model = torch.load(model_path)
        self.memory = loaded_model[0].to(device)
        self.gnn = loaded_model[1].to(device)
        self.classifier = loaded_model[2].to(device)
        self.neighbor_loader = loaded_model[3]



if __name__ == '__main__':
    """读取和保存路径"""

    data_name = 'SysClient0201'

    data_dir = './data/OPTC/using_data/evaluation'
    map_dir = './data/OPTC/map/evaluation'
    gt_path = './data/OPTC/ground_truth/ground_truth.txt'
    anomaly_dir = './anomaly_graph'

    weight_path = './weights/SysClient0201_epoch=25_lr=1e-05_loss=1.1693.pt'

    """数据加载"""
    test_data = torch.load(osp.join(data_dir, data_name+'.TemporalData'))
    node_uuid2index = torch.load(osp.join(map_dir, data_name + '_uuid2index'))
    '''gt加载'''
    # gt_node_uuid = set()
    # with open(gt_path, 'r', encoding='utf-8') as f:
    #     for line in f:
    #         gt_node_uuid.add(line.strip())

    max_num_node = len(node_uuid2index) // 2 + 1

    print(f'{max_num_node=}')

    print(f"num train data {len(test_data.msg)}")
    saved_nodes = set()
    print(f'Initiating class....')
    tester = Test(test_data, max_num_node + 2)

    # print(f'Transfer data....')
    # tester.data2vec()

    print(f'loading weight: {weight_path}')
    tester.load_model(weight_path)
    print(f'Start testing...')

    node_loss_list = tester.test()
    # weight_name =  weight_path.split('/')[-1].split('.')[0]
    torch.save(node_loss_list,osp.join(anomaly_dir, f'node_dic_list_{data_name}'))
    print(f'finish saving...')

    # node_dic = torch.load('./node_dic_list_model_saved_trace')
    # result = cal_metrics(gt_node_uuid, node_loss_list, gate, node_uuid2index)
    # print(f'calculating metrics...')
    # for metric, value in result.items():
    #     print(f"{metric}: {value:.4f}")