import csv
import os
import os.path as osp
import time
from Ours.scripts.utils import *
import torch
from torch_geometric.data import TemporalData
import networkx as nx
"""
    step3
    parsed_data -> using_data
    fix: map/uuid2index
    feat: graph/graph.graphml, map/coupling_node
"""
is_evaluation = False
benign_data_folder = './parsed_data/benign'
# data_path = './parsed_data/SysClient0201.csv'
benign_map_folder = './map/benign'
benign_graph_folder = './graph/benign'
write_benign_folder = './using_data/benign'
gt_path = './ground_truth/ground_truth.txt'

show_cnt = 2000

rel2id = {0: 'OPEN',
          'OPEN': 0,
          1: 'READ',
          'READ': 1,
          2: 'CREATE',
          'CREATE': 2,
          3: 'MESSAGE',
          'MESSAGE': 3,
          4: 'MODIFY',
          'MODIFY': 4,
          5: 'RENAME',
          'RENAME': 5,
          6: 'DELETE',
          'DELETE': 6,
          7: 'WRITE',
          'WRITE': 7,
          # 8: 'LOAD',
          # 'LOAD': 8,
          }
nodeType2id={0: 'PROCESS',
             'PROCESS': 0,
             1: 'FILE',
             'FILE': 1,
             2: 'FLOW',
             'FOLW': 2,
             # 3: 'MODULE',
             # 'MODULE': 3,
}


if __name__ == '__main__':

    for filename in os.listdir(benign_data_folder):
        hostname = filename.split('.')[0]
        """读取id映射"""
        node_uuid2index = torch.load(osp.join(benign_map_folder, hostname+'_uuid2index'))  # 从0开始
        assert len(node_uuid2index) % 2 == 0, f"error in len of node_uuid2index: {len(node_uuid2index)}"
        # 如果要插入新映射的起始id（因为第一个id是0）
        node_counter = len(node_uuid2index) / 2 # 节点id计数器，也是新增分裂节点的下标

        coupling_node_dic = {} # 这个歌用来存储节点的耦合次数，用于配合赋予新的节点id
        graph = nx.DiGraph() # 有向图

        if is_evaluation:
            # 会形成一个gt_set
            with open(gt_path,'r',encoding='utf-8') as gt_f:
                gt_list = [line.strip() for line in gt_f]

            gt_set = set(gt_list)
            assert len(gt_set) == len(gt_list), f'conflict in gt file, num gt_set is {len(gt_set)} while num gt_list is {len(gt_list)}'
            del gt_list # 节省点内存

        # set是无序集合，不能通过下标索引
        # for key in node_uuid2index.keys():
        #     print(key)
        # for i in range(int(len(node_uuid2index)/2)):
        #     print(set(node_uuid2index[int(i)])[1])
        src_list = []
        dst_list = []
        y_list = []
        msg_list = []
        t_list = []

        with open(osp.join(benign_data_folder, filename), 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            cnt = 0
            num_coupling = 0 # 埋点
            start_time = time.time()

            """读取预处理的日志流"""
            print(f'Constructing graph.....')
            for row in reader:
                # 从 CSV 文件中读取需要导入的数据
                actorID = row['actorID']
                actorname = row['actorname']
                objectID = row['objectID']
                objectname = row['objectname']
                action = row['action']
                timestamp = row['timestamp']
                pid = row['pid'],
                ppid = row['ppid'],
                object_type = row['object'],
                phrase = row['phrase']

                # 下面三个变量是一个特殊的元组类型，将其转化成普通字符串
                pid = list(pid)[0]
                ppid = list(ppid)[0]
                object_type = list(object_type)[0]
                cnt += 1

                srcId = frozenset((int(ppid), actorID))
                dstId = frozenset((int(pid), objectID))


                """图构建"""
                # 一旦发生变动，下面这两个Id都要保持最新
                src_nodeId = node_uuid2index[srcId]
                dst_nodeId = node_uuid2index[dstId]
                # 检查节点是否是被分裂的节点，如果是，将其更新成最新的节点
                if srcId in coupling_node_dic.keys():
                    actorID_new = actorID+'-'+str(coupling_node_dic[srcId])
                    src_nodeId = node_uuid2index[frozenset((int(ppid), actorID_new))]
                if dstId in coupling_node_dic.keys():
                    objectID_new = objectID+'-'+str(coupling_node_dic[dstId])
                    dst_nodeId = node_uuid2index[frozenset((int(pid), objectID_new))]

                # 插入节点，事先检查节点是否存在
                if src_nodeId not in graph:
                    gt = 0
                    if is_evaluation and actorID in gt_set:
                        gt = 1
                    graph.add_node(src_nodeId, name=actorname, node_type='PROCESS', gt=gt)
                if dst_nodeId not in graph:
                    gt = 0
                    if is_evaluation and objectID in gt_set:
                        gt =1
                    graph.add_node(dst_nodeId, name=objectname, node_type=object_type, gt=gt)

                if graph.in_degree(dst_nodeId) >=1:
                    # 更新冲突的目的节点（分裂）
                    dst_nodeId = node_counter
                    node_counter += 1
                    num_coupling +=1 #埋点
                    # 插入新分裂的节点
                    gt =0
                    if is_evaluation and objectID in gt_set:
                        gt = 1
                    graph.add_node(dst_nodeId,name=objectname,node_type=object_type,gt=gt)
                    graph.add_edge(src_nodeId, dst_nodeId, edge_type=action, timestamp=timestamp, phrase=phrase)

                    """更新索引"""
                    # 更新耦合次数索引
                    if dstId not in coupling_node_dic.keys():
                        coupling_node_dic[dstId] = 1
                    else:
                        coupling_node_dic[dstId] = coupling_node_dic[dstId] + 1 # 耦合次数加1
                    # 更新uuid和uuid索引
                    coupling_cnt = coupling_node_dic[dstId]
                    objectID_new = objectID+'-'+str(coupling_cnt) # e.p. uuid-1, uuid-112
                    dstId_new = frozenset((int(pid),objectID_new))
                    # 下面用新的uuid为了避免索引混乱
                    node_uuid2index[dst_nodeId] = dstId_new
                    node_uuid2index[dstId_new] = dst_nodeId

                else:
                    graph.add_edge(src_nodeId, dst_nodeId, edge_type=action, timestamp=timestamp, phrase=phrase)


                """将处理好的数据向量化，保存"""
                msg_list.append(torch.cat([str2tensor('PROCESS', actorname),
                                           edge2vec[rel2id[action]],  # 这个就是one-hot编码
                                           str2tensor(object_type, objectname)]))
                # 沿着第一个维度拼接(16,)(8,)(16,) -> (40,)
                # src_list.append(node_uuid2index[srcId])
                # dst_list.append(node_uuid2index[dstId])
                src_list.append(src_nodeId)
                dst_list.append(dst_nodeId)
                y_list.append(rel2id[action])
                t_list.append(int(datetime_to_timestamp_US(timestamp)))
                # print(f'{actorID=} {actorname=} {objectID=} {objectname=} {action=} {timestamp=} {pid=} {ppid=} {object_type=} {phrase=}')
                """计数计时"""
                if cnt % show_cnt == 0:
                    end_time = time.time()
                    print(f'Now {cnt} lines imported, using time {end_time - start_time} s.')
                    start_time = end_time

        """制作成时间数据集"""
        print(f'Transfer to TemporalData......')
        dataset = TemporalData()
        dataset.src = torch.tensor(src_list)
        dataset.dst = torch.tensor(dst_list)
        dataset.t = torch.tensor(t_list)
        dataset.msg = torch.vstack(msg_list)
        dataset.y = torch.tensor(y_list)

        dataset.src = dataset.src.to(torch.long)
        dataset.dst = dataset.dst.to(torch.long)
        dataset.t = dataset.t.to(torch.long)
        dataset.y = dataset.y.to(torch.long)
        dataset.msg = dataset.msg.to(torch.float)
        print(f'Saving dataset.....')
        torch.save(dataset,osp.join(write_benign_folder, hostname+'.TemporalData'))
        print(f'Saving nodeId map.....')
        torch.save(node_uuid2index, osp.join(benign_map_folder, hostname + '_uuid2index'))
        torch.save(coupling_node_dic, osp.join(benign_map_folder, hostname+'_coupling_node'))
        print(f'Saving graph.....')
        nx.write_graphml(graph, osp.join(benign_graph_folder, hostname+"_graph.graphml"))
        print(f'All done.')
        print(f'{num_coupling=}, len of uuid2index after={len(node_uuid2index)}') # 打印埋点