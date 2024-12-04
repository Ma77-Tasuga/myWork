import csv
import os.path as osp
import time
from Ours.scripts.utils import *
import torch
from torch_geometric.data import TemporalData

# TODO: 现在还没有实现解耦合
"""
    step3
    parsed_data -> using_data
    fix map: uuid2index
"""

data_folder = './parsed_data'
data_path = './parsed_data/SysClient0201.csv'
map_folder = './map'

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
    node_uuid2index = torch.load(osp.join(map_folder, 'uuid2index'))  # 从0开始
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

    with open(data_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        cnt = 0

        start_time = time.time()

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

            msg_list.append(torch.cat([str2tensor('PROCESS', actorname),
                                       edge2vec[rel2id[action]],  # 这个就是one-hot编码
                                       str2tensor(object_type, objectname)]))
            # 沿着第一个维度拼接(16,)(8,)(16,) -> (40,)
            src_list.append(node_uuid2index[srcId])
            dst_list.append(node_uuid2index[dstId])
            y_list.append(rel2id[action])
            t_list.append(int(datetime_to_timestamp_US(timestamp)))
            # print(f'{actorID=} {actorname=} {objectID=} {objectname=} {action=} {timestamp=} {pid=} {ppid=} {object_type=} {phrase=}')
            """计数计时"""
            if cnt % 2000 == 0:
                end_time = time.time()
                print(f'Now {cnt} lines imported, using time {end_time - start_time} s.')
                start_time = end_time

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

    torch.save(dataset,f'./using_data/optc.TemporalData')