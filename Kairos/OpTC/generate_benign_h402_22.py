from scripts.utils import *
from tqdm import tqdm
import torch
from torch_geometric.data import TemporalData

# TODO: 这里需要实现两个load，一个是节点映射，一个是event
"""
    step3.1
    根据主机和日期分别收集数据，存储在不同的文件中
"""


for day in tqdm(range(22 ,23)):
    # 根据timestamp查找对应日期的数据
    start_timestamp =datetime_to_timestamp_US('2019-09- ' +str(day) +'T00:00:00')
    end_timestamp =datetime_to_timestamp_US('2019-09- ' +str(day+1) +'T00:00:00')
    hostname ='SysClient0402'
    datalabel ='benign'
    # sql =f"""
    # select * from event_table
    # where
    #       timestamp>{start_timestamp} and timestamp<{end_timestamp}
    #       and hostname='{hostname}' and data_label='{datalabel}' ORDER BY timestamp;
    # """
    # cur.execute(sql)
    # events = cur.fetchall()
    print(f"{len(events)=}")



    node_set =set()
    node_uuid2index ={}
    temp_index =0
    for e in events:
        if e[3] not in node_uuid2path or e[0] not in node_uuid2path:
            continue

        if e[0] in node_uuid2index:
            pass
        else:
            node_uuid2index[e[0]] =temp_index
            node_uuid2index[temp_index ] =node_uuid2path[e[0]]
            temp_index +=1

        if e[3] in node_uuid2index:
            pass
        else:
            node_uuid2index[e[3] ] =temp_index
            node_uuid2index[temp_index ] =node_uuid2path[e[3]]
            temp_index +=1

    torch.save(node_uuid2index ,f'node_uuid2index_9_{day}_host={hostname}_datalabel={datalabel}')


    dataset = TemporalData()
    src = []
    dst = []
    msg = []
    t = []
    for e in (events):
        if e[3] in node_uuid2index and e[0] in node_uuid2index:
            # If the image path of the node is not recorded, then skip this edge
            src.append(node_uuid2index[e[0]])
            dst.append(node_uuid2index[e[3]])
            #     msg.append(torch.cat([torch.from_numpy(node2higvec_bn[i[0]]), rel2vec[i[2]], torch.from_numpy(node2higvec_bn[i[1]])] ))

            # 这里都是之前声明过的数据结构
            msg.append(torch.cat([str2tensor(e[1],node_uuid2path[e[0]]),
                                  edge2vec[e[2]],
                                  str2tensor(e[4],node_uuid2path[e[3]])
                                  ]))
            t.append(int(e[6]))

    dataset.src = torch.tensor(src)
    dataset.dst = torch.tensor(dst)
    dataset.t = torch.tensor(t)
    dataset.msg = torch.vstack(msg)
    dataset.src = dataset.src.to(torch.long)
    dataset.dst = dataset.dst.to(torch.long)
    dataset.msg = dataset.msg.to(torch.float)
    dataset.t = dataset.t.to(torch.long)
    torch.save(dataset, f"./data/evaluation/9_{day}_host={hostname}_datalabel={datalabel}.TemporalData")