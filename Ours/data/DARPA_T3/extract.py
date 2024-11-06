from os import path as osp

from ...scripts.utils import *


"""
    step3
    提取需要的数据
"""

open_dir = './parsed_data'
write_dir = './using_data'

selected_file_train = ['ta1-trace-e3-official-1.json.txt']
selected_file_test = ['ta1-trace-e3-official-1.json.4.txt']


for file in selected_file_train:

    node_uuid2index = {} # 存放uuid与index下标的映射，存在反向映射，key为str类型

    nodeType_list = [] # 节点类型表，与节点下表索引一致
    now_index = 0


    with open(osp.join(open_dir, file), 'r', encoding='utf-8') as f:

        # [srcId, srcType, dstId2, dstType2, edgeType, timestamp]
        lines = f.readlines()


        """这个循环构建uuid到index的双向映射，构建节点类型表，下标是index"""
        for line in lines:
            entry = line.strip().split('\t')

            src_type = entry[1]
            dst_type = entry[3]

            # edge_type = entry[4]
            # raw_time = entry[5]

            srcId = str(entry[0])
            dstId = str(entry[2])

            '''构建节点uuid到数字下标的索引，索引从0开始'''
            if srcId not in node_uuid2index:
                node_uuid2index[srcId] = now_index
                node_uuid2index[str(now_index)] = srcId
                now_index +=1

                nodeType_list.append(src_type)

            if dstId not in node_uuid2index:
                node_uuid2index[dstId] = now_index
                node_uuid2index[str(now_index)] = dstId
                now_index += 1

                nodeType_list.append(dst_type)

        src_list = []
        dst_list = []
        edgeType_list = []
        time_list = [] # darpa t3的时间戳本来就是nano数字类型，但可能要去掉最后的6为零
        """这个循环按顺序构建二维临接表，边类型表，时间戳表"""
        for line in lines:
            entry = line.strip().split('\t')

            # 临接表
            src_list.append(node_uuid2index[entry[0]])
            dst_list.append(node_uuid2index[entry[2]])
            edgeType_list.append(entry[4])  # 边类型
            time_list.append(entry[5]) # 原始时间戳

    # with open(osp.join(write_dir,file.split('.')[0]+''))


"""测试集"""
for file in selected_file_test:

    node_uuid2index = {}  # 存放uuid与index下标的映射，存在反向映射，key为str类型

    nodeType_list = []  # 节点类型表，与节点下表索引一致
    now_index = 0

    with open(osp.join(open_dir, file), 'r', encoding='utf-8') as f:

        # [srcId, srcType, dstId2, dstType2, edgeType, timestamp]
        lines = f.readlines()

        """这个循环构建uuid到index的双向映射，构建节点类型表，下标是index"""
        for line in lines:
            entry = line.strip().split('\t')

            src_type = entry[1]
            dst_type = entry[3]

            # edge_type = entry[4]
            # raw_time = entry[5]

            srcId = str(entry[0])
            dstId = str(entry[2])

            '''构建节点uuid到数字下标的索引，索引从0开始'''
            if srcId not in node_uuid2index:
                node_uuid2index[srcId] = now_index
                node_uuid2index[str(now_index)] = srcId
                now_index += 1

                nodeType_list.append(src_type)

            if dstId not in node_uuid2index:
                node_uuid2index[dstId] = now_index
                node_uuid2index[str(now_index)] = dstId
                now_index += 1

                nodeType_list.append(dst_type)

        src_list = []
        dst_list = []
        edgeType_list = []
        time_list = []
        """这个循环按顺序构建二维临接表，边类型表，时间戳表"""
        for line in lines:
            entry = line.strip().split('\t')

            # 临接表
            src_list.append(node_uuid2index[entry[0]])
            dst_list.append(node_uuid2index[entry[2]])
            edgeType_list.append(entry[4])  # 边类型
            time_list.append(entry[5])  # 原始时间戳


