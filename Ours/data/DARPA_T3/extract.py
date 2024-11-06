from os import path as osp

"""
    step3
    提取需要的数据
"""

open_dir = './parsed_data'
write_dir = './using_data'

selected_file_train = ['ta1-trace-e3-official-1.json.txt']
selected_file_test = ['ta1-trace-e3-official-1.json.4.txt']

node_id2type = {}

for file in selected_file_train:
    with open(osp.join(open_dir, file), 'r', encoding='utf-8') as f:

        # [srcId, srcType, dstId2, dstType2, edgeType, timestamp]

        for line in f:
            entry = line.strip().split('\t')

            node_id2type[entry[0]] = entry[1]
            node_id2type[entry[2]] = entry[3]

            edge_type = entry[4]
            raw_time = entry[5]

            srcId = entry[0]
            dstId = entry[2]
            