import csv
import time
import os.path as osp
import re

"""
    step2
"""
input_dir = './raw_data'
output_dir = './parsed_data'


def show(str):
    print(str + ' ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))


# 保存文件名path
# path_list = ['ta1-cadets-e3-official.json', 'ta1-cadets-e3-official-2.json', 'ta1-fivedirections-e3-official-2.json',
#              'ta1-theia-e3-official-1r.json', 'ta1-theia-e3-official-6r.json', 'ta1-trace-e3-official-1.json']
path_list = ['ta1-trace-e3-official-1.json']

# 用于模式匹配，匹配括号里的字符
pattern_uuid = re.compile(r'uuid\":\"(.*?)\"')
pattern_src = re.compile(r'subject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_dst1 = re.compile(r'predicateObject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_dst2 = re.compile(r'predicateObject2\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_type = re.compile(r'type\":\"(.*?)\"')
pattern_time = re.compile(r'timestampNanos\":(.*?),')

notice_num = 1000000

# node_type_set = set()
# edge_type_set = set()

for path in path_list:
    id_nodetype_map = {}
    for i in range(100):
        now_path = path + '.' + str(i)
        if i == 0: now_path = path
        if not osp.exists(osp.join(input_dir, now_path)): break
        f = open(osp.join(input_dir, now_path), 'r')
        show(now_path)
        cnt = 0
        for line in f:
            cnt += 1
            if cnt % notice_num == 0:
                print(cnt)
            if 'com.bbn.tc.schema.avro.cdm18.Event' in line or 'com.bbn.tc.schema.avro.cdm18.Host' in line: continue
            if 'com.bbn.tc.schema.avro.cdm18.TimeMarker' in line or 'com.bbn.tc.schema.avro.cdm18.StartMarker' in line: continue
            if 'com.bbn.tc.schema.avro.cdm18.UnitDependency' in line or 'com.bbn.tc.schema.avro.cdm18.EndMarker' in line: continue
            if len(pattern_uuid.findall(line)) == 0: print(line)
            uuid = pattern_uuid.findall(line)[0]
            subject_type = pattern_type.findall(line)

            if len(subject_type) < 1:
                if 'com.bbn.tc.schema.avro.cdm18.MemoryObject' in line:
                    id_nodetype_map[uuid] = 'MemoryObject'
                    continue
                if 'com.bbn.tc.schema.avro.cdm18.NetFlowObject' in line:
                    id_nodetype_map[uuid] = 'NetFlowObject'
                    continue
                if 'com.bbn.tc.schema.avro.cdm18.UnnamedPipeObject' in line:
                    id_nodetype_map[uuid] = 'UnnamedPipeObject'
                    continue

            id_nodetype_map[uuid] = subject_type[0]

    # for type in id_nodetype_map.values():
    #     node_type_set.add(type)

    not_in_cnt = 0
    for i in range(100):
        now_path = path + '.' + str(i)
        if i == 0: now_path = path
        if not osp.exists(osp.join(input_dir, now_path)): break
        f = open(osp.join(input_dir, now_path), 'r')
        fw = open(osp.join(output_dir, now_path) + '.txt', 'w')
        cnt = 0
        for line in f:
            cnt += 1
            if cnt % notice_num == 0:
                print(cnt)

            if 'com.bbn.tc.schema.avro.cdm18.Event' in line:
                pattern = re.compile(r'subject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
                edgeType = pattern_type.findall(line)[0]
                timestamp = pattern_time.findall(line)[0]
                srcId = pattern_src.findall(line)

                # edge_type_set.add(edgeType)

                if len(srcId) == 0: continue
                srcId = srcId[0]
                if not srcId in id_nodetype_map.keys():
                    not_in_cnt += 1
                    continue
                srcType = id_nodetype_map[srcId]
                dstId1 = pattern_dst1.findall(line)
                if len(dstId1) > 0 and dstId1[0] != 'null':
                    dstId1 = dstId1[0]
                    if not dstId1 in id_nodetype_map.keys():
                        not_in_cnt += 1
                        continue
                    dstType1 = id_nodetype_map[dstId1]
                    this_edge1 = str(srcId) + '\t' + str(srcType) + '\t' + str(dstId1) + '\t' + str(
                        dstType1) + '\t' + str(edgeType) + '\t' + str(timestamp) + '\n'
                    fw.write(this_edge1)

                dstId2 = pattern_dst2.findall(line)
                if len(dstId2) > 0 and dstId2[0] != 'null':
                    dstId2 = dstId2[0]
                    if not dstId2 in id_nodetype_map.keys():
                        not_in_cnt += 1
                        continue
                    dstType2 = id_nodetype_map[dstId2]
                    this_edge2 = str(srcId) + '\t' + str(srcType) + '\t' + str(dstId2) + '\t' + str(
                        dstType2) + '\t' + str(edgeType) + '\t' + str(timestamp) + '\n'
                    fw.write(this_edge2)
        fw.close()
        f.close()

    # with open(osp.join('./using_data/', path.split('.')[0], 'nodeType2id.csv'), 'w', encoding='utf-8') as fw_node, open(
    #         osp.join('./using_data/', path.split('.')[0], 'edgeType2id.csv'), 'w', encoding='utf-8') as fw_edge:
    #     write_list = []
    #     for item in node_type_set:
    #         cnt = 1
    #         write_list.append([str(item), str(cnt)])
    #         cnt += 1
    #     writer = csv.writer(fw_node)
    #     writer.writerows(write_list)
    #
    #     write_list = []
    #     for item in edge_type_set:
    #         cnt = 1
    #         write_list.append([str(item), str(cnt)])
    #         cnt += 1
    #     writer = csv.writer(fw_edge)
    #     writer.writerows(write_list)
