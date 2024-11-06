from os import walk
import json
from scripts.utils import *
from tqdm import tqdm
import csv

"""
    step2
"""

# 记录uuid的image_path的变量
node_uuid2path = {}

# 不知道有什么用
node_type = {'FILE',
             'FLOW',
             'MODULE',
             'PROCESS',
             'REGISTRY',
             'SHELL',
             'TASK',
             'THREAD',
             'USER_SESSION'
             }

reverse_edge_type = [
    "READ",
]

node_type_used = [
    'FILE',
    'FLOW',
    'PROCESS',
    #  'SHELL',
]

pid_split_symble = "#_"
host_split_symble = "_@"


def process_raw_dic(raw_dic):
    """
    对原始数据做一些处理
    :param raw_dic: 原始数据dic,是直接json.load()的一行
    :return: 处理后的数据dic
    """
    ans_dic = {}

    ans_dic['hostname'] = raw_dic['hostname'].split('.')[0]

    ans_dic['edge_type'] = raw_dic['action']
    ans_dic['src_id'] = raw_dic['actorID']
    ans_dic['dst_id'] = raw_dic['objectID']

    ans_dic['src_type'] = 'PROCESS'
    ans_dic['timestamp'] = datetime_to_timestamp_US(raw_dic['timestamp'])
    ans_dic['dst_type'] = raw_dic['object']

    try:
        node_uuid2path[ans_dic['src_id']] = ans_dic['hostname'] + host_split_symble + raw_dic['properties'][
            'image_path']

        if raw_dic['object'] == 'FLOW':
            temp_flow = f"{raw_dic['properties']['direction']}#{raw_dic['properties']['src_ip']}:{raw_dic['properties']['src_port']}->{raw_dic['properties']['dest_ip']}:{raw_dic['properties']['dest_port']}"
            node_uuid2path[ans_dic['dst_id']] = ans_dic['hostname'] + host_split_symble + temp_flow

        if raw_dic['object'] == 'FILE':
            node_uuid2path[ans_dic['dst_id']] = ans_dic['hostname'] + host_split_symble + raw_dic['properties'][
                'file_path']


    except:
        ans_dic = {}

    return ans_dic


def is_selected_hosts_benign(line):
    hosts = [
        'SysClient0201',
        'SysClient0402',
        'SysClient0660',
        'SysClient0501',
        'SysClient0051',
        'SysClient0209',
    ]
    flag = False
    for h in hosts:
        if h in line:
            flag = True
            break
    return flag


"""
    保存benign数据
"""
# folder path
dir_path = '/home/monk/datasets/OpTC_data/ecar/benign/'
# TODO：改一下这个路径
write_path = ''

res = []

# 遍历目录，返回文件路径，benign
for (dir_path, dir_names, file_names) in walk(dir_path):
    if dir_path[-1] != '/':
        dir_path += '/'
    #     print(f"{dir_path=}")
    #     print(f"{file_names=}")
    for f in file_names:
        temp_file_path = dir_path + f
        #         print(f"{temp_file_path=}")
        # 选择一些文件
        if "201-225" in temp_file_path or ("20-23Sep19" in temp_file_path and (
                "401-425" in temp_file_path or "651-675" in temp_file_path or "501-525" in temp_file_path or "51-75" in temp_file_path)):
            res.append(temp_file_path)

# 应该不存在没有解压的，不然下面的执行逻辑也有问题
# for r in tqdm(res):
#     # 如果没有解压就解压
#     if ".gz" in r:
#         os.system(f"gzip -d {r}")
#         print(f" {r} Finished！")

# 返回一个处理后的data_list，这里是按文件进行处理的
for file_path in res:

    edge_list = []

    with open(file_path) as f:
        for line in tqdm(f):
            line = line.replace('\\\\', '/')  # 替换\\为/
            temp_dic = json.loads(line.strip())
            hostname = temp_dic['hostname'].split('.')[0]
            # 筛选节点类型和主机
            if temp_dic['object'] in node_type_used and is_selected_hosts_benign(hostname):
                edge_list.append(process_raw_dic(temp_dic))  # 返回一个处理后的字典类型

        print(f'{len(edge_list)=}')
        data_list = []
        # 遍历存储
        for e in edge_list:
            try:
                data_list.append([
                    e['src_id'],
                    e['src_type'],
                    e['edge_type'],
                    e['dst_id'],
                    e['dst_type'],
                    e['hostname'],
                    e['timestamp'],
                    "benign",
                ])
            except:
                pass

        with open(write_path, 'w', newline='', encoding='utf-8') as wf:
            writer = csv.writer(wf)
            writer.writerow([])
            # TODO: fill this
            writer.writerows(data_list)



        # 写入数据库
        # sql = '''insert into event_table
        #                      values %s
        #         '''
        # ex.execute_values(cur, sql, data_list, page_size=10000)
        # connect.commit()

        print(f"{file_path} Finished! ")
        # Clear the tmp variables to release the memory.
        del edge_list
        del data_list

"""
    保存eval数据
"""
# folder path
dir_path = '/home/monk/datasets/OpTC_data/ecar/evaluation/'
# TODO: 改一下这个目录

res = []
for (dir_path, dir_names, file_names) in walk(dir_path):
    if dir_path[-1] != '/':
        dir_path += '/'
    for f in file_names:
        temp_file_path = dir_path + f
        #         print(f"{temp_file_path=}")
        if (
                "201-225" in temp_file_path or "401-425" in temp_file_path or "651-675" in temp_file_path or "501-525" in temp_file_path or "51-75" in temp_file_path):
            res.append(temp_file_path)


# for r in tqdm(res):
#     if ".gz" in r:
#         os.system(f"gzip -d {r}")
#         print(f" {r} Finished！")


# 这个和benign的不一样
def is_selected_hosts_eval(line):
    hosts = [
        'SysClient0201',
        'SysClient0402',
        'SysClient0660',
        'SysClient0501',
        'SysClient0051',
        'SysClient0207',
    ]
    flag = False
    for h in hosts:
        if h in line:
            flag = True
            break
    return flag


for file_path in res:

    edge_list = []

    with open(file_path) as f:
        for line in tqdm(f):
            line = line.replace('\\\\', '/')
            temp_dic = json.loads(line.strip())
            hostname = temp_dic['hostname'].split('.')[0]
            if temp_dic['object'] in node_type_used and is_selected_hosts_eval(hostname):
                edge_list.append(process_raw_dic(temp_dic))

        print(f'{len(edge_list)=}')
        data_list = []
        for e in edge_list:
            try:
                data_list.append([
                    e['src_id'],
                    e['src_type'],
                    e['edge_type'],
                    e['dst_id'],
                    e['dst_type'],
                    e['hostname'],
                    e['timestamp'],
                    "evaluation",
                ])
            except:
                pass

        # sql = '''insert into event_table values %s'''
        # ex.execute_values(cur, sql, data_list, page_size=10000)
        # connect.commit()

        print(f"{file_path} Finished! ")
        # Clear the tmp variables to release the memory.
        del edge_list
        del data_list

"""
    保存节点的id-path映射
"""

data_list = []
for n in node_uuid2path:
    try:
        data_list.append([
            n,
            node_uuid2path[n]
        ])
    except:
        pass

# sql = '''insert into nodeid2msg
#                      values %s
#         '''
# ex.execute_values(cur, sql, data_list, page_size=10000)
# connect.commit()

len(node_uuid2path)
