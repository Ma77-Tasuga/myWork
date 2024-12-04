import csv
import time
import json
import os.path as osp
import torch

"""
    step2 解析txt文件中的原始日志条目，生成一个csv文件以用于导入数据库
    raw_data -> parsed_data
    feat: nodeId_dic to map
"""


origin_file = './raw_data/SysClient0201.systemia.com.txt'  # 原始日志文件
num_show =1000 # 控制打印进度的日志行数
write_file = './parsed_data/SysClient0201.csv'
map_folder = './map'

"""检查是否存在不需要的字段"""
def is_valid_entry(entry) -> bool:
    valid_objects = {'PROCESS',
                     'FILE',
                     'FLOW',
                     # 'MODULE',
                     }
    invalid_actions = {'START', 'TERMINATE'}

    object_valid = entry['object'] in valid_objects
    action_valid = entry['action'] not in invalid_actions
    actor_object_different = entry['actorID'] != entry['objectID'] # 保证不是自我操作，会导致自环

    return object_valid and action_valid and actor_object_different


"""检查传入的data是否合规，并过滤可能的重复"""
def Traversal_Rules(data):
    filtered_data = {}

    for entry in data:
        if is_valid_entry(entry):
            key = (
                entry['action'],
                entry['actorID'],
                entry['objectID'],
                entry['object'],
                entry['pid'],
                entry['ppid']
            )
            filtered_data[key] = entry # 过滤重复

    return list(filtered_data.values())


def Sentence_Construction(entry):
    action = entry["action"]
    properties = entry['properties']
    object_type = entry['object']

    format_strings = {
        'PROCESS': "{parent_image_path} {action} {image_path} {command_line}",
        'FILE': "{image_path} {action} {file_path}",
        'FLOW': "{image_path} {action} {src_ip} {src_port} {dest_ip} {dest_port} {direction}",
        'MODULE': "{image_path} {action} {module_path}"
    }

    default_format = "{image_path} {action} {module_path}"

    try:
        format_str = format_strings.get(object_type, default_format)
        phrase = format_str.format(action=action, **properties) # action通过参数提供，其他的字段从properties里面找
    except KeyError:
        phrase = ''

    return phrase.split(' ')


"""event为从json中读取的字段，该函数为其添加两个字段actorname和objectname"""
def Extract_Semantic_Info(event):
    object_type = event['object']
    properties = event['properties']

    label_mapping = {
        "PROCESS": ('parent_image_path', 'image_path'),
        "FILE": ('image_path', 'file_path'),
        "MODULE": ('image_path', 'module_path'),
        "FLOW": ('image_path', 'dest_ip', 'dest_port')
    }

    label_keys = label_mapping.get(object_type, None) #最后一个参数表示没有匹配对象时返回的默认值
    # 从properties中获取上面定义的属性，这个属性根据节点类型存在差异
    if label_keys:
        labels = [properties.get(key) for key in label_keys]
        if all(labels): # 判断是否所有属性都匹配到了值
            event["actorname"], event["objectname"] = labels[0], ' '.join(labels[1:]) # 为evnet添加了字段
            # 如果是网络流的话，对象名称就用空格分开ip和端口
            return event
    return None


def transform(text):
    labeled_data = [event for event in (Extract_Semantic_Info(x) for x in text) if event]
    # 日志去重操作
    data = Traversal_Rules(labeled_data)

    phrases = [Sentence_Construction(x) for x in data if Sentence_Construction(x)] #返回特殊属性的一些关键部分
    for datum, phrase in zip(data, phrases): # datum是data的引用，而不是副本
        datum['phrase'] = phrase

    return data


def load_data(file_path):
    with open(file_path, 'r') as file:
        content = [json.loads(line.strip()) for line in file]

    return transform(content)




if __name__ == '__main__':
    """ 数据处理 """
    start_time = time.time()

    data = load_data(origin_file)
    end_time = time.time()
    data_len = len(data)
    print(f'Num of lines: {data_len}')
    print(f'Eclipse time: {end_time-start_time} s.')

    nodeId_list = []
    """ 写入csv """
    start_time = time.time()
    with open(write_file, 'w', newline='') as wf:
        writer = csv.writer(wf)

        # 写入CSV表头
        writer.writerow(['actorID', 'actorname', 'objectID', 'objectname', 'action', 'timestamp', 'pid', 'ppid', 'object', 'phrase'])

        # 写入每行数据
        for entry in data:
            # nodeId_list.append((entry['actorname'],entry['actorID']))
            # nodeId_list.append((entry['objectname'],entry['objectID']))
            nodeId_list.append(frozenset((entry['ppid'],entry['actorID'])))
            nodeId_list.append(frozenset((entry['pid'],entry['objectID'])))
            writer.writerow([
                entry['actorID'],
                entry['actorname'],
                entry['objectID'],
                entry['objectname'],
                entry['action'],
                entry['timestamp'],
                entry['pid'],
                entry['ppid'],
                entry['object'],
                entry['phrase']
            ])
    print(f"len nodeId_list:{len(nodeId_list)}")
    nodeId_list = list(set(nodeId_list)) # 去重
    print(f'len nodeId_set:{len(nodeId_list)}')
    """构建节点索引"""
    nodeId_dic = {}
    for i in range(len(nodeId_list)):
        nodeId_dic[nodeId_list[i]] = i
        nodeId_dic[i] = nodeId_list[i]
    print(f'len node_dic:{len(nodeId_dic)}')
    torch.save(nodeId_dic, osp.join(map_folder, 'uuid2index'))

    end_time = time.time()
    print(f"Data has been written to output.csv, using time {end_time-start_time} s.")