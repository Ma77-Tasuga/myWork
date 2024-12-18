import os.path as osp
import csv
import time

data_path = '../Ours/data/OPTC/raw_data/evaluation/SysClient0201.systemia.com.txt'
out_folder = './data'
search_pattern = r'"object":"SHELL"'

# search_list = []
# with open(data_path, 'r', encoding='utf-8') as f:
#     for line in f:
#         line = line.strip()
#         if search_pattern in line:
#             search_list.append(line)
# with open(osp.join(out_folder, 'search_svchost.txt'),'w',encoding='utf-8') as wf:
#     search_list = [e+'\n' for e in search_list]
#     print(len(search_list))
#     wf.writelines(search_list)
#     print(f'ok')

with open('../Ours/data/OPTC/parsed_data/evaluation/SysClient0201.csv', 'r', encoding='utf-8', newline='') as f:
    reader = csv.DictReader(f)
    cnt = 0
    num_coupling = 0  # 埋点
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

        print(timestamp)
        cnt +=1
        if cnt==500:
            break