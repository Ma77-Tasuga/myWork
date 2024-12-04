import os.path as osp

data_path = '../Ours/data/OPTC/raw_data/SysClient0201.systemia.com.txt'
out_folder = './data'
search_pattern = 'svchost.exe'

search_list = []
with open(data_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if search_pattern in line:
            search_list.append(line)
with open(osp.join(out_folder, 'search_svchost.txt'),'w',encoding='utf-8') as wf:
    print(len(search_list))
    search_list = [e+'\n' for e in search_list]
    print(len(search_list))
    wf.writelines(search_list)
    print(f'ok')


