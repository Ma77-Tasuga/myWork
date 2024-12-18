import os.path

raw_file_path = '../Ours/data/OPTC/raw_data/evaluation/SysClient0201.systemia.com.txt'
write_path = './data'


start_list = []
terminate_list = []
with open(raw_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        if 'START' in line:
            start_list.append(line.strip() + '\n')
        if 'TERMINATE' in line:
            terminate_list.append(line.strip() + '\n')

with open(os.path.join(write_path,'start.txt'), 'w', encoding='utf-8') as fs:
    fs.writelines(start_list)
with open(os.path.join(write_path, 'terminate.txt'), 'w', encoding='utf-8') as ft:
    ft.writelines(terminate_list)