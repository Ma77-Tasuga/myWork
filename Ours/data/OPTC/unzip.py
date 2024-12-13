import io
import gzip
import os.path as osp
import time
from Ours.scripts.config import is_evaluation
from os import walk
"""
    step1 解压
    zip_data -> raw_data
"""
if is_evaluation:
    data_folder = "./zip_data/evaluation"
    out_folder = "./raw_data/evaluation"
else:
    data_folder = "./zip_data/benign"
    out_folder = "./raw_data/benign"

# 传入的两个变量都应该是list
# 这个代码会以追加模式写入，请事先删除已经存在的输出文件（raw_data文件夹）
def extract_logs(filepath_list, hostid_list):
    for hostid in hostid_list:
        start_time = time.time()
        print(f'Now parsing {hostid=}')
        search_pattern = f'SysClient{hostid}'
        output_filename = f'SysClient{hostid}.systemia.com.txt'
        total_line_count = 0 # 打点，提取的总点数
        with open(osp.join(out_folder, output_filename), 'ab') as f:
            out = io.BufferedWriter(f)
            for filepath in filepath_list:
                search_line_count = 0 # 打点，匹配到的行数
                file_line_all = 0 # 打点，文件行数
                with gzip.open(filepath, 'rt', encoding='utf-8') as fin:
                    for line in fin:
                        file_line_all += 1 # 打点
                        if search_pattern in line:
                            search_line_count += 1 # 打点
                            out.write(line.encode('utf-8')) # 写入缓冲区
                total_line_count += search_line_count
                print(f'Extract {search_line_count} lines out of {file_line_all} from file {filepath} as host: {hostid}.') # 打印埋点
            out.flush() # 将剩余缓冲区写入文件
            print(f'Finish write to file {output_filename}, {total_line_count} lines in total.')
        end_time = time.time()
        parsing_time = end_time-start_time
        print(f'{parsing_time=} s.')


if __name__ == "__main__":
    res = []
    hostId_list = ['0201']
    for (dir_path, dir_names, file_names) in walk(data_folder):
        if dir_path[-1] != '/':
            dir_path += '/'

        for f in file_names:
            temp_file_path = dir_path + f

            res.append(temp_file_path)

    print(res)
    # do_res = []
    # for file in res:
    #     if '17-18Sep19' in file:
    #         do_res.append(file)

    extract_logs(res, hostId_list)