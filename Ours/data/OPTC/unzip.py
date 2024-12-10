import io
import gzip
import os.path as osp
import time

from os import walk
"""
    step1 解压
    zip_data -> raw_data
"""

data_folder = "./zip_data/benign"
out_folder = "./raw_data"


# 传入的两个变量都应该是list
def extract_logs(filepath_list, hostid_list):
    for hostid in hostid_list:
        start_time = time.time()
        print(f'{hostid=}')
        search_pattern = f'SysClient{hostid}'
        output_filename = f'SysClient{hostid}.systemia.com.txt'

        with open(osp.join(out_folder, output_filename), 'ab') as f:
            out = io.BufferedWriter(f)
            for filepath in filepath_list:
                with gzip.open(filepath, 'rt', encoding='utf-8') as fin:
                    for line in fin:
                        if search_pattern in line:
                            out.write(line.encode('utf-8')) # 写入缓冲区
            out.flush() # 将剩余缓冲区写入文件
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

    extract_logs(res, hostId_list)