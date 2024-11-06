from tqdm import tqdm
import os
from os import walk

"""
    step1
"""

# folder path
dir_path = '/home/xu_rui/workspace_xr/PIDS/myWork/Kairos/OpTC/data/raw_data/'
# TODO: check this file path

# list to store files name
res = []
# 返回给定目录下所有文件的文件路径
for (dir_path, dir_names, file_names) in walk(dir_path):
    if dir_path[-1] != '/':
        dir_path += '/'
        # print(f"{dir_path=}")
        # print(f"{file_names=}")
    for f in file_names:
        temp_file_path = dir_path + f
        # print(f"{temp_file_path=}")

        res.append(temp_file_path)

print(res)

# 对返回的文件路径进一步过滤、解压
# 新文件会替代原来的文件
for r in tqdm(res):
    if ("201-225" in r or "401-425" in r or "651-675" in r or "501-525" in r or "51-75" in r) and ".gz" in r:
        os.system(f"gzip -d {r}")
        print(f" {r} Finished！")