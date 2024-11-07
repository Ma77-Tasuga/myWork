import torch
import os.path as osp
from scripts.config import *

"""
    step1
    假设数据已下载并解压
"""

data_dir = './data/DARPA_T3/using_data/train'

if __name__ == '__main__':
    dataset = torch.load(osp.join(data_dir, 'trace.TemporalData'))

