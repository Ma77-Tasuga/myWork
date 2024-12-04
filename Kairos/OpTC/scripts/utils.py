import time
from datetime import datetime
from time import mktime
import pytz  # 时区计算
import xxhash
import torch
from torch import nn
from torch.nn import Linear
from sklearn.feature_extraction import FeatureHasher
import numpy as np
import math
import os
"""
    处理时间戳
"""


def datetime_to_ns_time(date):
    """
    :param date: str   format: %Y-%m-%d %H:%M:%S   e.g. 2013-10-10 23:40:00
    :return: nano timestamp
    """
    date, ns = date.split('.')

    timeArray = time.strptime(date, '%Y-%m-%dT%H:%M:%S')
    timeStamp = int(time.mktime(timeArray))
    timeStamp = timeStamp * 1000000000
    timeStamp += int(ns.split('Z')[0])
    return timeStamp


def datetime_to_timestamp_US(date):
    """
    :param date: str   format: %Y-%m-%d %H:%M:%S   e.g. 2013-10-10 23:40:00
    :return: nano timestamp
    """
    date = date.replace('-04:00', '')
    if '.' in date:
        date, ms = date.split('.')
    else:
        ms = 0
    tz = pytz.timezone('Etc/GMT+4')
    timeArray = time.strptime(date, "%Y-%m-%dT%H:%M:%S")
    dt = datetime.fromtimestamp(mktime(timeArray))
    timestamp = tz.localize(dt)
    timestamp = timestamp.timestamp()
    timeStamp = timestamp * 1000 + int(ms)
    return int(timeStamp)


def timestamp_to_datetime_US(ns):
    """
    :param date: str   format: %Y-%m-%d %H:%M:%S   e.g. 2013-10-10 23:40:00
    :return: nano timestamp
    """
    tz = pytz.timezone('US/Eastern')
    ms = ns % 1000
    ns /= 1000
    dt = pytz.datetime.datetime.fromtimestamp(int(ns), tz)
    s = dt.strftime('%Y-%m-%d %H:%M:%S')
    s += '.' + str(ms)
    #     s += '.' + str(int(int(ns) % 1000000000)).zfill(9)
    return s




"""
    处理节点特征
"""
encode_len = 16

FH_string = FeatureHasher(n_features=encode_len, input_type="string")
FH_dict = FeatureHasher(n_features=encode_len, input_type="dict")


# 根据path结构分级提取对应的路径，为了下游的hash构建索引
def path2higlist(p) -> list:
    l = []
    spl = p.strip().split('/')
    for i in spl:
        if len(l) != 0:
            l.append(l[-1] + '/' + i)
        else:
            l.append(i)
    #     print(l)
    return l


# 分层ip
def ip2higlist(p) -> list:
    l = []
    if "::" not in p:
        spl = p.strip().split('.')
        for i in spl:
            if len(l) != 0:
                l.append(l[-1] + '.' + i)
            else:
                l.append(i)
        #     print(l)
        return l
    else:
        spl = p.strip().split(':')
        for i in spl:
            if len(l) != 0:
                l.append(l[-1] + ':' + i)
            else:
                l.append(i)
        #     print(l)
        return l


# 把list拼成string，不加任何修饰
def list2str(l):
    s = ''
    for i in l:
        s += i
    return s


def str2tensor(msg_type, msg):
    if msg_type == 'FLOW':
        h_msg = list2str(ip2higlist(msg))
    else:
        h_msg = list2str(path2higlist(msg))
    vec = FH_string.transform([msg_type + h_msg]).toarray() # 二维数组（1，encode_len）
    vec = torch.tensor(vec).reshape(encode_len).float() # 一维张量
    #     print(h_msg)
    return vec


class TimeEncoder(torch.nn.Module):
    """
        时间编码层，设置out_channel为50
    """

    def __init__(self, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.lin = Linear(1, out_channels)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, t):
        return self.lin(t.view(-1, 1)).cos()


time_enc = TimeEncoder(50)


"""
    计算hash初始嵌入
"""
def hashgen(l):
    """Generate a single hash value from a list. @l is a list of
    string values, which can be properties of a node/edge. This
    function returns a single hashed integer value."""
    hasher = xxhash.xxh64()
    for e in l:
        hasher.update(e)
    return hasher.intdigest()

def tensor_find(t,x):
    t_np=t.numpy()
    idx=np.argwhere(t_np==x)
    return idx[0][0]+1


def std(t):
    t = np.array(t)
    return np.std(t)


def var(t):
    t = np.array(t)
    return np.var(t)


def mean(t):
    t = np.array(t)
    return np.mean(t)

criterion = nn.CrossEntropyLoss()

def cal_pos_edges_loss(link_pred_ratio):
    loss=[]
    for i in link_pred_ratio:
        loss.append(criterion(i,torch.ones(1)))
    return torch.tensor(loss)

def cal_pos_edges_loss_multiclass(link_pred_ratio,labels):
    loss=[]
    for i in range(len(link_pred_ratio)):
        loss.append(criterion(link_pred_ratio[i].reshape(1,-1),labels[i].reshape(-1)))
    return torch.tensor(loss)


"""
    计算异常分数
"""

def cal_train_IDF(find_str, file_list):
    include_count = 0
    for f_path in (file_list):
        f = open(f_path)
        if find_str in f.read():
            include_count += 1
    IDF = math.log(len(file_list) / (include_count + 1))
    return IDF


def cal_IDF(find_str, file_path, file_list):
    file_list = os.listdir(file_path)
    include_count = 0
    different_neighbor = set()
    for f_path in (file_list):
        f = open(file_path + f_path)
        if find_str in f.read():
            include_count += 1

    IDF = math.log(len(file_list) / (include_count + 1))

    return IDF, 1


def cal_IDF_by_file_in_mem(find_str, file_list):
    include_count = 0
    different_neighbor = set()
    for f in (file_list):
        if find_str in f:
            include_count += 1
    IDF = math.log(len(file_list) / (include_count + 1))
    return IDF


def cal_redundant(find_str, edge_list):
    different_neighbor = set()
    for e in edge_list:
        if find_str in str(e):
            different_neighbor.add(e[0])
            different_neighbor.add(e[1])
    return len(different_neighbor) - 2


def cal_anomaly_loss(loss_list, edge_list, file_path):
    if len(loss_list) != len(edge_list):
        print("error!")
        return 0
    count = 0
    loss_sum = 0
    loss_std = std(loss_list)
    loss_mean = mean(loss_list)
    edge_set = set()
    node_set = set()
    node2redundant = {}

    thr = loss_mean + 2.5 * loss_std

    print("thr:", thr)

    for i in range(len(loss_list)):
        if loss_list[i] > thr:
            count += 1
            src_node = edge_list[i][0]
            dst_node = edge_list[i][1]

            loss_sum += loss_list[i]

            node_set.add(src_node)
            node_set.add(dst_node)
            edge_set.add(edge_list[i][0] + edge_list[i][1])
    return count, loss_sum / count, node_set, edge_set
#     return count, count/len(loss_list)