import time
from datetime import datetime
from time import mktime
import pytz  # 时区计算
import numpy as np
import xxhash
import torch
from torch import nn
from sklearn.feature_extraction import FeatureHasher
from Ours.scripts.config import *

"""
    处理时间戳
    转换的最小单位是毫秒
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
    :param date: str   e.g. 2013-10-10T23:40:00.974-04:00
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
    :param date: nano_timestamps:str   format: 1533289579012(000000)
    :return: nano timestamp e.g. 2018-04-09 11:59:39.12
    """
    tz = pytz.timezone('US/Eastern')
    # ms = ns % 1000
    ms = str(ns)[-3:]
    ns /= 1000
    dt = pytz.datetime.datetime.fromtimestamp(int(ns), tz)
    s = dt.strftime('%Y-%m-%d %H:%M:%S')
    s += '.' + str(ms)
    #     s += '.' + str(int(int(ns) % 1000000000)).zfill(9)
    return s


"""
    计算初始嵌入
"""
encode_len = 16
FH_string = FeatureHasher(n_features=encode_len, input_type="string")
# FH输出的特征向量的维度为16


# 根据path结构分级提取对应的路径，为了下游的hash构建索引
def path2higlist(p) -> list:
    l = []
    spl = p.strip().split('/') # 这里是按照"/"分割的，注意元数据中的路径分割符号
    for i in spl:
        if len(l) != 0:
            l.append(l[-1] + '/' + i)
        else:
            l.append(i)
            # path
            # path/to
            # path/to/image
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
                # 192
                # 192.168
                # 192.168.0
                # 192.168.0.1
        #     print(l)
        return l
    else:
        spl = p.strip().split(':') # 这个应该不是处理端口
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
        h_msg = list2str(path2higlist(msg)) # 直接前后拼起来，不知道是为什么
    vec = FH_string.transform([msg_type + h_msg]).toarray()
    # 将输入数据转化为哈希表示形式，返回一个稀疏矩阵（通常是 scipy.sparse 类型）
    vec = torch.tensor(vec).reshape(encode_len).float()
    #     print(h_msg)
    return vec


edge2vec = torch.nn.functional.one_hot(torch.arange(0, edge_type_num), num_classes=edge_type_num)

"""
    计算用函数
"""

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

def hashgen(l):
    """Generate a single hash value from a list. @l is a list of
    string values, which can be properties of a node/edge. This
    function returns a single hashed integer value."""
    hasher = xxhash.xxh64()
    for e in l:
        hasher.update(e)
    return hasher.intdigest()


# def cal_pos_edges_loss(link_pred_ratio):
#     loss=[]
#     for i in link_pred_ratio:
#         loss.append(criterion(i,torch.ones(1)))
#     return torch.tensor(loss)
#
# def cal_pos_edges_loss_multiclass(link_pred_ratio,labels):
#     loss=[]
#     for i in range(len(link_pred_ratio)):
#         loss.append(criterion(link_pred_ratio[i].reshape(1,-1),labels[i].reshape(-1)))
#     return torch.tensor(loss)
"""
    metrics计算相关函数
"""
def cal_pos_ndoe_loss_multiclass(pred_ratio,labels):
    loss=[]
    criterion = nn.CrossEntropyLoss()
    for i in range(len(pred_ratio)):
        loss.append(criterion(pred_ratio[i].reshape(1,-1),labels[i].reshape(-1)))
    return torch.tensor(loss)

def cal_metrics(gt, node_dic, gate, node_uuid2index):
    gate = float(gate)
    anomaly_node = set()
    benign_node = set()
    for n in node_dic:
        if n['loss'] >= gate:
            anomaly_node.add(n['nodeId'])
        else:
            benign_node.add(n['nodeId'])

    benign_node.difference_update(anomaly_node) # 排除A中的重复元素

    FN = 0
    FP = 0
    TN = 0
    TP = 0
    for n in anomaly_node:
        n_uuid = node_uuid2index[str(n)]
        if n_uuid in gt:
            TP +=1
        else:
            FP +=1
    for n in benign_node:
        n_uuid = node_uuid2index[str(n)]
        if n_uuid in gt:
            FN +=1
        else:
            TN +=1

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0.0

    return {
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1_score,
        "Accuracy": accuracy,
        "TP":TP,
        "TN":TN,
        "FP":FP,
        "FN":FN
    }





if __name__ == '__main__':
    # print(datetime_to_ns_time())
    print(datetime_to_timestamp_US("2019-09-23T09:42:53.999-04:00"))
    print(timestamp_to_datetime_US(1569246173999))