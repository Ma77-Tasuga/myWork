import torch
from Ours.scripts.config import *
import networkx as nx


def cal_metrics(gt_node, benign_node, anomaly_node):
    # gate = float(gate)
    # anomaly_node = set()
    # benign_node = set()
    # for n in node_dic:
    #     if n['loss'] >= gate:
    #         anomaly_node.add(n['nodeId'])
    #     else:
    #         benign_node.add(n['nodeId'])
    #
    # benign_node.difference_update(anomaly_node) # 排除A中的重复元素

    FN = 0
    FP = 0
    TN = 0
    TP = 0
    # for n in anomaly_node:
    #     n_uuid = node_uuid2index[str(n)]
    #     if n_uuid in gt:
    #         TP +=1
    #     else:
    #         FP +=1
    # for n in benign_node:
    #     n_uuid = node_uuid2index[str(n)]
    #     if n_uuid in gt:
    #         FN +=1
    #     else:
    #         TN +=1
    for n in anomaly_node:
        if n in gt_node:
            TP +=1
        else:
            FP +=1
    for n in benign_node:
        if n in gt_node:
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
    loss_list_path = './anomaly_graph/node_dic_list_SysClient0201'
    graph_path = './data/OPTC/graph/evaluation/SysClient0201_graph.graphml'
    node_loss_list = torch.load(loss_list_path)


    anomaly_node_set = set()
    benign_node_set = set()
    all_node_set = set()
    for node in node_loss_list:
        if node['loss'] >= float(gate):
            anomaly_node_set.add(int(node['nodeId']))
            all_node_set.add(int(node['nodeId']))
        else:
            benign_node_set.add(int(node['nodeId']))
            all_node_set.add(int(node['nodeId']))

    benign_node_set = benign_node_set - anomaly_node_set

    node_count = len(all_node_set)
    anomaly_count = len(anomaly_node_set)
    benign_count = len(benign_node_set)
    print(f'{node_count=} {benign_count=} {anomaly_count=}')
    assert node_count == (anomaly_count+benign_count), "error in node set"

    print(f'Loading ground truth...')
    gt_node_anomaly_set = set()
    G = nx.read_graphml(graph_path)
    for nodeId, data in G.nodes(data=True):
        gt = data.get('gt', 0)
        if gt == 1:
            if '.' in nodeId:
                print(nodeId)
                nodeId = nodeId.split('.')[0]
            gt_node_anomaly_set.add(int(nodeId))
    print(f'num of gt node is {len(gt_node_anomaly_set)}')

    print(f'calculating metrics...')
    result = cal_metrics(gt_node_anomaly_set, benign_node_set, anomaly_node_set)

    for metric, value in result.items():
        print(f"{metric}: {value}")