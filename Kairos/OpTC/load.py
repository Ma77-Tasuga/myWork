"""
    step4
"""
import torch


# 这个csv没找到
label_df = pd.read_csv("./labels.csv")

nodes_attack = {}
edges_attack_list = []

for idx, row in label_df.iterrows():
    flag = False
    if row['objectID'] in node_uuid2path:
        nodes_attack[row['objectID']] = node_uuid2path[row['objectID']]
        flag = True
    if row['actorID'] in node_uuid2path:
        nodes_attack[row['actorID']] = node_uuid2path[row['actorID']]
        flag = True
    if flag and row['action'] in edge2vec:
        #         and row['action'] in edge2vec
        temp_dic = {}
        temp_dic['src_uuid'] = row['actorID']
        temp_dic['dst_uuid'] = row['objectID']
        temp_dic['edge_type'] = row['action']
        temp_dic['timestamp'] = datetime_to_timestamp_US(row['timestamp'])

        edges_attack_list.append(temp_dic)

# 看来他们的ground truth是包含边标注的
len(edges_attack_list)  # 33504
len(nodes_attack)  # 16047

graph_9_22_h201 = torch.load("./data/evaluation/9_22_host=SysClient0201_datalabel=benign.TemporalData")
graph_9_22_h402 = torch.load("./data/evaluation/9_22_host=SysClient0402_datalabel=benign.TemporalData")
graph_9_22_h660 = torch.load("./data/evaluation/9_22_host=SysClient0660_datalabel=benign.TemporalData")
graph_9_22_h501 = torch.load("./data/evaluation/9_22_host=SysClient0501_datalabel=benign.TemporalData")
graph_9_22_h051 = torch.load("./data/evaluation/9_22_host=SysClient0051_datalabel=benign.TemporalData")
graph_9_22_h209 = torch.load("./data/evaluation/9_22_host=SysClient0209_datalabel=benign.TemporalData")

graph_9_23_h201 = torch.load("./data/evaluation/9_23_host=SysClient0201_datalabel=evaluation.TemporalData")
graph_9_24_h201 = torch.load("./data/evaluation/9_24_host=SysClient0201_datalabel=evaluation.TemporalData")
graph_9_25_h201 = torch.load("./data/evaluation/9_25_host=SysClient0201_datalabel=evaluation.TemporalData")

graph_9_23_h402 = torch.load("./data/evaluation/9_23_host=SysClient0402_datalabel=evaluation.TemporalData")
graph_9_24_h402 = torch.load("./data/evaluation/9_24_host=SysClient0402_datalabel=evaluation.TemporalData")
graph_9_25_h402 = torch.load("./data/evaluation/9_25_host=SysClient0402_datalabel=evaluation.TemporalData")

graph_9_23_h660 = torch.load("./data/evaluation/9_23_host=SysClient0660_datalabel=evaluation.TemporalData")
graph_9_24_h660 = torch.load("./data/evaluation/9_24_host=SysClient0660_datalabel=evaluation.TemporalData")
graph_9_25_h660 = torch.load("./data/evaluation/9_25_host=SysClient0660_datalabel=evaluation.TemporalData")

graph_9_23_h501 = torch.load("./data/evaluation/9_23_host=SysClient0501_datalabel=evaluation.TemporalData")
graph_9_24_h501 = torch.load("./data/evaluation/9_24_host=SysClient0501_datalabel=evaluation.TemporalData")
graph_9_25_h501 = torch.load("./data/evaluation/9_25_host=SysClient0501_datalabel=evaluation.TemporalData")

graph_9_23_h051 = torch.load("./data/evaluation/9_23_host=SysClient0051_datalabel=evaluation.TemporalData")
graph_9_24_h051 = torch.load("./data/evaluation/9_24_host=SysClient0051_datalabel=evaluation.TemporalData")
graph_9_25_h051 = torch.load("./data/evaluation/9_25_host=SysClient0051_datalabel=evaluation.TemporalData")

graph_9_23_h207 = torch.load("./data/evaluation/9_23_host=SysClient0207_datalabel=evaluation.TemporalData")
graph_9_24_h207 = torch.load("./data/evaluation/9_24_host=SysClient0207_datalabel=evaluation.TemporalData")
graph_9_25_h207 = torch.load("./data/evaluation/9_25_host=SysClient0207_datalabel=evaluation.TemporalData")

graphs = [
    graph_9_22_h201,
    graph_9_22_h402,
    graph_9_22_h660,
    graph_9_22_h501,
    graph_9_22_h051,
    graph_9_22_h209,

    graph_9_23_h201,
    graph_9_24_h201,
    graph_9_25_h201,

    graph_9_23_h402,
    graph_9_24_h402,
    graph_9_25_h402,

    graph_9_23_h660,
    graph_9_24_h660,
    graph_9_25_h660,

    graph_9_23_h501,
    graph_9_24_h501,
    graph_9_25_h501,

    graph_9_23_h051,
    graph_9_24_h051,
    graph_9_25_h051,

    graph_9_23_h207,
    graph_9_24_h207,
    graph_9_25_h207,
]

edges_count=0 # 74989583
for g in graphs:
     edges_count+=len(g.t)

node_uuid2index_9_22_h201=torch.load("node_uuid2index_9_22_host=SysClient0201_datalabel=benign")
node_uuid2index_9_22_h402=torch.load("node_uuid2index_9_22_host=SysClient0402_datalabel=benign")
node_uuid2index_9_22_h660=torch.load("node_uuid2index_9_22_host=SysClient0660_datalabel=benign")
node_uuid2index_9_22_h501=torch.load("node_uuid2index_9_22_host=SysClient0501_datalabel=benign")
node_uuid2index_9_22_h051=torch.load("node_uuid2index_9_22_host=SysClient0051_datalabel=benign")
node_uuid2index_9_22_h209=torch.load("node_uuid2index_9_22_host=SysClient0209_datalabel=benign")


node_uuid2index_9_23_h201=torch.load("node_uuid2index_9_23_host=SysClient0201_datalabel=evaluation")
node_uuid2index_9_24_h201=torch.load("node_uuid2index_9_24_host=SysClient0201_datalabel=evaluation")
node_uuid2index_9_25_h201=torch.load("node_uuid2index_9_25_host=SysClient0201_datalabel=evaluation")

node_uuid2index_9_23_h402=torch.load("node_uuid2index_9_23_host=SysClient0402_datalabel=evaluation")
node_uuid2index_9_24_h402=torch.load("node_uuid2index_9_24_host=SysClient0402_datalabel=evaluation")
node_uuid2index_9_25_h402=torch.load("node_uuid2index_9_25_host=SysClient0402_datalabel=evaluation")

node_uuid2index_9_23_h660=torch.load("node_uuid2index_9_23_host=SysClient0660_datalabel=evaluation")
node_uuid2index_9_24_h660=torch.load("node_uuid2index_9_24_host=SysClient0660_datalabel=evaluation")
node_uuid2index_9_25_h660=torch.load("node_uuid2index_9_25_host=SysClient0660_datalabel=evaluation")

node_uuid2index_9_23_h501=torch.load("node_uuid2index_9_23_host=SysClient0501_datalabel=evaluation")
node_uuid2index_9_24_h501=torch.load("node_uuid2index_9_24_host=SysClient0501_datalabel=evaluation")
node_uuid2index_9_25_h501=torch.load("node_uuid2index_9_25_host=SysClient0501_datalabel=evaluation")

node_uuid2index_9_23_h051=torch.load("node_uuid2index_9_23_host=SysClient0051_datalabel=evaluation")
node_uuid2index_9_24_h051=torch.load("node_uuid2index_9_24_host=SysClient0051_datalabel=evaluation")
node_uuid2index_9_25_h051=torch.load("node_uuid2index_9_25_host=SysClient0051_datalabel=evaluation")

node_uuid2index_9_23_h207=torch.load("node_uuid2index_9_23_host=SysClient0207_datalabel=evaluation")
node_uuid2index_9_24_h207=torch.load("node_uuid2index_9_24_host=SysClient0207_datalabel=evaluation")
node_uuid2index_9_25_h207=torch.load("node_uuid2index_9_25_host=SysClient0207_datalabel=evaluation")


node_dics=[
    node_uuid2index_9_22_h201,
    node_uuid2index_9_22_h402,
    node_uuid2index_9_22_h660,
    node_uuid2index_9_22_h501,
    node_uuid2index_9_22_h051,
    node_uuid2index_9_22_h209,
    node_uuid2index_9_23_h201,
    node_uuid2index_9_24_h201,
    node_uuid2index_9_25_h201,
    node_uuid2index_9_23_h402,
    node_uuid2index_9_24_h402,
    node_uuid2index_9_25_h402,
    node_uuid2index_9_23_h660,
    node_uuid2index_9_24_h660,
    node_uuid2index_9_25_h660,
    node_uuid2index_9_23_h501,
    node_uuid2index_9_24_h501,
    node_uuid2index_9_25_h501,
    node_uuid2index_9_23_h051,
    node_uuid2index_9_24_h051,
    node_uuid2index_9_25_h051,
    node_uuid2index_9_23_h207,
    node_uuid2index_9_24_h207,
    node_uuid2index_9_25_h207,
]

nodes=set()
for dic in node_dics:
    for n in dic:
        if type(n)==str:
            nodes.add(n)

len(nodes) # 9485265