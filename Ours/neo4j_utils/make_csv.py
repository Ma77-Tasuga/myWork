import torch
from neo4j import GraphDatabase
import networkx as nx
import csv
import time
import os.path as osp
from Ours.scripts.utils import timestamp_to_datetime_US


"""从networkx导入"""
def import_nx_to_neo4j(nx_path, write_folder, node_uuid2index):
    G = nx.read_graphml(nx_path)

    with open(osp.join(write_folder,'name_node.csv'), 'w', newline='') as wn:
        writer = csv.writer(wn)

        writer.writerow(['nodeId', 'nodeName', 'nodeType', 'gt', 'nodeUuid'])

        cnt = 0
        num_nodes = G.number_of_nodes()
        # 这个Id类型可以是整数也可以是字符串，取决于一开始赋予的类型
        for nodeId, data in G.nodes(data=True):
            nodeName = data.get('name', 'Unknown')
            nodeType = data.get('node_type', 'Unknown')
            gt = data.get('gt', 0)

            str_nodeId = str(nodeId)
            if '.' in str_nodeId:
                str_nodeId = str_nodeId.split('.')[0]
                print(f'Id: {nodeId}, type: {type(nodeId)}')
            int_nodeId = int(str_nodeId)
            uuid = node_uuid2index[int_nodeId]

            writer.writerow([
                str_nodeId,
                nodeName,
                nodeType,
                gt,
                uuid
            ])

            cnt += 1

        print(f'Finish write {cnt} out of {num_nodes} rows to file.')

    with open(osp.join(write_folder,'name_edge.csv'), 'w', newline='') as we:
        writer = csv.writer(we)

        # 写入CSV表头
        writer.writerow(['src_nodeId', 'dst_nodeId', 'edge_type', 'timestamp', 'phrase'])

        cnt = 0
        num_edges = G.number_of_edges()
        # 然后收集所有的边数据，批量导入边
        for src_nodeId, dst_nodeId, data in G.edges(data=True):
            edge_type = data.get('edge_type', 'Unknown')
            timestamp = data.get('timestamp', 'Unknown')
            timestamp = timestamp_to_datetime_US(int(timestamp))

            phrase = data.get('phrase', 'Unknown')

            str_src_nodeId = str(src_nodeId)
            str_dst_nodeId = str(dst_nodeId)
            if '.' in str_src_nodeId:
                str_src_nodeId = str_src_nodeId.split('.')[0]
            if '.' in str_dst_nodeId:
                str_dst_nodeId = str_dst_nodeId.split('.')[0]

            writer.writerow([
                str_src_nodeId,
                str_dst_nodeId,
                edge_type,
                timestamp,
                phrase
            ])

            cnt += 1

        print(f'Finish write {cnt} out of {num_edges} rows to file.')


if __name__ == "__main__":

    csv_write_folder = './csv_file'

    nx_graph_path = '../data/OPTC/graph/evaluation/SysClient0201_name_graph.graphml'
    index_path = '../data/OPTC/map/evaluation/SysClient0201_name_uuid2index'
    node_uuid2index = torch.load(index_path)

    import_nx_to_neo4j(nx_graph_path, csv_write_folder, node_uuid2index)