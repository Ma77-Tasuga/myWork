import torch
from neo4j import GraphDatabase
import networkx as nx
import csv
import time
import os.path as osp
from Ours.scripts.utils import timestamp_to_datetime_US
"""
    负责向数据库中插入
"""
batch_num = 2000

class Neo4jImporter:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_node(self, nodeId, nodeName, nodeType, gt, nodeUuid):
        with self.driver.session() as session:
            session.write_transaction(self._create_node, nodeId, nodeName, nodeType, gt, nodeUuid)

    @staticmethod
    def _create_node(tx, nodeId, nodeName, nodeType, gt, nodeUuid):
        query = (
            """
            MERGE (n:Host0201_only {nodeId: $nodeId})
            ON CREATE SET n.name = $nodeName, n.type = $nodeType, n.gt = $gt, n.uuid = $nodeUuid
            """
        )
        tx.run(query, nodeId=nodeId, nodeName=nodeName, nodeType=nodeType, gt=gt, nodeUuid=nodeUuid)

    def create_edge(self, src_nodeId, dst_nodeId, edge_type, timestamp, phrase):
        with self.driver.session() as session:
            session.write_transaction(self._create_edge, src_nodeId, dst_nodeId, edge_type, timestamp, phrase)

    @staticmethod
    def _create_edge(tx, src_nodeId, dst_nodeId, edge_type, timestamp, phrase):
        query = (
            """
            MATCH (a:Host0201_only {nodeId: $src_nodeId})
            MATCH (b:Host0201_only {nodeId: $dst_nodeId})
            MERGE (a)-[r:ACTION_ONLY {type: $edge_type, timestamp: $timestamp, phrase: $phrase}]->(b)
            """
        )
        tx.run(query, src_nodeId=src_nodeId, dst_nodeId=dst_nodeId, edge_type=edge_type, timestamp=timestamp,
               phrase=phrase)

    def create_nodes_batch(self, nodes_data):
        with self.driver.session() as session:
            session.write_transaction(self._create_nodes_batch, nodes_data)

    @staticmethod
    def _create_nodes_batch(tx, nodes_data):
        query = """
        UNWIND $nodes AS node
        MERGE (n:Host0201_only {nodeId: node.nodeId})
        ON CREATE SET n.name = node.nodeName, n.type = node.nodeType, n.gt = node.gt, n.uuid = node.nodeUuid
        """
        tx.run(query, nodes=nodes_data)

    def create_edges_batch(self, edges_data):
        with self.driver.session() as session:
            session.write_transaction(self._create_edges_batch, edges_data)

    @staticmethod
    def _create_edges_batch(tx, edges_data):
        query = """
        UNWIND $edges AS edge
        MATCH (a:Host0201_only {nodeId: edge.src_nodeId})
        MATCH (b:Host0201_only {nodeId: edge.dst_nodeId})
        MERGE (a)-[r:ACTION_ONLY {type: edge.edge_type, timestamp: edge.timestamp, phrase: edge.phrase}]->(b)
        """
        tx.run(query, edges=edges_data)


"""从csv导入"""
# def import_csv_to_neo4j(file_path, neo4j_importer):
#     with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
#         reader = csv.DictReader(csvfile)
#         cnt = 0
#         start_time = time.time()
#         for row in reader:
#             # 从 CSV 文件中读取需要导入的数据
#             actorID = row['actorID']
#             actorname = row['actorname']
#             objectID = row['objectID']
#             objectname = row['objectname']
#             action = row['action']
#             timestamp = row['timestamp']
#             pid = row['pid'][0],
#             ppid = row['ppid'][0],
#             object_type = row['object'][0],
#             phrase = row['phrase']
#             # 调用 Neo4j 导入函数
#             neo4j_importer.create_node(actorID, actorname, objectID, objectname, action, timestamp, pid, ppid,
#                                        object_type, phrase)
#
#             cnt += 1
#             if cnt % 2000 == 0:
#                 end_time = time.time()
#                 print(f'Now {cnt} lines imported, using time {end_time - start_time} s.')
#                 start_time = end_time


"""从networkx导入"""
def import_nx_to_neo4j(nx_path, neo4j_importer, node_uuid2index):
    G = nx.read_graphml(nx_path)
    # 准备批量导入数据
    nodes_data = []

    cnt = 0
    # 先收集所有的节点数据，以便批量导入
    # for nodeId, data in G.nodes(data=True):
    #     nodeName = data.get('name', 'Unknown')
    #     nodeType = data.get('node_type', 'Unknown')
    #     gt = data.get('gt', 0)
    #
    #     str_nodeId = str(nodeId)
    #     if '.' in str_nodeId:
    #         str_nodeId = str_nodeId.split('.')[0]
    #     int_nodeId = int(str_nodeId)
    #     uuid = node_uuid2index[int_nodeId]
    #
    #     nodes_data.append({
    #         'nodeId': str_nodeId,
    #         'nodeName': nodeName,
    #         'nodeType': nodeType,
    #         'gt': gt,
    #         'nodeUuid': uuid
    #     })

        # cnt += 1
        # if cnt == 2000:
        #     neo4j_importer.create_nodes_batch(nodes_data)
        #     nodes_data = []
        #     cnt = 0

        # 调用create_node方法批量插入节点
        # neo4j_importer.create_node(str_nodeId, nodeName, nodeType, gt, uuid)
    # neo4j_importer.create_nodes_batch(nodes_data)

    start_time = time.time()
    # edges_data = []
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

        # edges_data.append({
        #     'src_nodeId': str_src_nodeId,
        #     'dst_nodeId': str_dst_nodeId,
        #     'edge_type': edge_type,
        #     'timestamp': timestamp,
        #     'phrase': phrase
        # })


        # 创建节点间的关系
        neo4j_importer.create_edge(str_src_nodeId, str_dst_nodeId, edge_type, timestamp, phrase)
        cnt += 1
        if cnt % batch_num == 0:
            # neo4j_importer.create_edges_batch(edges_data)
            end_time = time.time()
            print(f'Insert edges to neo4j, {cnt} out of {num_edges}, using {end_time-start_time} s.')
            edges_data = []
            start_time = time.time()

    # if len(edges_data) != 0:
    #     neo4j_importer.create_edges_batch(edges_data)
    #     end_time = time.time()
    #     print(f'Final insert {len(edges_data)}, using {end_time-start_time} s.')



if __name__ == "__main__":

    neo4j_uri = "neo4j://localhost:7687"
    neo4j_user = "neo4j"
    neo4j_password = "11111111"

    # database_name = 'SysClient0201'
    # 初始化 Neo4j 导入器
    importer = Neo4jImporter(neo4j_uri, neo4j_user, neo4j_password)

    # csv_file_path = '../SysClient0201.csv'
    nx_graph_path = '../data/OPTC/graph/evaluation/SysClient0201_only_graph.graphml'
    index_path = '../data/OPTC/map/evaluation/SysClient0201_only_uuid2index'
    node_uuid2index = torch.load(index_path)
    try:
        # 从 CSV 导入到 Neo4j
        # import_csv_to_neo4j(csv_file_path, importer)
        # print("CSV 导入成功！")

        # 从nx导入
        import_nx_to_neo4j(nx_graph_path, importer,node_uuid2index)
    finally:
        importer.close()
