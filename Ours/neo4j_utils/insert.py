from neo4j import GraphDatabase
import networkx as nx
import csv
import time

"""
    负责向数据库中插入
"""


class Neo4jImporter:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_node(self, nodeId, nodeName, nodeType, gt):
        with self.driver.session() as session:
            session.write_transaction(self._create_node, nodeId, nodeName, nodeType, gt)

    @staticmethod
    def _create_node(tx, nodeId, nodeName, nodeType, gt):
        query = (
            """
            MERGE (n:Host0201 {uuid: $nodeId})
            ON CREATE SET n.name = $nodeName, n.type = $nodeType, n.gt = $gt
            """
        )
        tx.run(query, nodeId=nodeId, nodeName=nodeName, nodeType=nodeType, gt=gt)

    def create_edge(self, src_nodeId, dst_nodeId, edge_type, timestamp, phrase):
        with self.driver.session() as session:
            session.write_transaction(self._create_edge, src_nodeId, dst_nodeId, edge_type, timestamp, phrase)

    @staticmethod
    def _create_edge(tx, src_nodeId, dst_nodeId, edge_type, timestamp, phrase):
        query = (
            """
            MATCH (a:Host0201 {uuid: $src_nodeId})
            MATCH (b:Host0201 {uuid: $dst_nodeId})
            MERGE (a)-[r:ACTION0201 {type: $edge_type, timestamp: $timestamp, phrase: $phrase}]->(b)
            """
        )
        tx.run(query, src_nodeId=src_nodeId, dst_nodeId=dst_nodeId, edge_type=edge_type, timestamp=timestamp,
               phrase=phrase)


"""从csv导入"""
def import_csv_to_neo4j(file_path, neo4j_importer):
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        cnt = 0
        start_time = time.time()
        for row in reader:
            # 从 CSV 文件中读取需要导入的数据
            actorID = row['actorID']
            actorname = row['actorname']
            objectID = row['objectID']
            objectname = row['objectname']
            action = row['action']
            timestamp = row['timestamp']
            pid = row['pid'][0],
            ppid = row['ppid'][0],
            object_type = row['object'][0],
            phrase = row['phrase']
            # 调用 Neo4j 导入函数
            neo4j_importer.create_node(actorID, actorname, objectID, objectname, action, timestamp, pid, ppid,
                                       object_type, phrase)

            cnt += 1
            if cnt % 2000 == 0:
                end_time = time.time()
                print(f'Now {cnt} lines imported, using time {end_time - start_time} s.')
                start_time = end_time


"""从networkx导入"""
def import_nx_to_neo4j(nx_path, neo4j_importer):
    G = nx.read_graphml(nx_path)

    # 先收集所有的节点数据，以便批量导入
    for nodeId, data in G.nodes(data=True):
        nodeName = data.get('name', 'Unknown')
        nodeType = data.get('type', 'Unknown')
        gt = data.get('gt', 0)

        # 调用create_node方法批量插入节点
        neo4j_importer.create_node(nodeId, nodeName, nodeType, gt)

    # 然后收集所有的边数据，批量导入边
    for src_nodeId, dst_nodeId, data in G.edges(data=True):
        edge_type = data.get('edge_type', 'Unknown')
        timestamp = data.get('timestamp', 'Unknown')
        phrase = data.get('phrase', 'Unknown')

        # 创建节点间的关系
        neo4j_importer.create_edge(src_nodeId, dst_nodeId, edge_type, timestamp, phrase)


if __name__ == "__main__":

    neo4j_uri = "neo4j://localhost:7687"
    neo4j_user = "neo4j"
    neo4j_password = "11111111"

    # database_name = 'SysClient0201'
    # 初始化 Neo4j 导入器
    importer = Neo4jImporter(neo4j_uri, neo4j_user, neo4j_password)

    # csv_file_path = '../SysClient0201.csv'
    nx_graph_path = '../data/OPTC/graph/graph.graphml'
    try:
        # 从 CSV 导入到 Neo4j
        # import_csv_to_neo4j(csv_file_path, importer)
        # print("CSV 导入成功！")

        # 从nx导入
        import_nx_to_neo4j(nx_graph_path, importer)
    finally:
        importer.close()
