
"""
    这个加载就是加载store_node_data里的所有数据，加载成原来的uuid映射path
"""

# Construct the map between nodeid and msg
# sql="select * from nodeid2msg;"
# cur.execute(sql)
# rows = cur.fetchall()

node_uuid2path={}  # nodeid => msg      node hash => nodeid
for i in tqdm(rows):
    node_uuid2path[i[0]]=i[1]