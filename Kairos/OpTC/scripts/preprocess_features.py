import torch

"""
    处理边特征
"""
#  边类型
edge_set = ['OPEN',
            'READ',
            'CREATE',
            'MESSAGE',
            'MODIFY',
            'START',
            'RENAME',
            'DELETE',
            'TERMINATE',
            'WRITE', ]

# 生成边类型的one-hot编码
edgevec = torch.nn.functional.one_hot(torch.arange(0, len(edge_set)), num_classes=len(edge_set))

# 保存边特征的数据结构
edge2vec = {}
for e in range(len(edge_set)):
    edge2vec[edge_set[e]] = edgevec[e]


# 边类型索引id
rel2id = {}
index = 1
for i in edge_set:
    rel2id[index] = i
    rel2id[i] = index
    index += 1

"""
 {1: 'OPEN',
  'OPEN': 1,
  2: 'READ',
  'READ': 2,
  3: 'CREATE',
  'CREATE': 3,
  4: 'MESSAGE',
  'MESSAGE': 4,
  5: 'MODIFY',
  'MODIFY': 5,
  6: 'START',
  'START': 6,
  7: 'RENAME',
  'RENAME': 7,
  8: 'DELETE',
  'DELETE': 8,
  9: 'TERMINATE',
  'TERMINATE': 9,
  10: 'WRITE',
  'WRITE': 10}
"""