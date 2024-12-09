import torch
from scripts.model import *
from tqdm import tqdm
"""
    step5
"""


# TODO: 补充一下dataload

"""load trian data"""

# train_data=graph_9_22_h660

"""load test data"""



rel2id={1: 'OPEN',
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

BATCH=1024
def train(train_data):


    memory.train()
    gnn.train()
    link_pred.train()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.
    saved_nodes=set()

    total_loss = 0

    for batch in train_data.seq_batches(batch_size=BATCH):
        optimizer.zero_grad()

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        n_id = torch.cat([src, pos_dst]).unique()

        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = memory(n_id)

        z = gnn(z, last_update, edge_index, train_data.t[e_id], train_data.msg[e_id])

        pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])

        y_pred = torch.cat([pos_out], dim=0)

        y_true=[]
        for m in msg:
            l=tensor_find(m[16:-16],1)-1 # 查找one-hot编码中1的位置
            y_true.append(l)

        y_true = torch.tensor(y_true)
        y_true=y_true.reshape(-1).to(torch.long)

        loss = criterion(y_pred, y_true)

        # Update memory and neighbor loader with ground-truth state.
        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

        loss.backward()
        optimizer.step()
        memory.detach()
        total_loss += float(loss) * batch.num_events
    return total_loss / train_data.num_events

train_graphs=[
    # graph_9_22_h201,
    # graph_9_22_h402,
    # graph_9_22_h660,
    # graph_9_22_h501,
    # graph_9_22_h051,
    # graph_9_22_h209,
]
print(f"{embedding_dim=}")
print(f"{gnn=}")
for epoch in tqdm(range(1, 11)):
    for g in train_graphs:
        loss = train(g)
        print(f'  Epoch: {epoch:02d}, Loss: {loss:.4f}')


memory.reset_state()  # Start with a fresxh memory.
neighbor_loader.reset_state()
model=[memory,gnn, link_pred,neighbor_loader]
torch.save(model,f"./models/model_saved_traindata=hosts_9_22.pt") # 保存权重
