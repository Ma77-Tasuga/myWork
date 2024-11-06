"""
    step6
"""
import torch
import os
from scripts.model import *
import time
from scripts.utils import *

@torch.no_grad()
def test_day_new(inference_data, path, nodeuuid2index):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

    memory.eval()
    gnn.eval()
    link_pred.eval()

    memory.reset_state()
    neighbor_loader.reset_state()

    time_with_loss = {}
    total_loss = 0
    edge_list = []

    unique_nodes = torch.tensor([])
    total_edges = 0

    start_time = int(inference_data.t[0])
    event_count = 0

    pos_o = []

    loss_list = []

    print("after merge:", inference_data)

    start = time.perf_counter()

    for batch in inference_data.seq_batches(batch_size=BATCH):

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        unique_nodes = torch.cat([unique_nodes, src, pos_dst]).unique()
        total_edges += BATCH

        n_id = torch.cat([src, pos_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = memory(n_id)

        z = gnn(z, last_update, edge_index, inference_data.t[e_id], inference_data.msg[e_id])

        pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])

        pos_o.append(pos_out)
        y_pred = torch.cat([pos_out], dim=0)

        y_true = []
        for m in msg:
            l = tensor_find(m[16:-16], 1) - 1
            y_true.append(l)
        y_true = torch.tensor(y_true)
        y_true = y_true.reshape(-1).to(torch.long)

        loss = criterion(y_pred, y_true)

        total_loss += float(loss) * batch.num_events

        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

        each_edge_loss = cal_pos_edges_loss_multiclass(pos_out, y_true)

        for i in range(len(pos_out)):
            srcnode = int(src[i])
            dstnode = int(pos_dst[i])

            srcmsg = str(nodeuuid2index[srcnode])
            dstmsg = str(nodeuuid2index[dstnode])
            t_var = int(t[i])
            edgeindex = tensor_find(msg[i][16:-16], 1)
            edge_type = rel2id[edgeindex]
            loss = each_edge_loss[i]

            temp_dic = {}
            temp_dic['loss'] = float(loss)
            temp_dic['srcnode'] = srcnode
            temp_dic['dstnode'] = dstnode
            temp_dic['srcmsg'] = srcmsg
            temp_dic['dstmsg'] = dstmsg
            temp_dic['edge_type'] = edge_type
            temp_dic['time'] = t_var

            edge_list.append(temp_dic)

        event_count += len(batch.src)
        if t[-1] > start_time + 60000 * 15:

            time_interval = timestamp_to_datetime_US(start_time) + "~" + timestamp_to_datetime_US(int(t[-1]))

            end = time.perf_counter()
            time_with_loss[time_interval] = {'loss': loss,

                                             'nodes_count': len(unique_nodes),
                                             'total_edges': total_edges,
                                             'costed_time': (end - start)}

            log = open(path + "/" + time_interval + ".txt", 'w')

            for e in edge_list:
                loss += e['loss']

            loss = loss / event_count
            print(
                f'Time: {time_interval}, Loss: {loss:.4f}, Nodes_count: {len(unique_nodes)}, Cost Time: {(end - start):.2f}s')
            edge_list = sorted(edge_list, key=lambda x: x['loss'], reverse=True)
            for e in edge_list:
                log.write(str(e))
                log.write("\n")
            event_count = 0
            total_loss = 0
            loss = 0
            start_time = t[-1]
            log.close()
            edge_list.clear()

    return time_with_loss