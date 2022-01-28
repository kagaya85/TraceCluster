import copy
import random

import torch
import numpy as np
import math
import queue

def drop_nodes(data):
    node_num, _ = data.x.size()  # x: num_node * num_node_features
    _, edge_num = data.edge_index.size()  # edge_index: 2 * edge_num
    drop_num = int(node_num / 10)

    # index  从 [0, node_num) 个点中选 drop_num 个点，并返回点的索引列表
    idx_drop = np.random.choice(node_num, drop_num, replace=False)
    # 不被删除的点的索引列表，保留点的索引列表
    idx_nondrop = [n for n in range(node_num) if not n in idx_drop]
    idx_dict = {idx_nondrop[n]: n for n in list(
        range(node_num - drop_num))}  # ???????  干嘛呢

    # data.x = data.x[idx_nondrop]
    edge_index = data.edge_index.numpy()

    edge_dict = {(edge_index[0][n], edge_index[1][n]): n for n in range(
        edge_num)}  # (from, to): edge_index colum idx

    # node_num = 4
    # edge_index = [[-1,0,1,2],
    # [0,1,2,0]]
    # edge_index = torch.tensor(edge_index, dtype=torch.long)
    # idx_drop = [0, 1]

    adj = torch.zeros((node_num, node_num))
    # 根据 edge_index，在有边的地方置为 1，没有边的地方保持 0，得到未 drop 的邻接矩阵
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero(as_tuple=False).t()

    data.edge_index = edge_index

    edge_index = data.edge_index.numpy()
    edge_attr = []
    for idx_edge in range(data.edge_index.size(1)):
        idx_column = edge_dict[(edge_index[0][idx_edge],
                                edge_index[1][idx_edge])]
        edge_attr.append(data.edge_attr[idx_column].numpy())

    edge_attr = np.asarray(edge_attr)
    data.edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
    # edge_index = [[edge_index[0, n], edge_index[1, n]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)] + [[n, n] for n in idx_nondrop]
    # data.edge_index = torch.tensor(edge_index).transpose_(0, 1)

    return data


def permute_edges(data):
    node_num, _ = data.x.size()  # [node_num, 300]
    _, edge_num = data.edge_index.size()  # [2, edge_num]
    permute_num = int(edge_num / 10)

    edge_index = data.edge_index.numpy()
    edge_dict = {(edge_index[0][n], edge_index[1][n]): n for n in range(
        edge_num)}  # (from, to): edge_index colum idx

    edge_index = data.edge_index.transpose(
        0, 1).numpy()  # [[from1, to1], [from2, to2]...]

    # 选择 permute_num 条边，[[from1, to1], [from2, to2]...]
    # permute_num 表示向量个数，2 表示每个向量的维度，(permute_num, 2) 表示维度。node_num 表示选择范围（从哪里选）
    idx_add = np.random.choice(node_num, (permute_num, 2))
    # idx_add = [[idx_add[0, n], idx_add[1, n]] for n in range(permute_num) if not (idx_add[0, n], idx_add[1, n]) in edge_index]

    # edge_index = np.concatenate((np.array([edge_index[n] for n in range(edge_num) if not n in np.random.choice(edge_num, permute_num, replace=False)]), idx_add), axis=0)
    # edge_index = np.concatenate((edge_index[np.random.choice(edge_num, edge_num-permute_num, replace=False)], idx_add), axis=0)

    # 删除边，删除 permute_num 条边，保留 edge_num-permute_num 条边
    edge_index = edge_index[np.random.choice(
        edge_num, edge_num - permute_num, replace=False)]
    # edge_index = [edge_index[n] for n in range(edge_num) if not n in np.random.choice(edge_num, permute_num, replace=False)] + idx_add
    data.edge_index = torch.tensor(edge_index).transpose_(0, 1)

    edge_index = data.edge_index.numpy()
    edge_attr = []
    for idx_edge in range(data.edge_index.size(1)):
        idx_column = edge_dict[(edge_index[0][idx_edge],
                                edge_index[1][idx_edge])]
        edge_attr.append(data.edge_attr[idx_column].numpy())

    edge_attr = np.asarray(edge_attr)
    data.edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return data


def subgraph(data):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    sub_num = int(node_num * 0.2)  # 子图大小，子图中点的个数

    edge_index = data.edge_index.numpy()
    edge_dict = {(edge_index[0][n], edge_index[1][n]): n for n in range(
        edge_num)}  # (from, to): edge_index colum idx

    idx_sub = [np.random.randint(node_num, size=1)[0]]  # 选中一个点作为起始点，放入子图集合
    # 选中的起点可到达的终点集合，即邻居
    idx_neigh = set([n for n in edge_index[1][edge_index[0] == idx_sub[0]]])

    count = 0
    while len(idx_sub) <= sub_num:
        count = count + 1
        if count > node_num:
            break
        if len(idx_neigh) == 0:  # 选中的点没有下游点
            break
        sample_node = np.random.choice(list(idx_neigh))  # 选择选中点的一个邻居，作为新选中点
        if sample_node in idx_sub:  # 新选中的点已经被选过
            continue
        idx_sub.append(sample_node)  # 将新选中点加入子图集合
        idx_neigh.union(  # 新选中点的邻居节点集合
            set([n for n in edge_index[1][edge_index[0] == idx_sub[-1]]]))

    idx_drop = [n for n in range(node_num) if not n in idx_sub]  # 丢弃非子图集合的节点
    idx_nondrop = idx_sub  # 保留节点为子图集合节点
    idx_dict = {idx_nondrop[n]: n for n in list(range(len(idx_nondrop)))}
    data.x = data.x[idx_nondrop]
    edge_index = data.edge_index.numpy()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0

    edge_index = adj.nonzero(as_tuple=False).t()

    data.edge_index = edge_index

    edge_index = data.edge_index.numpy()
    edge_attr = []
    for idx_edge in range(data.edge_index.size(1)):
        idx_column = edge_dict[(edge_index[0][idx_edge],
                                edge_index[1][idx_edge])]
        edge_attr.append(data.edge_attr[idx_column].numpy())

    edge_attr = np.asarray(edge_attr)
    data.edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
    # edge_index = [[edge_index[0, n], edge_index[1, n]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)] + [[n, n] for n in idx_nondrop]
    # data.edge_index = torch.tensor(edge_index).transpose_(0, 1)

    return data


def mask_nodes(data):
    node_num, feat_dim = data.x.size()
    mask_num = math.ceil(node_num / 2)
    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data.x[idx_mask] = torch.tensor(np.random.normal(
        loc=0.5, scale=0.5, size=(mask_num, feat_dim)), dtype=torch.float32)
    return data


def mask_edges(data):
    edge_num, feat_dim = data.edge_attr.size()
    mask_num = math.ceil(edge_num / 2)
    idx_mask = np.random.choice(edge_num, mask_num, replace=False)
    data.edge_attr[idx_mask] = torch.tensor(np.random.normal(
        loc=0.5, scale=0.5, size=(mask_num, feat_dim)), dtype=torch.float32)
    return data


def response_code_injection(data):
    edge_num, feat_dim = data.edge_attr.size()
    inject_num = math.ceil(edge_num / 2)
    idx_mask = np.random.choice(edge_num, inject_num, replace=False)
    for i in idx_mask:
        data.edge_attr[i][-1] = 1
    return data


class adjacency_edge:
    def __init__(self, to, edge_attr_id):
        self.to = to
        self.edge_attr_id = edge_attr_id


def time_error_injection(data, root_cause):
    trace = {}
    random_range = 10
    node_num = data.x.size(0)
    if node_num == 1:
        print('Can\'t inject time error because there is only one node!')
        return None
    for i in range(node_num):
        trace[i] = []
    for i in range(data.edge_index.size(1)):
        trace[int(data.edge_index[0][i])].append(adjacency_edge(int(data.edge_index[1][i]), i))
    inject_node = np.random.randint(1, node_num)
    flag = False
    end = False
    diff_sum = 0

    def dfs_for_time_error_injection(current_node, current_edge_attr):
        nonlocal flag, end, diff_sum
        if current_node == inject_node:
            flag = True
            if data.edge_attr[current_edge_attr][7] == 0:
                value = np.random.randint(1, random_range)
                if root_cause == 'request_and_response_duration':
                    data.edge_attr[current_edge_attr][1] += value
                elif root_cause == 'subSpan_duration':
                    data.edge_attr[current_edge_attr][2] += value
                if current_node != 1:
                    diff_sum += value
                data.edge_attr[current_edge_attr][3] += value
            else:
                end = True
            return
        for edge in trace[current_node]:
            dfs_for_time_error_injection(edge.to, edge.edge_attr_id)
            if end is True:
                return
            if flag is True and current_node != 0:
                if data.edge_attr[current_edge_attr][7] == 0:
                    value = np.random.randint(1, random_range)
                    if root_cause == 'request_and_response_duration':
                        data.edge_attr[current_edge_attr][1] += value
                    elif root_cause == 'subSpan_duration':
                        data.edge_attr[current_edge_attr][2] += value
                    if current_node != 1:
                        diff_sum += value
                    data.edge_attr[current_edge_attr][3] += value
                else:
                    end = True
                return
        return
    dfs_for_time_error_injection(0, None)
    data.edge_attr[0][3] += diff_sum
    for i in range(1, data.edge_attr.size(0)):
        data.edge_attr[i][5] = data.edge_attr[i][3]/data.edge_attr[0][3]
    return data


def permute_edges_for_subgraph(data):
    trace = {}
    node_num = data.x.size(0)
    if node_num == 1:
        print('Can\'t inject time error because there is only one node!')
        return None
    for i in range(node_num):
        trace[i] = []
    permuted_edge_id = np.random.choice(data.edge_index.size(1))
    edge_index = data.edge_index.numpy()
    edge_dict = {(edge_index[0][n], edge_index[1][n]): n for n in range(edge_index.shape[1])}
    subgraph_1 = [edge_index[0][permuted_edge_id]]
    subgraph_2 = [edge_index[1][permuted_edge_id]]
    edge_index = np.delete(edge_index, permuted_edge_id, axis=1)
    for i in range(edge_index.shape[1]):
        trace[int(edge_index[0][i])].append(adjacency_edge(int(edge_index[1][i]), i))

    def bfs(start_node, subgraph):
        que = queue.Queue()
        subgraph.append(start_node)
        que.put(start_node)
        while not que.empty():
            node = que.get()
            for i in range(len(trace[node])):
                que.put(trace[node][i].to)
                subgraph.append(trace[node][i].to)
        return list(set(subgraph))
    subgraph_1 = bfs(0, subgraph_1)
    subgraph_2 = bfs(subgraph_2[0],subgraph_2)
    if len(subgraph_1) >= len(subgraph_2):
        data.x = data.x[subgraph_1]
        idx_drop = [n for n in range(node_num) if not n in subgraph_1]  # 丢弃非子图集合的节点
    else:
        data.x = data.x[subgraph_2]
        idx_drop = [n for n in range(node_num) if not n in subgraph_2]  # 丢弃非子图集合的节点
    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0

    edge_index = adj.nonzero(as_tuple=False).t()

    data.edge_index = edge_index

    edge_index = data.edge_index.numpy()
    edge_attr = []
    for idx_edge in range(data.edge_index.size(1)):
        idx_column = edge_dict[(edge_index[0][idx_edge],
                                edge_index[1][idx_edge])]
        edge_attr.append(data.edge_attr[idx_column].numpy())

    edge_attr = np.asarray(edge_attr)
    data.edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    return data



