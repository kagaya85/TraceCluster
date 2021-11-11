
import torch
from torch_geometric.data import Dataset
from torch_geometric.data.data import Data
import numpy as np
from tqdm import tqdm
import json
import os
import os.path as osp

from typing import List, Tuple, Union
from itertools import repeat
from copy import deepcopy


class TraceClusterDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, aug=None):
        super(TraceClusterDataset, self).__init__(
            root, transform, pre_transform, pre_filter)

        # self.data, self.slices = torch.load(self.processed_paths[0])
        self.aug = aug

    @property
    def raw_dir(self):
        name = 'preprocessed'
        return osp.join(self.root, name)

    @property
    def processed_dir(self):
        name = 'processed'
        return osp.join(self.root, name)

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        # file_list = ['2021-10-13_16-57-51.json']
        # file_list = ['2021-11-09_21-10-48.json']
        # file_list = ['2021-11-08_19-21-29.json']
        file_list = ['2021-11-10_21-13-58.json']
        # file_list = ['1.json']

        path_list = []
        for file in file_list:
            path_list.append(osp.join(self.raw_dir, file))

        return path_list

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        file_list = []
        for file in os.listdir(self.processed_dir):
            if os.path.splitext(file)[1] == '.pt':
                if file in ['pre_filter.pt', 'pre_transform.pt']:
                    continue
                file_list.append(file)

        return file_list

    def download(self):
        pass

    def process(self):
        idx = 0
        data_list = []

        with open(self.raw_file_names[0], "r") as f:    # file name not list
            raw_data = json.load(f)

        for trace_id, trace in tqdm(raw_data.items()):
            node_feats = self._get_node_features(trace)
            edge_feats = self._get_edge_features(trace)
            edge_index = self._get_adjacency_info(trace)

            # dim check
            num_nodes_node_feats, _ = node_feats.size()
            num_nodes_edge_index = edge_index.max()+1    # 包括 0
            if num_nodes_node_feats != num_nodes_edge_index:
                print("Feature dismatch! num_nodes_node_feats: {}, num_nodes_edge_index: {}, trace_id: {}".format(
                    num_nodes_node_feats, num_nodes_edge_index, trace_id))

            num_edges_edge_feats, _ = edge_feats.size()
            _, num_edges_edge_index = edge_index.size()
            if num_edges_edge_feats != num_edges_edge_index:
                print("Feature dismatch! num_edges_edge_feats: {}, num_edges_edge_index: {}, trace_id: {}".format(
                    num_edges_edge_feats, num_edges_edge_index, trace_id))

            data = Data(
                x=node_feats,
                edge_index=edge_index,
                edge_attr=edge_feats,
                trace_id=trace_id,    # add trace_id for cluster
                # add time_stamp for DenStream
                time_stamp=trace["edges"]["0"][0]["startTime"],
                # time_stamp=list(trace["edges"].items())[0][1][0]["startTime"],
            )

            # test
            #if data.trace_id == '1e3c47720fe24523938fff342ebe6c0d.35.16288656971030003':
            #    data.edge_attr = data.edge_attr * 1000

            filename = osp.join(self.processed_dir, 'data_{}.pt'.format(idx))
            torch.save(data, filename)
            idx += 1
            # data_list.append(data)

        # if self.pre_filter is not None:
        #     data_list = [data for data in data_list if self.pre_filter(data)]

        # if self.pre_transform is not None:
        #     data_list = [self.pre_transform(data) for data in data_list]

        # datas, slices = self.collate(data_list)
        # torch.save((datas, slices), self.processed_paths[0])

    def _get_node_features(self, trace):
        """ 
        node features matrix
        This will return a matrix / 2d array of the shape
        [Number of Nodes, Node Feature size]
        """
        node_feats = []
        for span_id, attr in trace["vertexs"].items():
            node_feats.append(attr)

        node_feats = np.asarray(node_feats)
        return torch.tensor(node_feats, dtype=torch.float)

    def _get_edge_features(self, trace):
        """ 
        edge features matrix
        This will return a matrix / 2d array of the shape
        [Number of edges, Edge Feature size]
        """
        edge_feats = []
        for from_id, to_list in trace["edges"].items():
            for to in to_list:
                feat = []
                feat.append(to["duration"])
                edge_feats.append(feat)

        edge_feats = np.asarray(edge_feats)
        return torch.tensor(edge_feats, dtype=torch.float)

    def _get_adjacency_info(self, trace):
        """
        adjacency list
        [from1, from2, from3 ...] [to1, to2, to3 ...]
        """
        adj_list = [[], []]
        for from_id, to_list in trace["edges"].items():
            for to in to_list:
                to_id = to["vertexId"]
                adj_list[0].append(int(from_id))
                adj_list[1].append(int(to_id))

        return torch.tensor(adj_list, dtype=torch.long)

    def _get_node_labels(self, trace):
        """
        node label
        """
        pass

    def get_num_feature(self):
        data = torch.load(
            osp.join(self.processed_dir, 'data_0.pt'))

        # if hasattr(self.data, '__num_nodes__'):
        #     data.num_nodes = self.data.__num_nodes__[0]

        # 根据 slice 对整个数据集的 data 进行划分，得到每个 graph 的 data
        # for key in self.data.keys:
        #     item, slices = self.data[key], self.slices[key]
        #     if torch.is_tensor(item):
        #         s = list(repeat(slice(None), item.dim()))
        #         s[self.data.__cat_dim__(key, item)] = slice(
        #             slices[0], slices[0 + 1])
        #     else:
        #         s = slice(slices[0], slices[0 + 1])

        #     data[key] = item[s]

        _, num_feature = data.x.size()

        return num_feature

    def get(self, idx):
        data = torch.load(
            osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))

        # if hasattr(self.data, '__num_nodes__'):
        #     data.num_nodes = self.data.__num_nodes__[idx]

        # for key in self.data.keys:
        #     item, slices = self.data[key], self.slices[key]
        #     if torch.is_tensor(item):
        #         s = list(repeat(slice(None), item.dim()))
        #         s[self.data.__cat_dim__(key, item)] = slice(slices[idx],
        #                                                     slices[idx + 1])
        #     else:
        #         s = slice(slices[idx], slices[idx + 1])
        #     data[key] = item[s]

        if data.x.size(0) != data.edge_index.max()+1:
            print("dim dismatch !!!!!, idx: {}".format(idx))

        """
        edge_index = data.edge_index
        node_num = data.x.size()[0]
        edge_num = data.edge_index.size()[1]
        data.edge_index = torch.tensor([[edge_index[0, n], edge_index[1, n]] for n in range(edge_num) if edge_index[0, n] < node_num and edge_index[1, n] < node_num] + [[n, n] for n in range(node_num)], dtype=torch.int64).t()
        """

        # 为每个 node 添加一条自己指向自己的边
        #node_num = data.edge_index.max()
        #sl = torch.tensor([[n, n] for n in range(node_num)]).t()
        #data.edge_index = torch.cat((data.edge_index, sl), dim=1)

        if self.aug == 'dnodes':    # 删除部分点
            data_aug = drop_nodes(deepcopy(data))
        elif self.aug == 'pedges':    # 删除 or 增加部分边
            data_aug = permute_edges(deepcopy(data))
        elif self.aug == 'subgraph':    #
            data_aug = subgraph(deepcopy(data))
        elif self.aug == 'mask_nodes':    # 属性屏蔽
            data_aug = mask_nodes(deepcopy(data))
        elif self.aug == 'none':
            """
            if data.edge_index.max() > data.x.size()[0]:
                print(data.edge_index)
                print(data.x.size())
                assert False
            """
            data_aug = deepcopy(data)
            #data_aug.x = torch.ones((data.edge_index.max()+1, 1))

        elif self.aug == 'random2':
            n = np.random.randint(2)
            if n == 0:
                data_aug = drop_nodes(deepcopy(data))
            elif n == 1:
                data_aug = subgraph(deepcopy(data))
            else:
                print('sample error')
                assert False

        elif self.aug == 'random3':
            n = np.random.randint(3)
            if n == 0:
                data_aug = drop_nodes(deepcopy(data))
            elif n == 1:
                data_aug = permute_edges(deepcopy(data))
            elif n == 2:
                data_aug = mask_nodes(deepcopy(data))
            else:
                print('sample error')
                assert False

        elif self.aug == 'random4':
            n = np.random.randint(4)
            if n == 0:
                data_aug = drop_nodes(deepcopy(data))
            elif n == 1:
                data_aug = permute_edges(deepcopy(data))
            elif n == 2:
                data_aug = subgraph(deepcopy(data))
            elif n == 3:
                data_aug = mask_nodes(deepcopy(data))
            else:
                print('sample error')
                assert False

        else:
            print('augmentation error')
            assert False

        # print(data, data_aug)
        # assert False

        return data, data_aug

    def len(self) -> int:
        return len(self.processed_file_names)


def drop_nodes(data):

    node_num, _ = data.x.size()    # x: num_node * num_node_features
    _, edge_num = data.edge_index.size()    # edge_index: 2 * edge_num
    drop_num = int(node_num / 10)

    # index  从 [0, node_num) 个点中选 drop_num 个点，并返回点的索引列表
    idx_drop = np.random.choice(node_num, drop_num, replace=False)
    # 不被删除的点的索引列表，保留点的索引列表
    idx_nondrop = [n for n in range(node_num) if not n in idx_drop]
    idx_dict = {idx_nondrop[n]: n for n in list(
        range(node_num - drop_num))}    # ???????  干嘛呢

    # data.x = data.x[idx_nondrop]
    edge_index = data.edge_index.numpy()

    edge_dict = {(edge_index[0][n], edge_index[1][n]): n for n in range(
        edge_num)}    # (from, to): edge_index colum idx

    #node_num = 4
    # edge_index = [[-1,0,1,2],
    # [0,1,2,0]]
    #edge_index = torch.tensor(edge_index, dtype=torch.long)
    #idx_drop = [0, 1]

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

    node_num, _ = data.x.size()    # [node_num, 300]
    _, edge_num = data.edge_index.size()    # [2, edge_num]
    permute_num = int(edge_num / 10)

    edge_index = data.edge_index.numpy()
    edge_dict = {(edge_index[0][n], edge_index[1][n]): n for n in range(
        edge_num)}    # (from, to): edge_index colum idx

    edge_index = data.edge_index.transpose(
        0, 1).numpy()    # [[from1, to1], [from2, to2]...]

    # 选择 permute_num 条边，[[from1, to1], [from2, to2]...]
    # permute_num 表示向量个数，2 表示每个向量的维度，(permute_num, 2) 表示维度。node_num 表示选择范围（从哪里选）
    idx_add = np.random.choice(node_num, (permute_num, 2))
    # idx_add = [[idx_add[0, n], idx_add[1, n]] for n in range(permute_num) if not (idx_add[0, n], idx_add[1, n]) in edge_index]

    # edge_index = np.concatenate((np.array([edge_index[n] for n in range(edge_num) if not n in np.random.choice(edge_num, permute_num, replace=False)]), idx_add), axis=0)
    # edge_index = np.concatenate((edge_index[np.random.choice(edge_num, edge_num-permute_num, replace=False)], idx_add), axis=0)

    # 删除边，删除 permute_num 条边，保留 edge_num-permute_num 条边
    edge_index = edge_index[np.random.choice(
        edge_num, edge_num-permute_num, replace=False)]
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
    sub_num = int(node_num * 0.2)    # 子图大小，子图中点的个数

    edge_index = data.edge_index.numpy()
    edge_dict = {(edge_index[0][n], edge_index[1][n]): n for n in range(
        edge_num)}    # (from, to): edge_index colum idx

    idx_sub = [np.random.randint(node_num, size=1)[0]]    # 选中一个点作为起始点，放入子图集合
    # 选中的起点可到达的终点集合，即邻居
    idx_neigh = set([n for n in edge_index[1][edge_index[0] == idx_sub[0]]])

    count = 0
    while len(idx_sub) <= sub_num:
        count = count + 1
        if count > node_num:
            break
        if len(idx_neigh) == 0:    # 选中的点没有下游点
            break
        sample_node = np.random.choice(list(idx_neigh))    # 选择选中点的一个邻居，作为新选中点
        if sample_node in idx_sub:    # 新选中的点已经被选过
            continue
        idx_sub.append(sample_node)    # 将新选中点加入子图集合
        idx_neigh.union(    # 新选中点的邻居节点集合
            set([n for n in edge_index[1][edge_index[0] == idx_sub[-1]]]))

    idx_drop = [n for n in range(node_num) if not n in idx_sub]    # 丢弃非子图集合的节点
    idx_nondrop = idx_sub    # 保留节点为子图集合节点
    idx_dict = {idx_nondrop[n]: n for n in list(range(len(idx_nondrop)))}

    # data.x = data.x[idx_nondrop]
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
    mask_num = int(node_num / 10)

    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data.x[idx_mask] = torch.tensor(np.random.normal(
        loc=0.5, scale=0.5, size=(mask_num, feat_dim)), dtype=torch.float32)

    return data
