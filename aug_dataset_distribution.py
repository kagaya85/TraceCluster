import math
import time
from tqdm import tqdm
from collections import Counter

import torch
from torch_geometric.data import Dataset, InMemoryDataset
from torch_geometric.data.data import Data
import numpy as np
from tqdm import tqdm
import json
import os
import os.path as osp
import utils
import copy
import pickle
from itertools import repeat, product

from torch_geometric.data.separate import separate
from typing import List, Tuple, Union
from copy import deepcopy
from aug_method import *


class TraceDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, aug=None):
        super(TraceDataset, self).__init__(
            root, transform, pre_transform)
        self.aug = aug


    @property
    def kpi_features(self):
        return ['requestAndResponseDuration', 'workDuration', 'rawDuration']  # workDuration   subspanDuration

    @property
    def span_features(self):
        return ['timeScale', 'isParallel', 'callType', 'isError']  #  'childrenSpanNum', 'subspanNum',

    @property
    def edge_features(self):
        return self.kpi_features + self.span_features

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        file_list = []
        file_list.append(osp.join(self.root, 'preprocessed/0.json'))
        return file_list

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        file_list = []
        for file in os.listdir(self.processed_dir):
            if os.path.splitext(file)[1] == '.pt':
                if file in ['pre_filter.pt', 'pre_transform.pt']:
                    continue
                file_list.append(file)

        return file_list

    @property
    def processed_dir(self):
        name = 'processed'
        return osp.join(self.root, name)

    @property
    def normal_idx(self):
        with open(self.processed_dir + '/data_info.json', "r") as f:    # file name not list
            data_info = json.load(f)
            normal_idx = data_info['normal']
        return normal_idx

    @property
    def abnormal_idx(self):
        with open(self.processed_dir + '/data_info.json', "r") as f:  # file name not list
            data_info = json.load(f)
            abnormal_idx = data_info['abnormal']
        return abnormal_idx

    @property
    def aug_idx(self):
        with open(self.processed_dir + '/data_info.json', "r") as f:  # file name not list
            data_info = json.load(f)
            abnormal_idx = data_info['aug']
        return abnormal_idx

    @property
    def trace_classes(self):
        with open(self.processed_dir + '/data_info.json', "r") as f:  # file name not list
            data_info = json.load(f)
            trace_classes = data_info['trace_classes']
        return trace_classes

    @property
    def url_status_classes(self):
        with open(self.processed_dir + '/data_info.json', "r") as f:  # file name not list
            data_info = json.load(f)
            url_status_classes = data_info['url_status_classes']
        return url_status_classes

    @property
    def url_classes(self):
        with open(self.processed_dir + '/data_info.json', "r") as f:  # file name not list
            data_info = json.load(f)
            url_classes = data_info['url_classes']
        return url_classes

    def download(self):
        pass

    def process(self):

        idx = 0
        normal_idx = []
        abnormal_idx = []
        aug_idx = []
        class_list = []  # url status node_num
        url_status_class_list = []  # url status
        url_class_list = []
        # data_list = []
        num_features_stat = self._get_num_features_stat()
        operation_embedding = self._operation_embedding()

        rcs = ['ts-contacts-service', 'ts-execute-service', 'ts-config-service', 'ts-seat-service', 'ts-travel-service']

        print('load preprocessed data file:', self.raw_file_names[0])
        with open(self.raw_file_names[0], "r") as f:    # file name not list
            raw_data = json.load(f)

        for trace_id, trace in tqdm(raw_data.items()):

            if trace['rc'] in rcs:
                continue

            node_feats = self._get_node_features(trace, operation_embedding)
            edge_feats, edge_feats_stat = self._get_edge_features(trace, num_features_stat)
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

            # define class based on root url, normal/abnormal, node num
            trace_class = trace["edges"]["0"][0]["operation"] + str(trace['abnormal']) + str(node_feats.size(0))
            if trace_class not in class_list:
                class_list.append(trace_class)
            # define url status class based on root url, normal/abnormal
            url_status_class = trace["edges"]["0"][0]["operation"] + str(trace['abnormal'])
            if url_status_class not in url_status_class_list:
                url_status_class_list.append(url_status_class)

            # define url status class based on root url, normal/abnormal
            root_url_class = trace["edges"]["0"][0]["operation"]
            if root_url_class not in url_class_list:
                url_class_list.append(root_url_class)

            data = Data(
                x=node_feats,
                edge_index=edge_index,
                edge_attr=edge_feats,
                trace_id=trace_id,    # add trace_id for cluster
                # add time_stamp for DenStream
                time_stamp=trace["edges"]["0"][0]["startTime"],
                # time_stamp=list(trace["edges"].items())[0][1][0]["startTime"],
                y=trace['abnormal'],
                root_url=trace["edges"]["0"][0]["operation"],
                trace_class=class_list.index(trace_class),
                url_status_class=url_status_class_list.index(url_status_class),
                url_class=url_class_list.index(root_url_class),
                edge_attr_stat=edge_feats_stat
            )
            # data_list.append(data)

            # test
            # if data.trace_id == '1e3c47720fe24523938fff342ebe6c0d.35.16288656971030003':
            #    data.edge_attr = data.edge_attr * 1000

            if trace['abnormal'] == 0:
                normal_idx.append(idx)
            elif trace['abnormal'] == 1:
                abnormal_idx.append(idx)

            filename = osp.join(self.processed_dir, 'data_{}.pt'.format(idx))
            torch.save(data, filename)
            idx += 1

            if trace['abnormal'] == 1:
                continue

            n = np.random.randint(6)
            if n == 0:
                data_aug = time_error_injection(deepcopy(data), root_cause='requestAndResponseDuration',
                                                edge_features=self.edge_features)
            elif n == 1:
                data_aug = time_error_injection(deepcopy(data), root_cause='workDuration',
                                                edge_features=self.edge_features)
            elif n == 2:
                data_aug = response_code_injection(deepcopy(data), self.edge_features)
            elif n == 3:
                data_aug = span_order_error_injection(deepcopy(data))
            elif n == 4:
                data_aug = drop_several_nodes(deepcopy(data))
            elif n == 5:
                data_aug = add_nodes(deepcopy(data))

            aug_idx.append(idx)
            filename = osp.join(self.processed_dir, 'data_{}.pt'.format(idx))
            torch.save(data_aug, filename)
            idx += 1

            # data_list.append(data)

        # if self.pre_filter is not None:
        #     data_list = [data for data in data_list if self.pre_filter(data)]

        # if self.pre_transform is not None:
        #     data_list = [self.pre_transform(data) for data in data_list]

        # datas, slices = self.collate(data_list)
        # torch.save((datas, slices), self.processed_paths[0])

        datainfo = {'normal': normal_idx,
                    'abnormal': abnormal_idx,
                    'aug': aug_idx,
                    'trace_classes': class_list,
                    'url_status_classes': url_status_class_list,
                    'url_classes': url_class_list}

        with open(self.processed_dir+'/data_info.json', 'w', encoding='utf-8') as json_file:
            json.dump(datainfo, json_file)
            print('write data info success')


    def _operation_embedding(self):
        """
        get operation embedding
        """
        with open(self.root + '/preprocessed/embeddings.json', 'r') as f:
            operations_embedding = json.load(f)

        return operations_embedding


    def _get_num_features_stat(self):
        """
        calculate features stat
        """
        operations_stat_map = {}
        with open(self.root + '/preprocessed/operations.json', 'r') as f:
            operations_info = json.load(f)

        for key in operations_info.keys():
            stat_map = {}
            for feature in self.kpi_features:
                ops = operations_info[key][feature]
                ops_mean = np.mean(ops)
                ops_std = np.std(ops)
                stat_map[feature] = [ops_mean, ops_std]
            operations_stat_map[key] = stat_map

        return operations_stat_map


    def _get_node_features(self, trace, operation_embedding):
        """
        node features matrix
        This will return a matrix / 2d array of the shape
        [Number of Nodes, Node Feature size]
        """
        node_feats = []
        for span_id, attr in trace["vertexs"].items():
            if span_id == '0':
                node_feats.append(operation_embedding[attr])
            else:
                node_feats.append(operation_embedding[attr[1]])

        node_feats = np.asarray(node_feats)
        return torch.tensor(node_feats, dtype=torch.float)

    def _get_edge_features(self, trace, num_features_stat):
        """
        edge features matrix
        This will return a matrix / 2d array of the shape
        [Number of edges, Edge Feature size]
        """
        edge_feats = []
        edge_feats_stat = []
        for from_id, to_list in trace["edges"].items():
            for to in to_list:
                feat = []
                feat_stat = []
                # if from_id == 0:
                #     api_pair = 'root--->' + trace["vertexs"][to["vertexId"]][1].strip(trace["vertexs"][to["vertexId"]][0]+'/')
                # else:
                #     api_pair = trace["vertexs"][from_id][1].strip(
                #         trace["vertexs"][from_id][0] + '/') + '--->' + trace["vertexs"][to["vertexId"]][1].strip(
                #         trace["vertexs"][to["vertexId"]][0] + '/')

                for feature in self.kpi_features:
                    feature_num = self._z_score(to[feature], num_features_stat[to['operation']][feature])
                    # feature_num = self._z_score(to[feature], num_features_stat[api_pair][feature])
                    feat.append(feature_num)
                    # feat_stat.append(num_features_stat[api_pair][feature][0])
                    # feat_stat.append(num_features_stat[api_pair][feature][1])
                    feat_stat.append(num_features_stat[to['operation']][feature][0])
                    feat_stat.append(num_features_stat[to['operation']][feature][1])
                    # feat.append(to[feature])

                for feature in self.span_features:
                    if feature == 'isError':
                        feat.append(0.0 if to[feature] is False else 1.0)
                    else:
                        feat.append(float(to[feature]))

                edge_feats.append(feat)
                edge_feats_stat.append(feat_stat)

        edge_feats_stat = np.asarray(edge_feats_stat)
        edge_feats = np.asarray(edge_feats)
        return torch.tensor(edge_feats, dtype=torch.float), torch.tensor(edge_feats_stat, dtype=torch.float)

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

    def _z_score(self, raw, feature_stat):
        """
        calculate z-score
        """
        if feature_stat[1] == 0:
            z_socre = (raw - feature_stat[0]) / 1
        else:
            z_socre = (raw - feature_stat[0]) / feature_stat[1]
        return z_socre

    _dispatcher = {}

    def get(self, idx: int):
        data = torch.load(
            osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))

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

        if self.aug == 'permute_edges_for_subgraph':  # 随机删一条边，选较大子图
            data_aug_1 = permute_edges_for_subgraph(deepcopy(data))
            data_aug_2 = permute_edges_for_subgraph(deepcopy(data))
        elif self.aug == 'mask_nodes':    # 结点属性屏蔽
            data_aug_1 = mask_nodes(deepcopy(data))
            data_aug_2 = mask_nodes(deepcopy(data))
        elif self.aug == 'mask_edges':    # 边属性屏蔽
            data_aug_1 = mask_edges(deepcopy(data))
            data_aug_2 = mask_edges(deepcopy(data))
        elif self.aug == 'mask_nodes_and_edges':    # 结点与边属性屏蔽
            data_aug_1 = mask_nodes(deepcopy(data))
            data_aug_1 = mask_edges(data_aug_1)
            data_aug_2 = mask_nodes(deepcopy(data))
            data_aug_2 = mask_edges(data_aug_2)
        elif self.aug == "request_and_response_duration_time_error_injection":  # request_and_response_duration时间异常对比
            data_aug_1 = time_error_injection(deepcopy(data), root_cause='requestAndResponseDuration', edge_features=self.edge_features)
            data_aug_2 = time_error_injection(deepcopy(data), root_cause='requestAndResponseDuration', edge_features=self.edge_features)
        elif self.aug == 'work_duration_time_error_injection':   # subSpan_duration时间异常
            data_aug_1 = time_error_injection(deepcopy(data), root_cause='workDuration', edge_features=self.edge_features)
            data_aug_2 = time_error_injection(deepcopy(data), root_cause='workDuration', edge_features=self.edge_features)
        elif self.aug == 'response_code_error_injection':
            data_aug_1 = response_code_injection(deepcopy(data), self.edge_features)
            data_aug_2 = response_code_injection(deepcopy(data), self.edge_features)
        elif self.aug == 'span_order_error_injection':      # span顺序错误，随机交换几个结点位置
            data_aug_1 = span_order_error_injection(deepcopy(data))
            data_aug_2 = span_order_error_injection(deepcopy(data))
        elif self.aug == 'drop_several_nodes':      # 缺少span，去掉几个节点
            data_aug_1 = drop_several_nodes(deepcopy(data))
            data_aug_2 = drop_several_nodes(deepcopy(data))
        elif self.aug == 'add_nodes':       # 增加span
            data_aug_1 = add_nodes(deepcopy(data))
            data_aug_2 = add_nodes(deepcopy(data))
        elif self.aug == 'none':
            """
            if data.edge_index.max() > data.x.size()[0]:
                print(data.edge_index)
                print(data.x.size())
                assert False
            """
            # data_aug_1 = pickle.loads(pickle.dumps(data))
            # data_aug_1 = data
            # data_aug_2 = pickle.loads(pickle.dumps(data))
            #data_aug.x = torch.ones((data.edge_index.max()+1, 1))
            # data_aug_2 = data
            return data
        elif self.aug == 'random':
            n = np.random.randint(6)
            if n < 3:
                # view aug
                data_aug_1 = self._get_view_aug(data)
                data_aug_2 = self._get_view_aug(data)
            elif n >= 3:
                # anomaly aug
                data_aug_1 = self._get_anomaly_aug(data)
                data_aug_2 = self._get_anomaly_aug(data)
                # data_aug_1, data_aug_2 = self._get_anomaly_aug(data)
        elif self.aug == 'anomaly_random':
            data_aug_1, data_aug_2 = self._get_anomaly_aug(data)
        elif self.aug == 'view_random':
            data_aug_1 = self._get_view_aug(data)
            data_aug_2 = self._get_view_aug(data)
        else:
            print('no need for augmentation ')
            assert False

        return data, data_aug_1, data_aug_2

    def len(self) -> int:

        return len(self.processed_file_names)

    def _get_view_aug(self, data):
        n = np.random.randint(4)
        if n == 0:
            data_aug = mask_nodes(deepcopy(data))
        elif n == 1:
            data_aug = mask_edges(deepcopy(data))
        elif n == 2:
            data_aug = mask_nodes(deepcopy(data))
            data_aug = mask_edges(data_aug)
        elif n == 3:
            data_aug = permute_edges_for_subgraph(deepcopy(data))
        # elif n == 4:
        #     data_aug = subgraph(deepcopy(data))
        else:
            print('sample error')
            assert False
        return data_aug

    # def _get_anomaly_aug(self, data):
    #     n = np.random.randint(6)
    #     if n == 0:
    #         data_aug_1 = time_error_injection(deepcopy(data), root_cause='requestAndResponseDuration', edge_features=self.edge_features)
    #         data_aug_2 = time_error_injection(deepcopy(data), root_cause='requestAndResponseDuration', edge_features=self.edge_features)
    #     elif n == 1:
    #         data_aug_1 = time_error_injection(deepcopy(data), root_cause='workDuration', edge_features=self.edge_features)
    #         data_aug_2 = time_error_injection(deepcopy(data), root_cause='workDuration', edge_features=self.edge_features)
    #     elif n == 2:
    #         data_aug_1 = response_code_injection(deepcopy(data), self.edge_features)
    #         data_aug_2 = response_code_injection(deepcopy(data), self.edge_features)
    #     elif n == 3:
    #         data_aug_1 = span_order_error_injection(deepcopy(data))
    #         data_aug_2 = span_order_error_injection(deepcopy(data))
    #     elif n == 4:
    #         data_aug_1 = drop_several_nodes(deepcopy(data))
    #         data_aug_2 = drop_several_nodes(deepcopy(data))
    #     elif n == 5:
    #         data_aug_1 = add_nodes(deepcopy(data))
    #         data_aug_2 = add_nodes(deepcopy(data))
    #     else:
    #         print('sample error')
    #         assert False
    #     return data_aug_1, data_aug_2

    def _get_anomaly_aug(self, data):
        n = np.random.randint(6)
        if n == 0:
            data_aug = time_error_injection(deepcopy(data), root_cause='requestAndResponseDuration', edge_features=self.edge_features)
        elif n == 1:
            data_aug = time_error_injection(deepcopy(data), root_cause='workDuration', edge_features=self.edge_features)
        elif n == 2:
            data_aug = response_code_injection(deepcopy(data), self.edge_features)
        elif n == 3:
            data_aug = span_order_error_injection(deepcopy(data))
        elif n == 4:
            data_aug = drop_several_nodes(deepcopy(data))
        elif n == 5:
            data_aug = add_nodes(deepcopy(data))
        else:
            print('sample error')
            assert False
        return data_aug



if __name__ == '__main__':
    print("start...")
    dataset = TraceDataset(root=r"E:\Data\TraceCluster\02-09-data-aug")
    # dataset.aug = "add_nodes"
    # data, data_aug_1, data_aug_2 = dataset.get(5000)
    # dataset1 = deepcopy(dataset)
    # dataset1.aug = "response_code_error_injection"
    # data_aug_1 = dataset1.get(0)
    # print(data, '\n', data.edge_attr, data.edge_index)
    # print(data_aug_1, '\n', data_aug_1.edge_attr)
    # print(data_aug_2, '\n', data_aug_2.edge_attr)
    # start_time = time.time()
    # node_count = []
    # for i in tqdm(range(len(dataset))):
    #     data, _, _ = dataset.get(i)
    #     node_count.append(data.num_nodes)
        # if i % 10 == 0:
        #     print(i)
    # print(Counter(node_count))
    # print(time.time()-start_time)

