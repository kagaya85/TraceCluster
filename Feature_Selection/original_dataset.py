import torch
from torch_geometric.data import Dataset, InMemoryDataset
from torch_geometric.data.data import Data
import numpy as np
from tqdm import tqdm
import json
import os
import os.path as osp
# import utils
from itertools import repeat, product

from typing import List, Tuple, Union
from copy import deepcopy


class TraceDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(TraceDataset, self).__init__(
            root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def z_score_num_features(self):
        # return ['childrenSpanNum', 'requestAndResponseDuration', 'subspanDuration', 'rawDuration', 'subspanNum']
        return []

    @property
    def span_type_features(self):
        return ['timeScale', 'isParallel', 'callType']
        # return []

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        file_list = []
        file_list.append(osp.join(self.root, 'preprocessed/0.json'))
        return file_list

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:

        return ["processed.pt"]

    def download(self):
        pass

    def process(self):

        data_list = []
        num_features_stat = self._get_num_features_stat()
        operation_embedding = self._operation_embedding()

        print('load preprocessed data file:', self.raw_file_names[0])
        with open(self.raw_file_names[0], "r") as f:    # file name not list
            raw_data = json.load(f)

        for trace_id, trace in tqdm(raw_data.items()):
            node_feats = self._get_node_features(trace, operation_embedding)
            edge_feats = self._get_edge_features(trace, num_features_stat)
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
                y=trace['abnormal'],
                root_url=trace["edges"]["0"][0]["operation"]
            )
            data_list.append(data)

            # test
            # if data.trace_id == '1e3c47720fe24523938fff342ebe6c0d.35.16288656971030003':
            #    data.edge_attr = data.edge_attr * 1000

            # filename = osp.join(self.processed_dir, 'data_{}.pt'.format(idx))
            # torch.save(data, filename)
            # idx += 1
            # data_list.append(data)

        # if self.pre_filter is not None:
        #     data_list = [data for data in data_list if self.pre_filter(data)]

        # if self.pre_transform is not None:
        #     data_list = [self.pre_transform(data) for data in data_list]

        datas, slices = self.collate(data_list)
        torch.save((datas, slices), self.processed_paths[0])

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
            for feature in self.z_score_num_features:
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
        for from_id, to_list in trace["edges"].items():
            for to in to_list:
                feat = []

                for feature in self.z_score_num_features:
                    # feature_num = self._z_score(to[feature], num_features_stat[to['operation']][feature])
                    # feat.append(feature_num)
                    feat.append(to[feature])

                for feature in self.span_type_features:
                    feat.append(float(to[feature]))

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

    def _z_score(self, raw, feature_stat):
        """
        calculate z-score
        """
        z_socre = (raw - feature_stat[0]) / feature_stat[1]
        return z_socre


    def get_num_feature(self):
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[0]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key,
                                        item)] = slice(slices[0],
                                                       slices[0 + 1])
            else:
                s = slice(slices[0], slices[0 + 1])
            data[key] = item[s]
        _, num_feature = data.x.size()
        _, num_edge_feature = data.edge_attr.size()

        return num_feature, num_edge_feature


if __name__ == '__main__':
    print("start...")
    dataset = TraceDataset(root="./data")