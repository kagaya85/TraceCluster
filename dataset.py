from typing import Callable, List, Optional, Tuple, Union
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import Data
import pandas as pd
import numpy as np
import tqdm as tqdm
import json


class TraceDataset(InMemoryDataset):
    def __init__(self, root: Optional[str], transform: Optional[Callable], pre_transform: Optional[Callable], pre_filter: Optional[Callable]):
        super().__init__(root=root, transform=transform,
                         pre_transform=pre_transform, pre_filter=pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return ['trace.json']

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        data_list = []

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        with open(self.raw_file_names, "r") as f:
            raw_data = json.load(f)

        for trace_id, trace in tqdm(raw_data.items()):
            node_feats = self._get_node_features(trace)
            # node_label = self._get_node_labels(trace)
            edge_feats = self._get_edge_features(trace)
            edge_index = self._get_adjacency_info(trace)

            data = Data(
                x=node_feats,
                # y=node_label,
                edge_index=edge_index,
                edge_attr=edge_feats,
            )
            data_list.append(data)

        datas, slices = self.collate(data_list)
        torch.save((datas, slices), self.processed_paths[0])

    def _get_node_features(self, trace):
        """ 
        node features matrix
        This will return a matrix / 2d array of the shape
        [Number of Nodes, Node Feature size]
        """
        node_feats = []
        for span_id, attr in trace["vertexs"].items():
            feat = []
            feat.append(attr)
            node_feats.append(feat)

        node_feats = np.asarray(node_feats)
        return torch.tensor(node_feats, dtype=torch.float)

    def _get_edge_features(self, trace):
        """ 
        edge features matrix
        This will return a matrix / 2d array of the shape
        [Number of edges, Edge Feature size]
        """
        edge_feats = []
        for from_id, to_list in trace["edges"]:
            feat = []
            for to in to_list:
                feat.append[to["startTime"]]
                feat.append(to["duration"])
            edge_feats.append(feat)

        edge_feats = np.asarray(edge_feats)
        return torch.tensor(edge_feats, dtype=torch.float)

    def _get_adjacency_info(self, trace):
        """
        adjacency list
        """
        adj_list = []
        for from_id, to_list in trace["edges"]:
            for to in to_list:
                to_id = to["spanId"]
                adj_list.append([from_id, to_id])

        return adj_list

    def _get_node_labels(self, trace):
        """
        node label
        """
        pass

    # TODO use get methods
