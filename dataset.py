from typing import Callable, List, Optional, Tuple, Union
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import Data
import pandas as pd
import tqdm as tqdm


class TraceDataset(InMemoryDataset):
    def __init__(self, root: Optional[str], transform: Optional[Callable], pre_transform: Optional[Callable], pre_filter: Optional[Callable]):
        super().__init__(root=root, transform=transform,
                         pre_transform=pre_transform, pre_filter=pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        # TODO
        return ['trace.csv']

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

        raw_data = pd.read_csv(self.raw_paths[0])
        for idx, trace in tqdm(raw_data.iterrows()):
            node_feats = self._get_node_features(trace)
            node_label = self._get_node_labels(trace)
            edge_feats = self._get_edge_features(trace)
            edge_index = self._get_adjacency_info(trace)

            data = Data(
                x=node_feats,
                y=node_label,
                edge_index=edge_index,
                edge_attr=edge_feats,
            )
            data_list.append(data)

        datas, slices = self.collate(data_list)
        torch.save((datas, slices), self.processed_paths[0])

    def _get_node_features(trace):
        """ 
        node features matrix
        This will return a matrix / 2d array of the shape
        [Number of Nodes, Node Feature size]
        """
        pass

    def _get_edge_features(trace):
        """ 
        edge features matrix
        This will return a matrix / 2d array of the shape
        [Number of edges, Edge Feature size]
        """
        pass

    def _get_adjacency_info(trace):
        """
        adjacency list
        """
        pass

    def _get_node_labels(trace):
        """
        node label
        """
        pass
