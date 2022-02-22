import math
import sys
import time
from tqdm import tqdm

import torch
# from torch_geometric.data import Dataset, InMemoryDataset
from torch.utils.data import Dataset
from torch_geometric.data.data import Data
import numpy as np
from tqdm import tqdm
import json
import os
import os.path as osp
import torch.nn.functional as F

from torch_geometric.data.separate import separate
from typing import List, Tuple, Union
from copy import deepcopy
sys.path.append("..")
from aug_method import *


class TraceDataset(Dataset):

    def __init__(self, root):
        self.root = root
        self.api_seq = []
        self.original_api_seq = []
        self.time_seq = []
        self.y = []
        self.trace_id = []
        rcs = ['ts-contacts-service', 'ts-execute-service', 'ts-config-service', 'ts-seat-service', 'ts-travel-service']
        operation_embedding = self._operation_embedding()
        num_classes, _ = self.get_interface_num()

        api_dict = {}
        i = 0
        for api in operation_embedding.keys():
            api_dict[api] = i
            i += 1
        roots = self.root.split(',')
        with open(roots[0] + '/preprocessed/' + roots[1] + '.json', "r") as f:  # file name not list
            raw_data = json.load(f)

        for trace_id, trace in tqdm(raw_data.items()):
            if trace['rc'] in rcs:
                continue
            api_seq, time_seq = self._get_multimodal_lstm_input(trace, api_dict)
            one_hot_api_seq = F.one_hot(api_seq, num_classes).float()

            self.api_seq.append(one_hot_api_seq)
            self.original_api_seq.append(api_seq)
            self.time_seq.append(time_seq)
            self.trace_id.append(trace_id)
            self.y.append(trace['abnormal'])


    def _operation_embedding(self):
        """
        get operation embedding
        """
        with open(self.root.split(',')[0] + '/preprocessed/embeddings.json', 'r') as f:
            operations_embedding = json.load(f)

        return operations_embedding

    def _get_multimodal_lstm_input(self, trace, api_dict):
        api_seq = []
        time_seq = []
        spans = []
        for from_id, to_list in trace['edges'].items():
            for span in to_list:
                spans.append(span)
        spans = sorted(spans, key=lambda i: i['startTime'])
        for span in spans:
            api_seq.append(api_dict[span['service'] + '/' + span['operation']])
            time_seq.append(span['rawDuration'])
        api_seq = np.asarray(api_seq)
        time_seq = np.asarray(time_seq)
        return torch.tensor(api_seq, dtype=torch.long), torch.tensor(time_seq, dtype=torch.float)

    def __getitem__(self, index):
        return self.api_seq[index], self.original_api_seq[index], self.time_seq[index], self.trace_id[index], self.y[index]

    def __len__(self):
        return len(self.y)

    def get_interface_num(self):
        with open(self.root.split(',')[0] + '/preprocessed/embeddings.json', 'r') as f:
            data = json.load(f)
        return len(data.keys()), list(data.keys())


if __name__ == '__main__':
    print("start...")
    dataset = TraceDataset(root=r"/data/cyr/traceCluster_01,normal")
    # dataset.aug = None
    # data = dataset.get(0)
    # dataset1 = deepcopy(dataset)
    # dataset1.aug = 'permute_edges_for_subgraph'
    # data_aug_1 = dataset1.get(0)
    # print(data, '\n', data.edge_attr)
    # print(data_aug_1, '\n', data_aug_1.edge_attr)
    # start_time = time.time()
    # for i in range(len(dataset1)):
    #     dataset1.get(i)
    #     # if i % 10 == 0:
    #     #     print(i)
    # print(time.time()-start_time)
