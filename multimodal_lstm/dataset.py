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
from utils import get_unique_service_sequence

from torch_geometric.data.separate import separate
from typing import List, Tuple, Union
from copy import deepcopy
sys.path.append("..")
from aug_method import *


class TraceDataset(Dataset):

    def __init__(self, root, event_seq_root):
        self.root = root
        self.event_seq_root = event_seq_root
        self.api_seq = []
        self.original_api_seq = []
        self.time_seq = []
        self.y = []
        self.trace_id = []
        operation_embedding = self._operation_embedding()
        num_classes, api_dict = self.get_interface_num()

        roots = self.root.split(',')
        with open(roots[0] + roots[1] + '.json', "r") as f:  # file name not list
            raw_data = json.load(f)
        trace_data = {}
        min_time = 1000
        max_time = 0
        for trace_id, trace in tqdm(raw_data.items()):
            api_seq, time_seq, root_operation, min_time, max_time = self._get_multimodal_lstm_input(trace, min_time, max_time)
            trace_data[trace_id] = {'api_seq': api_seq, 'time_seq': time_seq,
                                    'root_operation': root_operation, 'y': trace['abnormal']}
            # _, trace_data = self.statistic_unique_trace_length(trace_data)
            # print(len(trace_data['api_seq']), trace_data['api_seq'])
            # print(len(trace_data['time_seq']), trace_data['time_seq'])
            # trace_data['api_seq'] = torch.tensor(np.asarray([api_dict[trace_data['api_seq']] for i in range(1, len(trace_data['api_seq']))]), dtype=torch.long)
            # trace_data['time_seq'] = torch.tensor(np.asarray(trace_data['time_seq']), dtype=torch.float)
            # one_hot_api_seq = F.one_hot(trace_data['api_seq'], num_classes).float()
            #
            # self.api_seq.append(one_hot_api_seq)
            # self.original_api_seq.append(trace_data['api_seq'])
            # self.time_seq.append(trace_data['time_seq'])
            # self.trace_id.append(trace_id)
            # self.y.append(trace['abnormal'])
        _, trace_data = self.statistic_unique_trace_length(trace_data)
        for trace_id, trace in tqdm(raw_data.items()):
            api_seq = torch.tensor(np.asarray([api_dict[trace_data[trace_id]['api_seq'][i]] for i in range(1, len(trace_data[trace_id]['api_seq']))]), dtype=torch.long)
            time_seq = torch.tensor(np.asarray([(trace_data[trace_id]['time_seq'][i]-min_time)/(max_time-min_time) for i in range(len(trace_data[trace_id]['time_seq']))]), dtype=torch.float)
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
        with open(self.root.split(',')[0] + 'embeddings.json', 'r') as f:
            operations_embedding = json.load(f)

        return operations_embedding

    def _get_multimodal_lstm_input(self, trace, min_time, max_time):
        api_seq = ['start']
        time_seq = []
        spans = []
        root_operation = trace['edges']['0'][0]['operation']
        for from_id, to_list in trace['edges'].items():
            for span in to_list:
                spans.append(span)
        spans = sorted(spans, key=lambda i: i['startTime'])
        for span in spans:
            api_seq.append(str(not span['isError']) + '_' + '_'.join(span['operation'].split('/')) + '_' + span['service'])
            # api_seq.append(span['service'] + '/' + span['operation'])
            time_seq.append(span['rawDuration'])
            min_time = min(span['rawDuration'], min_time)
            max_time = max(span['rawDuration'], max_time)

        # api_seq = np.asarray(api_seq)
        # time_seq = np.asarray(time_seq)
        # return torch.tensor(api_seq, dtype=torch.long), torch.tensor(time_seq, dtype=torch.float)
        return api_seq, time_seq, root_operation, min_time, max_time

    def __getitem__(self, index):
        return self.api_seq[index], self.original_api_seq[index], self.time_seq[index], self.trace_id[index], self.y[index]

    def __len__(self):
        return len(self.y)

    def get_interface_num(self):
        with open(self.event_seq_root + 'event_seq_set.json', 'r') as f:
            data = json.load(f)
        return len(data), data

    def statistic_unique_trace_length(self, trace_data):
        root_length_type = {}
        base_root_statistic = {}
        new_trace_data = {}
        for trace_id, trace in tqdm(trace_data.items()):
            root = trace['root_operation']
            trace_length = len(trace['api_seq'])
            if root_length_type.get(root) is None:
                root_length_type[root] = {}
                base_root_statistic[root] = {}
            if root_length_type[root].get(trace_length) is None:
                root_length_type[root][trace_length] = {}
                base_root_statistic[root][trace_length] = {}
            unique_service_sequence = get_unique_service_sequence(trace['api_seq'])
            if base_root_statistic[root][trace_length].get(unique_service_sequence) is None:
                root_length_type[root][trace_length][unique_service_sequence] = 0
                base_root_statistic[root][trace_length][unique_service_sequence] = trace['api_seq']
            root_length_type[root][trace_length][unique_service_sequence] += 1
            different_index = []
            different_time = {}
            base_service_seq = base_root_statistic[root][trace_length][unique_service_sequence]
            # get the different index and time between base and current trace
            for i in range(1, trace_length):
                if base_service_seq[i] != trace['api_seq'][i]:
                    different_index.append(i)
                    if different_time.get(trace['api_seq'][i]) is None:
                        different_time[trace['api_seq'][i]] = []
                    different_time[trace['api_seq'][i]].append(trace['time_seq'][i - 1])
            new_trace = {}
            new_trace['api_seq'] = base_service_seq
            new_trace['time_seq'] = trace['time_seq']
            for j in different_index:
                service = base_service_seq[j]
                new_trace['time_seq'][j - 1] = different_time[service][0]
                different_time[service].remove(different_time[service][0])
            new_trace_data[trace_id] = new_trace
        return base_root_statistic, new_trace_data


def write_event_seq_set(root_dir, train_filename, test_normal_filename, test_abnormal_filename):
    api_set = set()
    for filename in [train_filename, test_normal_filename, test_abnormal_filename]:
        with open(filename, "r") as f:
            raw_data = json.load(f)
        for trace_id, trace in tqdm(raw_data.items()):
            for from_id, to_list in trace['edges'].items():
                for span in to_list:
                    api_set.add(str(not span['isError']) + '_' + '_'.join(span['operation'].split('/')) + '_' + span['service'])
    api_dict = {}
    i = 0
    for api in api_set:
        api_dict[api] = i
        i += 1
    print(api_dict)
    with open(root_dir + 'event_seq_set.json', 'w') as f:
        json.dump(api_dict, f)


if __name__ == '__main__':
    print("start...")
    # dataset = TraceDataset(root=r"/data/cyr/data/,train_normal")
    write_event_seq_set(r"../0301-data/", r"../0301-data/train/preprocessed/0.json", r"../0301-data/test_normal/preprocessed/0.json", r"../0301-data/test_abnormal/preprocessed/0.json")

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
