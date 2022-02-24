import json
import sys

import numpy as np
from tqdm import tqdm


def get_api_seq_and_time_seq():
    with open(r'/data/cyr/traceCluster_01/preprocessed/normal.json', 'r') as file:
        raw_data = json.load(file)
        trace_data = {}
        print('getting trace data (api and time seq)...')
        for trace_id, trace in tqdm(raw_data.items()):
            root_operation = trace['edges']['0'][0]['operation']
            service_seq = ['start']
            spans = []
            for span in trace['edges'].values():
                spans.extend(span)
            spans = sorted(spans, key=lambda span: (span['startTime']))
            service_seq.extend([span['service'] for span in spans])
            time_seq = [span['rawDuration'] for span in spans]
            trace_data[trace_id] = {'service_seq': service_seq, 'time_seq': time_seq, 'root_operation': root_operation}
    return trace_data


def get_api_dict():
    api_dict = {}
    idx = 0
    with open(r'/data/cyr/traceCluster_01/preprocessed/embeddings.json', 'r') as f:
        data = json.load(f)
        for api in data.keys():
            api_dict[api] = idx
            idx += 1
    return api_dict


def get_common_seq_set(trace_data):
    seq_set = set()
    trace_set = set()
    print('getting common seq set...')
    # idx = 0
    for trace_id, trace in tqdm(trace_data.items()):
        # trace_set.add('->'.join(trace['service_seq']))
        for i in range(1, len(trace['service_seq'])):
            seq_set.add('->'.join(trace['service_seq'][:i + 1]))
        # idx += 1
        # if idx > 100:
        #     break
    # for trace in trace_set:
    #     print(trace)
    # sys.exit(0)
    return list(seq_set)


def statistic_unique_trace_length(trace_data):
    root_length_type = {}
    base_root_statistic = {}
    new_trace_data = {}
    for trace_id, trace in tqdm(trace_data.items()):
        root = trace['root_operation']
        trace_length = len(trace['service_seq'])
        if root_length_type.get(root) is None:
            root_length_type[root] = {}
            base_root_statistic[root] = {}
        if root_length_type[root].get(trace_length) is None:
            root_length_type[root][trace_length] = {}
            base_root_statistic[root][trace_length] = {}
        unique_service_sequence = get_unique_service_sequence(trace['service_seq'])
        if base_root_statistic[root][trace_length].get(unique_service_sequence) is None:
            root_length_type[root][trace_length][unique_service_sequence] = 0
            base_root_statistic[root][trace_length][unique_service_sequence] = trace['service_seq']
        root_length_type[root][trace_length][unique_service_sequence] += 1
        different_index = []
        different_time = {}
        base_service_seq = base_root_statistic[root][trace_length][unique_service_sequence]
        # get the different index and time between base and current trace
        for i in range(1, trace_length):
            if base_service_seq[i] != trace['service_seq'][i]:
                different_index.append(i)
                if different_time.get(trace['service_seq'][i]) is None:
                    different_time[trace['service_seq'][i]] = []
                different_time[trace['service_seq'][i]].append(trace['time_seq'][i-1])
        new_trace = {}
        new_trace['service_seq'] = base_service_seq
        new_trace['time_seq'] = trace['time_seq']
        for j in different_index:
            service = base_service_seq[j]
            new_trace['time_seq'][j-1] = different_time[service][0]
            different_time[service].remove(different_time[service][0])
        new_trace_data[trace_id] = new_trace
    return base_root_statistic, new_trace_data


def data_process_for_trace_anomaly(trace_data, seq_set, filename):
    length = len(seq_set)
    print('seq_set\'s length:', length)
    print('processing data and writing it to file...')
    with open(r'./data/' + filename, 'w') as file:
        for trace_id, trace in tqdm(trace_data.items()):
            output_seq = ['0' for i in range(length)]
            for i in range(1, len(trace['service_seq'])):
                output_seq[seq_set.index('->'.join(trace['service_seq'][:i + 1]))] = str(trace['time_seq'][i - 1])
            file.write(trace_id + ':' + ','.join(output_seq) + '\n')


def preprocess_for_npy(filename, output_filename):
    data = np.load(filename)
    idx = 0
    with open(output_filename, 'w')as file:
        for d in data:
            embed = [str(i) for i in d]
            file.write(str(idx) + ':' + ','.join(embed) + '\n')


if __name__ == '__main__':
    trace_data = get_api_seq_and_time_seq()
    seq_set = get_common_seq_set(trace_data)
    data_process_for_trace_anomaly(trace_data, seq_set, 'train')
