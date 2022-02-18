import json
import sys

import numpy as np
from tqdm import tqdm


def get_api_seq_and_time_seq():
    with open(r'/home/yanghong/data/trainticket/preprocessNew/normal.json', 'r') as file:
        raw_data = json.load(file)
        trace_data = {}
        print('getting trace data (api and time seq)...')
        for trace_id, trace in tqdm(raw_data.items()):
            service_seq = ['start']
            spans = []
            for span in trace['edges'].values():
                spans.extend(span)
            spans = sorted(spans, key=lambda span: span['startTime'])
            service_seq.extend([span['service'] for span in spans])
            time_seq = [span['rawDuration'] for span in spans]
            trace_data[trace_id] = {'service_seq': service_seq, 'time_seq': time_seq}
    return trace_data


def get_api_dict():
    api_dict = {}
    idx = 0
    with open(r'/home/yanghong/data/trainticket/preprocessNew/embeddings.json', 'r') as f:
        data = json.load(f)
        for api in data.keys():
            api_dict[api] = idx
            idx += 1
    return api_dict


def get_common_seq_set(trace_data):
    seq_set = set()
    print('getting common seq set...')
    for trace_id, trace in tqdm(trace_data.items()):
        # trace_set.add('->'.join(trace['service_seq']))
        for i in range(1, len(trace['service_seq'])):
            seq_set.add('->'.join(trace['service_seq'][:i + 1]))
    return list(seq_set)


def data_process_for_trace_anomaly(trace_data, seq_set, filename):
    length = len(seq_set)
    print('seq_set\'s length:', length)
    print('processing data and writing it to file...')
    with open(r'./data/' + filename, 'w') as file:
        for trace_id, trace in tqdm(trace_data.items()):
            output_seq = ['0' for i in range(length)]
            for i in range(1, len(trace['service_seq'])):
                output_seq[seq_set.index('->'.join(trace['service_seq'][:i + 1]))] = str(trace['time_seq'][i-1])
            file.write(trace_id + ':' + ','.join(output_seq) + '\n')


if __name__ == '__main__':
    trace_data = get_api_seq_and_time_seq()
    seq_set = get_common_seq_set(trace_data)
    data_process_for_trace_anomaly(trace_data, seq_set, 'train')
