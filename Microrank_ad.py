import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, precision_score
import random


def latency(raw, feature_stat):

    latency_span = feature_stat[0]+1.5*feature_stat[1]

    return latency_span


operations_stat_map = {}
with open('E:\Data\TraceCluster\\02-09-data-mem\preprocessed\operations.json', 'r') as f:
    operations_info = json.load(f)

for key in operations_info.keys():
    stat_map = {}
    ops = operations_info[key]['rawDuration']
    ops_mean = np.mean(ops)
    ops_std = np.std(ops)
    stat_map['rawDuration'] = [ops_mean, ops_std]
    operations_stat_map[key] = stat_map

rcs = ['ts-contacts-service', 'ts-execute-service', 'ts-config-service', 'ts-seat-service', 'ts-travel-service']


with open('E:\Data\TraceCluster\\02-09-data-mem\preprocessed\\0.json', "r") as f:    # file name not list
    raw_data = json.load(f)

ab_list = []
nor_list = []

for key in raw_data.keys():
    if raw_data[key]['abnormal'] == 0:
        nor_list.append(key)
    else:
        ab_list.append(key)

nor_list = random.sample(nor_list, int(0.2 * len(nor_list)))

true_label = []
predict_label = []

nor_list.extend(ab_list)

for trace_id, trace in tqdm(raw_data.items()):
    if trace['rc'] in rcs:
        continue
    if trace_id not in nor_list:
        continue
    edge_feats = []
    edge_feats_stat = []
    trace_latency = 0
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


            span_lantency = latency(to['rawDuration'], operations_stat_map[to['operation']]['rawDuration'])
            # feature_num = self._z_score(to[feature], num_features_stat[api_pair][feature])
            trace_latency += span_lantency
    if trace["edges"]['0'][0]['rawDuration'] > trace_latency:
        predict_label.append(1)
    else:
        predict_label.append(0)
    true_label.append(trace['abnormal'])

acc_test = accuracy_score(true_label, predict_label)
recall_test = recall_score(true_label, predict_label)
precision_test = precision_score(true_label, predict_label)

print(acc_test)
print(recall_test)
print(precision_test)

