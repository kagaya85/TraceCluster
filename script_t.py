import json
from collections import Counter
import numpy as np
import torch
import random
import copy
from copy import deepcopy
from tqdm import tqdm

# print(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# with open('E:\Data\TraceCluster\\0228-data\\normal.json', 'r') as f:
# #     dic = json.load(f)
# model_path = 'E:\Data\TraceCluster\log\\20epoch_CGConv_add_0301data_0.6normal_randomview\\'
# with open(model_path + 'train_info.json', "r") as f:  # file name not list
#     model_info = json.load(f)
#     num_layers = model_info['num_layers']
#     output_dim = model_info['output_dim']
#     # num_node_features = model_info['num_node_features']
#     # num_edge_features = model_info['num_edge_features']
#     gnn_type = model_info['gnn_type']
#     pooling_type = model_info['pooling_type']
#     aug = model_info['aug']
#     train_idx = model_info['train_idx']
#     test_idx = model_info['test_idx']

with open('G:\Data\TraceCluster\\0301-data\Sampling_22000_everytype\\test_traces.json', 'r') as f:
    raw_data = json.load(f)

# status_codes = []
# for trace_id, trace in tqdm(raw_data.items()):
#     if len(trace['vertexs'].keys()) == 5:
#         print(trace)
#         print(trace_id)
#         exit()
#     for from_id, to_list in trace["edges"].items():
#         for to in to_list:
#             status_codes.append(to['statusCode'])
# print(Counter(status_codes))
# exit()

# print(max(workDuration))
# print(min(workDuration))


# kpi_features = ['rawDuration', 'clientRequestAndResponseDuration']  # workDuration   subspanDuration  'requestAndResponseDuration', 'workDuration',

# span_features = ['timeScale', 'isParallel', 'callType', 'isError', 'start2startTimeScale', 'end2startTimeScale', 'workDuration', 'subspanNum']  #  'childrenSpanNum', 'subspanNum',

# rawDuration = processingTime
# clientRequestAndResponseDuration = networklatency
# callType = isAsynchronous
# timeScale = proportionProcessingTime
# start2startTimeScale = quantileStartTime
# end2startTimeScale = quantileEndTime
# workDuration = proportionLocalExecution
# subspanNum = sectionsLocalExecution
new_dic = {}

all_traceids = list(raw_data.keys())

random.shuffle(all_traceids)

for trace_id in all_traceids:
    new_trace = copy.deepcopy(raw_data[trace_id])
    for from_id, to_list in raw_data[trace_id]["edges"].items():
        for i in range(len(to_list)):
            to = to_list[i]

            caller_id = str(to['vertexId'])
            a = raw_data[trace_id]["edges"]
            if caller_id in a.keys():
                subspanNum = len(raw_data[trace_id]["edges"][caller_id]) + 1
            else:
                subspanNum = 1

            new_trace['edges'][from_id][i]['sectionsLocalExecution'] = float(subspanNum)

            if float(to['rawDuration']) == 0.0:
                new_trace['edges'][from_id][i]['proportionLocalExecution'] = 1.0
            else:
                new_trace['edges'][from_id][i]['proportionLocalExecution'] = float(to['workDuration']) / float(to['rawDuration'])

            new_trace['edges'][from_id][i]['quantileEndTime'] = to['end2startTimeScale']
            new_trace['edges'][from_id][i]['quantileStartTime'] = to['start2startTimeScale']
            new_trace['edges'][from_id][i]['proportionProcessingTime'] = to['timeScale']
            new_trace['edges'][from_id][i]['isAsynchronous'] = to['callType']
            new_trace['edges'][from_id][i]['rawNetworklatency'] = to['clientRequestAndResponseDuration']
            new_trace['edges'][from_id][i]['rawProcessingTime'] = to['rawDuration']
            if to['isError'] == False:
                code = 2
            elif to['isError'] == True:
                code = 5
            new_trace['edges'][from_id][i]['statusCode'] = code

            del new_trace['edges'][from_id][i]['requestDuration']
            del new_trace['edges'][from_id][i]['responseDuration']
            del new_trace['edges'][from_id][i]['clientRequestDuration']
            del new_trace['edges'][from_id][i]['clientResponseDuration']
    new_dic[trace_id] = new_trace

print(len(new_dic.keys()))

with open('G:\Data\TraceCluster\\0301-data\github_data\\tail-based_sampling.json', 'w', encoding='utf-8') as json_file:
    json.dump(new_dic, json_file)
    print('write data info success')

# for trace_id, trace in tqdm(raw_data.items()):
#     edge_feats = []
#     edge_feats_stat = []
#     for from_id, to_list in trace["edges"].items():
#         for to in to_list:
#             feat = []
#             feat_stat = []
#             if from_id == '0':
#                 api_pair = 'root--->' + trace["vertexs"][str(to["vertexId"])][1].replace(
#                     trace["vertexs"][str(to["vertexId"])][0] + '/', '')
#             else:
#                 api_pair = trace["vertexs"][from_id][1].replace(
#                     trace["vertexs"][from_id][0] + '/', '') + '--->' + trace["vertexs"][str(to["vertexId"])][1].replace(
#                     trace["vertexs"][str(to["vertexId"])][0] + '/', '')
#
#             for feature in self.kpi_features:
#                 # feature_num = self._z_score(to[feature], num_features_stat[to['operation']][feature])
#                 feature_num = self._z_score(to[feature], num_features_stat[api_pair][feature])
#                 feat.append(feature_num)
#                 feat_stat.append(num_features_stat[api_pair][feature][0])
#                 feat_stat.append(num_features_stat[api_pair][feature][1])
#                 # feat_stat.append(num_features_stat[to['operation']][feature][0])
#                 # feat_stat.append(num_features_stat[to['operation']][feature][1])
#                 # feat.append(to[feature])
#
#             for feature in self.span_features:
#                 if feature == 'isError':
#                     feat.append(0.0 if to[feature] is False else 1.0)
#                 elif feature == 'subspanNum':
#                     caller_id = str(to['vertexId'])
#                     a = trace["edges"]
#                     if caller_id in a.keys():
#                         subspanNum = len(trace["edges"][caller_id]) + 1
#                     else:
#                         subspanNum = 1
#                     feat.append(float(subspanNum))
#                 elif feature == 'workDuration':
#                     if float(to['rawDuration']) == 0.0:
#                         feat.append(1.0)
#                     else:
#                         feat.append(float(to['workDuration']) / float(to['rawDuration']))
#                 else:
#                     feat.append(float(to[feature]))
#
#             edge_feats.append(feat)
#             edge_feats_stat.append(feat_stat)
#
#     edge_feats_stat = np.asarray(edge_feats_stat)
#     edge_feats = np.asarray(edge_feats)





# with open(r'G:\Data\TraceCluster\0301-data\PERCH_random2000data\test_traces.json', 'r') as file_1:
#     raw_data = json.load(file_1)

# with open('E:\Data\TraceCluster\\0301-data\\normal.json', 'r') as f:
#     dic1 = json.load(f)
#
# with open('E:\Data\TraceCluster\\0301-data\\test_normal_0.4.json', 'r') as f:
#     dic2 = json.load(f)

# print(dic1.keys())
# dic.update(dic1)


# print(len(dic1.keys()))
# print(len(dic2.keys()))

#
# trace = {}
# trace_type_dic = {}
# ab_type_list = []
# nor_type_list = []
#
# # rc = ['ts-seat-service']
# rc = ['station_code_chaos', 'user_code_chaos']
#
# nor_trace_dic = {}
# ab_trace_dic = {}
#
#
#
# for key in all_traceids:
# if dic[key]['abnormal'] == 0:
#     t = dic[key]["edges"]["0"][0]["operation"] + '0' + str(len(dic[key]['vertexs'].keys()))
#     nor_type_list.append(t)
#     trace_type_dic[key] = t
# else:
#     t = dic[key]['rc'] + dic[key]["edges"]["0"][0]["operation"] + '1' + str(len(dic[key]['vertexs'].keys()))
#     ab_type_list.append(t)
#     trace_type_dic[key] = t
#
# # t = raw_data[key]['rc'] + raw_data[key]["edges"]["0"][0]["operation"] + str(dic[key]['abnormal'])
# # trace_type_dic[key] = t
#
# print(len(nor_type_list))
# print(len(ab_type_list))
#
# print(Counter(nor_type_list))
# print(Counter(ab_type_list))
# print(list(set(nor_type_list)))
# print(list(set(ab_type_list)))
# print(len(list(set(nor_type_list))))
# print(len(list(set(ab_type_list))))

# exit()
    # if dic[key]["edges"]["0"][0]["operation"] == 'POST:/api/v1/orderOtherService/orderOther/refresh':
    #
    # if dic[key]['rc'] in rc:
    # trace[key] = dic[key]["edges"]["0"][0]["operation"]
# dic.update(dic1)
# dic.update(dic2)

# print(len(dic.keys()))
#






# print((Counter(ab_list)))

# for key in all_traceids:
#     root_url = dic1[key]["edges"]["0"][0]["operation"]
#     if dic1[key]['abnormal'] == 0:
#         if root_url in root_url_dic.keys():
#             root_url_dic[root_url] += 1
#         else:
#             root_url_dic[root_url] = 1

# count_dic = {}
# train_normal_dic = {}
# test_normal_dic = {}
# train_normal_list = []
# test_normal_list = []
#
# for key in dic.keys():
#     if key in all_traceids:
#         continue
#     else:
#         test_normal_dic[key]=dic[key]
# for key in all_traceids:
#     root_url = dic1[key]["edges"]["0"][0]["operation"]
#     if root_url not in count_dic.keys():
#         count_dic[root_url] = 0
#     if count_dic[root_url] < 0.6*root_url_dic[root_url]:
#         train_normal_dic[key] = dic1[key]
#         count_dic[root_url] += 1
#         train_normal_list.append(root_url)
#     else:
#         test_normal_dic[key] = dic1[key]
#         test_normal_list.append(root_url)

# for key in dic1.keys():
#     trace_class = dic1[key]["edges"]["0"][0]["operation"] + str(dic1[key]['abnormal']) + str(
#         len(dic1[key]['vertexs'].keys()))
#     if dic1[key]['abnormal'] == 1:
#         ab_list.append(trace_class)
#     elif dic1[key]['abnormal'] == 0:
#         nor_list.append(trace_class)

# print(root_url_dic)
# print(len(train_normal_dic.keys()))
# print(len(test_normal_dic.keys()))
# print(Counter(train_normal_list))
# print(Counter(test_normal_list))


# with open('E:\Data\TraceCluster\\0301-data\\train_normal_0.6.json', 'w', encoding='utf-8') as json_file:
#     json.dump(train_normal_dic, json_file)
#     print('write data info success')
#
# with open('E:\Data\TraceCluster\\0301-data\\test_normal_0.4.json', 'w', encoding='utf-8') as json_file:
#     json.dump(test_normal_dic, json_file)
#     print('write data info success')

# train_normal = {}
# test_normal = {}
# idx = 0
# for trace_id, trace in tqdm(dic1.items()):
#     if idx in train_idx:
#         train_normal[trace_id] = trace
#     elif idx in test_idx:
#         test_normal[trace_id] = trace
#     idx += 1
# print(len(train_normal.keys()))
# print(len(test_normal.keys()))
# with open('E:\Data\TraceCluster\\0301-data\\train_normal.json', 'w', encoding='utf-8') as json_file:
#     json.dump(train_normal, json_file)
#     print('write data info success')

# with open('E:\Data\TraceCluster\\0301-data\\abnormal.json', 'r') as f:
#     dic = json.load(f)


# print(len(dic.keys()))
# print(len(dic1.keys()))

# dic.update(dic1)

# traceids = dic['traceids']

# test_trace = {}
# for traceid in traceids:
#     test_trace[traceid] = dic1[traceid]

# print(len(dic.keys()))


# train_data_ids = dic1.keys()

# print(len(train_data_ids))
# print(len(dic.keys()))

# print(dic['2e675902ffcf4d1f88a13d01f075afd6.41.16440841480077707'])

#
#



# for key in dic.keys():
#     if dic[key]['abnormal'] == 1:
#         ab_list.append(dic[key]["edges"]["0"][0]["operation"])
#     elif dic[key]['abnormal'] == 0:
#         nor_list.append(dic[key]["edges"]["0"][0]["operation"])

# print(Counter(nor_list))
# print(Counter(ab_list))

# print(len(dic.keys()))

# dic.update(dic1)
#




# with open('E:\Data\TraceCluster\\0222-data\PERCH\\test_data.json', 'w', encoding='utf-8') as json_file:
#     json.dump(test_data, json_file)
#     print('write data info success')

    # if dic[key]['abnormal'] == 1:
    #     ab_list.append(dic[key]["edges"]["0"][0]["operation"])
    # elif dic[key]['abnormal'] == 0:
    #     list.append(dic[key]["edges"]["0"][0]["operation"])


# print(Counter(ab_list))
# print(Counter(list))
#
# for key in dic.keys():
#     trace_class = dic[key]["edges"]["0"][0]["operation"] + str(dic[key]['abnormal']) + str(len(dic[key]['vertexs'].keys()))
#     trace_list.append(trace_class)
# print(Counter(trace_list))

# with open('E:\Data\TraceCluster\\0222-data\\change_test_abnormal.json', 'r') as f:
#     dic1 = json.load(f)

# dic.update(dic1)



# rcs = ['ts-contacts-service', 'ts-execute-service', 'ts-config-service', 'ts-seat-service', 'ts-travel-service']
# model_path = 'E:\Data\TraceCluster\log\\20epoch_CGConv_add_0222data_onlychaos_nomodify_viewaug\\'
#
# with open(model_path + 'train_info.json', "r") as f:  # file name not list
#     model_info = json.load(f)
#     num_layers = model_info['num_layers']
#     output_dim = model_info['output_dim']
#     # num_node_features = model_info['num_node_features']
#     # num_edge_features = model_info['num_edge_features']
#     gnn_type = model_info['gnn_type']
#     pooling_type = model_info['pooling_type']
#     aug = model_info['aug']
#     train_idx = model_info['train_idx']
#     test_idx = model_info['test_idx']


# idx = 0
#
# train_normal_dic = {}
# test_normal_dic = {}
# abnormal_dic = {}
#
# modify_idx = train_idx[:int(len(train_idx) * 0.1)]
# modify_dic = {}
#
# for trace_id, trace in dic.items():
#     if idx in train_idx:
#         train_normal_dic[trace_id] = trace
#     elif idx in test_idx:
#         if trace['abnormal'] == 0:
#             test_normal_dic[trace_id] = trace
#         elif trace['abnormal'] == 1:
#             abnormal_dic[trace_id] = trace
#     if idx in modify_idx:
#         trace1 = deepcopy(trace)
#         trace1['abnormal'] = 1
#         for key in trace1['edges'].keys():
#             trace1['edges'][key][0]['isError'] = True
#         modify_dic[trace_id+'1'] = trace1
#     idx += 1
#
# print(len(train_normal_dic.keys()))
# print(len(test_normal_dic.keys()))
# print(len(abnormal_dic.keys()))
# print(len(modify_dic.keys()))

# modify_dic.update(abnormal_dic)

# print(len(modify_dic.keys()))

# with open('D:\Code\TraceAnomaly\modified\TraceAnomaly\data\\chaos_normal.json', 'w', encoding='utf-8') as json_file:
#     json.dump(test_normal_dic, json_file)
#     print('write data info success')
#
# with open('D:\Code\TraceAnomaly\modified\TraceAnomaly\data\\normal.json', 'w', encoding='utf-8') as json_file:
#     json.dump(train_normal_dic, json_file)
#     print('write data info success')

# with open('D:\Code\TraceAnomaly\modified\TraceAnomaly\data\\chaos_abnormal_more_httperror_only.json', 'w',
#           encoding='utf-8') as json_file:
#     json.dump(modify_dic, json_file)
#     print('write data info success')

    # if dic[key]['rc'] in rcs:
    #     continue
    # list.append(dic[key]["edges"]["0"][0]["operation"])



# print(Counter(list))

# print(len(dic1.keys()))

# print(dic.keys())
# print(len(dic.keys()))

# rcs = ['ts-contacts-service', 'ts-execute-service', 'ts-config-service', 'ts-seat-service', 'ts-travel-service']
# #
# for key in dic.keys():
#     if dic[key]['rc'] in rcs:
#         dic[key]['abnormal'] = 0


#
# random.shuffle(list)
# train_normal = list[:int(len(list) * 0.6)]
#
# test_normal = list[int(len(list) * 0.6):]
#
# dic_test_abnormal = {}
# dic_test_normal = {}
# dic_train_normal = {}
#
# for key in dic.keys():
#     if key in train_normal:
#         dic_train_normal[key] = dic[key]
#     elif key in test_normal:
#         dic_test_normal[key] = dic[key]
#     elif key in ab_list:
#         dic_test_abnormal[key] = dic[key]
#
# print(len(list))
# print(len(train_normal))
# print(len(dic_train_normal.keys()))
# print(len(test_normal))
# print(len(dic_test_normal.keys()))
# print(len(ab_list))
# print(len(dic_test_abnormal.keys()))


#     print(dic[key])
#     print(dic[key].keys())
#     exit()


# test_normal = {}
# test_abnormal = {}
# for key in dic1.keys():
#     if dic1[key]['rc'] in rcs:
#         continue
#     if dic1[key]['abnormal'] == 1:
#         test_abnormal[key] = dic1[key]
#     else:
#         test_normal[key] = dic1[key]
#
# print(len(test_normal.keys()))
# print(len(test_abnormal.keys()))


#

# with open('E:\Data\TraceCluster\\0222-data\\test_abnormal.json', 'w', encoding='utf-8') as json_file:
#     json.dump(test_abnormal, json_file)
#     print('write data info success')

# for key in list:
#     dic.pop(key)


# from sklearn.neighbors import KernelDensity
#
# x = np.random.normal(0,1, (100,1))
# x_1 = np.random.normal(0,3, (50,1))
# rng = np.random.RandomState(42)
# X = rng.random_sample((100,1))
# rng_1 = np.random.RandomState(1500)
# x_1 = rng_1.random_sample((10,1))
# kde = KernelDensity(kernel='gaussian').fit(x)
# print(kde.get_params())
# log_density = kde.score_samples(x_1)
# scores = np.exp(log_density)
# pred_label = scores<0.01
# print(log_density)
# print(np.exp(log_density))
# print(pred_label)


#
# list = []
# for key in dic.keys():
#     print(dic[key])
#     exit()
#     if dic[key]['abnormal'] == 0:
#         list.append(key)
# for key in list:
#     dic.pop(key)

# dic.update(dic1)

# print(dic['5f736bfb4fcd4969b6225e36220362e0.39.16436745956820537'])
#
# print(dic['5f736bfb4fcd4969b6225e36220362e0.44.16434153140855385'])

# rcs = ['ts-contacts-service', 'ts-execute-service', 'ts-config-service', 'ts-seat-service', 'ts-travel-service']
#
# list = []
#
# for key in dic.keys():
#     if dic[key]['abnormal'] == 1 in rcs:
#         list.append(key)
# print(len(list))
# for key in list:
#     dic.pop(key)
    # print(dic[key])
    # exit()
# with open('E:\Data\TraceCluster\\0222-data\edge_data_mem\\all\preprocessed\\data_info.json', 'w', encoding='utf-8') as json_file:
#     json.dump(dic, json_file)
#     print('write data info success')
    # if len(dic[key]['vertexs'].keys()) == 1:
    #     print(key)

# print(dic['POST:/api/v1/travelservice/trips/left--->GET:/api/v1/ticketinfoservice/ticketinfo/{name}'])

# list = []
# nor_list = []
#
# rcs = ['ts-contacts-service', 'ts-execute-service', 'ts-config-service', 'ts-seat-service', 'ts-travel-service']
#
# for key in dic.keys():
#     if dic[key]['rc'] in rcs:
#         continue
#     # print(key)
#     # print(type(key))
#     if dic[key]['abnormal'] == 1:
#         list.append(dic[key]["edges"]["0"][0]["operation"])
#     else:
#         nor_list.append(dic[key]["edges"]["0"][0]["operation"])
#
# print(Counter(list))
# print(Counter(nor_list))
