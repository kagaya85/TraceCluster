import json
from collections import Counter
import numpy as np
import torch
import random
import copy
from copy import deepcopy



with open('E:\Data\TraceCluster\\0301-data\\all_final.json', 'r') as f:
    dic = json.load(f)

type1 = 'POST:/api/v1/orderOtherService/orderOther/refresh03'
type1_num = 1000
type1_ab = 'POST:/api/v1/orderOtherService/orderOther/refresh13'
type1_ab_num = 50
type2 = 'POST:/api/v1/travelservice/trips/left013'
type2_num = 1000
type2_ab = 'POST:/api/v1/travelservice/trips/left113'
type2_ab_num = 50
type2_ab_2 = 'POST:/api/v1/travelservice/trips/left19'
type2_ab_2_num = 25
type3 = 'POST:/api/v1/travel2service/trips/left09'
type3_num = 1000
type3_ab = 'POST:/api/v1/travel2service/trips/left19'
type3_ab_num = 50
type4 = 'POST:/api/v1/travelservice/trips/left_parallel013'
type4_num = 500
type4_ab = 'POST:/api/v1/travelservice/trips/left_parallel113'
type4_ab_num = 25
type5 = 'POST:/api/v1/preserveservice/preserve073'
type5_num = 1000
type5_ab = 'POST:/api/v1/preserveservice/preserve173'
type5_ab_num = 50
type6 = 'POST:/api/v1/travelplanservice/travelPlan/cheapest093'
type6_num = 500
type6_ab = 'POST:/api/v1/travelplanservice/travelPlan/cheapest193'
type6_ab_num = 25

trace_class_list = [type1, type2, type3, type4, type5, type6, type1_ab, type2_ab, type2_ab_2, type3_ab, type4_ab,
                    type5_ab, type6_ab]

trace_class_num_list = [type1_num, type2_num, type3_num, type4_num, type5_num, type6_num, type1_ab_num, type2_ab_num,
                        type2_ab_2_num, type3_ab_num, type4_ab_num, type5_ab_num, type6_ab_num]
count_list = [0] * len(trace_class_num_list)

traceid_list = []
id_classes_dic = {}

all_traceids = list(dic.keys())

random.shuffle(all_traceids)

test_traces = {}

for key in all_traceids:
    root_url = dic[key]["edges"]["0"][0]["operation"] + str(dic[key]['abnormal']) + str(len(dic[key]['vertexs'].keys()))
    if root_url in trace_class_list:
        indx = trace_class_list.index(root_url)
        if count_list[indx] < trace_class_num_list[indx]:
            traceid_list.append(key)
            id_classes_dic[key] = root_url
            count_list[indx] += 1
            test_traces[key] = dic[key]

print(len(traceid_list))
print(len(test_traces.keys()))

test_data = {
    'traceids': traceid_list,
    'id_classes': id_classes_dic,
    'trace_classes_index': trace_class_list,
    'trace_classes_num': trace_class_num_list
}

with open('E:\Data\TraceCluster\\0301-data\PERCH\\test_data.json', 'w', encoding='utf-8') as json_file:
    json.dump(test_data, json_file)
    print('write data info success')

with open('E:\Data\TraceCluster\\0301-data\PERCH\\test_traces.json', 'w', encoding='utf-8') as json_file:
    json.dump(test_traces, json_file)
    print('write data info success')

original_trace_class_list = [type1, type2, type3, type4, type5, type6]
original_traces_num = 0
original_traces = {}


# for key in all_traceids:
#     root_url = dic[key]["edges"]["0"][0]["operation"] + str(dic[key]['abnormal']) + str(len(dic[key]['vertexs'].keys()))
#     if root_url in original_trace_class_list:
#         if original_traces_num < 50:
#             original_traces[key] = dic[key]
#             original_traces_num += 1
#
# with open('E:\Data\TraceCluster\\0301-data\PERCH\\original_traces.json', 'w', encoding='utf-8') as json_file:
#     json.dump(original_traces, json_file)
#     print('write data info success')
