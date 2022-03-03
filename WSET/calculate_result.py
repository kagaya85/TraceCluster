from collections import Counter
import os
import re
import json
import pandas as pd
import time
import numpy as np

with open('E:\Data\TraceCluster\\0301-data\PERCH\\test_data.json', 'r') as f:
    test_data = json.load(f)

traceID_list = test_data['traceids']
class_list = test_data['id_classes']
trace_class_num_list = test_data['trace_classes_num']
trace_class_list = test_data['trace_classes_index']

oPERCH = [0] * len(trace_class_num_list)
mPERCH = [0] * len(trace_class_num_list)

oPERCH_path = './result/'
mPERCH_path = './result_simclr/'

oPERCH_file_names = []
mPERCH_file_names = []

for root, dirs, names in os.walk(oPERCH_path):
    patten = re.compile(r'result')
    for filename in names:
        match = patten.search(filename)
        if match:
            oPERCH_file_names.append(filename)

for root, dirs, names in os.walk(mPERCH_path):
    patten = re.compile(r'result')
    for filename in names:
        match = patten.search(filename)
        if match:
            mPERCH_file_names.append(filename)

print(oPERCH_file_names)
print(mPERCH_file_names)


for file in oPERCH_file_names:
    file_oSieve = open(oPERCH_path + file, "r")
    listOfLines = file_oSieve.readlines()
    file_oSieve.close()
    for idx, line in enumerate(listOfLines):
        if line.strip().split('\t')[3] == 'Sample':
            indx = trace_class_list.index(line.strip().split('\t')[4])
            oPERCH[indx] += 1

oPERCH = (np.array(oPERCH) / len(oPERCH_file_names)).tolist()

for file in mPERCH_file_names:
    file_mSieve = open(mPERCH_path + file, "r")
    listOfLines = file_mSieve.readlines()
    file_mSieve.close()
    for idx, line in enumerate(listOfLines):
        if line.strip().split('\t')[3] == 'Sample':
            indx = trace_class_list.index(line.strip().split('\t')[4])
            mPERCH[indx] += 1

mPERCH = (np.array(mPERCH) / len(mPERCH_file_names)).tolist()

time_str = str(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))

index = ['', 'num', 'oPERCH', 'mPERCH']
result_list = [trace_class_list,trace_class_num_list,oPERCH,mPERCH]
result = pd.DataFrame(index=index, data=result_list)
result.to_csv('./stat_result/'+'result_' + time_str + '.csv')

print(Counter(oPERCH))