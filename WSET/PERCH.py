import numpy as np
import pandas as pd

from utils import read_json
from xcluster.src.python.xcluster.models.PNode import PNode
from xcluster.src.python.xcluster.utils.Graphviz import Graphviz

import time


def create_dataset():
    global traceID_list
    traceID_fr = open('./newData/test_traceID.txt', 'r')
    S = traceID_fr.read()
    traceID_fr.close()
    traceID_list = [traceID for traceID in S.split(', ')]

    traces = read_json('./newData/preprocessed/0.json')

    # Get the traces
    traces_data = {}
    api_class = {}
    for traceID in traceID_list:
        traces_data[traceID] = traces[traceID]
        api_class[traceID] = "abnormal" if traces[traceID]["abnormal"]==1 else "normal"

    # with open(path, 'r') as f:
    #     traces_data = json.load(f)

    print("Finish reading traces.")

    # 所有操作的字母表
    alphabet = set()

    # 第一次遍历：得到字母表
    for trace_id, trace in traces_data.items():
        for start_vertex_id, edges in trace["edges"].items():
            for edge in edges:
                if edge["operation"]:
                    alphabet.add(edge["operation"])

    print("Finish building alphabet, length: %d" % len(alphabet))

    # 第二次遍历：得到每个trace的operation列表，计算字母表中每个操作出现的次数，作为每个trace对应的array
    dataset = []
    for trace_id, trace in traces_data.items():
        # 统计trace中每个operation
        c = []
        for start_vertex_id, edges in trace["edges"].items():
            for edge in edges:
                c.append(edge["operation"])

        vector = []
        for a in alphabet:
            vector.append(c.count(a))

        dataset.append((np.array(vector), api_class[trace_id], trace_id))

    print("Finish building dataset, length: %d" % len(dataset))

    return dataset


if __name__ == '__main__':
    dataset = create_dataset()

    root = PNode(exact_dist_thres=10)

    for i, pt in enumerate(dataset):
        root = root.insert(pt, collapsibles=None, L=float('inf'))

    print("Finish creating cluster tree.")
    Graphviz.write_tree('tree.dot', root)

    results = {}
    # 实验一：计算每个trace被采样的概率
    for leaf in root.leaves():
        trace_id = leaf.pts[0][2]
        label = leaf.pts[0][1]
        p = 1
        n = leaf
        while n is not None:
            p = p / (len(n.siblings()) + 1)
            n = n.parent
        results[trace_id] = (label, p)

    # 记录实验结果 trace_id    label    p
    # open test result
    time_str = str(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
    res_f = open('./result_' + time_str + '.txt', 'w')
    for traceID in traceID_list:
        res_content = traceID + '\t' + results[traceID][0] + '\t' + str(results[traceID][1]) + '\n'
        res_f.write(res_content)
    res_f.close()

    print("Done !")