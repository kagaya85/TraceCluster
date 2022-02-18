import numpy as np
import pandas as pd

import sys
sys.path.append("..")
from utils import read_json, f_n
from xcluster.src.python.xcluster.models.PNode import PNode
from xcluster.src.python.xcluster.utils.Graphviz import Graphviz


def create_dataset():
    normal_sample = pd.read_csv("/home/ws/Sifter/experiment/1/normal_sample.csv", index_col=0)["0"].tolist()
    abnormal_sample = pd.read_csv("/home/ws/Sifter/experiment/1/abnormal_sample.csv", index_col=0)["0"].tolist()
    # Get the traces
    file_name = "/home/ws/TraceCluster/data/preprocessed/trainticket/normal/0.json"
    normal_traces = read_json(file_name)
    abnormal_traces = {}
    for file in f_n:
        file_name = "/home/ws/TraceCluster/data/preprocessed/trainticket/" + file + "/0.json"
        traces = read_json(file_name)
        abnormal_traces.update(traces)
    traces_data = {}
    normal_index = 0
    abnormal_index = 0

    api_class = {}
    for i in range(1, 1001):
        if i % 10 == 0:
            id = abnormal_sample[abnormal_index]
            traces_data[id] = abnormal_traces[id]
            api_class[id] = "abnormal"
            abnormal_index += 1
        else:
            id = normal_sample[normal_index]
            traces_data[id] = normal_traces[id]
            api_class[id] = "normal"
            normal_index += 1

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
    i = 0
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
        i += 1

    print("Finish building dataset, length: %d" % len(dataset))

    return dataset


if __name__ == '__main__':
    dataset = create_dataset()

    root = PNode(exact_dist_thres=10)

    for i, pt in enumerate(dataset):
        root = root.insert(pt, collapsibles=None, L=float('inf'))

    print("Finish creating cluster tree.")
    Graphviz.write_tree('tree.dot', root)

    csv_data = []
    # 实验一：计算每个trace被采样的概率
    for leaf in root.leaves():
        trace_id = leaf.pts[0][2]
        p = 1
        n = leaf
        while n is not None:
            p = p / (len(n.siblings()) + 1)
            n = n.parent
        csv_data.append([trace_id, p])

    pd.DataFrame(csv_data, columns=["trace_id", "probability"]).to_csv("p.csv", mode="w")