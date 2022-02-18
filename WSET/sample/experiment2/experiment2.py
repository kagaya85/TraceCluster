import pandas as pd
import numpy as np

from sample.utils import read_json, f_n
from xcluster.src.python.xcluster.models.PNode import PNode
from xcluster.src.python.xcluster.utils.Graphviz import Graphviz


def create_dataset():
    data = pd.read_csv("/home/ws/Sifter/experiment/2/data.csv", index_col=0)["0"].tolist()
    abnormal_traces = {}
    for file in f_n:
        file_name = "/home/ws/TraceCluster/data/preprocessed/trainticket/" + file + "/0.json"
        traces = read_json(file_name)
        abnormal_traces.update(traces)
    traces_data = {}

    api_class = {}
    for i, trace in enumerate(data):
        if i < 20:
            api_class[trace] = "F13"
        elif i < 100:
            api_class[trace] = "F1"
        elif i < 250:
            api_class[trace] = "F8"
        elif i < 500:
            api_class[trace] = "F11"
        elif i < 1000:
            api_class[trace] = "F25"

        traces_data[trace] = abnormal_traces[trace]

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

        # TODO: api的种类怎么确定
        api_type = api_class[trace_id]
        dataset.append((np.array(vector), api_type, trace_id))
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

    result = {'F13': 0, 'F1': 0, 'F8': 0, 'F11': 0, 'F25': 0}

    for leaf in root.leaves():
        trace_id = leaf.pts[0][2]
        api_type = leaf.pts[0][1]
        p = 1
        n = leaf
        while n is not None:
            p = p / (len(n.siblings()) + 1)
            n = n.parent

        result[api_type] += p

    with open("result.txt", "w") as f:
        for api, p in result.items():
            f.write("%s\t%s\n" % (api, p))