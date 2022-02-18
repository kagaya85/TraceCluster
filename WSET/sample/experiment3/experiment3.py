from time import time
from threading import Timer

import numpy as np

from sample.utils import read_json, f_n
from xcluster.src.python.xcluster.models.PNode import PNode
from xcluster.src.python.xcluster.utils.Graphviz import Graphviz


def create_dataset():
    file_name = "/home/ws/TraceCluster/data/preprocessed/trainticket/normal/0.json"
    traces_data = read_json(file_name)
    for file in f_n:
        file_name = "/home/ws/TraceCluster/data/preprocessed/trainticket/" + file + "/0.json"
        traces = read_json(file_name)
        traces_data.update(traces)

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

        dataset.append((np.array(vector), i, trace_id))
        i += 1

    print("Finish building dataset, length: %d" % len(dataset))

    return dataset

class RepeatingTimer(Timer):
    def run(self):
        while not self.finished.is_set():
            self.function(*self.args, **self.kwargs)
            self.finished.wait(self.interval)

global_count = 0


def print_global_count(record):
    record.append(global_count)


if __name__ == '__main__':
    dataset = create_dataset()

    root = PNode(exact_dist_thres=10)

    record = []
    record2 = []
    inc = 5
    t = RepeatingTimer(inc, print_global_count, (record,))
    t.start()

    t2 = time()
    for i, pt in enumerate(dataset):
        global_count = i
        root = root.insert(pt, collapsibles=None, L=float('inf'))
        if global_count % 10 == 0:
            record2.append(time() - t2)
            t2 = time()

    t.cancel()
    print("Finish creating cluster tree.")
    Graphviz.write_tree('tree.dot', root)

    np.savetxt("record.txt", np.array(record))
    np.savetxt("record2.txt", np.array(record2))
