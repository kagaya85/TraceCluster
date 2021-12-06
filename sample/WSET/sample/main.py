import json
import os
import sys

sys.path.append(os.pardir)

import numpy as np
import random
import argparse
import ijson
from xcluster.src.python.xcluster.models.PNode import PNode
from xcluster.src.python.xcluster.test.pnode_dendrogram_purity_test import create_trees_w_purity_check
from xcluster.src.python.xcluster.utils.Graphviz import Graphviz


# 离线采样
def offline_sampling(root: PNode, n: int):
    s_trace = set()
    c_trace = 0

    while c_trace < n:
        node = root
        while not node.is_leaf():
            node = random.choice(node.children)
        if node not in s_trace:
            s_trace.add(node)
            c_trace += 1

    return s_trace


# 在线采样
def online_sampling(root: PNode, n: int, trace):
    if len(root.descendants()) is n:
        delete_unlikely_node(root)

    root.insert(trace, collapsibles=None, L=float('inf'))


def delete_unlikely_node(root: PNode):
    node = root
    while not node.is_leaf():
        node = max(node.children, key=lambda child: len(child.leaves()))

    node.deleted = True
    # node._update_params_recursively()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trace Clustering and Sampling')
    parser.add_argument('--input', '-i', type=str, help='Path to the dataset.',
                        default="../../../data/preprocessed/2021-10-13_16-57-51.json")
    parser.add_argument('--sample_num', '-n', type=int,
                        help='Num of sampled traces',
                        default=1000)

    args = parser.parse_args()

    print("Read Dataset")

    with open(args.input, 'r') as f:
        traces_data = list(ijson.items(f, ""))[0]

    print("Has Read Dataset")

    dataset = []

    # 所有操作的字母表
    alphabet = set()

    # 第一次遍历：得到字母表
    for trace_id, trace in traces_data.items():
        for start_vertex_id, edges in trace["edges"].items():
            for edge in edges:
                if edge["operation"]:
                    alphabet.add(edge["operation"])

    print(alphabet)

    # 第二次遍历：得到每个trace的operation列表，计算字母表中每个操作出现的次数，作为每个trace对应的array
    i = 0
    for trace_id, trace in traces_data.items():
        # 统计trace中每个operation
        count = []
        for start_vertex_id, edges in trace["edges"].items():
            for edge in edges:
                if edge["operation"]:
                    count.append(edge["operation"])

        vector = []
        for a in alphabet:
            vector.append(count.count(a))

        print(trace_id)
        dataset.append((np.array(vector), i, trace_id))
        i += 1

    root = create_trees_w_purity_check(dataset)

    Graphviz.write_tree('result/tree.dot', root)

    # 采样
    n = args.sample_num
    sampled_traces = offline_sampling(root, n)
    with open("result/sampled_traces.csv", "w") as f:
        for trace in sampled_traces:
            # 打印trace_id
            f.write("%s\n" % (trace[0][2]))
