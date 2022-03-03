import json
import random
import numpy as np

from xcluster.src.python.xcluster.models.PNode import PNode


f_n = [
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
    "F7",
    "F8",
    "F11",
    "F12",
    "F13",
    "F14",
    "F23",
    "F24",
    "F25",
]


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


def read_json(file):
    with open(file, 'r', encoding='utf8') as fp:
        data = json.load(fp)
    return data