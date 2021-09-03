# Kagaya kagaya85@outlook.com
import json
import os
import time
import pandas as pd
import numpy as np
from torch.nn import factory_kwargs
from torch_geometric.nn import glob
from tqdm import tqdm
import utils
from typing import List, Callable, Dict

data_path_list = [
    'data/raw/F01-02/ERROR_F012_SpanData2021-08-14_01-52-43.csv'
]

time_now_str = str(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))

embedding_word_list = np.load('./data/glove/wordsList.npy').tolist()
embedding_word_vector = np.load('./data/glove/wordVectors.npy')


class Item:
    def __init__(self) -> None:
        self.SPAN_ID = 'SpanId'
        self.PARENT_SPAN_ID = 'ParentSpan'
        self.TRACE_ID = 'TraceId'
        self.START_TIME = 'StartTime'
        self.END_TIME = 'EndTime'
        self.OPERATION = 'URL'
        self.DURATION = 'Duration'
        self.SPAN_TYPE = 'SpanType'
        self.SERVICE = 'Service'
        self.IS_ERROR = 'IsError'
        self.PEER = 'Peer'
        self.TYPE = 'SpanType'


ITEM = Item()


class Span:
    def __init__(self, raw_span=None) -> None:
        """
        convert raw span to span object
        """
        if raw_span is None:
            self.spanId = ''
            self.parentSpanId = ''
            self.traceId = ''
            self.spanType = ''
            self.startTime = 0
            self.duration = 0
            self.service = ''
            self.peer = ''
            self.operation = ''
            self.code = ''
            self.isError = False
        else:
            self.spanId = raw_span[ITEM.SPAN_ID]
            self.parentSpanId = raw_span[ITEM.PARENT_SPAN_ID]
            self.traceId = raw_span[ITEM.TRACE_ID]
            self.spanType = raw_span[ITEM.SPAN_TYPE]
            self.startTime = raw_span[ITEM.START_TIME]
            self.duration = raw_span[ITEM.DURATION]
            self.service = raw_span[ITEM.SERVICE]
            self.peer = raw_span[ITEM.PEER]
            self.operation = raw_span[ITEM.OPERATION]
            self.code = str(utils.boolStr2Int(raw_span[ITEM.IS_ERROR]))
            self.isError = utils.boolStr2Bool(raw_span[ITEM.IS_ERROR])


def load_span(pathList: list):
    """
    load sapn data from pathList
    """
    spansList = []

    for filepath in pathList:
        print(f"load span data from {filepath}")
        data_type = {ITEM.START_TIME: np.uint64, ITEM.END_TIME: np.uint64}
        spans = pd.read_csv(
            filepath, dtype=data_type
        ).drop_duplicates().dropna()
        spansList.append(spans)

    spanData = pd.concat(spansList, axis=0, ignore_index=True)
    spanData[ITEM.DURATION] = spanData[ITEM.END_TIME] - \
        spanData[ITEM.START_TIME]

    return spanData


def build_graph(trace: List[Span], time_normolize: Callable[[float], float]):
    """
    build trace graph from span list
    """

    vertexs = {0: embedding('start')}
    edges = {}

    spanIdMap = {'-1': 0}
    spanIdCounter = 1
    rootSpan = None

    trace.sort(key=lambda s: s.startTime)

    for span in trace:
        """
        (raph object contains Vertexs and Edges
        Edge: [(from, to, duration), ...]
        Vertex: [(id, nodestr), ...]
        """
        if span.parentSpanId == '-1':
            rootSpan = span

        if span.parentSpanId not in spanIdMap.keys():
            spanIdMap[span.parentSpanId] = spanIdCounter
            spanIdCounter += 1

        spanIdMap[span.spanId] = spanIdCounter
        spanIdCounter += 1

        spanId, parentSpanId = spanIdMap[span.spanId], spanIdMap[span.parentSpanId]

        # span id should be unique
        if spanId not in vertexs.keys():
            vertexs[spanId] = embedding('/'.join(
                [span.service, span.operation, span.code]))

        if parentSpanId not in edges.keys():
            edges[parentSpanId] = []

        edges[parentSpanId].append({
            'vertexId': spanId,
            'spanId': span.spanId,
            'startTime': span.startTime,
            'duration': time_normolize(span.duration),
            'isError': span.isError,
        })

    if rootSpan == None:
        return None

    graph = {
        'vertexs': vertexs,
        'edges': edges,
    }

    return graph


def save_data(graphs: Dict, filename: str):
    """
    save graph data to json file
    """
    filename = os.path.join(os.getcwd(), 'data',
                            'processed', time_now_str, filename)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'a+', encoding='utf-8') as fd:
        data_json = json.dumps(graphs, ensure_ascii=False, indent=4)
        fd.write(data_json)
        fd.write('\n')
    print(f"data saved in {filename}")


def str_process(s: str) -> str:
    words = ['ticket', 'order', 'name',
             'operation', 'spring', 'service', 'trip',
             'date', 'route', 'type', 'id', 'account', 'number']
    word_list = []
    s = s.replace('-', '/')
    s = s.replace('_', '/')
    s = s.replace('{', '')
    s = s.replace('}', '')
    s = s.lower()
    s = s.strip('/')

    for w in s.split('/'):
        for sub in utils.wordSplit(w, words):
            snake = utils.hump2snake(sub)
            word_list.append(snake)

    return '/'.join(word_list)


def trace_process(trace: List[Span]) -> List[Span]:
    operationMap = {}
    for span in trace:
        span.service = str_process(span.service)
        span.operation = str_process(span.operation)
        if span.spanType == "Entry":
            operationMap[span.parentSpanId] = span.operation

    for span in trace:
        # 替换Exit span的URL
        if span.spanType == "Exit" and span.spanId in operationMap.keys():
            span.operation = operationMap[span.spanId]

    return trace


def embedding(input: str) -> List[float]:
    words = input.split('/')
    vec_sum = []
    for w in words:
        if w in embedding_word_list:
            idx = embedding_word_list.index(w)
            vec = embedding_word_vector[idx]
            vec_sum.append(vec)

    return np.mean(np.array(vec_sum), axis=0).tolist()


def z_score(x: float, mean: float, std: float) -> float:
    """
    z_score normalize funciton 
    """
    return (x - mean) / std


def main():
    span_data = load_span(data_path_list)
    duration_mean = span_data[ITEM.DURATION].mean()
    duration_std = span_data[ITEM.DURATION].std()

    graph_map = {}
    print("processing...")

    for trace_id, trace_data in tqdm(span_data.groupby([ITEM.TRACE_ID])):
        trace = [Span(raw_span) for idx, raw_span in trace_data.iterrows()]
        graph = build_graph(trace_process(trace),
                            lambda x: z_score(x, duration_mean, duration_std))
        if graph == None:
            continue
        graph_map[trace_id] = graph

    save_data(graph_map, 'processed.json')


if __name__ == '__main__':
    main()
