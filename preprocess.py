# Kagaya kagaya85@outlook.com
import json
import os
import time
import pandas as pd
from pandas.core.frame import DataFrame
from datetime import datetime
import numpy as np
import argparse
from tqdm import tqdm
import utils
from typing import List, Callable, Dict
from multiprocessing import Pool, Queue, cpu_count
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager
from concurrent.futures import ProcessPoolExecutor, as_completed

data_path_list = [
    # Normal
    'data/raw/normal/normal0822_01/SUCCESS_SpanData2021-08-22_15-43-01.csv',
    'data/raw/normal/normal0822_02/SUCCESS2_SpanData2021-08-22_22-04-35.csv',
    'data/raw/normal/normal0823/SUCCESS2_SpanData2021-08-23_15-15-08.csv',

    # F01
    'data/raw/F01-01/SUCCESSF0101_SpanData2021-08-14_10-22-48.csv',
    'data/raw/F01-02/ERROR_F012_SpanData2021-08-14_01-52-43.csv',
    'data/raw/F01-03/SUCCESSerrorf0103_SpanData2021-08-16_16-17-08.csv',
    'data/raw/F01-04/SUCCESSF0104_SpanData2021-08-14_02-14-51.csv',
    'data/raw/F01-05/SUCCESSF0105_SpanData2021-08-14_02-45-59.csv',

    # F02
    'data/raw/F02-01/SUCCESS_errorf0201_SpanData2021-08-17_18-25-59.csv',
    'data/raw/F02-02/SUCCESS_errorf0202_SpanData2021-08-17_18-47-04.csv',
    'data/raw/F02-03/SUCCESS_errorf0203_SpanData2021-08-17_18-54-53.csv',
    'data/raw/F02-04/ERROR_SpanData.csv',
    'data/raw/F02-05/ERROR_SpanData.csv',
    'data/raw/F02-06/ERROR_SpanData.csv',

    # F03
    'data/raw/F03-01/ERROR_SpanData.csv',
    'data/raw/F03-02/ERROR_SpanData.csv',
    'data/raw/F03-03/ERROR_SpanData.csv',
    'data/raw/F03-04/ERROR_SpanData.csv',
    'data/raw/F03-05/ERROR_SpanData.csv',
    'data/raw/F03-06/ERROR_SpanData.csv',
    'data/raw/F03-07/ERROR_SpanData.csv',
    'data/raw/F03-08/ERROR_SpanData.csv',

    # F04
    'data/raw/F04-01/ERROR_SpanData.csv',
    'data/raw/F04-02/ERROR_SpanData.csv',
    'data/raw/F04-03/ERROR_SpanData.csv',
    'data/raw/F04-04/ERROR_SpanData.csv',
    'data/raw/F04-05/ERROR_SpanData.csv',
    'data/raw/F04-06/ERROR_SpanData.csv',
    'data/raw/F04-07/ERROR_SpanData.csv',
    'data/raw/F04-08/ERROR_SpanData.csv',
]

mm_data_path_list = [
    ''
]

time_now_str = str(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))

embedding_word_list = np.load('./data/glove/wordsList.npy').tolist()
embedding_word_vector = np.load('./data/glove/wordVectors.npy')


def normalize(x): return x


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
        self.CODE = 'Code'


ITEM = Item()


class Span:
    def __init__(self, raw_span: dict) -> None:
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
            if ITEM.CODE in raw_span.keys():
                self.code = raw_span[ITEM.CODE]
            else:
                self.code = str(utils.boolStr2Int(raw_span[ITEM.IS_ERROR]))
            self.isError = utils.boolStr2Bool(raw_span[ITEM.IS_ERROR])


def arguments():
    parser = argparse.ArgumentParser(description="Preporcess Argumentes.")
    parser.add_argument('--cores', dest='cores',
                        help='parallel processing core numberes', default=cpu_count())
    parser.add_argument('--wechat', help='use wechat data',
                        action='store_true')


def load_span(is_wechat: bool) -> List[DataFrame]:
    """
    load raw sapn data from pathList
    """
    raw_spans = []

    if is_wechat:
        # wechat data
        for filepath in mm_data_path_list:
            print(f"load wechat span data from {filepath}")
            with open(filepath, 'r') as f:
                raw_data = json.load(f)
                mmspans = raw_data['data']
                spans = {
                    ITEM.SPAN_ID: [],
                    ITEM.PARENT_SPAN_ID: [],
                    ITEM.TRACE_ID: [],
                    ITEM.SPAN_TYPE: [],
                    ITEM.START_TIME: [],
                    ITEM.DURATION: [],
                    ITEM.SERVICE: [],
                    ITEM.PEER: [],
                    ITEM.OPERATION: [],
                    ITEM.IS_ERROR: [],
                    ITEM.CODE: [],
                }

                # convert to dataframe
                for s in mmspans:
                    spans[ITEM.SPAN_ID].append(str(s['CalleeCmdID']))
                    spans[ITEM.PARENT_SPAN_ID].append(str(s['CallerCmdID']))
                    spans[ITEM.TRACE_ID].append(s['GraphIdBase64'])
                    spans[ITEM.SPAN_TYPE].append('EntrySpan')
                    st = datetime.strptime(s['TimeStamp'], '%Y-%m-%d %H:%M:%S')
                    spans[ITEM.START_TIME].append(int(datetime.timestamp(st)))
                    spans[ITEM.DURATION].append(int(s['CostTime']))
                    spans[ITEM.SERVICE].append(s['CalleeOssID'])
                    spans[ITEM.PEER].append(s['CallerOssID'])
                    # convert to operation string
                    spans[ITEM.OPERATION].append(
                        convert_operation_name(s['CalleeCmdID'])
                    )
                    spans[ITEM.IS_ERROR].append(
                        not utils.int2Bool(s['IFSuccess']))
                    spans[ITEM.CODE].append(
                        s['NetworkRet'] if s['NetworkRet'] != 0 else s['ServiceRet'])

                raw_spans.append(DataFrame(spans))

    else:
        # skywalking data
        for filepath in data_path_list:
            print(f"load span data from {filepath}")
            data_type = {ITEM.START_TIME: np.uint64, ITEM.END_TIME: np.uint64}
            spans = pd.read_csv(
                filepath, dtype=data_type
            ).drop_duplicates().dropna()
            spans[ITEM.DURATION] = spans[ITEM.END_TIME] - \
                spans[ITEM.START_TIME]
            raw_spans.append(spans)

    return raw_spans


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

        if span.spanId not in spanIdMap.keys():
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


def save_data(graphs: Dict):
    """
    save graph data to json file
    """
    filename = os.path.join(os.getcwd(), 'data',
                            'preprocessed', time_now_str+'.json')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'a+', encoding='utf-8') as fd:
        data_json = json.dumps(graphs, ensure_ascii=False)
        fd.write(data_json)
        fd.write('\n')
    print(f"data saved in {filename}")


def str_process(s: str) -> str:
    words = ['ticket', 'order', 'name', 'security',
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


def convert_operation_name(opid: int) -> str:
    # TODO
    return str(opid)


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
    z-score normalize funciton
    """
    return (x - mean) / std


def min_max(x: float, min: float, max: float) -> float:
    """
    min-max normalize funciton
    """
    return (x - min) / (max - min)


def task(shared_list, idx) -> dict:
    span_data = shared_list[idx]

    graph_map = {}
    for trace_id, trace_data in tqdm(span_data.groupby([ITEM.TRACE_ID])):
        trace = [Span(raw_span) for idx, raw_span in trace_data.iterrows()]
        graph = build_graph(trace_process(trace), normalize)
        if graph == None:
            continue
        graph_map[trace_id] = graph

    return graph_map


def main():
    args = arguments()
    print(f"parallel processing number: {args.cores}")

    # load all span
    raw_spans = load_span(args.wechat)

    # concat all span data in one list
    span_data = pd.concat(raw_spans, axis=0, ignore_index=True)
    # duration_mean = span_data[ITEM.DURATION].mean()
    # duration_std = span_data[ITEM.DURATION].std()
    duration_max = span_data[ITEM.DURATION].max()
    duration_min = span_data[ITEM.DURATION].min()
    del span_data

    global normalize
    def normalize(x): return min_max(x, duration_min, duration_max)

    result_map = {}

    # With shared memory
    with SharedMemoryManager as smm:
        sl = smm.ShareableList(raw_spans)
        with ProcessPoolExecutor(args.cores) as exe:
            data_size = len(sl)
            fs = [exe.submit(task, sl, idx)
                  for idx in range(data_size)]
            for fu in as_completed(fs):
                utils.mergeDict(result_map, fu.result())

    print("saving data...")
    save_data(result_map)


if __name__ == '__main__':
    main()
