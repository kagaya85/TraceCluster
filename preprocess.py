# Kagaya kagaya85@outlook.com
import json
import yaml
import os
from sys import getsizeof
import time
import pandas as pd
from pandas.core.frame import DataFrame
from datetime import datetime
import numpy as np
import argparse
from tqdm import tqdm
import utils
from typing import List, Callable, Dict
from multiprocessing import cpu_count, Manager, current_process
from concurrent.futures import ProcessPoolExecutor, as_completed
import requests
import wordninja
from transformers import AutoTokenizer

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

    # F05
    "data/raw/F05-02/ERROR_errorf0502_SpanData2021-08-10_13-53-38.csv",
    "data/raw/F05-03/ERROR_SpanData2021-08-07_20-34-09.csv",
    "data/raw/F05-04/ERROR_SpanData2021-08-07_21-02-22.csv",
    "data/raw/F05-05/ERROR_SpanData2021-08-07_21-28-23.csv",

    # F07
    "data/raw/F07-01/back0729/ERROR_SpanData2021-07-29_10-36-21.csv",
    "data/raw/F07-01/back0729/SUCCESS_SpanData2021-07-29_10-38-09.csv",
    "data/raw/F07-01/ERROR_errorf0701_SpanData2021-08-10_14-09-59.csv",
    "data/raw/F07-02/back0729/ERROR_SpanData2021-07-29_13-58-37.csv",
    "data/raw/F07-02/back0729/SUCCESS_SpanData2021-07-29_13-51-48.csv",
    "data/raw/F07-02/ERROR_errorf0702_SpanData2021-08-10_14-33-35.csv",
    "data/raw/F07-03/ERROR_SpanData2021-08-07_22-53-33.csv",
    "data/raw/F07-04/ERROR_SpanData2021-08-07_23-49-11.csv",
    "data/raw/F07-05/ERROR_SpanData2021-08-07_23-57-44.csv",

    # F08
    "data/raw/F08-01/ERROR_SpanData2021-07-29_19-15-36.csv",
    "data/raw/F08-01/SUCCESS_SpanData2021-07-29_19-16-01.csv",
    "data/raw/F08-02/ERROR_SpanData2021-07-30_10-13-04.csv",
    "data/raw/F08-02/SUCCESS_SpanData2021-07-30_10-13-46.csv",
    "data/raw/F08-03/ERROR_SpanData2021-07-30_12-07-36.csv",
    "data/raw/F08-03/SUCCESS_SpanData2021-07-30_12-07-23.csv",
    "data/raw/F08-04/ERROR_SpanData2021-07-30_14-20-15.csv",
    "data/raw/F08-04/SUCCESS_SpanData2021-07-30_14-22-24.csv",
    "data/raw/F08-05/ERROR_SpanData2021-07-30_11-00-30.csv",
    "data/raw/F08-05/SUCCESS_SpanData2021-07-30_11-01-05.csv",

    # F11
    "data/raw/F11-01/SUCCESSF1101_SpanData2021-08-14_10-18-35.csv",
    "data/raw/F11-02/SUCCESSerrorf1102_SpanData2021-08-16_16-57-36.csv",
    "data/raw/F11-03/SUCCESSF1103_SpanData2021-08-14_03-04-11.csv",
    "data/raw/F11-04/SUCCESSF1104_SpanData2021-08-14_03-35-38.csv",
    "data/raw/F11-05/SUCCESSF1105_SpanData2021-08-14_03-38-35.csv",

    # F12
    "data/raw/F12-01/ERROR_SpanData2021-08-12_16-17-46.csv",
    "data/raw/F12-02/ERROR_SpanData2021-08-12_16-24-54.csv",
    "data/raw/F12-03/ERROR_SpanData2021-08-12_16-36-33.csv",
    "data/raw/F12-04/ERROR_SpanData2021-08-12_17-04-34.csv",
    "data/raw/F12-05/ERROR_SpanData2021-08-12_16-49-08.csv",

    # F13
    "data/raw/F13-01/SUCCESSerrorf1301_SpanData2021-08-16_21-01-36.csv",
    "data/raw/F13-02/SUCCESS_SpanData2021-08-13_17-34-58.csv",
    "data/raw/F13-03/SUCCESSerrorf1303_SpanData2021-08-16_18-55-52.csv",
    "data/raw/F13-04/SUCCESSF1304_SpanData2021-08-14_10-50-42.csv",
    "data/raw/F13-05/SUCCESSF1305_SpanData2021-08-14_11-13-43.csv",

    # F14
    "data/raw/F14-01/SUCCESS_SpanData2021-08-12_14-56-41.csv",
    "data/raw/F14-02/SUCCESS_SpanData2021-08-12_15-24-50.csv",
    "data/raw/F14-03/SUCCESS_SpanData2021-08-12_15-46-08.csv",

    # F23
    "data/raw/F23-01/ERROR_SpanData2021-08-07_20-30-26.csv",
    "data/raw/F23-02/ERROR_SpanData2021-08-07_20-51-14.csv",
    "data/raw/F23-03/ERROR_SpanData2021-08-07_21-10-11.csv",
    "data/raw/F23-04/ERROR_SpanData2021-08-07_21-34-47.csv",
    "data/raw/F23-05/ERROR_SpanData2021-08-07_22-02-42.csv",

    # F24
    "data/raw/F24-01/ERROR_SpanData.csv",
    "data/raw/F24-02/ERROR_SpanData.csv",
    "data/raw/F24-03/ERROR_SpanData.csv",

    # F25
    "data/raw/F25-01/ERROR_SpanData2021-08-16_11-17-21.csv",
    "data/raw/F25-02/ERROR_SpanData2021-08-16_11-21-59.csv",
    "data/raw/F25-03/ERROR_SpanData2021-08-16_12-20-59.csv",
]

mm_data_path_list = [
    # 'data/raw/wechat/5-18/finer_data.json',
    # 'data/raw/wechat/5-18/finer_data2.json',
    # 'data/raw/wechat/8-2/data.json',
    # 'data/raw/wechat/8-3/data.json',
    # 'data/raw/wechat/11-9/data.json',
    'data/raw/wecaht/11-18/call_graph_2021-11-18_61266.csv'
]

time_now_str = str(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))

# wecath data flag
is_wechat = False
use_request = False
cache_file = './secrets/cache.json'


def normalize(x): return x


def embedding(input: str) -> List[float]:
    return []


# load name cache
cache = {}
mmapis = {}
service_url = ""
operation_url = ""
sn = ""


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

        if raw_span is not None:
            self.spanId = raw_span[ITEM.SPAN_ID]
            self.parentSpanId = raw_span[ITEM.PARENT_SPAN_ID]
            self.traceId = raw_span[ITEM.TRACE_ID]
            self.spanType = raw_span[ITEM.SPAN_TYPE]
            self.startTime = raw_span[ITEM.START_TIME]
            self.duration = raw_span[ITEM.DURATION]
            self.service = raw_span[ITEM.SERVICE]
            self.peer = raw_span[ITEM.PEER]
            self.operation = raw_span[ITEM.OPERATION]
            if ITEM.IS_ERROR in raw_span.keys():
                self.code = str(utils.boolStr2Int(raw_span[ITEM.IS_ERROR]))
                self.isError = utils.boolStr2Bool(raw_span[ITEM.IS_ERROR])
            if ITEM.CODE in raw_span.keys():
                self.code = str(raw_span[ITEM.CODE])


def arguments():
    parser = argparse.ArgumentParser(description="Preporcess Argumentes.")
    parser.add_argument('--cores', dest='cores',
                        help='parallel processing core numberes', default=cpu_count())
    parser.add_argument('--wechat', help='use wechat data',
                        action='store_true')
    parser.add_argument('--use-request', dest='use_request', help='use http request when replace id to name',
                        action='store_true')
    parser.add_argument('--normalize', dest='normalize',
                        help='normarlize method [zscore/minmax]', default='minmax')
    parser.add_argument('--embedding', dest='embedding',
                        help='word embedding method [bert/glove]', default='bert')
    return parser.parse_args()


def load_span() -> List[DataFrame]:
    """
    load raw sapn data from pathList
    """
    raw_spans = []

    if is_wechat:
        # wechat data
        for filepath in mm_data_path_list:
            print(f"load wechat span data from {filepath}")
            if filepath.endswith('.json'):
                with open(filepath, 'r') as f:
                    raw_data = json.load(f)
                mmspans = raw_data['data']
            elif filepath.endswith('.csv'):
                raw_data = pd.read_csv(filepath).drop_duplicates().dropna()
                mmspans = raw_data.iterrows()
            else:
                print(f'invalid file type: {filepath}')
                continue

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
            for s in tqdm(mmspans):
                spans[ITEM.SPAN_ID].append(str(s['CalleeNodeID']))
                spans[ITEM.PARENT_SPAN_ID].append(str(s['CallerNodeID']))
                spans[ITEM.TRACE_ID].append(s['GraphIdBase64'])
                spans[ITEM.SPAN_TYPE].append('EntrySpan')
                st = datetime.strptime(s['TimeStamp'], '%Y-%m-%d %H:%M:%S')
                spans[ITEM.START_TIME].append(int(datetime.timestamp(st)))
                spans[ITEM.DURATION].append(int(s['CostTime']))

                # 尝试替换id为name
                service_name = get_service_name(s['CalleeOssID'])
                if service_name == "":
                    spans[ITEM.SERVICE].append(str(s['CalleeOssID']))
                else:
                    spans[ITEM.SERVICE].append(service_name)

                spans[ITEM.OPERATION].append(
                    get_operation_name(s['CalleeCmdID'], service_name))

                peer_service_name = get_service_name(s['CallerOssID'])
                peer_cmd_name = get_operation_name(
                    s['CallerCmdID'], peer_service_name)

                if peer_service_name == "":
                    spans[ITEM.PEER].append(
                        '/'.join([str(s['CallerOssID']), peer_cmd_name]))
                else:
                    spans[ITEM.PEER].append(
                        '/'.join([peer_service_name, peer_cmd_name]))

                error_code = s['NetworkRet'] if s['NetworkRet'] != 0 else s['ServiceRet']
                spans[ITEM.IS_ERROR].append(utils.int2Bool(error_code))
                spans[ITEM.CODE].append(str(error_code))

            df = DataFrame(spans)
            raw_spans.extend(data_partition(df))

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

            raw_spans.extend(data_partition(spans))

    return raw_spans


def convert


def data_partition(data: DataFrame, size: int = 1024) -> List[DataFrame]:
    id_list = data[ITEM.TRACE_ID].unique()
    if len(id_list) < size:
        return [data]

    res = []
    for sub in [id_list[i:i+size] for i in range(0, len(id_list), size)]:
        df = data[data[ITEM.TRACE_ID].isin(sub)]
        res.append(df)

    return res


def build_graph(trace: List[Span], time_normolize: Callable[[float], float]) -> dict:
    """
    build trace graph from span list
    """

    trace.sort(key=lambda s: s.startTime)

    if is_wechat:
        return build_mm_graph(trace, time_normolize)

    return build_sw_graph(trace, time_normolize)


def build_sw_graph(trace: List[Span], time_normolize: Callable[[float], float]) -> dict:
    vertexs = {0: embedding('start')}
    edges = {}

    spanIdMap = {'-1': 0}
    spanIdCounter = 1
    rootSpan = None

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
            'parentSpanId': span.parentSpanId,
            'startTime': span.startTime,
            'duration': time_normolize(span.duration),
            'service': span.service,
            'operation': span.operation,
            'peer': span.peer,
            'isError': span.isError,
        })

    if rootSpan == None:
        return None

    graph = {
        'vertexs': vertexs,
        'edges': edges,
    }

    return graph


def build_mm_graph(trace: List[Span], time_normolize: Callable[[float], float]) -> dict:
    vertexs = {0: embedding('start')}
    edges = {0: []}

    spanIdMap = {}
    spanIdCounter = 1

    is_error = False
    tot_duration = 0
    for span in trace:
        """
        (raph object contains Vertexs and Edges
        Edge: [(from, to, duration), ...]
        Vertex: [(id, nodestr), ...]
        """

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

        # statistics
        if span.isError:
            is_error = True
        tot_duration = tot_duration + span.duration

        edges[parentSpanId].append({
            'vertexId': spanId,
            'parentSpanId': span.parentSpanId,
            'spanId': span.spanId,
            'startTime': span.startTime,
            'duration': time_normolize(span.duration),
            'service': span.service,
            'operation': span.operation,
            'peer': span.peer,
            'isError': span.isError,
        })

    # graph check，如果图是连通的，edge的起始点中应该只有一个不存在在vertexs集合里
    isolate_point_id = 0
    ipc = 0
    for id in edges:
        if id not in vertexs.keys():
            isolate_point_id = id
            ipc = ipc + 1
            # service = e['peer']
            # operation = cache['cmd_name'][service][e['parentSpanId']]
            # vertexs[id] = embedding('/'.join([service, operation, '0']))

    # if ipc != 1:
    #     return None

    edges[0] = edges.pop(isolate_point_id)

    graph = {
        'vertexs': vertexs,
        'edges': edges,
    }

    return graph


def get_mmapi() -> dict:
    api_file = './secrets/api.yaml'
    print(f"read api url from {api_file}")

    with open(api_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    return data


def load_name_cache() -> dict:
    with open(cache_file, 'r') as f:
        cache = json.load(f)
        print(f"load cache from {cache_file}")

    return cache


def save_data(graphs: Dict, embedding: str):
    """
    save graph data to json file
    """

    name = embedding + '_' + time_now_str+'.json'

    if is_wechat:
        filename = os.path.join(os.getcwd(), 'data',
                                'preprocessed', 'wechat', name)
    else:
        filename = os.path.join(os.getcwd(), 'data',
                                'preprocessed', name)
    print(f'prepare saving to {filename}')
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w', encoding='utf-8') as fd:
        json.dump(graphs, fd, ensure_ascii=False)

    print(f"data saved in {filename}")


def divide_word(s: str, sep: str = "/") -> str:
    if is_wechat:
        return sep.join(wordninja.split(s))

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

    return sep.join(word_list)


def trace_process(trace: List[Span], enable_word_division: bool) -> List[Span]:
    operationMap = {}
    for span in trace:
        if enable_word_division:
            span.service = divide_word(span.service)
            span.operation = divide_word(span.operation)
        if span.spanType == "Entry":
            operationMap[span.parentSpanId] = span.operation

    for span in trace:
        # 替换Exit span的URL
        if span.spanType == "Exit" and span.spanId in operationMap.keys():
            span.operation = operationMap[span.spanId]

    return trace


def get_operation_name(cmdid: int, module_name: str) -> str:
    global cache

    if module_name == "":
        return str(cmdid)

    if module_name not in cache['cmd_name'].keys():
        cache['cmd_name'][module_name] = {}

    if cmdid in cache['cmd_name'][module_name].keys():
        return cache['cmd_name'][module_name][cmdid]

    if use_request:
        params = {
            'sn': sn,
            'fields': 'interface_id,name,module_id,module_name,interface_id',
            'page': 1,
            'page_size': 1000,
            'where_module_name': module_name,
            'where_interface_id': cmdid,
        }

        try:
            rsp = requests.get(operation_url, timeout=10, params=params)
        except Exception as e:
            print(f"get operation name from cmdb failed:", e)
        else:
            if rsp.ok:
                datas = rsp.json()['data']
                if len(datas) > 0:
                    name = datas[0]['name']
                    cache['cmd_name'][module_name][cmdid] = name
                    return name
                # not found
                cache['cmd_name'][module_name][cmdid] = str(cmdid)
                return str(cmdid)
            print(f'cant get operation name, code:', rsp.status_code)

    return str(cmdid)


def get_service_name(ossid: int) -> str:
    global cache

    if ossid in cache['oss_name'].keys():
        return cache['oss_name'][ossid]

    if use_request:
        params = {
            'sn': sn,
            'fields': 'module_name,ossid,module_id',
            'where_ossid': ossid,
        }

        try:
            rsp = requests.get(service_url, timeout=10, params=params)
        except Exception as e:
            print(f"get service name from cmdb failed:", e)
        else:
            if rsp.ok:
                datas = rsp.json()['data']
                if len(datas) > 0:
                    name = datas[0]['module_name']
                    cache['oss_name'][ossid] = name
                    return name
                # not found
                cache['oss_name'][ossid] = str(ossid)
                return ""
            print(f'cant get name, code:', rsp.status_code)

    return ""


def save_name_cache(cache: dict):
    with open(cache_file, 'w') as f:
        json.dump(cache, f, indent=4)
        print('save cache success')


def glove_embedding() -> Callable[[str], List[float]]:
    embedding_word_list = np.load('./data/glove/wordsList.npy').tolist()
    embedding_word_vector = np.load('./data/glove/wordVectors.npy')

    def glove(input: str) -> List[float]:
        words = input.split('/')
        vec_sum = []
        for w in words:
            if w in embedding_word_list:
                idx = embedding_word_list.index(w)
                vec = embedding_word_vector[idx]
                vec_sum.append(vec)

        return np.mean(np.array(vec_sum), axis=0).tolist()

    return glove


def bert_embedding() -> Callable[[str], List[float]]:
    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, do_lower_case=True, cache_dir='./data/cache')

    def bert(input: str) -> List[float]:
        encoded = tokenizer(
            input, padding='max_length', max_length=100)

        return encoded['input_ids']

    return bert


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


def task(ns, idx, divide_word: bool = True) -> dict:
    span_data = ns.sl[idx]
    current = current_process()
    pos = current._identity[0]-1
    graph_map = {}
    for trace_id, trace_data in tqdm(span_data.groupby([ITEM.TRACE_ID]), desc="processing #{:0>2d}".format(idx), position=pos):
        trace = [Span(raw_span) for idx, raw_span in trace_data.iterrows()]
        graph = build_graph(trace_process(trace, divide_word), normalize)
        if graph == None:
            continue
        graph_map[trace_id] = graph

    return graph_map


def main():
    args = arguments()
    global is_wechat, use_request
    is_wechat = args.wechat
    use_request = args.use_request

    if is_wechat:
        global cache, mmapis, service_url, operation_url, sn
        cache = load_name_cache()
        if use_request:
            mmapis = get_mmapi()
            service_url = mmapis['api']['getApps']
            operation_url = mmapis['api']['getModuleInterface']
            sn = mmapis['sn']

    print(f"parallel processing number: {args.cores}")

    # load all span
    raw_spans = load_span()

    if is_wechat and use_request:
        save_name_cache(cache)

    # concat all span data in one list
    span_data = pd.concat(raw_spans, axis=0, ignore_index=True)
    global normalize
    if args.normalize == 'minmax':
        max_duration = span_data[ITEM.DURATION].max()
        min_duration = span_data[ITEM.DURATION].min()

        def normalize(x): return min_max(
            x, max_duration, min_duration)

    elif args.normalize == 'zscore':
        mean_duration = span_data[ITEM.DURATION].mean()
        std_duration = span_data[ITEM.DURATION].std()

        def normalize(x): return z_score(
            x, mean_duration, std_duration)

    else:
        print(f"invalid normalize method name: {args.embedding}")
        exit()
    del span_data

    global embedding
    if args.embedding == 'glove':
        embedding = glove_embedding()
        enable_word_division = True
    elif args.embedding == 'bert':
        embedding = bert_embedding()
        enable_word_division = False
    else:
        print(f"invalid embedding method name: {args.embedding}")
        exit()

    result_map = {}

    # With shared memory
    with Manager() as m:
        ns = m.Namespace()
        ns.sl = raw_spans
        with ProcessPoolExecutor(args.cores) as exe:
            data_size = len(raw_spans)
            fs = [exe.submit(task, ns, idx, enable_word_division)
                  for idx in range(data_size)]
            for fu in as_completed(fs):
                result_map = utils.mergeDict(result_map, fu.result())

    print("saving data..., map size: {}".format(getsizeof(result_map)))
    save_data(result_map, args.embedding)


if __name__ == '__main__':
    main()
