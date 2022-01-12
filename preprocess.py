# Kagaya kagaya85@outlook.com
import json
import yaml
import os
import sys
import time
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
import argparse
from tqdm import tqdm
import utils
from typing import List, Callable, Dict, Tuple
from multiprocessing import cpu_count, Manager, current_process
from concurrent.futures import ProcessPoolExecutor, as_completed
import requests
import wordninja
from transformers import AutoTokenizer, AutoConfig, AutoModel
from params import data_path_list, mm_data_path_list, mm_trace_root_list, chaos_dict

data_root = '/data/TraceCluster/raw'

time_now_str = str(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))

# wecath data flag
is_wechat = False
use_request = False
cache_file = './secrets/cache.json'
embedding_name = ''


def normalize(x): return x


def embedding(input: str) -> List[float]:
    return []


# load name cache
cache = {}
mmapis = {}
mm_root_map = {}
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
        self.code = '0'
        self.isError = False

        if raw_span is not None:
            self.spanId = raw_span[ITEM.SPAN_ID]
            self.parentSpanId = raw_span[ITEM.PARENT_SPAN_ID]
            self.traceId = raw_span[ITEM.TRACE_ID]
            self.spanType = raw_span[ITEM.SPAN_TYPE]
            self.startTime = raw_span[ITEM.START_TIME]
            self.duration = raw_span[ITEM.DURATION]
            self.service = str(raw_span[ITEM.SERVICE])
            self.peer = str(raw_span[ITEM.PEER])
            self.operation = str(raw_span[ITEM.OPERATION])
            if ITEM.IS_ERROR in raw_span.keys():
                self.code = str(utils.boolStr2Int(raw_span[ITEM.IS_ERROR]))
                self.isError = utils.any2bool(raw_span[ITEM.IS_ERROR])
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
    parser.add_argument('--max-num', dest='max_num',
                        default=100000, help='max trace number in saved file')
    return parser.parse_args()


def load_span() -> List[DataFrame]:
    """
    load raw sapn data from pathList
    """
    raw_spans = []

    if is_wechat:
        # wechat data

        # load root info
        global mm_root_map
        for path in mm_trace_root_list:
            path = os.path.join(data_root, 'wechat', path)
            clickstreams = pd.read_csv(path)
            for _, root in clickstreams.iterrows():
                mm_root_map[root['GraphIdBase64']] = {
                    'ossid': root['CallerOssID'],
                    'code': root['RetCode'],
                    'start_time': root['TimeStamp'],
                }

        # load trace info
        for filepath in mm_data_path_list:
            filepath = os.path.join(data_root, 'wechat', filepath)
            print(f"load wechat span data from {filepath}")
            if filepath.endswith('.json'):
                with open(filepath, 'r') as f:
                    raw_data = json.load(f)
                mmspans = raw_data['data']
            elif filepath.endswith('.csv'):
                raw_data = pd.read_csv(filepath).drop_duplicates()
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
            for i, s in tqdm(mmspans):
                spans[ITEM.SPAN_ID].append(
                    str(s['CalleeNodeID']) + str(['CalleeOssID']) + str(['CalleeCmdID']))
                spans[ITEM.PARENT_SPAN_ID].append(
                    str(s['CallerNodeID']) + str(s['CallerOssID']) + str(s['CallerCmdID']))
                spans[ITEM.TRACE_ID].append(s['GraphIdBase64'])
                spans[ITEM.SPAN_TYPE].append('EntrySpan')
                spans[ITEM.START_TIME].append(s['TimeStamp'])
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
            filepath = os.path.join(data_root, 'trainticket', filepath)
            print(f"load span data from {filepath}")

            data_type = {ITEM.START_TIME: np.uint64, ITEM.END_TIME: np.uint64}
            spans = pd.read_csv(
                filepath, dtype=data_type
            ).drop_duplicates().dropna()
            spans[ITEM.DURATION] = spans[ITEM.END_TIME] - \
                spans[ITEM.START_TIME]

            raw_spans.extend(data_partition(spans, 10240))

    return raw_spans


def data_partition(data: DataFrame, size: int = 1024) -> List[DataFrame]:
    id_list = data[ITEM.TRACE_ID].unique()
    if len(id_list) < size:
        return [data]

    res = []
    for sub in [id_list[i:i + size] for i in range(0, len(id_list), size)]:
        df = data[data[ITEM.TRACE_ID].isin(sub)]
        res.append(df)

    return res


def build_graph(trace: List[Span], time_normolize: Callable[[float], float], operation_map: dict):
    """
    build trace graph from span list
    """

    trace.sort(key=lambda s: s.startTime)

    if is_wechat:
        graph, str_set = build_mm_graph(trace, time_normolize)
    else:
        graph, str_set = build_sw_graph(trace, time_normolize, operation_map)

    str_set.add('start')
    return graph, str_set


def subspan_info(span: Span, children_span: list[Span]) -> Tuple(int, int, int):
    """
    returns subspan duration, subspan number, is_parallel (0-not parallel, 1-is parallel)
    """
    if len(children_span) == 0:
        return 0, 0
    total_duration = 0
    is_parallel = 0
    time_spans = []
    for child in children_span:
        time_spans.append(
            {"start": child.startTime, "end": child.startTime + child.duration})
    time_spans.sort(key=lambda s: s["start"])
    last_time_span = time_spans[0]
    last_length = -1

    while (len(time_spans) != last_length):
        last_length = len(time_spans)
        for time_span in time_spans:
            if time_span["start"] < last_time_span["end"]:
                if time_span != time_spans[0]:
                    is_parallel = 1
                    time_span["start"] = last_time_span["start"]
                    time_span["end"] = max(
                        time_span["end"], last_time_span["end"])
                    time_spans.remove(last_time_span)
            last_time_span = time_span
    subspanNum = len(time_spans) + 1

    for time_span in time_spans:
        total_duration += time_span["end"] - time_span["start"]
    if time_spans[0]["start"] == span.startTime:
        subspanNum -= 1
    if time_spans[-1]["end"] == span.startTime + span.duration:
        subspanNum -= 1

    return total_duration, subspanNum, is_parallel


def calculate_edge_features(current_span: Span, trace_duration: dict, spanChildrenMap: dict):
    # base features
    features = {
        'spanId': current_span.spanId,
        'parentSpanId': current_span.parentSpanId,
        'startTime': current_span.startTime,
        'rawDuration': current_span.duration,
        'service': current_span.service,
        'operation': current_span.operation,
        'peer': current_span.peer,
        'isError': current_span.isError,
    }

    if spanChildrenMap.get(current_span.spanId) is None:
        features["childrenSpanNum"] = 0
        features["requestDuration"] = 0
        features["responseDuration"] = 0
        features["subspanDuration"] = 0
        features["timeScale"] = round(
            (current_span.duration / (trace_duration["end"] - trace_duration["start"])), 4)
        features["subspanNum"] = 0
        features["requestAndResponseDuration"] = 0
        features["isParallel"] = 0
        features["callType"] = 0 if current_span.spanType == "Entry" else 1
        features["statusCode"] = current_span.code
        return features

    children_span = spanChildrenMap[current_span.spanId]
    request_and_response_duration = 0.0
    request_duration = 0.0
    response_duration = 0.0
    children_duration = 0.0
    subspan_duration = 0.0
    subspan_num = 0.0
    min_time = sys.maxsize - 1
    max_time = -1

    for child in children_span:
        if child.startTime < min_time:
            min_time = child.startTime
        if child.startTime + child.duration > max_time:
            max_time = child.startTime + child.duration
        if child.spanType == "Exit":
            if spanChildrenMap.get(child.spanId) is not None:
                grandChild = spanChildrenMap[child.spanId][0]
                children_duration += grandChild.duration
                request_duration += (grandChild.startTime - child.startTime)
                response_duration += (child.duration -
                                      request_duration - grandChild.duration)
                request_and_response_duration += (
                    child.duration - grandChild.duration)
        if child.spanType == "Producer":
            if spanChildrenMap.get(child.spanId) is not None:
                grandChild = spanChildrenMap[child.spanId][0]
                children_duration += grandChild.duration
                if grandChild.startTime + grandChild.duration > trace_duration["end"]:
                    trace_duration["end"] = grandChild.startTime + \
                        grandChild.duration

    subspan_duration, subspan_num, is_parallel = subspan_info(
        current_span, children_span)

    features["callType"] = 0 if current_span.spanType == "Entry" else 1
    features["isParallel"] = is_parallel
    features["statusCode"] = current_span.code
    features["childrenSpanNum"] = len(children_span)
    features["requestDuration"] = request_duration
    features["responseDuration"] = response_duration
    features["requestAndResponseDuration"] = request_and_response_duration
    features["workDuration"] = current_span.duration - subspan_duration
    features["subspanNum"] = subspan_num
    features["timeScale"] = round(
        (current_span.duration / (trace_duration["end"] - trace_duration["start"])), 4)

    return features


def check_abnormal_span(span: Span) -> bool:
    start_hour = time.localtime(span.startTime).tm_hour

    if start_hour in chaos_dict.keys() and span.service.startswith(chaos_dict.get(start_hour)):
        return True

    return False


def build_sw_graph(trace: List[Span], time_normolize: Callable[[float], float], operation_map: dict):
    vertexs = {0: 'start'}
    edges = {}
    str_set = set()
    trace_duration = {}

    spanIdMap = {'-1': 0}
    spanIdCounter = 1
    rootSpan = None
    spanMap = {}
    spanChildrenMap = {}

    # generate span dict
    for span in trace:
        spanMap[span.spanId] = span
        if span.parentSpanId not in spanChildrenMap.keys():
            spanChildrenMap[span.parentSpanId] = []
        spanChildrenMap[span.parentSpanId].append(span)

    # remove local span
    for span in trace:
        if span.spanType != 'Local':
            continue

        if spanMap.get(span.parentSpanId) is None:
            return None, str_set
        else:
            local_span_children = spanChildrenMap[span.spanId]
            local_span_parent = spanMap[span.parentSpanId]
            spanChildrenMap[local_span_parent.spanId].remove(span)
            for child in local_span_children:
                child.parentSpanId = local_span_parent.spanId
                spanChildrenMap[local_span_parent.spanId].append(child)

    is_abnormal = 0
    # process other span
    for span in trace:
        """
        (raph object contains Vertexs and Edges
        Edge: [(from, to, duration), ...]
        Vertex: [(id, nodestr), ...]
        """

        # skip client span
        if span.spanType in ['Exit', 'Producer', 'Local']:
            continue

        if check_abnormal_span(span):
            is_abnormal = 1

        # get the parent server span id
        if span.parentSpanId == '-1':
            rootSpan = span
            trace_duration["start"] = span.startTime
            trace_duration["end"] = span.startTime + span.duration
            parentSpanId = '-1'
        else:
            if spanMap.get(span.parentSpanId) is None:
                return None, str_set
            parentSpanId = spanMap[span.parentSpanId].parentSpanId

        if parentSpanId not in spanIdMap.keys():
            spanIdMap[parentSpanId] = spanIdCounter
            spanIdCounter += 1

        if span.spanId not in spanIdMap.keys():
            spanIdMap[span.spanId] = spanIdCounter
            spanIdCounter += 1

        vid, pvid = spanIdMap[span.spanId], spanIdMap[parentSpanId]

        # span id should be unique
        if vid not in vertexs.keys():
            opname = '/'.join([span.service, span.operation])
            vertexs[vid] = [span.service, opname]
            str_set.add(span.service)
            str_set.add(opname)

        if pvid not in edges.keys():
            edges[pvid] = []

        # get features of the edge directed to current span
        operation_select_keys = ['childrenSpanNum', 'requestDuration', 'responseDuration',
                                 'requestAndResponseDuration', 'workDuration', 'subspanNum',
                                 'duration', 'rawDuration', 'timeScale']

        feats = calculate_edge_features(
            span, trace_duration, spanChildrenMap)
        feats['vertexId'] = vid
        feats['duration'] = time_normolize(span.duration)

        if span.operation not in operation_map.keys():
            operation_map[span.operation] = {}
            for key in operation_select_keys:
                operation_map[span.operation][key] = []
        for key in operation_select_keys:
            operation_map[span.operation][key].append(feats[key])

        edges[pvid].append(feats)

    if rootSpan == None:
        return None, str_set

    graph = {
        'abnormal': is_abnormal,
        'vertexs': vertexs,
        'edges': edges,
    }

    return graph, str_set


def build_mm_graph(trace: List[Span], time_normolize: Callable[[float], float]):
    traceId = trace[0].traceId

    str_set = set()
    spanId2Idx = {}
    tailIdx = 0
    parentNum = {}
    graph = []
    vertexs = {}
    edges = {}

    for span in trace:
        if span.parentSpanId not in spanId2Idx.keys():
            spanId2Idx[span.parentSpanId] = tailIdx
            graph.append([])
            tailIdx = tailIdx + 1
            parentNum[span.parentSpanId] = 0

        graph[spanId2Idx[span.parentSpanId]].append(span)

        if span.spanId not in spanId2Idx.keys():
            spanId2Idx[span.spanId] = tailIdx
            graph.append([])
            tailIdx = tailIdx + 1
            parentNum[span.spanId] = 0

        parentNum[span.spanId] = parentNum[span.spanId] + 1

    # add root node
    if traceId in mm_root_map.keys():
        root_span_id = '0'
        root_ossid = mm_root_map[traceId]['ossid']
        root_code = mm_root_map[traceId]['code']
        root_start_time = mm_root_map[traceId]['start_time']
        root_service_name = get_service_name(root_ossid)
        if root_service_name == "":
            root_service_name = str(root_ossid)

        # check root number
        root_spans = []
        for spanId, num in parentNum.items():
            if num == 0:
                root_spans.append(spanId)

        if len(root_spans) > 1:
            # add root info
            spanId2Idx[root_span_id] = tailIdx
            graph.append([])
            rspanId = root_span_id
            for spanId in root_spans:
                # add edge
                root_duration = 0
                for span in graph[spanId2Idx[spanId]]:
                    root_duration = root_duration + span.duration

                graph[tailIdx].append(Span({
                    ITEM.TRACE_ID: traceId,
                    ITEM.SPAN_ID: spanId,
                    ITEM.PARENT_SPAN_ID: root_span_id,
                    ITEM.START_TIME: root_start_time,
                    ITEM.DURATION: root_duration,
                    ITEM.SERVICE: "root",
                    ITEM.OPERATION: "start",
                    ITEM.SPAN_TYPE: 'EntrySpan',
                    ITEM.PEER: "{}/{}".format(root_service_name, "start"),
                    ITEM.CODE: 0,
                    ITEM.IS_ERROR: False,
                }))
        else:
            rspanId = root_spans[0]

    vertexs[spanId2Idx[rspanId]] = [root_service_name, "start"]
    str_set.add(root_service_name)

    for idx, spans in enumerate(graph):
        if len(spans) == 0:
            continue

        edges[idx] = []
        for span in spans:
            vertexs[spanId2Idx[span.spanId]] = [span.service, span.peer]
            str_set.add(span.service)
            str_set.add(span.peer)
            edges[idx].append({
                'vertexId': spanId2Idx[span.spanId],
                'parentSpanId': span.parentSpanId,
                'spanId': span.spanId,
                'startTime': span.startTime,
                'duration': time_normolize(span.duration),
                'service': span.service,
                'operation': span.operation,
                'peer': span.peer,
                'isError': utils.any2bool(root_code),
            })

    graph = {
        'vertexs': vertexs,
        'edges': edges,
    }

    return graph, str_set


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


def save_data(graphs: Dict, idx: str = ''):
    """
    save graph data to json file
    """
    filepath = generate_save_filepath(idx)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    print("saving data..., map size: {}".format(sys.getsizeof(graphs)))
    with open(filepath, 'w', encoding='utf-8') as fd:
        json.dump(graphs, fd, ensure_ascii=False)

    print(f"{len(graphs)} traces data saved in {filepath}")


def generate_save_filepath(name: str) -> str:
    filename = embedding_name + '_' + time_now_str + '/' + name + '.json'

    if is_wechat:
        filepath = os.path.join(os.getcwd(), 'data',
                                'preprocessed', 'wechat', filename)
    else:
        filepath = os.path.join(os.getcwd(), 'data',
                                'preprocessed', 'trainticket', filename)

    return filepath


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
                    return str(name)
                # not found
                cache['cmd_name'][module_name][cmdid] = str(cmdid)
                return cmdid
            print(f'cant get operation name, code:', rsp.status_code)

    return str(cmdid)


def get_service_name(ossid: int) -> str:
    global cache

    if ossid in cache['oss_name'].keys():
        return str(cache['oss_name'][ossid])

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
                    name = str(datas[0]['module_name'])
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
        model_name, do_lower_case=True, cache_dir='./data/cache'
    )

    model = AutoModel.from_pretrained(
        model_name, output_hidden_states=True, cache_dir='./data/cache'
    )

    def bert(input: str) -> List[float]:
        inputs = tokenizer(
            input, padding='max_length', max_length=100, return_tensors="pt")

        outputs = model(**inputs)

        return outputs.pooler_output.tolist()[0]

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


def task(ns, idx, divide_word: bool = True):
    span_data = ns.sl[idx]
    current = current_process()
    pos = current._identity[0] - 1
    graph_map = {}
    str_set = set()
    operation_map = {}
    for trace_id, trace_data in tqdm(span_data.groupby([ITEM.TRACE_ID]), desc="processing #{:0>2d}".format(idx),
                                     position=pos):
        trace = [Span(raw_span) for idx, raw_span in trace_data.iterrows()]
        graph, sset = build_graph(trace_process(
            trace, divide_word), normalize, operation_map)
        if graph == None:
            continue
        graph_map[trace_id] = graph
        str_set = set.union(str_set, sset)

    return (graph_map, str_set, operation_map)


def main():
    args = arguments()
    global is_wechat, use_request, embedding_name
    is_wechat = args.wechat
    use_request = args.use_request
    embedding_name = args.embedding

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

        def normalize(x):
            return min_max(
                x, max_duration, min_duration)

    elif args.normalize == 'zscore':
        mean_duration = span_data[ITEM.DURATION].mean()
        std_duration = span_data[ITEM.DURATION].std()

        def normalize(x):
            return z_score(
                x, mean_duration, std_duration)

    else:
        print(f"invalid normalize method name: {embedding_name}")
        exit()
    del span_data

    global embedding
    if embedding_name == 'glove':
        embedding = glove_embedding()
        enable_word_division = True
    elif embedding_name == 'bert':
        embedding = bert_embedding()
        enable_word_division = False
    else:
        print(f"invalid embedding method name: {embedding_name}")
        exit()

    result_map = {}
    operation_map = {}
    name_set = set()
    file_idx = 0

    # With shared memory
    with Manager() as m:
        ns = m.Namespace()
        ns.sl = raw_spans
        with ProcessPoolExecutor(args.cores) as exe:
            data_size = len(raw_spans)
            fs = [exe.submit(task, ns, idx, enable_word_division)
                  for idx in range(data_size)]
            for fu in as_completed(fs):
                (graphs, sset, temp_operation_map) = fu.result()
                result_map = utils.mergeDict(result_map, graphs)
                operation_map = utils.mergeOperation(
                    temp_operation_map, operation_map)
                name_set = set.union(name_set, sset)
                # control the data size
                if len(result_map) > args.max_num:
                    save_data(result_map, str(file_idx))
                    file_idx = file_idx + 1
                    result_map = {}

    if len(result_map) > 0:
        save_data(result_map, str(file_idx))

    print('start generate embedding file')
    name_dict = {}
    for name in tqdm(name_set):
        name_dict[name] = embedding(name)

    embd_filepath = generate_save_filepath('embedding')
    with open(embd_filepath, 'w', encoding='utf-8') as fd:
        json.dump(name_dict, fd, ensure_ascii=False)
    print(f'embedding data saved in {embd_filepath}')

    operation_filepath = generate_save_filepath('operations')
    with open(operation_filepath, 'w', encoding='utf-8') as fo:
        json.dump(operation_map, fo, ensure_ascii=False)
    print(f'operations data saved in {operation_filepath}')

    print('preprocess finished :)')


if __name__ == '__main__':
    main()
