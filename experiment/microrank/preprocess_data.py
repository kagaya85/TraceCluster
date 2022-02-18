import sys
from typing import List, Callable, Dict
from pandas.core.frame import DataFrame
import pandas as pd
from copy import deepcopy
import numpy as np
import time
sys.path.append("../..")
from preprocess import Span, load_mm_span

root_index = '-1'

span_list = []
mm_root_map = {}
is_wechat = True


def fix_root(span_list: List[Span], mm_root_map: dict):
    cur_tid = ''
    has_root = False
    spans = []
    for span in span_list:
        if span.traceId != cur_tid:
            cur_tid = span.traceId
            if has_root == False:
                root = mm_root_map[cur_tid]
                root_span = Span()
                root_span.spanId = root['spanid']
                root_span.parentSpanId = '-1'
                root_span.traceId = cur_tid
                root_span.spanType = 'Entry'
                root_span.startTime = root['start_time']
                root_span.duration = root['cost_time']
                root_span.service = root['ossid']
                root_span.operation = root['cmdid']
                root_span.code = root['code']
                spans.append(root_span)

            has_root = False

        spans.append(span)        
    return spans


def get_span() -> List[Span]:
    clickstream_list = [
        'trace_mmfindersynclogicsvr/click_stream_2022-01-17_23629.csv',
        'trace_mmfindersynclogicsvr/click_stream_2022-01-18_23629.csv',
        ]
    callgraph_list = [
        'trace_mmfindersynclogicsvr/tmp17.csv',
        'trace_mmfindersynclogicsvr/tmp18.csv',
        ]

    global mm_root_map, span_list
    if len(span_list) == 0:
        mm_root_map, span_data = load_mm_span(clickstream_list, callgraph_list)
        span_data = pd.concat(span_data, axis=0, ignore_index=True)
        span_data = span_data.groupby('TraceId').apply(lambda x:x.sort_values('StartTime', ascending=True)).reset_index(drop=True)
        span_list = [Span(raw_span) for _, raw_span in span_data.iterrows()]
        if is_wechat:
            span_list = fix_root(span_list, mm_root_map)

    return span_list


def get_normal_span() -> List[Span]:
    clickstream_list = ['trace_mmfindersynclogicsvr/click_stream_2022-01-17_23629.csv']
    callgraph_list = ['trace_mmfindersynclogicsvr/tmp17.csv']

    print('load normal span...')
    root_map, span_data = load_mm_span(clickstream_list, callgraph_list)
    span_data = pd.concat(span_data, axis=0, ignore_index=True)
    span_data = span_data.groupby('TraceId').apply(lambda x:x.sort_values('StartTime', ascending=True)).reset_index(drop=True)
    spans = [Span(raw_span) for _, raw_span in span_data.iterrows()] 
    spans = fix_root(spans, root_map)
    return spans


def is_root_span(span: Span):
    return span.parentSpanId == '-1'

'''
  Query all the service_operation from the input span_list
  :arg
     span_list: should be a long time span_list to get all operation
  :return
       the operation list and operation list dict
'''


def get_service_operation_list(span_list: List[Span]) -> List[str]:
    operation_list = []

    for span in span_list:
        operation = span.operation
        if operation not in operation_list:
            operation_list.append(operation)

    return operation_list


"""
   Calculate the mean of duration and variance for each span_list 
   :arg
       operation_list: contains all operation
       span_list: should be a long time span_list
   :return
       operation dict of the mean of and variance 
       {
           # operation: {mean, variance}
           "Currencyservice_Convert": [600, 3]}
       }   
"""


def get_operation_slo(service_operation_list: List[str], span_list: List[Span]):
    template = {
        'parent': '',  # parent span
        'operation': '',  # current servicename_operation
        'duration': 0  # duration of current operation
    }

    traceid = span_list[0].traceId
    filter_data = {}
    temp = {}
    normal_trace = True
    root_id = ''

    def check_filter_data():
        for spanid in temp:
            if temp[spanid]['parent'] == root_index:
                if temp[spanid]['duration'] > 1000000:
                    print("filter data because duration > 1000ms")
                    print(temp)
                    return False
        return True

    for span in span_list:
        if traceid == span.traceId:
            spanid = span.spanId
            temp[spanid] = deepcopy(template)
            temp[spanid]['duration'] = span.duration
            temp[spanid]['operation'] = span.operation

            if is_root_span(span):
                temp[spanid]['parent'] = root_index
                root_id = spanid
            else:
                parentId = span.parentSpanId
                if parentId not in temp:
                    parentId = root_id

                temp[spanid]['parent'] = parentId

                if parentId in temp:
                    temp[parentId]['duration'] -= temp[spanid]['duration']
                else:
                    normal_trace = False

        elif traceid != span.spanId and len(temp) > 0:
            if check_filter_data() and normal_trace:
                filter_data[traceid] = temp

            traceid = span.traceId
            normal_trace = True
            spanid = span.spanId
            temp = {}
            temp[spanid] = deepcopy(template)
            temp[spanid]['duration'] = span.duration
            temp[spanid]['operation'] = span.operation
            if is_root_span(span):
                temp[spanid]['parent'] = root_index
                root_id = spanid
            else:
                parentId = span.parentSpanId
                if parentId not in temp:
                    parentId = root_id
                temp[spanid]['parent'] = parentId
                if parentId in temp:
                    temp[parentId]['duration'] -= temp[spanid]['duration']
                else:
                    normal_trace = False

    # The last trace
    if len(temp) > 1:
        if check_filter_data() and normal_trace:
            filter_data[traceid] = temp

    duration_dict = {}
    """
    {'frontend_Recv.': [1961, 1934, 1316, 1415, 1546, 1670, 1357, 2099, 2789, 1832, 1270, 1242, 2230, 1386],
      'recommendationservice_ListProducts': [3576, 7127, 4387, 19657, 5158, 4563, 4167, 8822, 4507],
    """
    for operation in service_operation_list:
        duration_dict[operation] = []

    for traceid in filter_data:
        single_trace = filter_data[traceid]

        for spanid in single_trace:
            duration_dict[single_trace[spanid]['operation']].append(
                single_trace[spanid]['duration'])

    operation_slo = {}
    """
    {'frontend_Recv.': [2.903, 10.0949], 'frontend_GetSupportedCurrencies': [8.1019, 16.2973], }
    """
    for operation in service_operation_list:
        operation_slo[operation] = []

    for operation in service_operation_list:
        operation_slo[operation].append(
            round(np.mean(duration_dict[operation]) / 1000.0, 4))
        #operation_slo[operation].append(round(np.percentile(duration_dict[operation], 90) / 1000.0, 4))
        operation_slo[operation].append(
            round(np.std(duration_dict[operation]) / 1000.0, 4))

    return operation_slo


'''
   Query the operation and duration in span_list for anormaly detector 
   :arg
       operation_list: contains all operation
       operation_dict:  { "operation1": 1, "operation2":2 ... "operationn": 0, "duration": 666}
       span_list: all the span_list in one anomaly detection interval (1 min or 30s)
   :return
       { 
          traceid: {
              operation1: 1
              operation2: 2
          }
       }
'''


def get_operation_duration_data(operation_list: List[str], span_list: List[Span]):
    operation_dict = {}
    trace_id = span_list[0].traceId

    def init_dict(trace_id):
        if trace_id not in operation_dict:
            operation_dict[trace_id] = {}
            for operation in operation_list:
                operation_dict[trace_id][operation] = 0
            operation_dict[trace_id]['duration'] = 0

    for span in span_list:
        operation_name = span.operation

        init_dict(span.traceId)

        if trace_id == span.traceId:
            operation_dict[trace_id][operation_name] += 1

            if is_root_span(span):
                operation_dict[trace_id]['duration'] += span.duration

        else:
            trace_id = span.traceId
            operation_dict[trace_id][operation_name] += 1

            if is_root_span(span):
                operation_dict[trace_id]['duration'] += span.duration

    return operation_dict


'''
   Query the pagerank graph
   :arg
       trace_list: anormaly_traceid_list or normaly_traceid_list
       span_list:  异常点前后两分钟 span_list
   
   :return
       operation_operation 存储子节点 Call graph
       operation_operation[operation_name] = [operation_name1 , operation_name1 ] 

       operation_trace 存储trace经过了哪些operation, 右上角 coverage graph
       operation_trace[traceid] = [operation_name1 , operation_name2]

       trace_operation 存储 operation被哪些trace 访问过, 左下角 coverage graph
       trace_operation[operation_name] = [traceid1, traceid2]  
       
       pr_trace: 存储trace id 经过了哪些operation，不去重
       pr_trace[traceid] = [operation_name1 , operation_name2]
'''


def get_pagerank_graph(trace_list: List[str], span_list: List[Span]):
    template = {
        'parent': '',  # parent span
        'operation': '',  # current servicename_operation
    }

    if len(trace_list) > 0:
        traceid = trace_list[0]
    else:
        traceid = span_list[0].traceId
    filter_data = {}
    temp = {}

    operation_operation = {}
    operation_trace = {}
    trace_operation = {}
    pr_trace = {}

    for span in span_list:
        operation_name = span.operation
        if span.traceId in trace_list:
            if traceid == span.traceId:
                spanid = span.spanId
                temp[spanid] = deepcopy(template)
                temp[spanid]['operation'] = span.operation
                temp[spanid]['parent'] = span.parentSpanId

            elif traceid != span.spanId and len(temp) > 0:
                filter_data[traceid] = temp

                traceid = span.traceId
                spanid = span.spanId
                temp = {}
                temp[spanid] = deepcopy(template)
                temp[spanid]['operation'] = span.operation
                temp[spanid]['parent'] = span.parentSpanId

            if len(temp) > 1:
                filter_data[traceid] = temp

            """
            operation_operation 
            operation_operation[operation_name] = [operation_name1 , operation_name1 ] 

            operation_trace
            operation_trace[traceid] = [operation_name1 , operation_name1]

            trace_operation
            trace_operation[operation_name] = [traceid1, traceid2]
            """
            if operation_name not in operation_operation:
                operation_operation[operation_name] = []
                trace_operation[operation_name] = []

            if span.traceId not in operation_trace:
                operation_trace[span.traceId] = []
                pr_trace[span.traceId] = []

            pr_trace[span.traceId].append(operation_name)

            if operation_name not in operation_trace[span.traceId]:
                operation_trace[span.traceId].append(operation_name)
            if span.traceId not in trace_operation[operation_name]:
                trace_operation[operation_name].append(span.traceId)

    for traceid in filter_data:
        single_trace = filter_data[traceid]
        if traceid in trace_list:
            for spanid in single_trace:
                parent_id = single_trace[spanid]["parent"]
                if parent_id != "":
                    if parent_id not in single_trace:
                        continue
                    if single_trace[spanid]["operation"] not in operation_operation[
                            single_trace[parent_id]["operation"]]:
                        operation_operation[single_trace[parent_id]["operation"]].append(
                            single_trace[spanid]["operation"])

    return operation_operation, operation_trace, trace_operation, pr_trace


if __name__ == '__main__':
    def timestamp(datetime):
        timeArray = time.strptime(datetime, "%Y-%m-%d %H:%M:%S")
        ts = int(time.mktime(timeArray)) * 1000
        # print(ts)
        return ts

    start = '2020-08-28 14:56:43'
    end = '2020-08-28 14:57:44'

    span_list = get_span(start=timestamp(start), end=timestamp(end))
    # print(span_list)
    operation_list = get_service_operation_list(span_list)
    print(operation_list)
    operation_slo = get_operation_slo(operation_list, span_list)
    print(operation_slo)
