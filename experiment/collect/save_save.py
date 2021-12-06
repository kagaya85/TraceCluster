import requests
import csv
import time
from tqdm import tqdm


def time_check(time):
    hour = time // 10000
    minute = (time - hour * 10000) // 100
    second = time - hour * 10000 - minute * 100

    if second >= 60:
        second = second - 60
        minute = minute + 1
    if minute >= 60:
        minute = minute - 60
        hour = hour + 1
    return hour * 10000 + minute * 100 + second


def request_for_traces(start, end, date, state, host):
    if start < 100000:
        start_time = '0' + str(start)
    else:
        start_time = start
    if end < 100000:
        end_time = '0' + str(end)
    else:
        end_time = end
    if not state == "ERROR" and not state == "SUCCESS":
        state = "ALL"

    payload = {
        "query": "query queryTraces($condition: TraceQueryCondition) {\n  data: queryBasicTraces(condition: $condition) {"
                 "\n    traces {\n      key: segmentId\n      endpointNames\n      duration\n      start\n      isError\n "
                 "     traceIds\n    }\n    total\n  }}",
        "variables": {
            "condition": {
                "queryDuration": {
                    "start": f"{date} {start_time}",
                    "end": f"{date} {end_time}",
                    "step": "SECOND"},
                "traceState": f"{state}",
                "paging": {
                    "pageNum": 1,
                    "pageSize": 10000,
                    "needTotal": True
                },
                "queryOrder": "BY_DURATION"
            }
        }
    }

    headers = {
        "Host": f"{host}",
        "Connection": "keep-alive",
        "Pragma": "no-cache",
        "Cache-Control": "no-cache",
        "Accept": "application/json, text/plain, */*",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36",
        "Content-Type": "application/json;charset=UTF-8",
        "Origin": f"http://{host}",
        "Referer": f"http://{host}/trace",
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Cookie": "SESSIONID=7EA8B09F0D6D4604A3CB8FFD96A46401"
    }

    r = requests.post(f'http://{host}/graphql', json=payload, headers=headers)
    info_dict = r.json()
    return info_dict


def write_head(path):
    csv_headers = ['StartTime', 'EndTime', 'URL', 'SpanType', 'Service', 'SpanId', 'TraceId', 'Peer', 'ParentSpan',
                   'Component', 'IsError']

    with open(path, 'a+', encoding='utf-8', newline='') as f1:
        print(f"write file {path} header")
        f_csv = csv.writer(f1)
        f_csv.writerow(csv_headers)


def query_trace_ids(argpair):
    global trace_data, state

    start_time = argpair[0]
    temp_start_time = start_time
    end_time = argpair[1]
    date = argpair[2]
    sw_host = argpair[3]

    redo_list_traceid = []
    while temp_start_time < end_time:
        print(temp_start_time)
        temp_end_time = time_check(temp_start_time + 29)
        try:
            info_dict = request_for_traces(
                temp_start_time, temp_end_time, date, state, sw_host)
            traces = info_dict['data']['data']['traces']
            # print(len(total_log))
        except:
            redo_list_traceid.append(
                (temp_start_time, temp_end_time, date, state, sw_host))
            # 报错
            temp_start_time = time_check(temp_end_time + 1)
            continue

        for l in traces:
            for traceId in l['traceIds']:
                trace_id_host_pair = (traceId, sw_host)
                trace_data.add(trace_id_host_pair)

        temp_start_time = time_check(temp_end_time + 1)

    if len(redo_list_traceid) > 1:
        print(f"redo_list_traceid length {len(redo_list_traceid)}")

    for item in redo_list_traceid:
        start_time = item[0]
        end_time = item[1]
        date = item[2]
        state = item[3]
        sw_host = item[4]

        try:
            info_dict = request_for_traces(
                start_time, end_time, date, state, sw_host)
            traces = info_dict['data']['data']['traces']
        except:
            print(
                f"request pair {(start_time, end_time, date, state, sw_host)} error again!")
            continue

        for l in traces:
            for traceId in l['traceIds']:
                trace_id_host_pair = (traceId, sw_host)
                trace_data.add(trace_id_host_pair)


def query_trace_data_and_save(path, trace_data):
    redo_pair_list = []

    with open(path, 'a+', encoding='utf-8', newline='') as f1:
        f_csv = csv.writer(f1)
        for pair in tqdm(trace_data):
            traceId = pair[0]
            host = pair[1]

            payload = {
                "query": "query queryTrace($traceId: ID!) {\n  trace: queryTrace(traceId: $traceId) {\n    spans {\n      "
                         "traceId\n      segmentId\n      spanId\n      parentSpanId\n      refs {\n        traceId\n "
                         "parentSegmentId\n        parentSpanId\n        type\n      }\n      serviceCode\n      "
                         "serviceInstanceName\n      startTime\n      endTime\n      endpointName\n      type\n      "
                         "peer\n      component\n      isError\n      layer\n      tags {\n        key\n        value\n   "
                         "}\n      logs {\n        time\n        data {\n          key\n          value\n        }\n      "
                         "}\n    }\n  }\n  }",
                "variables": {"traceId": traceId}}

            headers = {
                "Host": f"{host}",
                "Connection": "keep-alive",
                "Pragma": "no-cache",
                "Cache-Control": "no-cache",
                "Accept": "application/json, text/plain, */*",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36",
                "Content-Type": "application/json;charset=UTF-8",
                "Origin": f"http://{host}",
                "Referer": f"http://{host}/trace",
                "Accept-Encoding": "gzip, deflate",
                "Accept-Language": "zh-CN,zh;q=0.9",
                "Cookie": "SESSIONID=7EA8B09F0D6D4604A3CB8FFD96A46401"
            }

            try:
                r = requests.post(
                    f'http://{host}/graphql', json=payload, headers=headers)
                info_dict = r.json()
                spans = info_dict['data']['trace']['spans']
            except Exception as e:
                print(f"exception {e}")
                redo_pair_list.append(pair)
                continue

            # pprint(spans)
            for span in spans:
                # 判断peer
                if span['type'] == 'Exit':
                    peer = span['peer'].split(':')[0]
                else:
                    peer = span['serviceCode']

                # 判断parentid，如果是segment开头，找ref中的parentSegmentId是否存在
                parentspan = -1
                if span['parentSpanId'] == -1:
                    if span['refs']:
                        parentspan = span['refs'][0]['parentSegmentId'] + \
                            '.' + str(span['refs'][0]['parentSpanId'])
                    else:
                        parentspan = -1
                else:
                    parentspan = str(span['segmentId']) + \
                        '.' + str(span['parentSpanId'])

                f_csv.writerow((span['startTime'], span['endTime'], span['endpointName'], span['type'], span['serviceCode'],
                                str(span['segmentId']) + '.' + str(span['spanId']
                                                                   ), span['traceId'], peer, parentspan,
                                span['component'], span['isError']))

    return redo_pair_list


trace_data = set()
#state = "ERROR"
state = "SUCCESS"


def main():
    argpairs = [
        (155132, 235500, "2021-08-23", "47.103.205.96:8080"),
    ]

    path = state + '_SpanData' + \
        str(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())) + ".csv"

    for argpair in argpairs:
        print(f"START argpair {argpair}")
        query_trace_ids(argpair)

    write_head(path)

    redo_list = query_trace_data_and_save(path, trace_data)
    if len(redo_list) > 1:
        print(f"redo pair list length {len(redo_list)}. try redo them")
        query_trace_data_and_save(path, redo_list)

    print("finished!")


if __name__ == '__main__':
    main()
