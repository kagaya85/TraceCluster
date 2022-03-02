import numpy as np
import math
from anormaly_detector import trace_list_partition
from anormaly_detector import system_anormaly_detect
from preprocess_data import get_normal_span, get_operation_duration_data
from preprocess_data import get_span
from preprocess_data import get_operation_slo
from preprocess_data import get_service_operation_list
from preprocess_data import get_pagerank_graph
from pagerank import trace_pagerank
from anormaly_detector import trace_list_partition
import time
from dateutil.parser import parse



def timestamp(datetime):
    timeArray = time.strptime(str(datetime), "%Y-%m-%d %H:%M:%S")
    ts = int(time.mktime(timeArray)) * 1000
    # print(ts)
    return ts


def calculate_spectrum_without_delay_list(anomaly_result, normal_result, anomaly_list_len, normal_list_len,
                                          top_max, normal_num_list, anomaly_num_list, spectrum_method):
    spectrum = {}

    for node in anomaly_result:
        spectrum[node] = {}
        # spectrum[node]['ef'] = anomaly_result[node] * anomaly_list_len
        # spectrum[node]['nf'] = anomaly_list_len - anomaly_result[node] * anomaly_list_len
        spectrum[node]['ef'] = anomaly_result[node] * anomaly_num_list[node]
        spectrum[node]['nf'] = anomaly_result[node] * \
            (anomaly_list_len - anomaly_num_list[node])
        if node in normal_result:
            #spectrum[node]['ep'] = normal_result[node] * normal_list_len
            #spectrum[node]['np'] = normal_list_len - normal_result[node] * normal_list_len
            spectrum[node]['ep'] = normal_result[node] * normal_num_list[node]
            spectrum[node]['np'] = normal_result[node] * \
                (normal_list_len - normal_num_list[node])
        else:
            spectrum[node]['ep'] = 0.0000001
            spectrum[node]['np'] = 0.0000001

    for node in normal_result:
        if node not in spectrum:
            spectrum[node] = {}
            #spectrum[node]['ep'] = normal_result[node] * normal_list_len
            #spectrum[node]['np'] = normal_list_len - normal_result[node] * normal_list_len
            spectrum[node]['ep'] = (
                1 + normal_result[node]) * normal_num_list[node]
            spectrum[node]['np'] = normal_list_len - normal_num_list[node]
            if node not in anomaly_result:
                spectrum[node]['ef'] = 0.0000001
                spectrum[node]['nf'] = 0.0000001

    # print('\n Micro Rank Spectrum raw:')
    # print(json.dumps(spectrum))
    result = {}

    for node in spectrum:
        # Dstar2
        if spectrum_method == "dstar2":
            result[node] = spectrum[node]['ef'] * spectrum[node]['ef'] / \
                (spectrum[node]['ep'] + spectrum[node]['nf'])
        # Ochiai
        elif spectrum_method == "ochiai":
            result[node] = spectrum[node]['ef'] / \
                math.sqrt((spectrum[node]['ep'] + spectrum[node]['ef']) * (
                    spectrum[node]['ef'] + spectrum[node]['nf']))

        elif spectrum_method == "jaccard":
            result[node] = spectrum[node]['ef'] / (spectrum[node]['ef'] + spectrum[node]['ep']
                                                   + spectrum[node]['nf'])

        elif spectrum_method == "sorensendice":
            result[node] = 2 * spectrum[node]['ef'] / \
                (2 * spectrum[node]['ef'] + spectrum[node]
                 ['ep'] + spectrum[node]['nf'])

        elif spectrum_method == "m1":
            result[node] = (spectrum[node]['ef'] + spectrum[node]
                            ['np']) / (spectrum[node]['ep'] + spectrum[node]['nf'])

        elif spectrum_method == "m2":
            result[node] = spectrum[node]['ef'] / (2 * spectrum[node]['ep'] + 2 * spectrum[node]['nf'] +
                                                   spectrum[node]['ef'] + spectrum[node]['np'])
        elif spectrum_method == "goodman":
            result[node] = (2 * spectrum[node]['ef'] - spectrum[node]['nf'] - spectrum[node]['ep']) / \
                (2 * spectrum[node]['ef'] + spectrum[node]
                 ['nf'] + spectrum[node]['ep'])
        # Tarantula
        elif spectrum_method == "tarantula":
            result[node] = spectrum[node]['ef'] / (spectrum[node]['ef'] + spectrum[node]['nf']) / \
                (spectrum[node]['ef'] / (spectrum[node]['ef'] + spectrum[node]['nf']) +
                 spectrum[node]['ep'] / (spectrum[node]['ep'] + spectrum[node]['np']))
        # RussellRao
        elif spectrum_method == "russellrao":
            result[node] = spectrum[node]['ef'] / \
                (spectrum[node]['ef'] + spectrum[node]['nf'] +
                 spectrum[node]['ep'] + spectrum[node]['np'])

        # Hamann
        elif spectrum_method == "hamann":
            result[node] = (spectrum[node]['ef'] + spectrum[node]['np'] - spectrum[node]['ep'] - spectrum[node]
                            ['nf']) / (spectrum[node]['ef'] + spectrum[node]['nf'] + spectrum[node]['ep'] + spectrum[node]['np'])

        # Dice
        elif spectrum_method == "dice":
            result[node] = 2 * spectrum[node]['ef'] / \
                (spectrum[node]['ef'] + spectrum[node]
                 ['nf'] + spectrum[node]['ep'])

        # SimpleMatching
        elif spectrum_method == "simplematcing":
            result[node] = (spectrum[node]['ef'] + spectrum[node]['np']) / (spectrum[node]
                                                                            ['ef'] + spectrum[node]['np'] + spectrum[node]['nf'] + spectrum[node]['ep'])

        # RogersTanimoto
        elif spectrum_method == "rogers":
            result[node] = (spectrum[node]['ef'] + spectrum[node]['np']) / (spectrum[node]['ef'] +
                                                                            spectrum[node]['np'] + 2 * spectrum[node]['nf'] + 2 * spectrum[node]['ep'])

    # Top-n节点列表
    top_list = []
    score_list = []
    for index, score in enumerate(sorted(result.items(), key=lambda x: x[1], reverse=True)):
        if index < top_max + 6:
            top_list.append(score[0])
            score_list.append(score[1])
            #print('%-50s: %.8f' % (score[0], score[1]))
    return top_list, score_list


def online_anomaly_detect_RCA(slo, operation_list):
# while True:
    # current_time = datetime.datetime.strptime(datetime.datetime.now().strftime(
    #     "%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S")-datetime.timedelta(minutes=1)

    # start_time = current_time - datetime.timedelta(seconds=60)
    anormaly_flag = system_anormaly_detect(
        slo=slo, operation_list=operation_list)

    if anormaly_flag:
        middle_span_list = get_span()
        operation_count = get_operation_duration_data(
            operation_list, middle_span_list)
        anomaly_list, normal_list = trace_list_partition(
            operation_count=operation_count, slo=slo)

        print("anomaly_list", len(anomaly_list))
        print("normal_list", len(normal_list))
        print("total", len(normal_list) + len(anomaly_list))

        if len(anomaly_list) == 0 or len(normal_list) == 0:
            print('list is empty')
            return
        operation_operation, operation_trace, trace_operation, pr_trace \
            = get_pagerank_graph(normal_list, middle_span_list)

        normal_trace_result, normal_num_list = trace_pagerank(operation_operation, operation_trace, trace_operation,
                                                                pr_trace, False)

        a_operation_operation, a_operation_trace, a_trace_operation, a_pr_trace \
            = get_pagerank_graph(anomaly_list, middle_span_list)
        anomaly_trace_result, anomaly_num_list = trace_pagerank(a_operation_operation, a_operation_trace,
                                                                a_trace_operation, a_pr_trace,
                                                                True)
        top_list, score_list = calculate_spectrum_without_delay_list(anomaly_result=anomaly_trace_result,
                                                                        normal_result=normal_trace_result,
                                                                        anomaly_list_len=len(
                                                                            anomaly_list),
                                                                        normal_list_len=len(
                                                                            normal_list),
                                                                        top_max=5,
                                                                        anomaly_num_list=anomaly_num_list,
                                                                        normal_num_list=normal_num_list,
                                                                        spectrum_method="dstar2")
        print('top_list:', top_list)
        print('score_list:', score_list)
        return



def main():
    span_list = get_span()
    # print(span_list)
    operation_list = get_service_operation_list(span_list)
    print('operation list:', operation_list)
    # normal_span_list = get_normal_span()
    slo = get_operation_slo(
        service_operation_list=operation_list, span_list=span_list)
    print('slo:', slo)
    online_anomaly_detect_RCA(slo, operation_list)


if __name__ == '__main__':
    main()