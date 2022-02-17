from preprocess_data import get_operation_slo
from preprocess_data import get_span
from preprocess_data import get_operation_duration_data
from preprocess_data import get_service_operation_list
import time

'''
   Input long time trace data and get the slo of operation
   :arg
       date: format 2020-08-14 or 2020-08-*
       start_time  end_time  expect more than one hours traces
       
   :return
       operation dict of the mean of and variance  
       {
           # operation: {mean, variance}
           "Currencyservice_Convert": [600, 3]}
       }        
'''


def get_slo():
    span_list = get_span()
    operation_list = get_service_operation_list(span_list)
    slo = get_operation_slo(
        service_operation_list=operation_list, span_list=span_list)
    return slo


'''
   Input short time trace data and calculate the expect_duration.
   expect_duration = operation1 * mean_duration1 + variation_duration1 +
                    operation2 * mean_duration2 + variation_duration2
   if expect_duration < real_duration  error                 
   :arg
       date: format 2020-08-14 or 2020-08-*
       start_time end_time  expect 30s or 1min traces
   :return
       if error_rate > 1%:
          return True    
'''


def system_anormaly_detect(slo, operation_list):
    span_list = get_span()
    if len(span_list) == 0:
        print("Error: Current span list is empty ")
        return False
    #operation_list = get_service_operation_list(span_list)
    operation_count = get_operation_duration_data(operation_list, span_list)

    anormaly_trace = 0
    total_trace = 0
    for trace_id in operation_count:
        total_trace += 1
        real_duration = float(operation_count[trace_id]['duration']) / 1000.0
        expect_duration = 0.0
        for operation in operation_count[trace_id]:
            if "duration" == operation:
                continue
            expect_duration += operation_count[trace_id][operation] * (
                slo[operation][0] + 1.5 * slo[operation][1])

        if real_duration > expect_duration:
            anormaly_trace += 1

    print("anormaly_trace", anormaly_trace)
    print("total_trace", total_trace)
    print()
    if anormaly_trace > 8:
        anormaly_rate = float(anormaly_trace) / total_trace
        print("anormaly_rate", anormaly_rate)
        return True

    else:
        return False


'''
   Determine single trace state
   :arg
       operation_list: operation_count[traceid] # list of operation of single trace
       slo: slo list
   
   :return
        if real_duration > expect_duration:
            return True
        else:
            return False    
'''


def trace_anormaly_detect(operation_list, slo):
    expect_duration = 0.0
    real_duration = float(operation_list['duration']) / 1000.0
    for operation in operation_list:
        if operation == "duration":
            continue
        expect_duration += operation_list[operation] * \
            (slo[operation][0] + slo[operation][1])

    if real_duration > expect_duration + 50:
        return True
    else:
        return False


'''
   Partition all the trace list in operation_count to normal_list and abnormal_list
   :arg
       operation_count: all the trace operation
       operation_count[traceid][operation] = 1
   :return
       normal_list: normal traceid list
       abnormal_list: abnormal traceid list
       
'''


def trace_list_partition(operation_count, slo):
    normal_list = []  # normal traceid list
    abnormal_list = []  # abnormal traceid list
    for traceid in operation_count:
        abnormal = trace_anormaly_detect(
            operation_list=operation_count[traceid], slo=slo)
        if abnormal:
            abnormal_list.append(traceid)
        else:
            normal_list.append(traceid)

    return abnormal_list, normal_list
