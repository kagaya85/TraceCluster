demo = {"0": [
    {"vertexId": 1, "parentSpanId": "21752", "spanId": "23928", "startTime": 1637200909, "duration": 0.5806451612903226,
     "service": "21384", "operation": "59", "peer": "61266/0", "isError": True},
    {"vertexId": 2, "parentSpanId": "21752", "spanId": "24755", "startTime": 1637200909, "duration": 0.967741935483871,
     "service": "30566", "operation": "1001", "peer": "61266/0", "isError": True},
    {"vertexId": 4, "parentSpanId": "21752", "spanId": "23928", "startTime": 1637200909,
     "duration": 0.5806451612903226,
     "service": "21384", "operation": "59", "peer": "61266/0", "isError": True}],
    "1": [
        {"vertexId": 3, "parentSpanId": "21752", "spanId": "23928", "startTime": 1637200909,
         "duration": 0.5806451612903226,
         "service": "21384", "operation": "59", "peer": "61266/0", "isError": True},
        {"vertexId": 4, "parentSpanId": "21752", "spanId": "23928", "startTime": 1637200909,
         "duration": 0.5806451612903226,
         "service": "21384", "operation": "59", "peer": "61266/0", "isError": True}],
    "2": [
        {"vertexId": 3, "parentSpanId": "21752", "spanId": "23928", "startTime": 1637200909,
         "duration": 0.5806451612903226,
         "service": "21384", "operation": "59", "peer": "61266/0", "isError": True},
        {"vertexId": 4, "parentSpanId": "21752", "spanId": "23928", "startTime": 1637200909,
         "duration": 0.5806451612903226,
         "service": "21384", "operation": "59", "peer": "61266/0", "isError": True}],
    "3": [
        {"vertexId": 4, "parentSpanId": "21752", "spanId": "23928", "startTime": 1637200909,
         "duration": 0.5806451612903226,
         "service": "21384", "operation": "59", "peer": "61266/0", "isError": True}, ],
}


def slide(edges, data, window_size):
    if len(data) == window_size:
        return [data]
    if len(data) == 0:
        for v, l in edges.items():
            first = [int(v)]
            rest = slide(edges, first, window_size)
            for r in rest:
                data.append(r)
        return data
    else:
        last = str(data[-1])
        windows = []
        if last not in edges:
            return windows
        for v in edges[last]:
            temp = data[:]
            temp.append(v["vertexId"])
            rest = slide(edges, temp, window_size)
            for r in rest:
                windows.append(r)
        return windows


def get_vocab(trace):
    vertices = trace['vertexs'].keys()
    v_index = [int(v) for v in vertices]
    vocab_size = max(v_index)
    return vocab_size


def subsequence(trace, window_size):
    edges = trace['edges']
    if window_size < 2:
        print("window_size < 2")
        exit(-1)
    data = slide(edges, [], window_size)
    return data


def split(data):
    context = []
    target = []
    for d in data:
        pos = int(len(d) / 2)
        target.append(d[pos])
        del d[pos]
        context.append(d)
    return context, target


'''
TEST
'''


def get_abnormal_trace():
    trace = {"vertexs": {"0": [1], "1": [1], "2": [1], "3": [1], "4": [1]}, "edges": demo}
    return trace
    # vocab_size = get_vocab(trace)
    # data = subsequence(trace, 3)
    # print(vocab_size)
    # print(data)
    # context, target = split(data)
    # return context, target
