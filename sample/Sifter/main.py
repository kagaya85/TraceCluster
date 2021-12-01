import json
import random
import matplotlib.pyplot as plt
from collections import deque
from Model import CBOW, forward_backward
from preprocess import subsequence, split, get_vocab, get_abnormal_trace


def read_json(file):
    with open(file, 'r', encoding='utf8')as fp:
        dic = json.load(fp)
    return dic


if __name__ == '__main__':
    file_name = "../data/preprocessed/2021-10-13_16-57-51.json"
    print("Reading...")
    traces = read_json(file_name)
    print(len(traces))
    # Paths of length
    window_size = 3
    max_vocab_size = 0
    # Calculate total vocabulary size
    for trace in traces.values():
        vocab_size = get_vocab(trace)
        if vocab_size > max_vocab_size:
            max_vocab_size = vocab_size
    # Manually inserting anomalous trace
    traces["abnormal_trace"] = get_abnormal_trace()
    # Embedding dimension
    P = 10
    model = CBOW(max_vocab_size, P, window_size - 1)
    # Previously seen k traces' loss
    k = 50
    window = deque(maxlen=k)
    # Sampling traces
    ids = []
    # For picture
    loss_y = []
    alpha_y = []
    for trace_id, trace in traces.items():
        # Get the slide windows of trace and split them into contexts + targets
        data = subsequence(trace, window_size)
        if len(data) == 0:
            # print(id)
            # print("==========================")
            continue
        print("id:{}".format(trace_id))
        context, target = split(data)

        # Forward to get the loss and backward to train the model
        loss = forward_backward(model, context, target)
        window.append(loss)
        print("loss:{}".format(loss))

        # Calculate sampling probability
        alpha = 0.01  # Sampling rate
        losses = list(window)
        min_loss = min(losses)
        weight = [loss - min_loss for loss in losses]
        total_weight = sum(weight)
        if total_weight != 0:
            alpha = (weight[-1] / total_weight) * len(weight) * alpha
        print("alpha:{}".format(alpha))
        r = random.randint(0, 100) / 100
        print("random:{}".format(r))
        if r < alpha:
            ids.append(trace_id)
        print("==========================")
        # For picture
        loss_y.append(loss)
        alpha_y.append(alpha)

    print(ids)
    # For picture
    # x = range(len(loss_y))
    # plt.plot(x, loss_y, label="loss")
    # plt.plot(x, alpha_y, label="alpha")
    # plt.legend()
    # plt.show()
