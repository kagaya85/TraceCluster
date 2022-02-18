import csv
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.mixture import GaussianMixture
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils import rnn
import argparse
from sklearn.metrics import roc_auc_score

import yappi

# Device configuration
from dataset import DealDataset, get_num_classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def judge_bool(out_time, recon_time_seq, distance_max, distance_min, out_node, origin_node_seq, data_length,
               num_candidates, trace_bool, error_type, traceId):
    bool_count = [1 for x in range(len(out_time))]
    time_value = []
    single_bool = [1 for x in range(len(out_time))]
    for i in range(len(out_time)):
        e_max = 0
        labels.append(trace_bool)
        for j in range(data_length[i]):
            e = (out_time[i, j] - recon_time_seq[i, j]) ** 2
            if e > e_max:
                e_max = e
        scores.append(e_max)
        time_value.append(float(e_max))
        if e_max > distance_max:
            bool_count[i] = 0

    for i in range(len(out_node)):
        for j in range(data_length[i]):
            values, indices = out_node[i][j].topk(num_candidates, dim=0, largest=True, sorted=True)
            o = origin_node_seq[i][j].to(device)
            if o not in indices:
                single_bool[i] = 0
                bool_count[i] = 0

    for i in range(len(out_time)):
        writer.writerow([trace_bool, bool_count[i], single_bool[i], time_value[i], traceId[i], error_type[i]])

    return len(bool_count) - sum(bool_count)


def collate_fn(batch):
    batch = sorted(batch, key=lambda i: len(i.api_seq), reverse=True)
    data_length = [len(row.api_seq) for row in batch]
    api_batch = [row.api_seq for row in batch]
    recon_api_batch = rnn.pad_sequence([row.api_seq for row in batch], batch_first=True)
    time_batch = [row.time_seq for row in batch]
    recon_time_batch = rnn.pad_sequence([row.time_seq for row in batch], batch_first=True)
    origin_data_batch = [row.original_api_seq for row in batch]
    error_trace_batch = [row.y for row in batch]
    trace_id_batch = [row.trace_id for row in batch]
    return api_batch, recon_api_batch, time_batch, recon_time_batch, data_length, origin_data_batch, error_trace_batch, trace_id_batch

class Model(nn.Module):
    def __init__(self, input_size, num_layers, num_keys):
        super(Model, self).__init__()
        self.num_layers = num_layers
        self.hidden_size_node = 100
        self.hidden_size_time = 1
        self.hidden_size_con = input_size + 1
        self.lstm_node = nn.LSTM(input_size, self.hidden_size_node, num_layers, batch_first=True)
        self.lstm_time = nn.LSTM(1, self.hidden_size_time, num_layers, batch_first=True)
        self.lstm_con = nn.LSTM(self.hidden_size_node + self.hidden_size_time, self.hidden_size_con, num_layers,
                                batch_first=True)
        self.fc_con = nn.Linear(hidden_size, num_keys + 1)
        # self.fc_node = nn.Linear(hidden_size, num_keys)
        # self.fc_time = nn.Linear(hidden_size, 1)

    def forward(self, x, y, x_len):
        h0_x = torch.zeros(self.num_layers, x.size(0), self.hidden_size_node).to(device)
        c0_x = torch.zeros(self.num_layers, x.size(0), self.hidden_size_node).to(device)
        packed_x = nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=True).to(device)
        out_x, _ = self.lstm_node(packed_x, (h0_x, c0_x))
        out_x, _ = nn.utils.rnn.pad_packed_sequence(out_x, batch_first=True)

        h0_y = torch.zeros(self.num_layers, y.size(0), self.hidden_size_time).to(device)
        c0_y = torch.zeros(self.num_layers, y.size(0), self.hidden_size_time).to(device)
        packed_y = nn.utils.rnn.pack_padded_sequence(y, x_len, batch_first=True).to(device)
        out_y, _ = self.lstm_time(packed_y, (h0_y, c0_y))
        out_y, _ = nn.utils.rnn.pad_packed_sequence(out_y, batch_first=True)

        # cat
        in_sec = torch.cat((out_x, out_y), -1)
        h0_con = torch.zeros(self.num_layers, in_sec.size(0), self.hidden_size_con).to(device)
        c0_con = torch.zeros(self.num_layers, in_sec.size(0), self.hidden_size_con).to(device)
        out, _ = self.lstm_con(in_sec, (h0_con, c0_con))

        # out = self.fc_con(out[:, -1, :])
        return out


if __name__ == '__main__':
    batch_size = 64
    learning_rate = 0.0001
    # input_size = 1
    model_path = 'model/Adam_batch_size=64_epoch=20.pt'
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_layers', default=1, type=int)
    parser.add_argument('-hidden_size', default=64, type=int)
    parser.add_argument('-num_candidates', default=9, type=int)
    args = parser.parse_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    num_candidates = args.num_candidates

    train_data = DealDataset(root="./train")
    normal_test_data = DealDataset(root='./test/normal')
    abnormal_test_data = DealDataset(root='./test/abnormal')
    num_classes, _ = get_num_classes()
    input_size = num_classes
    model = Model(input_size, num_layers, num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True,
                                  collate_fn=collate_fn)
    normal_test_dataloader = DataLoader(normal_test_data, batch_size=batch_size, shuffle=True, pin_memory=True,
                                        collate_fn=collate_fn)
    abnormal_test_dataloader = DataLoader(abnormal_test_data, batch_size=batch_size, shuffle=True, pin_memory=True,
                                          collate_fn=collate_fn)

    # test model 像deeplog一样算fp和tp，时间要进去一个一个算
    test_start_time = time.time()
    scores = []
    idx_label_score = []
    FP = 0
    TP = 0

    # PROFILE START
    yappi.set_clock_type("cpu")
    yappi.start()
    # PROFILE START

    # 计算rt高斯分布
    print("Start calculate gaussian distribute")
    squared_error_distances = []
    with torch.no_grad():
        for step, (node_seq, recon_node_seq, time_seq, recon_time_seq, data_length, origin_node_seq, _, _) in enumerate(
                train_dataloader):
            recon_node_seq = recon_node_seq.to(device)
            recon_time_seq = recon_time_seq.clone().detach().view(len(recon_time_seq), -1, 1).to(device)
            recon_time_seq = recon_time_seq.to(device)
            out = model(recon_node_seq, recon_time_seq, data_length)
            out_node, out_time = torch.split(out, [num_classes, 1], 2)

            for i in range(len(out_time)):
                for j in range(data_length[i]):
                    squared_error_distances.append([(out_time[i, j] - recon_time_seq[i, j]) ** 2])

    gmm = GaussianMixture(random_state=0).fit(squared_error_distances)
    distance_max = gmm.means_[0][0] + 1.96 * ((gmm.covariances_[0][0][0] / len(squared_error_distances)) ** 0.5)
    distance_min = gmm.means_[0][0] - 1.96 * ((gmm.covariances_[0][0][0] / len(squared_error_distances)) ** 0.5)
    print("Finish calculate gaussian distribute")
    print(distance_max)
    print("test-model error")

    labels = []
    scores = []

    # 计算fp,rt判断是否在高斯分布95%置信区间，node每个判断是否为topk
    with open(f'result_{num_candidates}.csv', 'a+', newline='') as f:
        headers = ['labels', 'predict', 'node_topk', 'time_max', 'traceId', 'error_type']
        writer = csv.writer(f)
        writer.writerow(headers)
        with torch.no_grad():
            for data in normal_test_dataloader:
                node_seq, recon_node_seq, time_seq, recon_time_seq, data_length, origin_node_seq, error_type, traceId = data
                recon_node_seq = recon_node_seq.to(device)
                recon_time_seq = recon_time_seq.clone().detach().view(len(recon_time_seq), -1, 1).to(device)
                recon_time_seq = recon_time_seq.to(device)
                out = model(recon_node_seq, recon_time_seq, data_length)
                out_node, out_time = torch.split(out, [num_classes, 1], 2)

                FP += judge_bool(out_time, recon_time_seq, distance_max, distance_min, out_node, origin_node_seq,
                                 data_length,
                                 num_candidates, 1, error_type, traceId)

        # 计算tp
        with torch.no_grad():
            for data in abnormal_test_dataloader:
                node_seq, recon_node_seq, time_seq, recon_time_seq, data_length, origin_node_seq, error_type, traceId = data
                recon_node_seq = recon_node_seq.to(device)
                recon_time_seq = recon_time_seq.clone().detach().view(len(recon_time_seq), -1, 1).to(device)
                recon_time_seq = recon_time_seq.to(device)
                out = model(recon_node_seq, recon_time_seq, data_length)
                out_node, out_time = torch.split(out, [num_classes, 1], 2)

                TP += judge_bool(out_time, recon_time_seq, distance_max, distance_min, out_node, origin_node_seq,
                                 data_length,
                                 num_candidates, 0, error_type, traceId)

    elapsed_time = time.time() - test_start_time
    print('elapsed_time: {:.3f}s'.format(elapsed_time))
    test_auc = roc_auc_score(labels, scores)
    print('auc: {:.3f}'.format(test_auc))
    # Compute precision, recall and F1-measure
    FN = len(abnormal_test_data) - TP
    P = 100 * TP / (TP + FP)
    R = 100 * TP / (TP + FN)
    F1 = 2 * P * R / (P + R)
    print(
        'false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(
            FP, FN, P, R, F1))
    print('Finished Predicting')


    # PROFILE END
    print("Print Profile things.....\n\n\n\n\n\n\n")

    yappi.get_func_stats().print_all()
    yappi.get_thread_stats().print_all()

    print("End Print Profile things!")
    # PROFILE END
