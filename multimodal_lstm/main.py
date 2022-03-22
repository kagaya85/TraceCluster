import csv
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.mixture import GaussianMixture
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils import rnn
import argparse
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import multiprocessing

import yappi

# Device configuration
from dataset import TraceDataset
# from dataset import DealDataset, get_num_classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""""""""
def judge_bool(out_time, recon_time_seq, distance_max, distance_min, out_node, origin_node_seq, data_length,
               num_candidates, trace_bool, error_type, traceId):
    bool_count = [0 for x in range(len(out_time))]
    for i in range(len(out_time)):
        e_max = 0
        labels.append(trace_bool)
        for j in range(data_length[i]):
            e = (out_time[i, j] - recon_time_seq[i, j]) ** 2
            if e > e_max:
                e_max = e
        scores.append(e_max)
        if e_max > distance_max:
            bool_count[i] = 1

    for i in range(len(out_node)):
        if bool_count[i] != 1:
            for j in range(data_length[i]):
                values, indices = out_node[i][j].topk(num_candidates, dim=0, largest=True, sorted=True)
                o = origin_node_seq[i][j].to(device)
                if o not in indices:
                    bool_count[i] = 1

    for i in range(len(out_time)):
        writer.writerow([trace_bool, bool_count[i], traceId[i], traceId[i], error_type[i]])

    return sum(bool_count)
"""""


def collate_fn(batch):
    batch = sorted(batch, key=lambda i: len(i[0]), reverse=True)
    data_length = [len(row[0]) for row in batch]
    api_batch = [row[0] for row in batch]
    recon_api_batch = rnn.pad_sequence([row[0] for row in batch], batch_first=True)
    time_batch = [row[2] for row in batch]
    recon_time_batch = rnn.pad_sequence([row[2] for row in batch], batch_first=True)
    origin_data_batch = [row[1] for row in batch]
    error_trace_batch = [row[4] for row in batch]
    trace_id_batch = [row[3] for row in batch]
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

    # Hyperparameters
    # num_classes = 281
    num_epochs = 50
    batch_size = 64
    learning_rate = 0.0001
    # input_size = 1
    model_dir = 'model'
    log = 'Adam_batch_size={}_epoch={}'.format(str(batch_size), str(num_epochs))
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_layers', default=1, type=int)
    parser.add_argument('-hidden_size', default=64, type=int)
    parser.add_argument('-num_candidates', default=9, type=int)
    args = parser.parse_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    num_candidates = args.num_candidates

    train_data = TraceDataset(root=r"/data/cyr/traceCluster_01,normal")
    train_data.aug = 'none'
    # normal_test_data = DealDataset(root='./test/normal')
    # abnormal_test_data = DealDataset(root='./test/abnormal')
    num_classes, _ = train_data.get_interface_num()
    input_size = num_classes
    model = Model(input_size, num_layers, num_classes).to(device)

    print("Start Load Train Data")

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True,
                                  collate_fn=collate_fn)
    print("End Load Train Data")

    # normal_test_dataloader = DataLoader(normal_test_data, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=collate_fn)
    # abnormal_test_dataloader = DataLoader(abnormal_test_data, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=collate_fn)
    # print(len(dataloader))
    # for step, (data_batch, labels_batch, data_length) in enumerate(dataloader):
    #   print(data_batch)

    # Loss and optimizer
    criterion_node = nn.CrossEntropyLoss()
    criterion_time = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    start_time = time.time()
    total_step = len(train_dataloader)
    print(f"len(train_dataloader) {total_step}")


    # PROFILE START
    yappi.set_clock_type("cpu")
    yappi.start()
    # PROFILE START


    for epoch in range(num_epochs):  # Loop over the dataset multiple times
        train_loss = 0

        start_time_epoch = time.time()
        for step, (api_seq, recon_api_seq, time_seq, recon_time_seq, data_length, origin_node_seq, _, _) in tqdm(enumerate(
                train_dataloader)):
            recon_api_seq = recon_api_seq.to(device)
            recon_time_seq = recon_time_seq.clone().detach().view(len(recon_time_seq), -1, 1).to(device)
            recon_time_seq = recon_time_seq.to(device)
            out = model(recon_api_seq, recon_time_seq, data_length)
            out_node, out_time = torch.split(out, [num_classes, 1], 2)


            loss_node = 0
            #print(f"calculate node loss: step{step}/{total_step}")
            for i in range(len(out_node)):
                # print(out_node[i], out_node.size())
                # print(len(origin_node_seq[i]), len(out_node[i]) - len(origin_node_seq[i]))
                out_node_split, _ = torch.split(out_node[i],
                                                [len(origin_node_seq[i]), len(out_node[i]) - len(origin_node_seq[i])],
                                                0)
                loss_node = criterion_node(out_node_split, origin_node_seq[i].to(device)) + loss_node
            #print(f"calculate time loss: step{step}/{total_step}")
            loss_time = criterion_time(out_time, recon_time_seq.to(device))
            #print(f"loss :node:{loss_node}, time{loss_time}")
            loss_con = loss_node + loss_time

            # Backward and optimize
            optimizer.zero_grad()
            loss_con.backward()
            train_loss += loss_con.item()
            optimizer.step()
        end_time_epoch = time.time()
        time_used = end_time_epoch - start_time_epoch

        print('Epoch [{}/{}], train_loss: {:.4f}  time_used {:.4f}'.format(epoch + 1, num_epochs, train_loss / total_step, time_used))
    elapsed_time = time.time() - start_time
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), model_dir + '/' + log + '.pt')
    print('elapsed_time: {:.3f}s'.format(elapsed_time))
    print('Finished Training')


    # PROFILE END
    print("Print Profile things.....\n\n\n\n\n\n\n")

    yappi.get_func_stats().print_all()
    yappi.get_thread_stats().print_all()

    print("End Print Profile things!")
    # PROFILE END
