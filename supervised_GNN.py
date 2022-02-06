import random
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from torch.nn import ModuleList, Embedding
from torch.nn import Sequential, ReLU, Linear, LeakyReLU, Tanh
from torch_geometric.loader import DataLoader
from torch_geometric.nn import PNAConv, BatchNorm, global_add_pool, global_mean_pool, CGConv, GatedGraphConv, GlobalAttention, Set2Set, GATConv
from sklearn.metrics import f1_score, recall_score, precision_score
from torch.utils.data import Subset
from utils import get_target_label_idx
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from aug_dataset_mem import TraceDataset


class CGNNet(nn.Module):
    def __init__(self, num_layers=2):
        super(CGNNet, self).__init__()

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(num_layers):
            conv = CGConv(channels=20, dim=8)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(20))

        self.batch_norms1 = BatchNorm(60)

        self.ggnn= GatedGraphConv(out_channels=20, num_layers=2).to(device)
        self.gat1 = GATConv(in_channels=20, out_channels=60)
        self.gat2 = GATConv(in_channels=60, out_channels=16)
        self.mlp = nn.Sequential(Linear(20, 32), Tanh(), Linear(32, 8), Tanh(), Linear(8, 1))
        # self.mlp = nn.Sequential(Linear(768, 1))
        self.readout = GlobalAttention(gate_nn=nn.Sequential(
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 1)))
        self.readout_2 = Set2Set(in_channels=20, processing_steps=2)

    def forward(self, x, edge_index, edge_attr, batch):

        # x1 = self.ggnn(x, edge_index)

        # x = self.gat1(x, edge_index)
        # x = self.batch_norms1(x)
        # x = self.gat2(x, edge_index)

        xs = []
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = batch_norm(conv(x, edge_index, edge_attr))
            xs.append(x)

        xpool = [global_add_pool(x, batch) for x in xs]

        # x = torch.cat(xpool, 1)
        x = xpool[-1]
        # x = global_add_pool(x,batch)
        # x = self.readout_2(x,batch)
        # x = self.readout(x,batch)

        return self.mlp(x)


if __name__ == '__main__':
    learning_rate = 0.001
    epochs = 20
    dataset = TraceDataset(root='../Data/TraceCluster/data', aug='none')
    normal_classes = [0]
    abnormal_classes = [1]

    # target_class = dataset.url_classes.index('POST:/api/v1/travelservice/trips/left_parallel')
    # target_idx = get_target_label_idx(dataset.data.url_class.clone().data.cpu().numpy(), target_class)
    #
    # normal_idx = get_target_label_idx(dataset.data.y.clone().data.cpu().numpy(), normal_classes)
    # abnormal_idx = get_target_label_idx(dataset.data.y.clone().data.cpu().numpy(), abnormal_classes)
    #
    # normal_idx = list(set(normal_idx).intersection(set(target_idx)))
    # abnormal_idx = list(set(abnormal_idx).intersection(set(target_idx)))

    normal_idx = dataset.normal_idx
    abnormal_idx = dataset.abnormal_idx

    random.shuffle(abnormal_idx)
    random.shuffle(normal_idx)

    train_normal = normal_idx[:int(len(normal_idx) * 0.5)]
    test_normal = normal_idx[int(len(normal_idx) * 0.5):]
    train_abnormal = abnormal_idx[:int(len(abnormal_idx) * 0.5)]
    test_abnormal = abnormal_idx[int(len(abnormal_idx) * 0.5):]

    train_dataset = Subset(dataset, train_normal + train_abnormal)
    test_dataset = Subset(dataset, test_abnormal + test_normal)

    device = torch.device('cuda')
    model = CGNNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

    model.train()


    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        total = 0

        for data in tqdm(train_loader):
            start_time = time.time()
            data = data.to(device)
            optimizer.zero_grad()

            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = criterion(torch.squeeze(out), data.y.float())

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total += 1

        total_loss = total_loss / total
        print('Epoch: %3d/%3d, Train Loss: %.5f' % (epoch + 1, epochs, total_loss))

    ys, preds = [], []
    for data in test_loader:
        data = data.to(device)
        ys.append(data.y.cpu())
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        preds.append((out > 0).float().cpu())

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    print('F1score')
    print(f1_score(y, pred) if pred.sum() > 0 else 0)
    print('Recall')
    print(recall_score(y, pred) if pred.sum() > 0 else 0)
    print('Precision')
    print(precision_score(y, pred) if pred.sum() > 0 else 0)

    # check graph representation
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    model.eval()

    ret = []
    y = []
    trace_class = []
    url_status_class = []
    url_class_list = []
    trace_ids = []
    with torch.no_grad():
        for data in tqdm(dataloader):
            # data = data[0]
            data.to(device)
            x = model(data.x, data.edge_index, data.edge_attr, data.batch)

            ret.append(x.cpu().numpy())
            y.append(data.y.cpu().numpy())
            trace_class.append(data.trace_class.cpu().numpy())
            url_status_class.append(data.url_status_class.cpu().numpy())
            url_class_list.append(data.url_class.cpu().numpy())
    ret = np.concatenate(ret, 0)
    y = np.concatenate(y, 0)
    trace_class = np.concatenate(trace_class, 0)
    url_status_class = np.concatenate(url_status_class, 0)
    url_class_list = np.concatenate(url_class_list, 0)

    tsne = TSNE()
    x = tsne.fit_transform(ret)
    plt.scatter(x[:, 0], x[:, 1], c=y, marker='o', s=10, cmap=plt.cm.Spectral)
    plt.show()


    exit()
