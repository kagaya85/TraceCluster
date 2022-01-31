import random
import time
import torch
from tqdm import tqdm

import torch.nn as nn
import numpy as np

from torch.nn import ModuleList, Embedding
from torch.nn import Sequential, ReLU, Linear, LeakyReLU, Tanh
from torch_geometric.loader import DataLoader
from torch_geometric.nn import PNAConv, BatchNorm, global_add_pool, global_mean_pool, CGConv, GatedGraphConv, GlobalAttention, Set2Set, GATConv
from sklearn.metrics import f1_score, recall_score, precision_score
from torch.utils.data import Subset
from simclr import SIMCLR

from aug_dataset_big import TraceDataset

# def get_target_label_idx(labels, targets):
#     """
#         Get the indices of labels that are included in targets.
#         :param labels: array of labels
#         :param targets: list/tuple of target labels
#         :return: list with indices of target labels
#         """
#     return np.argwhere(np.isin(labels, targets)).flatten().tolist()


if __name__ == '__main__':
    learning_rate = 0.001
    epochs = 50
    dataset = TraceDataset(root='../Data/TraceCluster/big_data', aug='mask_nodes')
    normal_classes = [0]
    abnormal_classes = [1]

    # normal_idx = get_target_label_idx(dataset.data.y.clone().data.cpu().numpy(), normal_classes)
    # abnormal_idx = get_target_label_idx(dataset.data.y.clone().data.cpu().numpy(), abnormal_classes)

    normal_idx = dataset.normal_idx
    abnormal_idx = dataset.abnormal_idx

    random.shuffle(abnormal_idx)
    random.shuffle(normal_idx)

    train_normal = normal_idx[:int(len(normal_idx) * 0.2)]
    test_normal = normal_idx[int(len(normal_idx) * 0.2):]
    train_abnormal = abnormal_idx[:int(len(abnormal_idx) * 0.4)]
    test_abnormal = abnormal_idx[int(len(abnormal_idx) * 0.4):]

    # train_dataset = Subset(dataset, [46945])

    train_dataset = Subset(dataset, train_normal + train_abnormal)
    test_dataset = Subset(dataset, test_abnormal + test_normal)

    device = torch.device('cuda')
    model = SIMCLR().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=10)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

    model.train()

    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        total = 0

        for data in tqdm(train_loader):
            data_o, data_aug_1, data_aug_2 = data
            # data, data_aug_1 = data

            # data, data_aug = data

            # data_aug = data_aug.to(device)
            # data = data.to(device)

            data_aug_1 = data_aug_1.to(device)
            data_aug_2 = data_aug_2.to(device)

            optimizer.zero_grad()

            # out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            # out_aug = model(data_aug.x, data_aug.edge_index, data_aug.edge_attr, data_aug.batch)

            out_aug_1 = model(data_aug_1.x, data_aug_1.edge_index, data_aug_1.edge_attr, data_aug_1.batch)
            out_aug_2 = model(data_aug_2.x, data_aug_2.edge_index, data_aug_2.edge_attr, data_aug_2.batch)

            loss = model.loss_cal(out_aug_1, out_aug_2)
            # loss = model.loss_cal(out_aug_1, out)
            #

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total += 1

        total_loss = total_loss / total
        print(
            'Epoch: %3d/%3d, Train Loss: %.5f, Time: %.5f' % (epoch + 1, epochs, total_loss, time.time() - start_time))

    ys, preds = [], []
    for data in test_loader:
        data = data.to(device)
        ys.append(data.y.cpu())
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        preds.append((out > 0).float().cpu())

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    print(f1_score(y, pred) if pred.sum() > 0 else 0)
    print(recall_score(y, pred) if pred.sum() > 0 else 0)
    print(precision_score(y, pred) if pred.sum() > 0 else 0)
    exit()