import random
import time
import torch
from tqdm import tqdm
import argparse
import os
import logging
import json

import torch.nn as nn
import numpy as np

from torch.nn import ModuleList, Embedding
from torch.nn import Sequential, ReLU, Linear, LeakyReLU, Tanh
from torch_geometric.loader import DataLoader
from torch_geometric.nn import PNAConv, BatchNorm, global_add_pool, global_mean_pool, CGConv, GatedGraphConv, \
    GlobalAttention, Set2Set, GATConv
from sklearn.metrics import f1_score, recall_score, precision_score
from torch.utils.data import Subset
from simclr import SIMCLR
from utils import get_target_label_idx

from aug_dataset import TraceDataset


def main():
    # param
    learning_rate = 0.001
    epochs = 20
    normal_classes = [0]
    abnormal_classes = [1]
    batch_size = 32
    num_workers = 10
    num_layers = 2
    gnn_type = 'CGConv'  # GATConv  TransformerConv  CGConv
    pooling_type = 'mean'  # mean  add

    aug = 'random'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = TraceDataset(root='../Data/TraceCluster/big_data', aug=aug)

    # output dim relay on gnn_type
    output_dim = dataset.num_node_features if gnn_type == 'CGConv' else 16

    normal_idx = dataset.normal_idx
    abnormal_idx = dataset.abnormal_idx

    random.shuffle(abnormal_idx)
    random.shuffle(normal_idx)

    train_normal = normal_idx[:int(len(normal_idx) * 0.8)]
    test_normal = normal_idx[int(len(normal_idx) * 0.8):]
    train_abnormal = abnormal_idx[:int(len(abnormal_idx) * 0)]
    test_abnormal = abnormal_idx[int(len(abnormal_idx) * 0):]

    train_dataset = Subset(dataset, train_normal + train_abnormal)
    test_dataset = Subset(dataset, test_abnormal + test_normal)

    model = SIMCLR(num_layers=num_layers, input_dim=dataset.num_node_features, output_dim=output_dim,
                   num_edge_attr=dataset.num_edge_features, gnn_type=gnn_type,
                   pooling_type=pooling_type).to(device)

    # normal_idx = get_target_label_idx(dataset.data.y.clone().data.cpu().numpy(), normal_classes)
    # abnormal_idx = get_target_label_idx(dataset.data.y.clone().data.cpu().numpy(), abnormal_classes)

    # Set up logging
    output_path = f"../Data/TraceCluster/log/" + time.strftime("%Y-%m-%d_%Hh-%Mm-%Ss", time.localtime()) \
                  + '{}epoch_{}_{}'.format(epochs, gnn_type, pooling_type)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = output_path + '/log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # logging param
    logger.info('----------------------')
    logger.info("dataset size: {}".format(len(dataset)))
    logger.info("node feature number: {}".format(dataset.num_node_features))
    logger.info("edge feature number: {}".format(dataset.num_edge_features))
    logger.info('batch_size: {}'.format(batch_size))
    logger.info('learning_rate: {}'.format(learning_rate))
    logger.info('num_gc_layers: {}'.format(num_layers))
    logger.info("epochs: {}".format(epochs))
    logger.info("gnn_type: {}".format(gnn_type))
    logger.info('pooling_type: {}'.format(pooling_type))
    logger.info('output_dim: {}'.format(output_dim))
    logger.info('aug: {}'.format(aug))
    logger.info('----------------------')

    # sava param
    train_info = {
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'num_layers': num_layers,
        'epochs': epochs,
        'gnn_type': gnn_type,
        'pooling_type': pooling_type,
        'num_node_features': dataset.num_node_features,
        'num_edge_features': dataset.num_edge_features,
        'output_dim': output_dim,
        'aug': aug,
        'train_idx': train_normal + train_abnormal,
        'test_idx': test_normal + test_abnormal
    }
    with open(output_path + '/train_info.json', 'w', encoding='utf-8') as json_file:
        json.dump(train_info, json_file)
        logger.info('write train info success')

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # train
    model.train()

    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        total = 0

        for data in tqdm(train_loader):
            data_o, data_aug_1, data_aug_2 = data

            data_aug_1 = data_aug_1.to(device)
            data_aug_2 = data_aug_2.to(device)

            optimizer.zero_grad()

            out_aug_1 = model(data_aug_1.x, data_aug_1.edge_index, data_aug_1.edge_attr, data_aug_1.batch)
            out_aug_2 = model(data_aug_2.x, data_aug_2.edge_index, data_aug_2.edge_attr, data_aug_2.batch)

            loss = model.loss_cal(out_aug_1, out_aug_2)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total += 1

        total_loss = total_loss / total
        logger.info(
            'Epoch: %3d/%3d, Train Loss: %.5f, Time: %.5f' % (epoch + 1, epochs, total_loss, time.time() - start_time))

    save_model_path = output_path + '/{}_{}'.format(gnn_type, pooling_type) + ".model"
    logger.info(f"save model to {save_model_path}")
    torch.save(model.state_dict(), save_model_path)

    exit()


if __name__ == '__main__':
    main()
