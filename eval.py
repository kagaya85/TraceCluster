import torch
import json
import logging
import random

from simclr import SIMCLR
from aug_dataset_big import TraceDataset
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from sklearn.svm import SVC, LinearSVC, OneClassSVM
from eval_method import oc_svm_classify


def main():
    model_path = '../Data/TraceCluster/log/50epoch-CGConv-mean/'
    batch_size = 128
    normal_classes = [0]
    abnormal_classes = [1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = TraceDataset(root='../Data/TraceCluster/big_data', aug='none')

    with open(model_path + 'train_info.json', "r") as f:  # file name not list
        model_info = json.load(f)
        num_layers = model_info['num_layers']
        output_dim = model_info['output_dim']
        # num_node_features = model_info['num_node_features']
        # num_edge_features = model_info['num_edge_features']
        gnn_type = model_info['gnn_type']
        pooling_type = model_info['pooling_type']
        aug = model_info['aug']
        train_idx = model_info['train_idx']
        # test_idx = model_info['test_idx']

    normal_idx = dataset.normal_idx
    abnormal_idx = dataset.abnormal_idx

    random.shuffle(abnormal_idx)
    random.shuffle(normal_idx)

    test_normal = normal_idx[int(len(normal_idx) * 0.8):]
    test_abnormal = abnormal_idx[int(len(abnormal_idx) * 0):]
    test_idx = test_abnormal + test_normal
    # test_idx = list(set(normal_idx + abnormal_idx).intersection(set(train_idx)))

    eval_dataset = Subset(dataset, test_idx)
    train_dataset = Subset(dataset, train_idx)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    model = SIMCLR(num_layers=num_layers, input_dim=dataset.num_node_features, output_dim=output_dim,
                   num_edge_attr=dataset.num_edge_features, gnn_type=gnn_type,
                   pooling_type=pooling_type).to(device)
    model.load_state_dict(torch.load(model_path + '{}_{}'.format(gnn_type, pooling_type) + ".model"))

    emb_train, y_train = model.encoder.get_embeddings(train_dataloader)
    emb_test, y_test = model.encoder.get_embeddings(eval_dataloader)

    oc_svm_classify(emb_train, emb_test, y_train, y_test)


if __name__ == '__main__':
    main()
