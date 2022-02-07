import torch
import json

import numpy as np

from simclr import SIMCLR
from aug_dataset_mem import TraceDataset
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from utils import get_target_label_idx

def main():
    model_path = '../Data/TraceCluster/log/02_06_20epoch_CGConv_mean_no_anomaly_z-score/'
    batch_size = 128
    normal_classes = [0]
    abnormal_classes = [1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = TraceDataset(root='../Data/TraceCluster/data', aug='none')

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

    # target_class = dataset.url_classes.index('{GET}/api/v1/adminbasicservice/adminbasic/prices')
    # target_idx = get_target_label_idx(dataset.data.url_class.clone().data.cpu().numpy(), target_class)
    # # dataset = Subset(dataset, target_idx)
    #
    # normal_idx = get_target_label_idx(dataset.data.y.clone().data.cpu().numpy(), normal_classes)
    # abnormal_idx = get_target_label_idx(dataset.data.y.clone().data.cpu().numpy(), abnormal_classes)
    #
    # normal_idx = list(set(normal_idx).intersection(set(target_idx)))
    # abnormal_idx = list(set(abnormal_idx).intersection(set(target_idx)))
    #
    # train_normal = normal_idx[:int(len(normal_idx) * 0.7)]
    # test_normal = normal_idx[int(len(normal_idx) * 0.7):]
    # train_abnormal = abnormal_idx[:int(len(abnormal_idx) * 0)]
    # test_abnormal = abnormal_idx[int(len(abnormal_idx) * 0):]
    #
    # train_dataset = Subset(dataset, train_normal + train_abnormal)
    # eval_dataset = Subset(dataset, test_abnormal + test_normal)

    normal_idx = dataset.normal_idx
    abnormal_idx = dataset.abnormal_idx
    #
    # random.shuffle(abnormal_idx)
    # random.shuffle(normal_idx)
    #
    # train_normal = normal_idx[:int(len(normal_idx) * 0.8)]
    # train_abnormal = abnormal_idx[:int(len(abnormal_idx) * 0)]
    # test_normal = normal_idx[int(len(normal_idx) * 0.8):]
    # test_abnormal = abnormal_idx[int(len(abnormal_idx) * 0):]
    # train_idx = train_normal + train_abnormal

    # test_idx = test_abnormal + test_normal
    test_idx = list(set(normal_idx + abnormal_idx).difference(set(train_idx)))

    eval_dataset = Subset(dataset, test_idx)
    train_dataset = Subset(dataset, train_idx)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, num_workers=10)

    model = SIMCLR(num_layers=num_layers, input_dim=dataset.num_node_features, output_dim=output_dim,
                   num_edge_attr=dataset.num_edge_features, gnn_type=gnn_type,
                   pooling_type=pooling_type).to(device)
    model.load_state_dict(torch.load(model_path + '{}_{}'.format(gnn_type, pooling_type) + ".model"))
    model.eval()

    emb_train, y_train, trace_class_train, url_status_class_train, url_class_list_train = model.encoder.get_embeddings(train_dataloader)
    emb_test, y_test, trace_class_test, url_status_class_test, url_class_list_test = model.encoder.get_embeddings(eval_dataloader)

    np.save(model_path + 'emb_train', emb_train)
    np.save(model_path + 'y_train', y_train)
    np.save(model_path + 'trace_class_train', trace_class_train)
    np.save(model_path + 'url_status_class_train', url_status_class_train)
    np.save(model_path + 'url_class_list_train', url_class_list_train)

    np.save(model_path + 'emb_test', emb_test)
    np.save(model_path + 'y_test', y_test)
    np.save(model_path + 'trace_class_test', trace_class_test)
    np.save(model_path + 'url_status_class_test', url_status_class_test)
    np.save(model_path + 'url_class_list_test', url_class_list_test)



if __name__ == '__main__':
    main()
