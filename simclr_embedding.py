import torch
import json
import random
import matplotlib.pyplot as plt

import numpy as np

from simclr import SIMCLR
from aug_dataset_mem import TraceDataset
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from utils import get_target_label_idx
from eval_method import oc_svm_classify, lof_detection, evaluate_embedding, isforest_classify
from sklearn.manifold import TSNE

def main():
    model_path = 'E:\Data\TraceCluster\log\\2-week-test\\final_test\\20epoch_CGConv_mean_2layers_32batch_random_0301data_select_normal\\'
    batch_size = 128
    normal_classes = [0]
    abnormal_classes = [1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eval_normal_dataset = TraceDataset(root='E:\Data\TraceCluster\\0301-data\\final_test_normal_mem', aug='none')
    eval_abnormal_dataset = TraceDataset(root='E:\Data\TraceCluster\\0301-data\\abnormal_nocode_mem', aug='none')
    dataset = TraceDataset(root='E:\Data\TraceCluster\\0301-data\\final_train_normal_mem', aug='none')

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
        test_idx = model_info['test_idx']

    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    eval_normal_dataloader = DataLoader(eval_normal_dataset, batch_size=batch_size, shuffle=True)
    eval_abnormal_dataloader = DataLoader(eval_abnormal_dataset, batch_size=batch_size, shuffle=True)
    # eval_abnormal_dataloader = DataLoader(eval_abnormal_dataset, batch_size=batch_size, shuffle=True)

    model = SIMCLR(num_layers=num_layers, input_dim=dataset.num_node_features, output_dim=output_dim,
                   num_edge_attr=dataset.num_edge_features, gnn_type=gnn_type,
                   pooling_type=pooling_type).to(device)
    model.load_state_dict(torch.load(model_path + '{}_{}'.format(gnn_type, pooling_type) + ".model"))
    model.eval()

    emb_train, y_train, trace_class_train, url_status_class_train, url_class_list_train = model.encoder.get_embeddings(train_dataloader)
    emb_normal_test, y_normal_test, trace_class_test, url_status_class_test, url_class_list_test = model.encoder.get_embeddings(eval_normal_dataloader)
    emb_ab_test, y_ab_test, trace_class_test, url_status_class_test, url_class_list_test = model.encoder.get_embeddings(
        eval_abnormal_dataloader)
    # emb_ab1_test, y_ab1_test, _, _, _ = model.encoder.get_embeddings(
    #     eval_ab1_dataloader)

    emb_test = np.append(emb_normal_test, emb_ab_test, axis=0)
    y_test = np.append(y_normal_test, y_ab_test)

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

    np.save(model_path + 'emb_ab_test', emb_ab_test)
    np.save(model_path + 'y_ab_test', y_ab_test)

    np.save(model_path + 'emb_normal_test', emb_normal_test)
    np.save(model_path + 'y_normal_test', y_normal_test)

    # lof_detection(emb_train, emb_test, y_train, y_test, [])
    # oc_svm_classify(emb_train, emb_test, y_train, y_test, nu=0.01, kernel='poly')
    # oc_svm_classify(emb_train, emb_test, y_train, y_test, nu=0.01, kernel='linear')
    # oc_svm_classify(emb_train, emb_test, y_train, y_test, nu=0.01, kernel='rbf')
    # oc_svm_classify(emb_train, emb_test, y_train, y_test, nu=0.03, kernel='poly')
    # oc_svm_classify(emb_train, emb_test, y_train, y_test, nu=0.03, kernel='linear')
    # oc_svm_classify(emb_train, emb_test, y_train, y_test, nu=0.03, kernel='rbf')
    # oc_svm_classify(emb_train, emb_test, y_train, y_test, nu=0.05, kernel='poly')
    # oc_svm_classify(emb_train, emb_test, y_train, y_test, nu=0.05, kernel='linear')
    # oc_svm_classify(emb_train, emb_test, y_train, y_test, nu=0.05, kernel='rbf')
    # oc_svm_classify(emb_train, emb_test, y_train, y_test, nu=0.09, kernel='poly')
    # oc_svm_classify(emb_train, emb_test, y_train, y_test, nu=0.09, kernel='linear')
    # oc_svm_classify(emb_train, emb_test, y_train, y_test, nu=0.09, kernel='rbf')
    # isforest_classify(emb_train, emb_test, y_train, y_test)


    # target_class = dataset.url_classes.index('POST:/api/v1/orderservice/order/refresh')
    # target_idx = get_target_label_idx(dataset.data.url_class.clone().data.cpu().numpy(), target_class)
    # dataset = Subset(dataset, target_idx)
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

    # normal_idx = dataset.normal_idx
    # abnormal_idx = dataset.abnormal_idx

    # train_normal_idx = list(set(normal_idx).intersection(set(train_idx)))
    # test_idx = test_idx = list(set(normal_idx + abnormal_idx).difference(set(train_normal_idx)))

    #
    # random.shuffle(abnormal_idx)
    # random.shuffle(normal_idx)
    #
    # train_normal = normal_idx[:int(len(normal_idx) * 0.6)]
    # train_abnormal = abnormal_idx[:int(len(abnormal_idx) * 0)]
    # test_normal = normal_idx[int(len(normal_idx) * 0.6):]
    # test_abnormal = abnormal_idx[int(len(abnormal_idx) * 0):]
    # train_idx = train_normal + train_abnormal
    #
    # test_idx = test_abnormal + test_normal
    # test_idx = list(set(normal_idx + abnormal_idx).difference(set(train_idx)))

    # test_normal_idx = list(set(normal_idx).intersection(set(test_idx)))
    # test_abnormal_idx = list(set(abnormal_idx).intersection(set(test_idx)))

    # eval_dataset = Subset(dataset, test_idx)
    # tsne = TSNE()
    # data_embedding = np.concatenate([emb_train, emb_test])
    # labels = np.concatenate([y_train, y_test])
    # trace_class = np.concatenate([trace_class_train, trace_class_test])
    # url_status_class = np.concatenate([url_status_class_train, url_status_class_test])
    # url_class_list = np.concatenate([url_class_list_train, url_class_list_test])
    # x = tsne.fit_transform(data_embedding)
    # plt.scatter(x[:, 0], x[:, 1], c=labels, marker='o', s=10, cmap=plt.cm.Spectral)
    # plt.show()

if __name__ == '__main__':
    main()
