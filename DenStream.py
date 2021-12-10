from math import atan
import os.path as osp
from copy import copy
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
import warnings
# from core.encoders import *

from model import simclr
from dataset import TraceClusterDataset
#from aug import TUDataset_aug as TUDataset
from torch_geometric.loader import DataLoader
import sys
import json
from torch import optim

from cortex_DIM.nn_modules.mi_networks import MIFCNet, MI1x1ConvNet
from losses import *

from train import arguments
# from arguments import arg_parse
from torch_geometric.transforms import Constant
from tqdm import tqdm
import pdb

from sklearn.cluster import DBSCAN    # 聚类
from sklearn.manifold import TSNE    # 降维
from sklearn.neighbors import kneighbors_graph
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances


from itertools import cycle, islice

import matplotlib.pyplot as plt

from cluster.DenStream_master.DenStream import DenStream


import random


def setup_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':

    args = arguments()
    setup_seed(args.seed)

    batch_size = args.batch_size

    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data')

    dataset = TraceClusterDataset(path, aug='none')    # 不要 shuffle，按照时间顺序
    dataset_eval = TraceClusterDataset(path, aug='none')
    print("dataset length: {}".format(len(dataset)))

    print("dataset num_features: {}".format(dataset.get_num_feature()))
    try:
        dataset_num_features = dataset.get_num_feature()
    except:
        dataset_num_features = 1

    dataloader = DataLoader(dataset, batch_size=batch_size)
    dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size)

    # #############################################################################
    # Load the pre-trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = simclr(args.hidden_dim, args.num_gc_layers,
                   args.prior, dataset_num_features).to(device)
    model.load_state_dict(torch.load(
        args.save_path + '/' + 'model_weights_epoch20.pth'))    # 做一个软链接？映射？到 latest
    model.eval()

    # #############################################################################
    # Create cluster objects
    denstream = DenStream(eps=0.3, lambd=0.1, beta=0.5, mu=11)

    print('================')
    print('batch_size: {}'.format(batch_size))
    print('num_features: {}'.format(dataset_num_features))
    print('hidden_dim: {}'.format(args.hidden_dim))
    print('num_gc_layers: {}'.format(args.num_gc_layers))
    print('================')

    # #############################################################################
    # Online Stage
    count = 0
    traceid_index = {}    # 需要改进
    # timestamp_list = []
    X_output_gnn = torch.Tensor([]).to(device)    # 需要改进

    # x_test = []

    # x_class0_0 = []
    # x_class0_1 = []
    # x_class0_2 = []
    # x_class0_3 = []

    # x_class1_0 = []
    # x_class1_1 = []
    # x_class1_2 = []
    # x_class1_3 = []

    for data in tqdm(dataloader):
        # print('start')
        data = data[0]
        data = data.to(device)

        timestamp_list = []

        x = model(data.x, data.edge_index, data.edge_attr,
                  data.batch)    # 每个图的特征均表示为一个 tensor
        if data.x.size(0) != data.batch.size(0):
            print("error: x and batch dim dismatch !")

        X_output_gnn = torch.cat((X_output_gnn, x), 0)

        for idx in range(x.size(0)):
            traceid_index[str(count*batch_size+idx)] = data['trace_id'][idx]

            # if data['trace_id'][idx] == '1e3c47720fe24523938fff342ebe6c0d.35.16288656971030003':    # abnormal
            #     x_test = x[idx]
            # if data['trace_id'][idx] == '1e3c47720fe24523938fff342ebe6c0d.35.16288657098040005':    # class 0
            #     x_class0_0 = x[idx]
            # if data['trace_id'][idx] == '1e3c47720fe24523938fff342ebe6c0d.35.16288658127040021':    # class 0
            #     x_class0_1 = x[idx]
            # if data['trace_id'][idx] == '1e3c47720fe24523938fff342ebe6c0d.35.16288659736030045':    # class 0
            #     x_class0_2 = x[idx]
            # if data['trace_id'][idx] == '1e3c47720fe24523938fff342ebe6c0d.35.16288661525040073':    # class 0
            #     x_class0_3 = x[idx]

            # if data['trace_id'][idx] == '3dcc96ad77fe45dfae8436f31379e7ad.38.16294251479940163':    # class 1
            #     x_class1_0 = x[idx]
            # if data['trace_id'][idx] == '3dcc96ad77fe45dfae8436f31379e7ad.38.16294251729850247':    # class 1
            #     x_class1_1 = x[idx]
            # if data['trace_id'][idx] == '3dcc96ad77fe45dfae8436f31379e7ad.38.16294252025170439':    # class 1
            #     x_class1_2 = x[idx]
            # if data['trace_id'][idx] == '3dcc96ad77fe45dfae8436f31379e7ad.38.16294252307460683':    # class 1
            #     x_class1_3 = x[idx]

            timestamp_list.append(data['time_stamp'][idx])

        X_DS_Input = copy(x).detach().cpu().numpy()
        denstream.partial_fit(X_DS_Input, timestamp_list)
        # print(f"Number of p_micro_clusters is {len(denstream.p_micro_clusters)}")
        # print(f"Number of o_micro_clusters is {len(denstream.o_micro_clusters)}")

        count += 1

    # # compute distance
    # # class 0
    # dist = euclidean_distances(x_test.detach().cpu().numpy().reshape(1, -1), x_class0_0.detach().cpu().numpy().reshape(1, -1))
    # print("Distance between x_test and x_class0_0 is: {}".format(dist))
    # dist = euclidean_distances(x_test.detach().cpu().numpy().reshape(1, -1), x_class0_1.detach().cpu().numpy().reshape(1, -1))
    # print("Distance between x_test and x_class0_1 is: {}".format(dist))
    # dist = euclidean_distances(x_test.detach().cpu().numpy().reshape(1, -1), x_class0_2.detach().cpu().numpy().reshape(1, -1))
    # print("Distance between x_test and x_class0_2 is: {}".format(dist))
    # dist = euclidean_distances(x_test.detach().cpu().numpy().reshape(1, -1), x_class0_3.detach().cpu().numpy().reshape(1, -1))
    # print("Distance between x_test and x_class0_3 is: {}".format(dist))
    # # class 1
    # dist = euclidean_distances(x_test.detach().cpu().numpy().reshape(1, -1), x_class1_0.detach().cpu().numpy().reshape(1, -1))
    # print("Distance between x_test and x_class1_0 is: {}".format(dist))
    # dist = euclidean_distances(x_test.detach().cpu().numpy().reshape(1, -1), x_class1_1.detach().cpu().numpy().reshape(1, -1))
    # print("Distance between x_test and x_class1_1 is: {}".format(dist))
    # dist = euclidean_distances(x_test.detach().cpu().numpy().reshape(1, -1), x_class1_2.detach().cpu().numpy().reshape(1, -1))
    # print("Distance between x_test and x_class1_2 is: {}".format(dist))
    # dist = euclidean_distances(x_test.detach().cpu().numpy().reshape(1, -1), x_class1_3.detach().cpu().numpy().reshape(1, -1))
    # print("Distance between x_test and x_class1_3 is: {}".format(dist))

    # tensor --> list  X_input: (num_samples, num_features_graph)
    X_input_db = X_output_gnn.detach().cpu().numpy()

    if len(X_input_db) != len(dataset):
        print("error: sample miss! len(X_input_db) is {} but len(dataset) is {} !".format(
            len(X_input_db), len(dataset)))

    print('================')
    print('num_samples: {}'.format(len(X_input_db)))    # 数据集中样本个数
    # 每个图表示特征维数 hidden-dim*num-gc-layers
    print('num_features: {}'.format(len(X_input_db[0])))
    print('================')

    # #############################################################################
    # Data preprocessing
    # 降维 TSNE
    X_embedded = TSNE(n_components=2, init='pca',
                      random_state=0).fit_transform(X_input_db)
    for idx in range(X_embedded.shape[0]):
        plt.plot(X_embedded[idx][0], X_embedded[idx][1],
                 'o', markeredgecolor='k', markersize=6)
    plt.title("t-SNE embedding of the digits")
    plt.savefig('./TSNE_res.jpg')
    plt.show()

    # #############################################################################
    # Offline stage
    # Compute DenStream
    # default eps=0.3 min_samples=10
    # db = DBSCAN(eps=0.3, min_samples=10, metric='euclidean', metric_params=None).fit(X_input_db)    # eps 和 min_samples 两个超参如何设置
    '''
    eps 表示邻域半径
    min_samples 表示核心点阈值
    '''
    labels = denstream.predict(X_input_db)

    # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)    # (num_samples, )
    # core_samples_mask[db.core_sample_indices_] = True    # 核心对象对应的位置为 True
    # labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)    # 聚类后类别个数
    n_noise_ = list(labels).count(-1)    # 噪声样本个数

    # #############################################################################
    # Print out result
    print('Estimated number of clusters: %d' % n_clusters_)
    for cluster_idx in range(n_clusters_):
        print("Cluster_id {}:".format(cluster_idx))
        for sample_idx in range(len(dataset)):
            if labels[sample_idx] == cluster_idx:
                print("Trace_id: {}".format(traceid_index[str(sample_idx)]))

                # if traceid_index[str(sample_idx)] == '1e3c47720fe24523938fff342ebe6c0d.35.16288656971030003':
                #     print("############################################################################################################")
        print('\n')

    print('\n')
    print('Estimated number of noise points: %d' % n_noise_)
    print("Noise:")
    for sample_idx in range(len(dataset)):
        if labels[sample_idx] == -1:
            print("Trace_id: {}".format(traceid_index[str(sample_idx)]))

    #print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    #print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    #print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    # print("Adjusted Rand Index: %0.3f"
    #      % metrics.adjusted_rand_score(labels_true, labels))
    # print("Adjusted Mutual Information: %0.3f"
    #      % metrics.adjusted_mutual_info_score(labels_true, labels))
    #print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X_input, labels))

    # #############################################################################
    # Plot result
    '''
    import matplotlib.pyplot as plt    # 2 维的图吗？还是画成 3 维的

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
            for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
    '''