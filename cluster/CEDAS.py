from math import atan
import os.path as osp
from copy import copy
from numpy.lib.function_base import rot90
from numpy.linalg.linalg import _norm_dispatcher
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

# from sklearn.cluster import DBSCAN    # 聚类
from sklearn.manifold import TSNE    # 降维
from sklearn.neighbors import kneighbors_graph
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances


from itertools import cycle, islice

import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

from CEDAS_master.CEDAS import CEDAS, distance

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

    batch_size = 1
    
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data')

    dataset = TraceClusterDataset(path, aug='none')    # 不要 shuffle，按照时间顺序
    # dataset_eval = TraceClusterDataset(path, aug='none')
    print("dataset length: {}".format(len(dataset)))

    print("dataset num_features: {}".format(dataset.get_num_feature()))
    try:
        dataset_num_features = dataset.get_num_feature()
    except:
        dataset_num_features = 1

    dataloader = DataLoader(dataset, batch_size=batch_size)
    # dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size)


    # #############################################################################
    # Load the pre-trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = simclr(args.hidden_dim, args.num_gc_layers, args.prior, dataset_num_features).to(device)
    model.load_state_dict(torch.load(args.save_path + '/' + 'model_weights_epoch20.pth'))    # 做一个软链接？映射？到 latest
    model.eval()


    # #############################################################################
    # Create cluster objects
    cedas = CEDAS(
            r0=0.2,    # ?????????????????
            decay=0.001,    # ??????????????????
            threshold=5,    # ??????????????????
        )
    

    print('================')
    print('batch_size: {}'.format(batch_size))
    print('num_features: {}'.format(dataset_num_features))
    print('hidden_dim: {}'.format(args.hidden_dim))
    print('num_gc_layers: {}'.format(args.num_gc_layers))
    print('================')


    
    traceid_index = {}    # 需要改进
    # timestamp_list = []
    X_output_gnn = torch.Tensor([]).to(device)    # 需要改进

    
    for idx, data in tqdm(enumerate(dataloader)):
        # print('start')
        data = data[0]
        data = data.to(device)

        timestamp_list = []
            
        x = model(data.x, data.edge_index, data.edge_attr, data.batch)    # 每个图的特征均表示为一个 tensor
        # if data.x.size(0) != data.batch.size(0):
        #     print("error: x and batch dim dismatch !")
        
        if idx == 0:
            # #############################################################################
            # 1. Initialization
            cedas.initialization(x.detach().cpu().numpy())
        else:
            X_output_gnn = torch.cat((X_output_gnn, x), 0)

            traceid_index[str(idx)] = data['trace_id']
            timestamp_list.append(data['time_stamp'])

            # for idx in range(x.size(0)):
            #     traceid_index[str(count*batch_size+idx)] = data['trace_id'][idx]
            #     timestamp_list.append(data['time_stamp'][idx])
        
        
            cedas.changed_cluster = None
            # #############################################################################
            # 2. Update Micro-Clusters
            cedas.update(x.detach().cpu().numpy())
            # #############################################################################
            # 3. Kill Clusters
            cedas.kill()

            if cedas.changed_cluster and cedas.changed_cluster.count > cedas.threshold:
                # #############################################################################
                # 4. Update Cluster Graph
                cedas.update_graph()

    
    X_input_db = X_output_gnn.detach().cpu().numpy()    # tensor --> list  X_input: (num_samples, num_features_graph)

    if len(X_input_db) != len(dataset):
        print("error: sample miss! len(X_input_db) is {} but len(dataset) is {} !".format(len(X_input_db), len(dataset)))
    
    print('================')
    print('num_samples: {}'.format(len(X_input_db)))    # 数据集中样本个数
    print('num_features: {}'.format(len(X_input_db[0])))    # 每个图表示特征维数 hidden-dim*num-gc-layers
    print('================')



    # #############################################################################
    # 降维 TSNE
    X_embedded = TSNE(n_components=2, init='pca', random_state=0).fit_transform(X_input_db)
    for idx in range(X_embedded.shape[0]):
        plt.plot(X_embedded[idx][0], X_embedded[idx][1], 'o', markeredgecolor='k', markersize=6)
    plt.title("t-SNE embedding of the digits")
    plt.savefig('./TSNE_res_1.jpg')
    plt.show()



    print('Estimated number of clusters: {}'.format(len(cedas.get_macro_cluster())))


    for idx, data in enumerate(dataloader):
        data = data[0]
        data = data.to(device)
            
        x = model(data.x, data.edge_index, data.edge_attr, data.batch)    # 每个图的特征均表示为一个 tensor
        
        classid = -1
        # find nearest micro cluster
        nearest_cluster = min(
            cedas.micro_clusters,
            key=lambda cluster: distance(x.detach().cpu().numpy(), cluster.centre),
        )
        min_dist = distance(x.detach().cpu().numpy(), nearest_cluster.centre)

        if nearest_cluster.count > cedas.threshold and min_dist < cedas.r0:
            for i, macroC in enumerate(cedas.get_macro_cluster()):
                if nearest_cluster in macroC:
                    classid = i
                    break

        print("id: {}, Trace_id: {}, Class_id: {}".format(str(idx), data['trace_id'], str(classid)))