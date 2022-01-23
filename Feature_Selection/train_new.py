from ast import NotIn, arg
from audioop import bias
from copy import deepcopy
import torch
import argparse
import random
import os
import numpy as np
from tqdm import tqdm

from original_dataset import TraceDataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score

from model import Encoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def arguments():
    parser = argparse.ArgumentParser(description="gnn argumentes")
    parser.add_argument('--wechat', dest='wechat',
                        help='use wechat data', action='store_true', default=False)
    parser.add_argument('--dataset', dest='dataset',
                        help='use other preprocessed data dirpath, eg. /data/TraceCluster/preprocessed/trainticket', default="/data/TraceCluster/preprocessed/trainticket")
    parser.add_argument('--local', dest='local', action='store_const',
                        const=True, default=False)
    parser.add_argument('--glob', dest='glob', action='store_const',
                        const=True, default=False)
    parser.add_argument('--prior', dest='prior', action='store_const',
                        const=True, default=False)
    parser.add_argument('--lr', dest='lr', type=float, default=0.001,
                        help='learning rate, default 0.001')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=2,
                        help='number of graph convolution layers before each pooling')
    parser.add_argument('--gnn-type', dest='gnn_type', type=str, default='TransformerConv',
                        help='choose a GNN type, eg. TransformerConv, GATConv, CGConv')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=16,    # 32
                        help='hidden dim number, default 16')
    parser.add_argument('--aug', type=str, default='random3')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--classes', dest='classes', type=str, default='binary',
                        help='binary classification or multi classification, eg. binary or multi')
    parser.add_argument('--save-to', dest='save_path',
                        default='./weights', help='weights save path')
    parser.add_argument('--epochs', dest='epochs', type=int, default=20,    # 20
                        help='')
    parser.add_argument('--log-interval', dest='log_interval', type=int, default=1,
                        help='log interval.')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=64,
                        help='batch size')    # 128
    return parser.parse_args()


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def main():
    args = arguments()

    # accuracies = {'val': [], 'test': []}
    epochs = args.epochs
    # log_interval = args.log_interval
    batch_size = args.batch_size
    lr = args.lr

    set_random_seed(args.seed)
    dataroot = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), '.', 'data')

    # init dataset
    dataset = TraceDataset(root=dataroot)
    # get dataset index
    # index = [i for i in range(len(dataset))]
    # np.random.shuffle(index)

    # fw=open('./Feature_Selection/data/indexList.txt', 'w')
    # fw.write(str(index).replace('[', '').replace(']', ''))
    # fw.close()

    fr=open('./Feature_Selection/data/indexList.txt', 'r')
    S = fr.read()
    index = [int(index_item) for index_item in S.split(', ')]

    dataset_List = []
    dataset_List_0 = []
    dataset_List_1 = []
    # create trainSet and evalSet
    print("Create trainSet and evalSet ...")
    for index_item in tqdm(index):
        if dataset[index_item].root_url == '{POST}/api/v1/inside_pay_service/inside_payment':
            dataset_List.append(index_item)
            if dataset[index_item].y == 0:
                dataset_List_0.append(index_item)
            elif dataset[index_item].y == 1:
                dataset_List_1.append(index_item)

    index_train = dataset_List_0[:int(7*len(dataset_List_0)/10)] + dataset_List_1[:int(7*len(dataset_List_1)/10)]
    index_eval = dataset_List_0[int(7*len(dataset_List_0)/10):] + dataset_List_1[int(7*len(dataset_List_1)/10):]

    # index_eval_random = random.sample(dataset_List_0[int(7*len(dataset_List_0)/10):], int(len(dataset_List_0[int(7*len(dataset_List_0)/10):])/10))
    # traceID_list = [dataItem.trace_id for dataItem in dataset[index_eval_random]]
    # fw_eval=open('./Feature_Selection/data/traceIDList_eval.txt', 'w')
    # fw_eval.write(str(traceID_list).replace('[', '').replace(']', '').replace('\'', ''))
    # fw_eval.close()

    fr_eval=open('./Feature_Selection/data/traceIDList_eval.txt', 'r')
    S_eval = fr_eval.read()
    traceID_list_eval = [traceID for traceID in S_eval.split(', ')]


    print("URL dataset size:", len(dataset_List))
    print("URL dataset size y=0:", len(dataset_List_0))
    print("URL dataset size y=1:", len(dataset_List_1)) 

    dataset_train = dataset[index_train]
    dataset_eval = dataset[index_eval]

    print('----------------------')
    print("dataset size:", len(dataset))
    print("dataset_train size:", len(dataset_train))
    print("dataset_eval size:", len(dataset_eval))
    print("feature number:", dataset.get_num_feature()[0])
    print("edge feature number:", dataset.get_num_feature()[1])
    print("batch_size: {}".format(batch_size))
    print("lr: {}".format(lr))
    print("hidden_dim: {}".format(args.hidden_dim))
    print("num_gc_layers: {}".format(args.num_gc_layers))
    print("gnn_type: {}".format(args.gnn_type))
    print('----------------------')


    # get feature dim
    try:
        dataset_num_features, num_edge_feature = dataset.get_num_feature()
        if (num_edge_feature == 0 and args.gnn_type == 'TransformerConv') or (num_edge_feature == 0 and args.gnn_type == 'GATConv'):
            num_edge_feature = None
    except:
        dataset_num_features = 1

    # init dataloader
    dataloader = DataLoader(dataset, batch_size=1)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size)
    # dataloader_eval = DataLoader(dataset_eval, batch_size=len(dataset_eval))
    dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size)
    
    # get classes
    if args.classes == 'binary':
        num_classes = 2
    elif args.classes == 'multi':
        multiLabel = {}
        classCount = 0
        for data in tqdm(dataloader):
            if data.root_url[0] not in multiLabel:
                multiLabel[data.root_url[0]] = [classCount, 1]
                classCount = classCount + 1
            else:
                multiLabel[data.root_url[0]][1] = multiLabel[data.root_url[0]][1] + 1
        num_classes = len(multiLabel)


    # init model
    model = Encoder(dataset_num_features, args.hidden_dim, args.num_gc_layers, num_classes, num_edge_feature, args.gnn_type).to(device)

    # init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(),lr=0.02)

    # init loss function
    loss_func = torch.nn.CrossEntropyLoss()

    # training model
    for epoch in range(1, epochs+1):
        print("Epoch {} training ...".format(epoch))
        model.train()
        loss_sum = 0.0
        for data in tqdm(dataloader_train):
            data = data.to(device)
            if num_edge_feature == None or num_edge_feature == 0:
                data.edge_attr = None
            x = model(data.x, data.edge_index, data.edge_attr, data.batch)    # batchsize*2
            if args.classes == 'binary':
                y = data.y    # batchsize
            elif args.classes == 'multi':
                y = torch.Tensor([multiLabel[data.root_url[i]][0] for i in range(len(data.root_url))]).to(device).long()    # batchsize

            loss = loss_func(x, y)
            loss_sum = loss_sum + loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print("Accuracy score is {}".format(accuracy_score(model.predict(data.x, data.edge_index, data.batch).cpu().numpy(), y.cpu().numpy())))

        print("Epoch: {}, loss: {}".format(epoch, loss_sum/batch_size))

    # test model
    print("Test model ...")
    model.eval()
    pred_y_x10 = torch.Tensor([]).to(device)
    pred_y_x1 = torch.Tensor([]).to(device)
    true_y_x10 = torch.Tensor([]).to(device)
    true_y_x1 = torch.Tensor([]).to(device)
    for data in dataloader_eval:
        data = data.to(device)
        if args.classes == 'binary':
            # original edge_attr
            true_y_x1 = torch.cat([true_y_x1, data.y], dim=0)
            # edge_attr x10
            true_y_bias = []
            for traceID in data.trace_id:
                if traceID in traceID_list_eval:
                    true_y_bias.append(1)
                else:
                    true_y_bias.append(0)
            true_y_x10 = torch.cat([true_y_x10, data.y+torch.Tensor(true_y_bias).to(device)], dim=0)
                       
        elif args.classes == 'multi':
            true_y = torch.cat([true_y, torch.Tensor([multiLabel[data.root_url[i]][0] for i in range(len(data.root_url))]).to(device).long()], dim=0) 
        
        if num_edge_feature != None and num_edge_feature != 0:
            # original edge_attr
            pred_y_x1 = torch.cat([pred_y_x1, model.predict(data.x, data.edge_index, data.edge_attr, data.batch)], dim=0) 
            # edge_attr x10
            edge_attr_bias = []
            for traceID in data.trace_id:
                if traceID in traceID_list_eval:
                    edge_attr_bias.append([10])
                else:
                    edge_attr_bias.append([1])
            pred_y_x10 = torch.cat([pred_y_x10, model.predict(data.x, data.edge_index, torch.Tensor(edge_attr_bias).to(device)*data.edge_attr, data.batch)], dim=0)
        elif num_edge_feature == None or num_edge_feature == 0:
            data.edge_attr = None 
    
    if num_edge_feature != None and num_edge_feature != 0:
        accuracyScore_1 = accuracy_score(pred_y_x1.cpu().numpy(), true_y_x1.cpu().numpy())
        print("Accuracy score 1 (x1) is {}".format(accuracyScore_1))
    elif num_edge_feature == None or num_edge_feature == 0:
        print("Accuracy score 1 is not available !")
    accuracyScore_2 = accuracy_score(pred_y_x10.cpu().numpy(), true_y_x10.cpu().numpy())
    print("Accuracy score 2 (x10) is {}".format(accuracyScore_2))



if __name__ == '__main__':
    main()