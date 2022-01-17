from ast import NotIn
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
    dataset = TraceDataset(root=dataroot).shuffle()
    dataset_train = dataset[:int(7*len(dataset)/10)]
    dataset_eval = dataset[int(7*len(dataset)/10):]
    # dataset_eval = TraceDataset(root=dataroot)

    print('----------------------')
    print("dataset size:", len(dataset))
    print("dataset_train size:", len(dataset_train))
    print("dataset_eval size:", len(dataset_eval))
    print("feature number:", dataset.get_num_feature()[0])
    print("edge feature number:", dataset.get_num_feature()[1])
    print('batch_size: {}'.format(batch_size))
    print('lr: {}'.format(lr))
    print('hidden_dim: {}'.format(args.hidden_dim))
    print('num_gc_layers: {}'.format(args.num_gc_layers))
    print('----------------------')

    # get feature dim
    try:
        dataset_num_features, num_edge_feature = dataset.get_num_feature()
    except:
        dataset_num_features = 1

    # init dataloader
    dataloader = DataLoader(dataset, batch_size=1)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size)
    dataloader_eval = DataLoader(dataset_eval, batch_size=len(dataset_eval))
    
    # get classes
    if args.classes == 'binary':
        num_classes = 2
    elif args.classes == 'multi':
        multiLabel = {}
        classCount = 0
        for data in tqdm(dataloader):
            if data.root_url[0] not in multiLabel:
                multiLabel[data.root_url[0]] = classCount
                classCount = classCount + 1
        num_classes = len(multiLabel)


    # init model
    model = Encoder(dataset_num_features, args.hidden_dim, args.num_gc_layers, num_classes, num_edge_feature).to(device)

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
            x = model(data.x, data.edge_index, data.edge_attr, data.batch)    # batchsize*2
            if args.classes == 'binary':
                y = data.y    # batchsize
            elif args.classes == 'multi':
                y = torch.Tensor([multiLabel[data.root_url[i]] for i in range(len(data.root_url))]).to(device).long()    # batchsize

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
    data = [data for data in dataloader_eval][0].to(device)
    if args.classes == 'binary':
        y = data.y
    elif args.classes == 'multi':
        y = torch.Tensor([multiLabel[data.root_url[i]] for i in range(len(data.root_url))]).to(device).long()
    print("Accuracy score is {}".format(accuracy_score(model.predict(data.x, data.edge_index, data.edge_attr, data.batch).cpu().numpy(), y.cpu().numpy())))



if __name__ == '__main__':
    main()