import torch
import argparse
import random
import os
import numpy as np
from tqdm import tqdm

from dataset import TraceClusterDataset
from model import simclr, GcnInfomax

from torch_geometric.data import DataLoader
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn import preprocessing
from sklearn.metrics import accuracy_score


def arguments():
    parser = argparse.ArgumentParser(description="GNN Argumentes.")
    parser.add_argument('--dataset', dest='dataset', help='Dataset')
    parser.add_argument('--local', dest='local', action='store_const',
                        const=True, default=False)
    parser.add_argument('--glob', dest='glob', action='store_const',
                        const=True, default=False)
    parser.add_argument('--prior', dest='prior', action='store_const',
                        const=True, default=False)
    parser.add_argument('--lr', dest='lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=2,
                        help='Number of graph convolution layers before each pooling')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=16,    # 32
                        help='')
    parser.add_argument('--aug', type=str, default='random3')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--save-to', dest='save_path',
                        default='./weights', help='Save path.')
    parser.add_argument('--epochs', dest='epochs', type=int, default=20,
                        help='')
    parser.add_argument('--log-interval', dest='log_interval', type=int, default=1,
                        help='Log interval.')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=64,
                        help='Batch size.')    # 128

    return parser.parse_args()


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def svc_classify(x, y, search):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    accuracies_val = []
    for train_index, test_index in kf.split(x, y):
        # test
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
        if search:
            params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            classifier = GridSearchCV(
                SVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))

        val_size = len(test_index)
        test_index = np.random.choice(
            train_index, val_size, replace=False).tolist()
        train_index = [i for i in train_index if not i in test_index]

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
        if search:
            params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            classifier = GridSearchCV(
                SVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies_val.append(accuracy_score(
            y_test, classifier.predict(x_test)))

    return np.mean(accuracies_val), np.mean(accuracies)


def evaluate_embedding(embeddings, labels, search=True):
    labels = preprocessing.LabelEncoder().fit_transform(labels)
    x, y = np.array(embeddings), np.array(labels)

    acc = 0
    acc_val = 0

    _acc_val, _acc = svc_classify(x, y, search)
    if _acc_val > acc_val:
        acc_val = _acc_val
        acc = _acc

    print(acc_val, acc)

    return acc_val, acc


def main():
    args = arguments()
    print('----------------------')

    # accuracies = {'val': [], 'test': []}
    epochs = args.epochs
    # log_interval = args.log_interval
    batch_size = args.batch_size
    lr = args.lr

    set_random_seed(args.seed)
    dataroot = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), '.', 'data')

    # init dataset
    dataset = TraceClusterDataset(
        root=dataroot, aug=args.aug).shuffle()
    dataset_eval = TraceClusterDataset(
        root=dataroot, aug='none').shuffle()
    print("dataset size:", len(dataset))

    # get feature dim
    print("feature number:", dataset.get_num_feature())
    try:
        dataset_num_features = dataset.get_num_feature()
    except:
        dataset_num_features = 1

    # init dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size)
    # dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size)

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = simclr(args.hidden_dim, args.num_gc_layers,
                   args.prior, dataset_num_features).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print('batch_size: {}'.format(batch_size))
    print('lr: {}'.format(lr))
    print('hidden_dim: {}'.format(args.hidden_dim))
    print('num_gc_layers: {}'.format(args.num_gc_layers))
    print('----------------------')

    # model.eval()
    # emb, y = model.encoder.get_embeddings(dataloader_eval)
    # print('embedding shape:', emb.shape)
    # print('y shape:', y.shape)

    # training
    for epoch in range(1, epochs+1):
        loss_all = 0
        model.train()
        for data in tqdm(dataloader):

            # print('start')
            data, data_aug = data
            optimizer.zero_grad()

            node_num, _ = data.x.size()
            data = data.to(device)

            x = model(data.x, data.edge_index, data.edge_attr, data.batch)
            if data.x.size(0) != data.batch.size(0):
                print("error: x and batch dim dismatch !")
            # print("x.shape: {}".format(x.shape))

            # 对 data.x 点特征进行处理
            if args.aug == 'dnodes' or args.aug == 'pedges' or args.aug == 'subgraph' or args.aug == 'mask_nodes' or args.aug == 'random2' or args.aug == 'random3' or args.aug == 'random4':
                # node_num_aug, _ = data_aug.x.size()
                edge_idx = data_aug.edge_index.numpy()
                _, edge_num = edge_idx.shape
                idx_not_missing = [n for n in range(node_num) if (
                    n in edge_idx[0] or n in edge_idx[1])]

                node_num_aug = len(idx_not_missing)
                data_aug.x = data_aug.x[idx_not_missing]

                data_aug.batch = data.batch[idx_not_missing]

                idx_dict = {idx_not_missing[n]: n for n in range(
                    node_num_aug)}    # 每个未丢的点所对应的位置

                # 将点的序列变成连续的，由于之前数据增强（如 dnodes）将部分点删除使得点序列不连续，edge_index.max()+1 不能表示 node_num，故这几步用于对齐点序列
                edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]]
                            for n in range(edge_num) if not edge_idx[0, n] == edge_idx[1, n]]

                if len(edge_idx) != 0:
                    data_aug.edge_index = torch.tensor(
                        edge_idx).transpose_(0, 1)

            data_aug = data_aug.to(device)

            '''
            print(data.edge_index)
            print(data.edge_index.size())
            print(data_aug.edge_index)
            print(data_aug.edge_index.size())
            print(data.x.size())
            print(data_aug.x.size())
            print(data.batch.size())
            print(data_aug.batch.size())
            pdb.set_trace()
            '''

            x_aug = model(data_aug.x, data_aug.edge_index,
                          data_aug.edge_attr, data_aug.batch)
            # print("x_aug.shape: {}".format(x.aug.shape))

            # print(x)
            # print(x_aug)
            # x_aug = x

            loss = model.loss_cal(x, x_aug)
            # print("loss: {}".format(loss))
            loss_all += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()
            # print('batch')
        print('Epoch {}, Loss {}'.format(epoch, loss_all / len(dataloader)))

        # save model
        filename = os.path.join(
            args.save_path, 'model_weights_epoch{}.pth'.format(epoch))
        torch.save(model.state_dict(), filename)
        print(f"Saving model to {filename}")


if __name__ == '__main__':
    main()
