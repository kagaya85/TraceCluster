import torch
import argparse
import random
import os
import numpy as np
import json

from dataset import TraceClusterDataset
from model import simclr

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
    parser.add_argument('--lr', dest='lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=5,
                        help='Number of graph convolution layers before each pooling')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=32,
                        help='')
    parser.add_argument('--aug', type=str, default='dnodes')
    parser.add_argument('--seed', type=int, default=0)

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
    accuracies = {'val': [], 'test': []}
    epochs = 20
    log_interval = 10
    batch_size = 128

    args = arguments()
    set_random_seed(args.seed)

    dataroot = os.path.join('.', 'data', 'processed')
    dirname = '2021-09-02_13-06-3' if args.dataset == '' else args.dataseet

    # init dataset
    dataset = TraceClusterDataset(
        root=dataroot, name=dirname, aug=args.aug).shuffle()
    dataset_eval = TraceClusterDataset(
        root=dataroot, name=dirname, aug='none').shuffle()
    print("dataset size:", len(dataset))
    print("feature numbers:", dataset.get_num_feature())

    try:
        feat_num = dataset.get_num_feature()
    except:
        feat_num = 1

    # init dataloader
    loader = DataLoader(dataset, batch_size=batch_size)
    loader_eval = DataLoader(dataset_eval, batch_size=batch_size)

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = simclr(args.hidden_dim, args.num_gc_layers,
                   args.prior, feat_num).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print('----------------------')
    print('lr: {}'.format(args.lr))
    print('feat_num: {}'.format(feat_num))
    print('hidden_dim: {}'.format(args.hidden_dim))
    print('num_gc_layers: {}'.format(args.num_gc_layers))
    print('----------------------')

    model.eval()
    emb, y = model.encoder.get_embeddings(loader_eval)
    print('embedding shape:', emb.shape)
    print('y shape:', y.shape)

    # training
    for epoch in range(1, epochs+1):
        loss_all = 0
        model.train()
        for data in loader:
            data, data_aug = data
            optimizer.zero_grad()

            node_num, _ = data.x.size()
            data = data.to(device)
            x = model(data.x, data.edge_index, data.batch, data.num_graphs)

            if args.aug == 'dnodes' or args.aug == 'subgraph' or args.aug == 'random2' or args.aug == 'random3' or args.aug == 'random4':
                edge_idx = data_aug.edge_index.numpy()
                _, edge_num = edge_idx.shape
                idx_not_missing = [n for n in range(node_num) if (
                    n in edge_idx[0] or n in edge_idx[1])]

                node_num_aug = len(idx_not_missing)
                data_aug.x = data_aug.x[idx_not_missing]

                data_aug.batch = data.batch[idx_not_missing]
                idx_dict = {idx_not_missing[n]: n for n in range(node_num_aug)}
                edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]]
                            for n in range(edge_num) if not edge_idx[0, n] == edge_idx[1, n]]
                data_aug.edge_index = torch.tensor(edge_idx).transpose_(0, 1)

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
                          data_aug.batch, data_aug.num_graphs)

            # print(x)
            # print(x_aug)
            loss = model.loss_cal(x, x_aug)
            print(loss)
            loss_all += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()

        print('Epoch {}, Loss {}'.format(epoch, loss_all / len(loader)))

        if epoch % log_interval == 0:
            model.eval()
            emb, y = model.encoder.get_embeddings(loader_eval)
            acc_val, acc = evaluate_embedding(emb, y)
            accuracies['val'].append(acc_val)
            accuracies['test'].append(acc)
            # print(accuracies['val'][-1], accuracies['test'][-1])

    tpe = ('local' if args.local else '') + ('prior' if args.prior else '')
    with open('logs/log_' + args.DS + '_' + args.aug, 'a+') as f:
        s = json.dumps(accuracies)
        f.write('{},{},{},{},{},{},{}\n'.format(args.DS, tpe,
                args.num_gc_layers, epochs, log_interval, args.lr, s))
        f.write('\n')


if __name__ == '__main__':
    main()
