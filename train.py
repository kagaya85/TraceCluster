from typing import Callable, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import argparse
import random
from torch.random import seed
from torch_geometric.data import DataLoader
from torch_scatter import scatter_mean

from dataset import TraceDataset
from model import Net


def arguments():
    parser = argparse.ArgumentParser(description="GNN Argumentes.")
    parser.add_argument('--dataset', dest='dataset', help='Dataset')
    parser.add_argument('--local', dest='local', action='store_const',
                        const=True, default=False)
    parser.add_argument('--glob', dest='glob', action='store_const',
                        const=True, default=False)
    parser.add_argument('--prior', dest='prior', action='store_const',
                        const=True, default=False)
    parser.add_argument('--lr', dest='lr', type=float,
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


def main():
    args = arguments()
    set_random_seed(args.seed)

    dataroot = os.path.join('.', 'data', 'processed')
    dirname = '2021-09-02_13-06-3' if args.dataset == '' else args.dataseet

    # init dataset
    dataset = TraceDataset(root=dataroot, name=dirname, use_node_attr=True)
    print("dataset size:", len(dataset))
    print("feature numbers:", dataset.get_num_feature())

    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for data in loader:
        x = scatter_mean(data.x, data.batch, dim=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    _, pred = model(data).max(dim=1)
    correct = float(pred[data.test_mask].eq(
        data.y[data.test_mask]).sum().item())

    acc = correct / data.test_mask.sum().item()
    print('Accuracy: {:.4f}'.format(acc))


if __name__ == '__main__':
    main()
