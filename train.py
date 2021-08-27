from typing import Callable, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_scatter import scatter_mean

from dataset import TraceDataset
from model import Net

dataset = TraceDataset(root='/tmp/tracedata',
                       name='trace', use_node_attr=True)


def main():
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
