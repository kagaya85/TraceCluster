import os.path as osp

import time
import torch
import torch.nn.functional as F
import torch.nn as nn
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch.nn import ModuleList, Embedding
from torch.nn import Sequential, ReLU, Linear, LeakyReLU
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.utils import degree
from torch_geometric.datasets import ZINC, MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import PNAConv, BatchNorm, global_add_pool, global_mean_pool


epochs = 20
learning_rate = 0.001
path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', 'HIV')

dataset = MoleculeNet(root=path, name='HIV')

train_dataset = dataset[:30000]
test_dataset = dataset[30000:]

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128)

deg = torch.zeros(11, dtype=torch.long)
for data in train_dataset:
    d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
    deg += torch.bincount(d, minlength=deg.numel())


class PNANet(nn.Module):
    def __init__(self, num_layers=2):
        super(PNANet, self).__init__()

        self.node_emb = AtomEncoder(emb_dim=100)
        self.edge_emb = BondEncoder(emb_dim=100)

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']


        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(num_layers):
            conv = PNAConv(in_channels=100, out_channels=100, aggregators=aggregators, scalers=scalers,
                           deg=deg, edge_dim=100)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(100))
        self.mlp = nn.Sequential(Linear(100, 40), LeakyReLU(), Linear(40, 20), LeakyReLU(), Linear(20, 1))

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.leaky_relu(batch_norm(conv(x, edge_index, edge_attr)))

        x = global_mean_pool(x,batch)
        return self.mlp(x)



# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = torch.device('cpu')
model = PNANet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.BCEWithLogitsLoss()

model.train()


for epoch in range(epochs):
    start_time = time.time()
    total_loss = 0
    total = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = criterion(out, data.y)
        total_loss += loss.item()
        total += 1
        optimizer.step()

    total_loss = total_loss / total
    print('Epoch: %3d/%3d, Train Loss: %.5f' % (epoch + 1, epochs, total_loss))










