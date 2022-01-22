import torch
import torch.nn as nn

from torch.nn import ModuleList
import torch_geometric
from torch_geometric.nn import BatchNorm, GATConv, global_mean_pool
import numpy as np




class Encoder(nn.Module):
    def __init__(self, num_layers=2, input_dim=20, output_dim=16):
        super(Encoder, self).__init__()

        self.convs = ModuleList()
        self.batch_norms = ModuleList()

        conv_1 = GATConv(in_channels=input_dim, out_channels=input_dim*3)
        conv_2 = GATConv(in_channels=input_dim*3, out_channels=output_dim)
        bn_1 = BatchNorm(input_dim*3)
        bn_2 = BatchNorm(output_dim)

        self.convs.append(conv_1)
        self.convs.append(conv_2)

        self.batch_norms.append(bn_1)
        self.batch_norms.append(bn_2)

    def forward(self, x, edge_index, edge_attr, batch):

        xs = []
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = batch_norm(conv(x, edge_index))
            xs.append(x)

        x = global_mean_pool(x, batch)
        return x

    def get_embeddings(self, dataloader):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ret = []
        y = []
        trace_ids = []
        with torch.no_grad():
            for data in dataloader:
                data = data[0]
                data.to(device)
                x = self.forward(data)

                ret.append(x.cpu().numpy())
                y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y
    
    


class SIMCLR(nn.Module):
    def __init__(self, num_layers=2, input_dim=20, output_dim=16):
        super(SIMCLR, self).__init__()

        self.output_dim = output_dim

        self.encoder = Encoder(num_layers=num_layers, input_dim=input_dim, output_dim=output_dim)

        self.proj_head = nn.Sequential(nn.Linear(output_dim, output_dim), nn.LeakyReLU(),
                                       nn.Linear(output_dim, output_dim))
        self.init_emb()


    def init_emb(self):
        initrange = -1.5 / self.output_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, edge_attr, batch):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # batch_size = data.num_graphs
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)

        y = self.encoder(x, edge_index, edge_attr, batch)

        y = self.proj_head(y)

        return y

    def loss_cal(self, x, x_aug):

        T = 0.2
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        return loss


