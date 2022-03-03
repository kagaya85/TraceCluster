import torch
import torch.nn as nn

from torch.nn import ModuleList
from torch.nn import Sequential, ReLU, Linear, LeakyReLU, Tanh
import torch_geometric
from torch_geometric.nn import BatchNorm, GATConv, global_mean_pool, TransformerConv, CGConv, global_add_pool, GINConv
import numpy as np
from tqdm import tqdm



class Encoder(nn.Module):
    def __init__(self, num_layers=2, input_dim=20, output_dim=16, num_edge_attr=8,
                 gnn_type='CGConv', pooling_type='mean'):
        super(Encoder, self).__init__()

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.gnn_type = gnn_type
        self.num_layer = num_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pooling_type = pooling_type

        if gnn_type == 'GATConv':
            # GATConv
            conv_0 = GATConv(in_channels=input_dim, out_channels=output_dim*3)
            conv_1 = GATConv(in_channels=output_dim*3, out_channels=output_dim)
            bn_0 = BatchNorm(output_dim*3)
            bn_1 = BatchNorm(output_dim)
        elif gnn_type == 'GIN':
            # GIN
            gin_mlp1 = Sequential(Linear(input_dim, input_dim), Tanh(), Linear(input_dim, input_dim))
            gin_mlp2 = Sequential(Linear(input_dim, input_dim), Tanh(), Linear(input_dim, input_dim))
            conv_0 = GINConv(gin_mlp1)
            conv_1 = GINConv(gin_mlp2)
            bn_0 = BatchNorm(input_dim)
            bn_1 = BatchNorm(input_dim)
        else:
            print('gnn type error')
            assert False


        self.convs.append(conv_0)
        self.convs.append(conv_1)

        self.batch_norms.append(bn_0)
        self.batch_norms.append(bn_1)

    def forward(self, x, edge_index, batch):

        xs = []

        if self.gnn_type == 'GATConv':
            for conv, batch_norm in zip(self.convs, self.batch_norms):
                x = batch_norm(conv(x, edge_index))
                xs.append(x)
            if self.pooling_type == 'mean':
                x = global_mean_pool(x, batch)
            elif self.pooling_type == 'add':
                x = global_add_pool(x, batch)
            else:
                print('pooling type error')
                assert False
        elif self.gnn_type == 'GIN':
            for conv, batch_norm in zip(self.convs, self.batch_norms):
                x = torch.tanh(conv(x, edge_index))
                x = batch_norm(x)
                xs.append(x)
                # if i == 2:
                # feature_map = x2

            xpool = [global_add_pool(x, batch) for x in xs]
            x = torch.cat(xpool, 1)

        return x

    def get_embeddings(self, dataloader):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ret = []
        y = []
        trace_class = []
        url_status_class = []
        url_class_list = []
        trace_ids = []
        with torch.no_grad():
            for data in tqdm(dataloader):
                # data = data[0]
                data.to(device)
                x = self.forward(data.x, data.edge_index, data.batch)

                ret.append(x.cpu().numpy())
                y.append(data.y.cpu().numpy())
                trace_class.append(data.trace_class.cpu().numpy())
                url_status_class.append(data.url_status_class.cpu().numpy())
                url_class_list.append(data.url_class.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        trace_class = np.concatenate(trace_class, 0)
        url_status_class = np.concatenate(url_status_class, 0)
        url_class_list = np.concatenate(url_class_list, 0)
        return ret, y, trace_class, url_status_class, url_class_list


class SIMCLR(nn.Module):
    def __init__(self, num_layers=2, input_dim=20, output_dim=16, num_edge_attr=8, gnn_type='CGConv',
                 pooling_type='mean'):
        super(SIMCLR, self).__init__()

        self.gnn_type = gnn_type
        self.num_layer = num_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pooling_type = pooling_type

        self.encoder = Encoder(num_layers=num_layers, input_dim=input_dim, output_dim=output_dim,
                               num_edge_attr=num_edge_attr, gnn_type=gnn_type, pooling_type=pooling_type)

        self.proj_head = nn.Sequential(nn.Linear(output_dim, output_dim), nn.Tanh(),
                                       nn.Linear(output_dim, output_dim))
        self.init_emb()


    def init_emb(self):
        initrange = -1.5 / self.output_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, batch):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # batch_size = data.num_graphs
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)

        y = self.encoder(x, edge_index, batch)

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


