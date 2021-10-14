import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.nn.modules import loss
from torch_geometric.nn import TransformerConv, global_add_pool, global_mean_pool, GATConv
from losses import local_global_loss


class Encoder(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers):
        super(Encoder, self).__init__()

        self.num_gc_layers = num_gc_layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        '''
        for i in range(num_gc_layers):
            if i:
                # in_channel, out_channel, edge_idm
                conv = TransformerConv(
                    in_channels=dim, out_channels=dim, edge_dim=1)
            else:
                conv = TransformerConv(
                    in_channels=num_features, out_channels=dim, edge_dim=1)
            bn = nn.BatchNorm1d(dim)
            self.convs.append(conv)
            self.bns.append(bn)
        '''
        
        # GNN
        # TransformerConv
        conv_0 = TransformerConv(in_channels = num_features, out_channels = dim*4, edge_dim = 1)
        conv_1 = TransformerConv(in_channels = dim*4, out_channels = dim, edge_dim = 1)
        # GATConv
        # conv_0 = GATConv(in_channels = num_features, out_channels = dim*4, edge_dim = 1)
        # conv_1 = GATConv(in_channels = dim*4, out_channels = dim, edge_dim = 1)
        # GENConv
        # conv_0 = GENConv(in_channels = num_features, out_channels = dim*4, edge_dim = 1)
        # conv_1 = GENConv(in_channels = dim*4, out_channels = dim, edge_dim = 1)


        self.convs.append(conv_0)
        self.convs.append(conv_1)
            

        # conv = GINConv(nn)
            
        # BN
        bn_0 = torch.nn.BatchNorm1d(dim*4)
        bn_1 = torch.nn.BatchNorm1d(dim)
        self.bns.append(bn_0)
        self.bns.append(bn_1)
        
        
        

    def forward(self, x, edge_index, edge_attr, batch):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(device)

        xs = []
        for i in range(self.num_gc_layers):

            # x = F.relu(self.convs[i](x, edge_index, edge_attr))
            x = self.convs[i](x, edge_index, edge_attr)    # --> prelu
            
            x = self.bns[i](x)
            xs.append(x)

        # xpool = [global_add_pool(x, batch) for x in xs]
        xpool = [global_mean_pool(x, batch) for x in xs]
        
        x = torch.cat(xpool, 1)
        # x = xpool[-1]

        return x, torch.cat(xs, 1)

    def get_embeddings(self, loader):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ret = []
        y = []

        with torch.no_grad():
            for data in loader:

                data = data[0]
                data.to(device)
                x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0], 1)).to(device)
                x, _ = self.forward(x, edge_index, edge_attr, batch)

                ret.append(x.cpu().numpy())
                y.append(data.y.cpu().numpy())

        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y

    def get_embeddings_v(self, loader):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ret = []
        y = []

        with torch.no_grad():
            for n, data in enumerate(loader):
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0], 1)).to(device)
                x_g, x = self.forward(x, edge_index, batch)
                x_g = x_g.cpu().numpy()
                ret = x.cpu().numpy()
                y = data.edge_index.cpu().numpy()
                print(data.y)
                if n == 1:
                    break

        return x_g, ret, y


class GcnInfomax(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, prior, dataset_num_features, alpha=0.5, beta=1., gamma=.1):
        super(GcnInfomax, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.prior = prior

        # self.embedding_dim = mi_units = hidden_dim * num_gc_layers
        self.embedding_dim = hidden_dim + hidden_dim * 4
        self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)

        self.local_d = FF(self.embedding_dim)
        self.global_d = FF(self.embedding_dim)
        # self.local_d = MI1x1ConvNet(self.embedding_dim, mi_units)
        # self.global_d = MIFCNet(self.embedding_dim, mi_units)

        if self.prior:
            self.prior_d = PriorDiscriminator(self.embedding_dim)

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
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

        y, M = self.encoder(x, edge_index, edge_attr, batch)

        g_enc = self.global_d(y)
        l_enc = self.local_d(M)

        mode = 'fd'
        measure = 'JSD'
        loss = local_global_loss(
            l_enc, g_enc, edge_index, batch, measure)

        if self.prior:
            prior = torch.rand_like(y)
            term_a = torch.log(self.prior_d(prior)).mean()
            term_b = torch.log(1.0 - self.prior_d(y)).mean()
            PRIOR = - (term_a + term_b) * self.gamma
        else:
            PRIOR = 0

        return loss + PRIOR


class simclr(torch.nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, prior, dataset_num_features, alpha=0.5, beta=1., gamma=.1):
        super(simclr, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.prior = prior

        # self.embedding_dim = mi_units = hidden_dim * num_gc_layers
        self.embedding_dim = hidden_dim + hidden_dim * 4
        #self.embedding_dim = hidden_dim
    
        self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)

        self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(self.embedding_dim, self.embedding_dim))

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
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

        y, M = self.encoder(x, edge_index, edge_attr, batch)

        y = self.proj_head(y)

        return y

    def loss_cal(self, x, x_aug):
        T = 0.2
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)    # 对每一列求 1 范数
        x_aug_abs = x_aug.norm(dim=1)    # 1 范数

        sim_matrix = torch.einsum(
            'ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        return loss


class GlobalDiscriminator(nn.Module):
    def __init__(self, args, input_dim):
        super().__init__()

        self.l0 = nn.Linear(32, 32)
        self.l1 = nn.Linear(32, 32)
        self.l2 = nn.Linear(512, 1)

    def forward(self, y, M, data):

        adj = Variable(data['adj'].float(), requires_grad=False).cuda()
        # h0 = Variable(data['feats'].float()).cuda()
        batch_num_nodes = data['num_nodes'].int().numpy()
        M, _ = self.encoder(M, adj, batch_num_nodes)
        # h = F.relu(self.c0(M))
        # h = self.c1(h)
        # h = h.view(y.shape[0], -1)
        h = torch.cat((y, M), dim=1)
        h = F.relu(self.l0(h))
        h = F.relu(self.l1(h))
        return self.l2(h)


class PriorDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.l0 = nn.Linear(input_dim, input_dim)
        self.l1 = nn.Linear(input_dim, input_dim)
        self.l2 = nn.Linear(input_dim, 1)

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))


class FF(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # self.c0 = nn.Conv1d(input_dim, 512, kernel_size=1)
        # self.c1 = nn.Conv1d(512, 512, kernel_size=1)
        # self.c2 = nn.Conv1d(512, 1, kernel_size=1)
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, input_dim)
        # self.c0 = nn.Conv1d(input_dim, 512, kernel_size=1, stride=1, padding=0)
        # self.c1 = nn.Conv1d(512, 512, kernel_size=1, stride=1, padding=0)
        # self.c2 = nn.Conv1d(512, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)
