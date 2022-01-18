from re import T
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.nn.modules import loss
from torch_geometric.nn import GATConv, TransformerConv, CGConv, global_mean_pool


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_gc_layers, num_classes, num_edge_attr):
        super(Encoder, self).__init__()

        self.num_gc_layers = num_gc_layers
        self.num_classes = num_classes
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        # GATConv
        # conv_0 = GATConv(in_channels=input_dim, out_channels=output_dim*4)
        # conv_1 = GATConv(in_channels=output_dim*4, out_channels=output_dim)
        # TransformerConv
        conv_0 = TransformerConv(in_channels=input_dim, out_channels=output_dim*4, edge_dim=num_edge_attr)
        conv_1 = TransformerConv(in_channels=output_dim*4, out_channels=output_dim, edge_dim=num_edge_attr)
        # CGConv
        # conv_0 = CGConv(channels=input_dim, dim=num_edge_attr)
        # conv_1 = CGConv(channels=output_dim*4, dim=num_edge_attr)
        self.convs.append(conv_0)
        self.convs.append(conv_1)

        bn_0 = torch.nn.BatchNorm1d(output_dim*4)
        bn_1 = torch.nn.BatchNorm1d(output_dim)
        self.bns.append(bn_0)
        self.bns.append(bn_1)

        self.fc = torch.nn.Linear(output_dim, num_classes)

    def forward(self, x, edge_index, edge_attr, batch):
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(device)

        xs = []
        for i in range(self.num_gc_layers):
            x = self.convs[i](x=x, edge_index=edge_index, edge_attr=edge_attr)
            x = self.bns[i](x)
            xs.append(x)

        xpool = [global_mean_pool(x, batch) for x in xs]

        # x = torch.cat(xpool, 1)
        x = xpool[-1]
        
        output = self.fc(x)

        return output

    def predict(self, x, edge_index, edge_attr, batch):
        prediction = torch.max(F.softmax(self.forward(x, edge_index, edge_attr, batch), dim=1), 1)[1] 
        return prediction
        
        
        '''
        pred = F.softmax(self.forward(x, edge_index, batch))
        ans = []
        for t in pred:
            if t[0]>t[1]:
                ans.append(0)
            else:
                ans.append(1)
        return torch.tensor(ans)
        '''