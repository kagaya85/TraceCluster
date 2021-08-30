import torch
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
from torch_scatter import scatter_mean
import torch.nn.functional as F


class Net(torch.nn.Module):
    def __init__(self, feature_size, model_params):
        super(Net, self).__init__()

        self.conv1 = GCNConv(feature_size, 16)
        self.conv2 = GCNConv(16, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        x = F.log_softmax(x, dim=1)
        return x
