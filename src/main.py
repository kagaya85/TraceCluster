import torch

from torch_geometric.data import Dataset
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_scatter import scatter_mean


class TraceDataset(Dataset):
    def __init__(self, root: Optional[str], transform: Optional[Callable], pre_transform: Optional[Callable], pre_filter: Optional[Callable]):
        super().__init__(root=root, transform=transform,
                         pre_transform=pre_transform, pre_filter=pre_filter)

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return super().raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return super().processed_file_names

    def download(self):
        pass

    def process(self):
        return super().process()


class SAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(SAGEConv, self).__init__(aggr='max')  # "Max" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.act = torch.nn.ReLU()
        self.update_lin = torch.nn.Linear(
            in_channels + out_channels, in_channels, bias=False)
        self.update_act = torch.nn.ReLU()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j):
        # x_j has shape [E, in_channels]

        x_j = self.lin(x_j)
        x_j = self.act(x_j)

        return x_j

    def update(self, aggr_out, x):
        # aggr_out has shape [N, out_channels]

        new_embedding = torch.cat([aggr_out, x], dim=1)

        new_embedding = self.update_lin(new_embedding)
        new_embedding = self.update_act(new_embedding)

        return new_embedding


dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for data in loader:
    x = scatter_mean(data.x, data.batch, dim=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
