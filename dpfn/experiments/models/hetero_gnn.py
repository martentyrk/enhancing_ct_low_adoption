import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear, GraphConv


class HeteroGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('app_user', 'connect', 'target'): GraphConv(num_features, hidden_channels, add_self_loops=False),
                ('non_app_user', 'connect', 'target'): GraphConv(num_features, hidden_channels, add_self_loops=False),
                ('observation', 'connect', 'target'): GraphConv(num_features, hidden_channels, add_self_loops=False),
            }, aggr='sum')
            self.convs.append(conv)

        self.mlp = Linear(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        for hetero_conv in self.convs:
            for conv in hetero_conv.convs.values():
                if hasattr(conv, 'reset_parameters'):
                    conv.reset_parameters()

        # Reset parameters for linear layers
        self.mlp.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, data):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        x = self.mlp(x_dict['target'])
        x = self.lin(x)
        return x
