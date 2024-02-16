"""
version 1.0
date 2021/02/04
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, Sequential
from torch_geometric.nn.conv import GCNConv, GraphConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GCN_SiLU(nn.Module):
    def __init__(self, num_features, nhid=32, dropout=0.05, n_layers = 1):
        super(GCN_SiLU, self).__init__()
        self.num_features = num_features
        
        gcn_layers = [
            (GCNConv(num_features, nhid), 'x, edge_index -> x'),
            # (GraphConv(num_features, nhid, aggr='max'), 'x, edge_index -> x'),
            (nn.Dropout(dropout), 'x -> x'),
            (nn.SELU(inplace=True), 'x -> x')
        ]
        
        # for _ in range(n_layers - 1):
        #     gcn_layers.append((nn.Dropout(dropout), 'x -> x'))
        #     gcn_layers.append((GraphConv(nhid, nhid), 'x, edge_index -> x'))
        #     gcn_layers.append((nn.ReLU(inplace=True), 'x -> x'))
        
        self.gcn = Sequential('x, edge_index', gcn_layers)
        
        # Leaf network for observations
        # obs_layers = [
        #     nn.Linear(2, nhid),
        #     nn.Dropout(dropout),
        #     nn.ReLU(),
        # ]
    
        # for _ in range(n_layers - 1):
        #     obs_layers.append(nn.Linear(nhid, nhid))
        #     obs_layers.append(nn.Dropout(0.1))
        #     obs_layers.append(nn.ReLU())
            
        # obs_layers.append(nn.Linear(nhid, nhid))
        
        # self.leaf_network_obs = nn.Sequential(*obs_layers)
        
        backbone_layers = [
            nn.Linear(nhid, nhid),
            nn.Dropout(dropout),
            nn.SELU(),
        ]
        
        for _ in range(n_layers - 1):
            backbone_layers.append(nn.Linear(nhid, nhid))
            backbone_layers.append(nn.Dropout(dropout))
            backbone_layers.append(nn.SELU())

        backbone_layers.append(nn.Linear(nhid, 1))
        self.backbone_fc = nn.Sequential(*backbone_layers)

    def reset_parameters(self):
        for layer in self.gcn:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        
        for layer in self.backbone_fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1.414)
                layer.bias.data.fill_(0)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # TODO find root nodes
        # root_filter = x[:, 4] == 1
        x = self.gcn(x, edge_index)
        # TODO: apply filter
        # x = x[root_filter]

        x = global_add_pool(x, batch)
        
        logits = self.backbone_fc(x)

        return logits