"""
version 1.0
date 2021/02/04
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, Sequential
from torch_geometric.nn.conv import GCNConv

class GCN(nn.Module):
    def __init__(self, num_features, nhid=64, dropout=0.1, n_layers = 1):
        super(GCN, self).__init__()
        
        gcn_layers = [
            (GCNConv(num_features, nhid), 'x, edge_index, edge_weight -> x'),
            # (GraphConv(num_features, nhid, aggr='max'), 'x, edge_index -> x'),
            (nn.Dropout(dropout), 'x -> x'),
            (nn.ReLU(inplace=True), 'x -> x')
        ]
        
        self.gcn = Sequential('x, edge_index, edge_weight', gcn_layers)
        
        edge_layers = [
            nn.Linear(5, nhid),
            nn.ReLU(),
            nn.Linear(nhid, nhid),
            nn.ReLU(),
            nn.Linear(nhid, 1),
        ]
        
        self.edge_mlp = nn.Sequential(*edge_layers)
        
        
        backbone_layers = [
            nn.Linear(nhid, nhid),
            nn.Dropout(dropout),
            nn.ReLU(),
        ]
        
        for _ in range(n_layers - 1):
            backbone_layers.append(nn.Linear(nhid, nhid))
            backbone_layers.append(nn.Dropout(dropout))
            backbone_layers.append(nn.ReLU())

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
                
        for layer in self.edge_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1.414)
                layer.bias.data.fill_(0)

    def forward(self, data):
        x, edge_index, batch, edge_weights = data.x, data.edge_index, data.batch, data.edge_attr
        
        #Ensure all weights are non negative.
        edge_w = F.relu(self.edge_mlp(edge_weights).squeeze(1))
        
        x = self.gcn(x, edge_index, edge_weight=edge_w)
        
        x = global_add_pool(x, batch)
        
        logits = self.backbone_fc(x)

        return logits