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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
        
        
# class GCN_Weights(nn.Module):
#     def __init__(self, num_features, nhid=32, dropout=0.1, n_layers = 1):
#         super(GCN_Weights, self).__init__()
        
#         self.msg_weights = nn.Parameter(torch.randn(3), requires_grad=True)
        
#         gcn_layers = [
#             (GCNConv(num_features, nhid), 'x, edge_index, edge_weight -> x'),
#             (nn.Dropout(dropout), 'x -> x'),
#             (nn.ReLU(inplace=True), 'x -> x')
#         ]
        
        
#         self.gcn = Sequential('x, edge_index, edge_weight', gcn_layers)
        
        
#         backbone_layers = [
#             nn.Linear(nhid, nhid),
#             nn.Dropout(dropout),
#             nn.ReLU(),
#         ]
        
#         for _ in range(n_layers - 1):
#             backbone_layers.append(nn.Linear(nhid, nhid))
#             backbone_layers.append(nn.Dropout(dropout))
#             backbone_layers.append(nn.ReLU())

#         backbone_layers.append(nn.Linear(nhid, 1))
#         self.backbone_fc = nn.Sequential(*backbone_layers)

#     def reset_parameters(self):
#         for layer in self.gcn:
#             if hasattr(layer, 'reset_parameters'):
#                 layer.reset_parameters()
        
#         for layer in self.backbone_fc:
#             if isinstance(layer, nn.Linear):
#                 nn.init.xavier_uniform_(layer.weight, gain=1.414)
#                 layer.bias.data.fill_(0)

#     def forward(self, data):
#         x, edge_index, batch, known_mask, unk_mask, obs_mask = data.x, data.edge_index, data.batch, data.known_mask, data.unk_mask, data.obs_mask
#         known_mask = known_mask
#         unk_mask = unk_mask
#         obs_mask = obs_mask
        
#         edge_weights = torch.ones_like(edge_index[0], dtype=torch.float)
        
#         softmax_params = F.softmax(self.msg_weights, dim=0)

#         known_param = softmax_params[0]
#         unk_param = softmax_params[1]
#         obs_param = softmax_params[2]
        
#         edge_weights[known_mask] = known_param
#         edge_weights[unk_mask] = unk_param
#         edge_weights[obs_mask] = obs_param
        
#         x = self.gcn(x, edge_index=edge_index, edge_weight=edge_weights)

#         x = global_add_pool(x, batch)
        
#         logits = self.backbone_fc(x)

#         return logits


class GCN_Weights(nn.Module):
    def __init__(self, num_features, nhid=32, dropout=0.1, n_layers = 1):
        super(GCN_Weights, self).__init__()
        self.msg_weights = nn.Parameter(torch.randn(3), requires_grad=True)
        
        self.emb = nn.Linear(num_features - 1, 63)
        
        gcn_layers = [
            (GCNConv(64, nhid), 'x, edge_index, edge_weight -> x'),
            # (GraphConv(num_features, nhid, aggr='max'), 'x, edge_index -> x'),
            (nn.Dropout(dropout), 'x -> x'),
            (nn.ReLU(inplace=True), 'x -> x')
        ]
        
        
        self.gcn = Sequential('x, edge_index, edge_weight', gcn_layers)


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
                
        nn.init.xavier_uniform_(self.emb.weight, gain=1.414)
        self.emb.bias.data.fill_(0)

    
    def forward(self, data):
        x, edge_index, batch, known_mask, unk_mask, obs_mask = data.x, data.edge_index, data.batch, data.known_mask, data.unk_mask, data.obs_mask
        known_mask = known_mask.to(torch.int)
        unk_mask = unk_mask.to(torch.int)
        obs_mask = obs_mask.to(torch.int)
        
        edge_weights = torch.ones_like(edge_index[0], dtype=torch.float)
        
        softmax_params = F.softmax(self.msg_weights, dim=0)
        
        known_param = softmax_params[0]
        unk_param = softmax_params[1]
        obs_param = softmax_params[2]
        
        edge_weights[known_mask] = known_param
        edge_weights[unk_mask] = unk_param
        edge_weights[obs_mask] = obs_param
        
        x_fn = x[:, 0].unsqueeze(1)
        x_rest = x[:, 1:]
        
        x_rest = self.emb(x_rest)
        
        x = torch.cat((x_fn, x_rest), 1)
        
        x = self.gcn(x, edge_index=edge_index, edge_weight=edge_weights)

        x = global_add_pool(x, batch)
        
        logits = self.backbone_fc(x)

        return logits