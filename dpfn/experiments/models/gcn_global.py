"""
version 1.0
date 2021/02/04
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import global_add_pool, Sequential
from torch_geometric.nn.conv import GCNConv, GraphConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    


class GCN_W_Global(nn.Module):
    def __init__(self, num_features, nhid=32, dropout=0.1, n_layers = 1):
        super(GCN_W_Global, self).__init__()
        self.msg_weights = nn.Parameter(torch.randn(3), requires_grad=True)
        self.fn_embeddings = nn.Sequential(
            nn.Linear(3, 16),  # first linear layer (input size 2, output size 16)
            nn.GELU(),  # ReLU activation
            nn.Linear(16, 3)  # second linear layer (input size 16, output size adjusted)
        )
        
        # out features = 64 - num_features (usually 7) + 2 (fn + age embeds)
        self.emb = nn.Linear(2, 64 - num_features + 2)
        
        gcn_layers = [
            (GCNConv(64, nhid - 3), 'x, edge_index, edge_weight -> x'),
            # (GraphConv(num_features, nhid, aggr='max'), 'x, edge_index -> x'),
            (nn.Dropout(dropout), 'x -> x'),
            (nn.GELU(), 'x -> x'),
        ]
        
        self.gcn = Sequential('x, edge_index, edge_weight', gcn_layers)


        backbone_layers = [
            nn.Linear(nhid, nhid),
            nn.Dropout(dropout),
            nn.GELU(),
        ]
        
        for _ in range(n_layers - 1):
            backbone_layers.append(nn.Linear(nhid, nhid))
            backbone_layers.append(nn.Dropout(dropout))
            backbone_layers.append(nn.GELU())

        backbone_layers.append(nn.Linear(nhid, 1))
        self.backbone_fc = nn.Sequential(*backbone_layers)

    def forward(self, data):
        x, edge_index, batch, known_mask, unk_mask, obs_mask = data.x, data.edge_index, data.batch, data.known_mask, data.unk_mask, data.obs_mask
        
        fn_averages = data.fn_averages.view(-1, 2)
        infection_rate = data.infection_rates.view(-1, 1)
        extra_data = torch.concat((fn_averages, infection_rate), axis=1)
        
        known_mask = known_mask.to(torch.int)
        unk_mask = unk_mask.to(torch.int)
        obs_mask = obs_mask.to(torch.int)
        
        edge_weights = torch.ones_like(edge_index[0], dtype=torch.float)
        
        softmax_params = F.softmax(self.msg_weights, dim=0)
        
        edge_weights[known_mask] = softmax_params[0]
        edge_weights[unk_mask] = softmax_params[1]
        edge_weights[obs_mask] = softmax_params[2]
        
        x_fn_age = self.emb(x[:, :2])
        x_rest = x[:, 2:]
        
        x = torch.cat((x_fn_age, x_rest), 1)
        
        x = self.gcn(x, edge_index=edge_index, edge_weight=edge_weights)
        
        fn_embeddings = self.fn_embeddings(extra_data)

        x = global_add_pool(x, batch)
        
        x = torch.cat((x, fn_embeddings), axis=1)
        
        logits = self.backbone_fc(x)

        return logits