import torch
from torch import nn
import torch.nn.functional as F

class DeepSet(nn.Module):
    """Model."""

    def __init__(self, num_features, n_layers, nhid=64):
        super().__init__()


        def make_linear(arg_in, arg_out):
            return nn.Linear(arg_in, arg_out)

        def make_linear_bn(arg_in, arg_out, batch_layer_size):
            """Creates a sequence of Linear -> BatchNorm -> Dropout -> ReLU."""
            return nn.Sequential(
                nn.Linear(arg_in, arg_out),
                nn.BatchNorm1d(batch_layer_size),  # Batch normalization layer
                nn.Dropout(0.1),
                nn.ReLU()
            )
        # Leaf network for contacts
        layers = [
            make_linear_bn(num_features, nhid, 900)]
        for _ in range(n_layers - 1):
            layers.append(make_linear_bn(nhid, nhid, 900))
            
        layers.append(make_linear(nhid, nhid))
        self.leaf_network_contact = nn.Sequential(*layers)
        del layers

        # Leaf network for observations
        layers = [
            make_linear_bn(num_features, nhid, 14)]
        for _ in range(n_layers - 1):
            layers.append(make_linear_bn(nhid, nhid, 14))
            
        layers.append(make_linear(nhid, nhid))
        self.leaf_network_obs = nn.Sequential(*layers)
        del layers

        # Backbone network
        layers = [
            make_linear_bn(nhid, nhid, nhid)]
        for _ in range(n_layers):
            layers.append(make_linear_bn(nhid, nhid, nhid))
            
        layers.append(make_linear(nhid, 1))
        self.backbone_network = nn.Sequential(*layers)
        
        
    def reset_parameters(self):
        for layer in self.leaf_network_contact:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1.414)
                layer.bias.data.fill_(0)
                
        for layer in self.leaf_network_obs:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1.414)
                layer.bias.data.fill_(0)
                
        for layer in self.backbone_network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1.414)
                layer.bias.data.fill_(0)

    def forward(self, x):
        """Forward pass."""
        mask = x[:, :, 0] >= 0
        row_sum = torch.sum(mask, dim=1, keepdim=True) + 1E-9
        mask = (mask / row_sum).unsqueeze(-1).detach()

        # Apply leaf network
        node_features = torch.cat((
            self.leaf_network_contact(x[:, :900, :]),
            self.leaf_network_obs(x[:, 900:, :]),
        ), dim=1)
        backbone_features = torch.sum(node_features * mask, dim=1)

        # Apply backbone network
        logits = torch.squeeze(self.backbone_network(backbone_features), 1)

        logits = logits
        return F.sigmoid(logits)