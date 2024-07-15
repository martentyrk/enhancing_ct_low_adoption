import torch
import torch.nn as nn


class NeuralImputation(nn.Module):
    def __init__(self, num_features, hidden_layers=[64, 128, 128, 64, 32]):
        super(NeuralImputation, self).__init__()
        prev_dim = num_features
        layers = []
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, data):
        x = self.model(data)
        return x
