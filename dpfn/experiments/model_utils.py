from experiments.models import GCN, GraphCN, DeepSet, GCN_SiLU, GCN_Weights
import torch
from constants import GRAPH_MODELS
from feature_propagation import feature_propagation

def get_model(model_name, n_layers, nhid=64):
    if model_name in ['gcn']:
        return GCN(num_features=3, n_layers=n_layers, nhid=nhid)
    elif model_name in ['graphcn']:
        return GraphCN(num_features=7, n_layers=n_layers, nhid=nhid)
    elif model_name in ['set']:
        return DeepSet(num_features=5, n_layers=n_layers)
    elif model_name in ['gcn_silu']:
        return GCN_SiLU(num_features=7, n_layers=1, nhid=nhid)
    elif model_name in ['gcn_weight']:
        return GCN_Weights(num_features=7, n_layers=1, nhid=nhid)
    
    
def make_predictions(model, loader, model_type, device, feature_prop=False):
    all_preds = []
    with torch.no_grad():
        if model_type in GRAPH_MODELS:
            for data in loader:
                if feature_prop:
                    data.x = feature_propagation(data)
                data = data.to(device)
                predictions = model(data).squeeze(1)
                all_preds.extend(predictions.cpu().numpy())

        elif model_type in ['set']:
            for data in loader:
                X = data.to(device)
                predictions = model(X)
                all_preds.extend(predictions.cpu().numpy())
    return all_preds
