from experiments.models import GCN, GraphCN, DeepSet
import torch

def get_model(model_name, n_layers):
    if model_name in ['gcn']:
        return GCN(num_features=7, n_layers=n_layers)
    elif model_name in ['graphcn']:
        return GraphCN(num_features=7, n_layers=n_layers)
    elif model_name in ['set']:
        return DeepSet(num_features=5, n_layers=n_layers)
    
    
def make_predictions(model, loader, model_type, device):
    all_preds = []
    with torch.no_grad():
        if model_type in ['gcn', 'graphcn']:
            for data in loader:
                data = data.to(device)
                predictions = model(data).squeeze(1)
                all_preds.extend(predictions.cpu().numpy())
                
        elif model_type in ['set']:
            for data in loader:
                X = data.to(device)
                predictions = model(X)
                all_preds.extend(predictions.cpu().numpy())
    return all_preds
