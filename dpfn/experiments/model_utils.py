from experiments.models import GCN, GraphCN, DeepSet, GCN_SiLU, GCN_Weights, NeuralImputation, HeteroGNN, GCN_W_Global
import torch
from constants import GRAPH_MODELS
from feature_propagation import feature_propagation

def get_model(model_name, n_layers, nhid=64, num_features=5):
    if model_name in ['gcn']:
        return GCN(num_features=num_features, n_layers=n_layers, nhid=nhid)
    elif model_name in ['graphcn']:
        return GraphCN(num_features=num_features, n_layers=n_layers, nhid=nhid)
    elif model_name in ['set']:
        return DeepSet(num_features=num_features, n_layers=n_layers)
    elif model_name in ['gcn_silu']:
        return GCN_SiLU(num_features=num_features, n_layers=1, nhid=nhid)
    elif model_name in ['gcn_weight']:
        return GCN_Weights(num_features=num_features, n_layers=1, nhid=nhid)
    elif model_name in ['hetero_gnn']:
        return HeteroGNN(num_features=num_features, hidden_channels=nhid, out_channels=1, num_layers=1)
    elif model_name in ['gcn_global']:
        return GCN_W_Global(num_features=num_features, nhid=nhid, n_layers=n_layers)
    
    
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


def get_neural_imp_model(model_path, simulator, device):
    if simulator == 'covasim':
        num_features = 8
    elif simulator == 'abm':
        num_features = 7
    model_base = NeuralImputation(num_features=num_features)
    saved_model = torch.load(f"dpfn/config/feature_imp_configs/{model_path}", map_location=torch.device(device))
    model_base.load_state_dict(saved_model)
    
    return model_base