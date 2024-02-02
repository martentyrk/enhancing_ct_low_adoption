from experiments.models import GCN, GraphCN

def get_model(model_name, n_layers):
    if model_name in ['gcn']:
        return GCN(num_features=7, n_layers=n_layers)
    elif model_name in ['graphcn']:
        return GraphCN(num_features=7, n_layers=n_layers)