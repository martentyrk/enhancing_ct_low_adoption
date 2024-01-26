from experiments.models import GCN

def get_model(model_name):
    if model_name in ['gcn']:
        return GCN(num_features=7)