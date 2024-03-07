import torch
from torch import Tensor
from torch_scatter import scatter_add
from torch_geometric.typing import Adj


def feature_propagation(data):
    x = data.x
    non_app_users_mask = torch.ones_like(x).bool()
    non_app_users_indeces = torch.where(x[:, -1] == 0, 1, 0).bool()
    
    non_app_users_mask[non_app_users_indeces == 1] = torch.tensor([False, False, True, True, True, True, True])
    x_features = x.clone()
    x_features[~non_app_users_mask] = float('nan')
    
    x_features = filling(data.edge_index, x_features, non_app_users_mask)
    
    return x_features
    
    
def filling(edge_index, x, missing_feature_mask, num_iterations=40):
    propagation_model = FeaturePropagation(num_iterations=num_iterations)
    
    return propagation_model.propagate(x=x, edge_index=edge_index, mask=missing_feature_mask)




class FeaturePropagation(torch.nn.Module):
    def __init__(self, num_iterations: int):
        super(FeaturePropagation, self).__init__()
        self.num_iterations = num_iterations

    def propagate(self, x: Tensor, edge_index: Adj, mask: Tensor) -> Tensor:
        # out is inizialized to 0 for missing values. However, its initialization does not matter for the final
        # value at convergence
        out = x
        if mask is not None:
            out = torch.zeros_like(x)
            out[mask] = x[mask]

        n_nodes = x.shape[0]
        adj = self.get_propagation_matrix(out, edge_index, n_nodes)
        for _ in range(self.num_iterations):
            # Diffuse current features
            out = torch.sparse.mm(adj, out)
            # Reset original known features
            out[mask] = x[mask]

        return out

    def get_propagation_matrix(self, x, edge_index, n_nodes):
        # Initialize all edge weights to ones if the graph is unweighted)
        edge_index, edge_weight = get_symmetrically_normalized_adjacency(edge_index, n_nodes=n_nodes)
        adj = torch.sparse.FloatTensor(edge_index, values=edge_weight, size=(n_nodes, n_nodes)).to(edge_index.device)

        return adj
    
    
    
def get_symmetrically_normalized_adjacency(edge_index, n_nodes):
    """
    Given an edge_index, return the same edge_index and edge weights computed as
    \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2}.
    """
    edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=n_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    DAD = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, DAD