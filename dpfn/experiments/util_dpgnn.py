"""Utility functions for DPGNN experiments."""
from dpfn import logger
import functools


@functools.lru_cache(maxsize=1)
def get_dpgnn_model(model_fpath):
  """Get the neural network model for DPGNN."""
  # Load pytorch model
  import torch  # pylint: disable=import-outside-toplevel
  from dpgnn import util_model  # pylint: disable=import-outside-toplevel
  cfg = {
    'dirname': 'data/100k_abm', 'do_neural_augment': 1, 'layerwidth': 64,
    'learning_rate': 0.002, 'learning_rate_decay': -1, 'lr_warmup': 1,
    'model_type': 2, 'num_epochs': 40, 'num_features': 2, 'num_layers': 8,
    'probab1': 0.04, 'seed': 70, 'spectral_norm_decay': 1,
    'upper_spectral_norm': 1, 'batch_size': 256, 'weight_decay': 1e-09,
    'configname': 'graph', 'cpu_count': 40, 'num_power_iterations': 1}
  model = util_model.SplitGraphNetwork(cfg)

  # Load state dict here
  fname = f'results/dpgnn_models/{model_fpath}'

  model.load_state_dict(torch.load(fname, map_location=torch.device('cpu')))
  model.eval()
  logger.info(model)
  return model
