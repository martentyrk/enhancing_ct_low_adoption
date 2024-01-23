import argparse
from dpfn import logger
import os
import json


def balance_dataset(path: str):
    interactions_list = [[], [], []]
    path_split = path.split('/')
    dirpath = path_split[:-1]
    fname_og = path_split[-1]
    
    with open(path) as f:
        for line in f:
            line_data = json.loads(line.rstrip('\n'))
    
    
    
    
    fname_balanced = os.path.join(path, f'balanced_{fname_og}.jl')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Compare statistics acrosss inference methods')
    parser.add_argument('--path', type=str, default="dpfn/data/val_app_users")

    args = parser.parse_args()
    logger.info('Initializing ABMInMemoryDataset with path: %s', str(args.path))
    
    logger.info('File saved!')