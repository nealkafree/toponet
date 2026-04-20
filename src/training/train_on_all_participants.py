import json
import sys

from . import logger, eegnet, cross_validation

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_paths = logger.synchronise_dataset(config)

spatial_reg = sys.argv[1]
config['spatial']['spatial_regularization'] = spatial_reg

model = eegnet.EEGNet

for dataset in dataset_paths:
    config['train_data_path'] = dataset_paths[dataset]['train']
    config['test_data_path'] = dataset_paths[dataset]['test']
    config['participant'] = dataset

    cross_validation.train_with_cross_validation(model, config, disable_logs=True)