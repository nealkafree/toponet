import json
import sys
import os

from training import eegnet, cross_validation

with open('config.json', 'r') as f:
    config = json.load(f)

participant = sys.argv[2]

config['spatial']['spatial_regularization'] = float(sys.argv[1])

model = eegnet.EEGNet

config['train_data_path'] = os.path.join(config['preprocessed_data'], participant, 'train.pkl')
config['test_data_path'] = os.path.join(config['preprocessed_data'], participant, 'test.pkl')
config['participant'] = participant

cross_validation.train_with_cross_validation(model, config, disable_logs=True)