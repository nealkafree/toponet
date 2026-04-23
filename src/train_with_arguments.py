import argparse
import json
import os

from training import eegnet, cross_validation

parser = argparse.ArgumentParser()
parser.add_argument("--participant")
parser.add_argument("--regularization")
parser.add_argument("--random_state")
parser.add_argument("--device")
args = parser.parse_args()

with open('config.json', 'r') as f:
    config = json.load(f)

if args.participant:
    config['participant'] = args.participant

if args.regularization:
    config['spatial']['spatial_regularization'] = args.regularization

if args.random_state:
    config['training']['random_seed'] = args.random_state

if args.device:
    config['training']['device'] = args.device

config['train_data_path'] = os.path.join(config['preprocessed_data'], config['participant'], 'train.pkl')
config['test_data_path'] = os.path.join(config['preprocessed_data'], config['participant'], 'test.pkl')

model = eegnet.EEGNet

cross_validation.train_with_cross_validation(model, config, disable_logs=True)
