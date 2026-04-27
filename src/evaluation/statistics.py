import os.path

import numpy as np
from libpysal.weights import lat2W
from esda.moran import Moran
import torch

from src.training import eegnet, io_functions


def moran_i(model, data, model_config) -> float:
    """
    Calculates Moran's I spatial autocorrelation metric.
    """
    x = model.spatial_activation(data)

    x = x.reshape((model_config['spatial_grid_width'], model_config['spatial_grid_width']))

    # Create the matrix of weights
    w = lat2W(model_config['spatial_grid_width'], model_config['spatial_grid_width'])

    # Return the metric
    return Moran(x, w).I


def apply_statistic(stat_function, model_paths, config):
    stats = {}
    for name, path in model_paths.items():
        accum = 0
        for file in os.listdir(path):
            trained_model = eegnet.EEGNet(**config['model'])
            trained_model.load_state_dict(torch.load(os.path.join(path, file), weights_only=True))
            trained_model.eval()

            participant = name.split('_')[0]
            test_data = io_functions.load_file(os.path.join(config['preprocessed_data'], participant, 'test.pkl'))

            with torch.inference_mode():
                for example in test_data:
                    accum += stat_function(trained_model, example, config['model'])

        stats[name] = accum / len(os.listdir(path))

    return stats
