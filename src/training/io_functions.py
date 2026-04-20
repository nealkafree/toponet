import os
import pickle


def prepare_model_dir(wandb_logger, config):
    dir_name = '_'.join([wandb_logger.name, wandb_logger.id])
    dir_path = os.path.join(config['trained_models_path'], dir_name)
    os.makedirs(dir_path, exist_ok=True)

    return dir_path


def load_file(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data
