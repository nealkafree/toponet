import statistics
import os

import wandb


def set_up_logger(config):
    name = '_'.join([config['participant'],
                     str(config['spatial']['spatial_regularization']),
                     str(config['training']['random_seed'])])
    logger_config = {
        "model": config['model'],
        "spatial": config['spatial'],
        "training": {
            "learning_rate": config['training']['learning_rate'],
            "random_seed": config['training']['random_seed'],
        }
    }

    wandb_run = wandb.init(project=config['project_name'], name=name, settings={'silent': True}, config=logger_config)
    model_artifact = wandb.Artifact(
        name=name,
        type="model",
        metadata=logger_config,
    )
    return wandb_run, model_artifact

def log_summary(wandb_logger, test_accuracies, checkpoint_epochs):
    wandb_logger.summary['mean_test_accuracy'] = statistics.mean(test_accuracies.values())
    wandb_logger.summary['test_accuracies'] = test_accuracies
    wandb_logger.summary['std_test_accuracy'] = statistics.stdev(test_accuracies.values())
    wandb_logger.summary['best_checkpoint_epochs'] = checkpoint_epochs


def synchronise_dataset(config):
    collections = wandb.Api().artifact_collections(
        type_name="dataset", project_name=config['project_name']
    )

    dataset_paths = {}
    for dataset in collections:
        if not dataset.name in os.listdir(config['preprocessed_data']):
            print(f'Downloading dataset for {dataset.name}')
            api = wandb.Api()
            artifact = api.artifact(f"{config['project_name']}/{dataset.name}:latest")
            artifact.download(os.path.join(config['preprocessed_data'], dataset.name))

        dataset_paths[dataset.name] = {
            'test': os.path.join(config['preprocessed_data'], dataset.name, 'test.pkl'),
            'train': os.path.join(config['preprocessed_data'], dataset.name, 'train.pkl')
        }

    return dataset_paths
