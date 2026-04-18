import statistics

import wandb


def set_up_logger(config):
    name = '_'.join([config['participant'],
                     config['model']['spatial_regularization'],
                     config['training']['random_seed']])
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
    wandb_logger.summary['mean_test_accuracy'] = statistics.mean(test_accuracies)
    wandb_logger.summary['max_test_accuracy'] = max(test_accuracies)
    wandb_logger.summary['std_test_accuracy'] = statistics.stdev(test_accuracies)
    wandb_logger.summary['best_checkpoint_epochs'] = checkpoint_epochs
