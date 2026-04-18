import os.path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.model_selection import KFold

from . import train
from . import logger


def train_with_cross_validation(model_class, config: dict, train_data: list, test_data: list, disable_logs=False):
    """
    Trains several models on provided data using 10-fold cross validation.
    :param model_class: class of models we want to train.
    :param config: parameters for models. ['model'] - parameters of model, ['spatial'] - parameters for spatial component.
    :param train_data:
    :param test_data:
    :param disable_logs: False to print logs, True if not.
    """
    device = torch.device(config['training']['device'])

    wandb_logger, model_artifact = logger.set_up_logger(config)

    dir_name = '_'.join(wandb_logger.name, wandb_logger.id)
    dir_path = os.path.join(config['trained_models_path'], dir_name)

    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=8)

    test_accuracies = []
    checkpoint_epochs = {}
    # Prepare cross-validation folds
    kfolds = KFold(n_splits=10, shuffle=True, random_state=config['training']['random_seed'])
    for i, (train_idx, val_idx) in tqdm(enumerate(kfolds.split(train_data)), total=10, disable=disable_logs):

        # Prepare train and validation data loaders
        train_subset = Subset(train_data, train_idx)
        val_subset = Subset(train_data, val_idx)
        train_loader = DataLoader(train_subset, batch_size=64, shuffle=False, num_workers=8)
        validation_loader = DataLoader(val_subset, batch_size=64, shuffle=False, num_workers=8)

        # Setting seeds for weight initialization
        torch.manual_seed(config['training']['random_seed'])
        torch.cuda.manual_seed(config['training']['random_seed'])

        # Train one model. We get the best checkpoint
        model = model_class(**config['model']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
        loss_fn = torch.nn.CrossEntropyLoss()

        best_checkpoint, training_history = train.train_model(model, train_loader, validation_loader,
                                                              loss_fn, optimizer, wandb_logger, i,
                                                              config['spatial'], disable_logs=disable_logs,
                                                              stop_gap=config['training']['stop_gap'],
                                                              max_epochs=config['training']['max_epochs'])

        # Load the best checkpoint
        model = model_class(**config['model']).to(device)
        model.load_state_dict(best_checkpoint['state_dict'])

        # Test the best checkpoint and save results
        test_acc, test_loss_data = train.step(model, test_loader, loss_fn, optimizer,
                                              **config['spatial'], test=True)

        test_accuracies.append(test_acc)
        checkpoint_epochs[f'fold_{i}'] = best_checkpoint['epoch']

        # Save models on the disk
        torch.save(best_checkpoint['state_dict'], os.path.join(dir_path, f'fold_{i}.pth'))

        # Save models in wandb
        model_artifact.add_file(os.path.join(dir_path, f'fold_{i}.pth'))

        # resulting_models.append({
        #     'model': copy.deepcopy(model.state_dict()), # artefact
        #     'test_accuracy': test_acc, # to summary
        #     'test_loss': test_loss_data["performance_loss"] + test_loss_data["spatial_loss"], # to summary
        #     'epoch': best_checkpoint['epoch'], # to summary
        #     'training_history': training_history,
        # })

    # Log summary of the training
    logger.log_summary(wandb_logger, test_accuracies, checkpoint_epochs)
    wandb_logger.log_artifact(model_artifact)

    wandb_logger.finish()
