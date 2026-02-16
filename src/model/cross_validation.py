import copy

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.model_selection import KFold

from . import train


def train_with_cross_validation(model_class, config: dict, train_data: list, test_data: list,
                                disable_logs=False) -> list:
    """
    Trains several models on provided data using 10-fold cross validation.
    :param model_class: class of models we want to train.
    :param config: parameters for models. ['model'] - parameters of model, ['spatial'] - parameters for spatial component.
    :param train_data:
    :param test_data:
    :param disable_logs: False to print logs, True if not.
    :return: trained models with additional information.
    """
    device = torch.device(config['training']['device'])

    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    resulting_models = []
    # Prepare cross-validation folds
    kfolds = KFold(n_splits=10, shuffle=True, random_state=config['training']['random_seed'])
    for i, (train_idx, val_idx) in tqdm(enumerate(kfolds.split(train_data)), total=10, disable=disable_logs):
        if not disable_logs:
            print(f'Training on fold {i}')

        # Prepare train and validation data loaders
        train_subset = Subset(train_data, train_idx)
        val_subset = Subset(train_data, val_idx)
        train_loader = DataLoader(train_subset, batch_size=64, shuffle=False)
        validation_loader = DataLoader(val_subset, batch_size=64, shuffle=False)

        # Setting seeds for weight initialization
        torch.manual_seed(config['training']['random_seed'])
        torch.cuda.manual_seed(config['training']['random_seed'])

        # Train one model. We get the best checkpoint
        model = model_class(**config['model']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
        loss_fn = torch.nn.CrossEntropyLoss()
        best_checkpoint, training_history = train.train_model(model, train_loader, validation_loader,
                                                              loss_fn, optimizer,
                                                              config['spatial'], disable_logs=disable_logs,
                                                              stop_gap=config['training']['stop_gap'],
                                                              max_epochs=config['training']['max_epochs'])

        # Load the best checkpoint
        model = model_class(**config['model']).to(device)
        model.load_state_dict(best_checkpoint['state_dict'])

        # Test the best checkpoint and save results
        test_acc, test_loss_data = train.step(model, test_loader, loss_fn, optimizer,
                                              **config['spatial'], test=True)

        resulting_models.append({
            'model': copy.deepcopy(model.state_dict()),
            'test_accuracy': test_acc,
            'test_loss': test_loss_data["performance_loss"] + test_loss_data["spatial_loss"],
            'epoch': best_checkpoint['epoch'],
            'training_history': training_history,
        })

    return resulting_models
