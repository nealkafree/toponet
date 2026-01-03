import copy

import torch
from tqdm import tqdm

from . import loss

# Todo: move to config
DEVICE = torch.device("cuda:0")


def step(model, data, loss_fn, optimizer, spatial_regularization=0.0, spatial_grid_width=6, test=False):
    """
    Performs one training or test step (depends on parameters).
    :param model:
    :param data:
    :param loss_fn:
    :param optimizer:
    :param spatial_regularization: regularization parameter for spatial loss.
    :param spatial_grid_width: width of the grid for spatial loss.
    :param test: True if test step, False if train step.
    :return: Accuracy of predictions and average loss during the training or testing cycle.
    """
    # Set model for training or testing
    if test:
        model.eval()
    else:
        model.train()

    # y_preds = torch.empty(0, device=DEVICE)
    # y_targets = torch.empty(0, device=DEVICE)
    loss_data = {
        'performance_loss': 0.0,
        'spatial_loss': 0.0
    }
    dynamic_regularization = 0
    correct_predictions = 0
    total_samples = 0

    # Set mode for training or testing
    with torch.inference_mode(test):
        for X, y in data:
            X, y = X.to(DEVICE), y.to(DEVICE)
            X = X.unsqueeze(1)

            # Forward pass
            y_logits = model(X)

            if test:
                # We can't balance losses in test mode, so we only calculate performance loss
                performance_loss = loss_fn(y_logits, y)
                spatial_loss = torch.tensor(0.0, device=DEVICE)
            else:
                # Calculate balanced loss based on performance loss and spatial loss
                # Todo: move loss to an object
                performance_loss, spatial_loss, reg_update = loss.balanced_loss(loss_fn,
                                                                                y_logits,
                                                                                y,
                                                                                model,
                                                                                dynamic_regularization,
                                                                                spatial_regularization,
                                                                                spatial_grid_width)
                dynamic_regularization = reg_update
                full_loss = performance_loss + spatial_loss
                # Zero gradients
                optimizer.zero_grad()
                # Backpropagation
                full_loss.backward()
                # Parameters update
                optimizer.step()

            # Making predictions
            yp = torch.argmax(y_logits, dim=1)

            # Accumulate loss
            loss_data['performance_loss'] += performance_loss.item() * y.size(0)
            loss_data['spatial_loss'] += spatial_loss.item() * y.size(0)
            # Accumulate accuracy
            total_samples += y.size(0)
            correct_predictions += (yp == y).sum().item()

            # # Saving predictions
            # y_preds = torch.cat((y_preds, yp), dim=0)
            # # Saving targets
            # y_targets = torch.cat((y_targets, y), dim=0)

    # Calculate average loss and accuracy for the epoch
    loss_data['performance_loss'] = loss_data['performance_loss'] / total_samples
    loss_data['spatial_loss'] = loss_data['spatial_loss'] / total_samples
    acc = correct_predictions / total_samples
    return acc, loss_data


def train_model(model, train_loader, validation_loader, epochs, loss_fn, optimizer,
                spatial_regularization=0.0, spatial_grid_width=6, disable_logs=False):
    """
    Trains model for set amount of epochs.
    :param model:
    :param train_loader:
    :param validation_loader:
    :param epochs:
    :param loss_fn:
    :param optimizer:
    :param spatial_regularization: regularization parameter for spatial loss.
    :param spatial_grid_width: width of the grid for spatial loss.
    :param disable_logs: False to print logs, True if not.
    :return: Highest performing checkpoint, Logs with metrics for every epoch.
    """
    best_acc = 0
    best_checkpoint = {}
    training_history = {
        'train_loss': [], 'train_sp_loss': [],
        'valid_loss': [], 'valid_sp_loss': [],
        'train_acc': [], 'valid_acc': [],
    }
    for epoch in tqdm(range(epochs), disable=disable_logs):
        # Train step
        train_acc, train_loss_data = step(model, train_loader, loss_fn, optimizer,
                                          spatial_regularization=spatial_regularization,
                                          spatial_grid_width=spatial_grid_width,
                                          test=False)

        # Test step
        validation_acc, test_loss_data = step(model, validation_loader, loss_fn, optimizer,
                                              spatial_regularization=spatial_regularization,
                                              spatial_grid_width=spatial_grid_width,
                                              test=True)

        if not disable_logs:
            print(
                f'After epoch {epoch}, avg training loss is {train_loss_data["loss"] + train_loss_data["sp_loss"]:.4f}, avg validation loss is {test_loss_data["loss"] + test_loss_data["sp_loss"]:.4f}, acc on train set is {train_acc * 100:.2f}% and acc on validation set is {validation_acc * 100:.2f}%')

        # Saving logs
        training_history['train_loss'].append(train_loss_data['performance_loss'])
        training_history['train_sp_loss'].append(train_loss_data['spatial_loss'])
        training_history['valid_loss'].append(test_loss_data['performance_loss'])
        training_history['valid_sp_loss'].append(test_loss_data['spatial_loss'])
        training_history['train_acc'].append(train_acc)
        training_history['valid_acc'].append(validation_acc)

        # Saving model with the highest validation accuracy
        if validation_acc > best_acc:
            best_acc = validation_acc
            best_checkpoint = {
                'epoch': epoch,
                'state_dict': copy.deepcopy(model.state_dict()),
                'accuracy': validation_acc,
                'loss': test_loss_data["performance_loss"] + test_loss_data["spatial_loss"],
            }
    return best_checkpoint, training_history
