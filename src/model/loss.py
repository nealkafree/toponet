import torch
import numpy as np

# Todo: move to config
MOVING_AVERAGE_PARAMETER = 0.1
DEVICE = torch.device("cuda:0")


def spatial_loss(weights: torch.Tensor, grid_width: int):
    """
    Calculates the spatial loss for a set of weights.
    :param weights: weights tensor.
    :param grid_width: width of our imagined grid. Works with any grid of square shape.
    """
    # This is a transformation which helps to calculate indexes of "neighbouring" neurons.
    neighbour_transformations = np.array(
        [-grid_width - 1, -grid_width, -grid_width + 1, -1, +1, grid_width - 1, grid_width, grid_width + 1])

    sp_loss = 0.0
    num_neighbours = 0
    for i in range(weights.shape[0]):
        # We calculate indexes and filter those out of scope.
        neighbours = neighbour_transformations + i
        neighbour_filter = (neighbours >= 0) & (neighbours < weights.shape[0]) & ~ (
                (i % grid_width == grid_width - 1) & (neighbours % grid_width == 0)) & ~ (
                (i % grid_width == 0) & (neighbours % grid_width == grid_width - 1))
        neighbours = neighbours[neighbour_filter]

        # We calculate squared distance between weights of the neuron and its neighbours.
        for neighbour in neighbours:
            distance = torch.linalg.norm(weights[i] - weights[neighbour])
            sp_loss += distance

        num_neighbours += len(neighbours)

    # We return an average distance.
    return sp_loss / num_neighbours if num_neighbours > 0 else sp_loss


def balanced_loss(loss_fn, y_logits, y, model, dynamic_regularization, spatial_regularization, spatial_grid_width):
    """
    Calculates performance loss, spatial loss and balances them based on their gradient magnitudes.
    """
    # Performance loss
    performance_loss = loss_fn(y_logits, y)

    # Spatial loss
    if spatial_regularization != 0:
        # Computing gradients for a performance loss on a linear layer with spatial constraint
        grad = torch.autograd.grad(performance_loss, model.linear.weight, retain_graph=True)[0]

        # Calculating spatial loss
        sp_loss = spatial_loss(model.linear.weight, spatial_grid_width)

        # Computing gradients for a spatial loss on a linear layer
        sp_grad = torch.autograd.grad(sp_loss, model.linear.weight, retain_graph=True)[0]

        # Implementing regularization to balance losses
        current_regularization = (spatial_regularization * torch.linalg.matrix_norm(grad).item()
                                  / (torch.linalg.matrix_norm(sp_grad).item() + 0.000000001))

        # Updating dynamic regularization
        if dynamic_regularization == 0:
            dynamic_regularization = current_regularization
        else:
            dynamic_regularization = MOVING_AVERAGE_PARAMETER * current_regularization + (
                    1 - MOVING_AVERAGE_PARAMETER) * dynamic_regularization

        # Balanced spatial loss
        sp_loss = sp_loss * dynamic_regularization
    else:
        # If spatial regularization is zero, we don't use a spatial constraint
        sp_loss = torch.tensor(0.0, device=DEVICE)

    return performance_loss, sp_loss, dynamic_regularization
