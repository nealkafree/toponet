import torch
import numpy as np


class BalancedLoss:
    def __init__(self, performance_loss, model, spatial_regularization, moving_average):
        self.performance_loss = performance_loss
        self.model = model
        self.dynamic_regularization = 0
        self.moving_average = moving_average
        self.spatial_regularization = spatial_regularization

    def spatial_loss(self, weights: torch.Tensor, grid_width: int):
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

    def calculate_loss(self, y_logits, y):
        """
        Calculates performance loss, spatial loss and balances them based on their gradient magnitudes.
        :param y_logits: logits tensor.
        :param y: target tensor.
        :return: balanced performance and spatial losses.
        """
        # Performance loss
        performance_loss = self.performance_loss(y_logits, y)

        # Spatial loss

        # Computing gradients for a performance loss on a linear layer with spatial constraint
        grad = torch.autograd.grad(performance_loss, self.model.linear.weight, retain_graph=True)[0]

        # Calculating spatial loss
        sp_loss = self.spatial_loss(self.model.linear.weight, self.model.spatial_grid_width)

        # Computing gradients for a spatial loss on a linear layer
        sp_grad = torch.autograd.grad(sp_loss, self.model.linear.weight, retain_graph=True)[0]

        # Implementing regularization to balance losses
        current_regularization = (self.spatial_regularization * torch.linalg.matrix_norm(grad).item()
                                  / (torch.linalg.matrix_norm(sp_grad).item() + 0.000000001))

        # Updating dynamic regularization
        if self.dynamic_regularization == 0:
            dynamic_regularization = current_regularization
        else:
            self.dynamic_regularization = self.moving_average * current_regularization + (
                    1 - self.moving_average) * self.dynamic_regularization

        # Balanced spatial loss
        sp_loss = sp_loss * self.dynamic_regularization

        return performance_loss, sp_loss
