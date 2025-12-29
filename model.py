import torch
from torch import nn
import torch.nn.functional as F


class ConstrainedConv2d(nn.Conv2d):
    """
    Implementation of maximum norm constraint for Conv2D layer
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight.clamp(max=1.0), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class ConstrainedLinear(nn.Linear):
    """
    Implementation of maximum norm constraint for Linear layer
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.clamp(max=0.5), self.bias)


class EEGNet(nn.Module):
    def __init__(self, sensors: int, samples: int, num_classes: int, filter_size=64, f1=8, depth=2, f2=16,
                 spatial_layer_size=36, dropout=0.25):
        super().__init__()
        self.block1 = nn.Sequential(
            # This layer does 1d convolutions on data from sensors.
            nn.Conv2d(in_channels=1, out_channels=f1, kernel_size=(1, filter_size), padding='same',
                      bias=False),
            nn.BatchNorm2d(f1),

            # Depthwise convolution. Looks for spatial patterns and combines all data from sensors together.
            # We use our own implementation of convolution layer because pytorch doesn't have weight constraints implemented.
            ConstrainedConv2d(in_channels=f1, out_channels=f1 * depth, kernel_size=(sensors, 1), padding='valid',
                              groups=f1, bias=False),
            nn.BatchNorm2d(f1 * depth),
            nn.ELU(),

            # Average pooling to reduce sampling rate
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            # Implementation of Separable convolution.
            # First we use Depthwise convolution (separately for all channels).
            nn.Conv2d(in_channels=f1 * depth, out_channels=f1 * depth, kernel_size=(1, 16), padding='same',
                      groups=f1 * depth, bias=False),

            # Next we combine channels with Pointwise convolution.
            nn.Conv2d(in_channels=f1 * depth, out_channels=f2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),

            # Average pooling to reduce sampling rate
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
            nn.Flatten(),
        )
        # We have to add one dense layer in order to implement topographical constraints.
        # We use our own implementation again to introduce weights constraint.
        self.linear = ConstrainedLinear(in_features=f2 * (samples // 32), out_features=spatial_layer_size)

        # Classifier layer
        self.classifier = nn.Linear(in_features=spatial_layer_size, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.linear(x)
        return self.classifier(x)
