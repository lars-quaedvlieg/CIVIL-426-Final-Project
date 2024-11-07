from typing import List

import torch
import torch.nn as nn


class FullyConnectedModule(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims=None, dropout: float = 0.0):
        """
        Fully Connected Network for Control Variates.

        Args:
            input_dim (int): Dimension of the input features.
            output_dim (int): Dimension of the output (1 for each control variate).
            hidden_dims (List[int]): List defining the size of each hidden layer.
            dropout (float): Dropout rate applied after each hidden layer.
        """
        super(FullyConnectedModule, self).__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]
        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = h_dim
        layers.append(nn.Linear(input_dim, output_dim))
        self.fc = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
