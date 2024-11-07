import torch
import torch.nn as nn
from s5 import S5Block


class S5Backbone(nn.Module):
    def __init__(self, input_dim: int, state_dim: int, num_layers: int = 3, dropout: float = 0.0):
        """
        S5 Backbone Module.

        Args:
            input_dim (int): Dimension of input features.
            state_dim (int): Dimension of state representation in each S5 block.
            num_layers (int): Number of stacked S5 blocks.
            dropout (float): Dropout rate applied to each S5 block.
        """
        super(S5Backbone, self).__init__()
        self.s5_layers = nn.ModuleList([
            S5Block(input_dim, state_dim=state_dim, bidir=False, ff_dropout=dropout,
                    attn_dropout=dropout) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for s5_layer in self.s5_layers:
            x = s5_layer(x)
        return x
