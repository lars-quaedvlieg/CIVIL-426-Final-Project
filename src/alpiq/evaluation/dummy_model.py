from typing import List

import torch
import torch.nn as nn

from s5 import S5Block

from prettytable import PrettyTable


def print_num_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    total_trainable_params = 0
    for name, parameter in model.named_parameters():
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
        if parameter.requires_grad:
            total_trainable_params += params
    print(table)
    print(f"Total Params: {total_params}")
    print(f"Total Trainable Params: {total_trainable_params}")
    return total_params

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

class CausalModel(nn.Module):
    def __init__(
            self,
            input_dim: int,
            num_operating_modes: int,
            sequence_length: int,
            embedding_dim: int = 16,
            state_dim: int = 64,
            num_s5_layers: int = 3,
            num_control_variates: int = 3,
            s5_dropout: float = 0.0,
            fc_hidden_dims: [] = None,
            fc_dropout: float = 0.0,
    ):
        """
        Causal Model for Anomaly Detection using S5 blocks and FC layers.

        Args:
            input_dim (int): Dimension of input signals.
            num_operating_modes (int): Dimension of operating modes.
            embedding_dim (int): Dimension of embedding layer for control variables.
            state_dim (int): State dimension for S5 backbone.
            num_s5_layers (int): Number of S5 layers in the backbone.
            num_control_variates (int): Number of control variates (FC networks).
            s5_dropout (float): Dropout rate for the S5 backbone.
            fc_hidden_dims (List[int]): Hidden layer sizes for each FC network.
            fc_dropout (float): Dropout rate for each FC network.
        """
        super(CausalModel, self).__init__()
        # Embedding layer for control variables (Op_x)
        if fc_hidden_dims is None:
            fc_hidden_dims = [64, 32]

        self.sequence_length = sequence_length
        self.embedding = nn.Embedding(num_operating_modes, embedding_dim)

        # S5 Backbone
        concat_dim = input_dim + embedding_dim
        self.s5_backbone = S5Backbone(concat_dim, state_dim, num_layers=num_s5_layers,
                                      dropout=s5_dropout)

        # Control Variate FC networks
        fc_input_dim = concat_dim + 1  # We add 1, since we concatenate the current control values to the SSM outputs
        self.control_variates = nn.ModuleList([
            FullyConnectedModule(fc_input_dim, 1, hidden_dims=fc_hidden_dims, dropout=fc_dropout)
            for _ in range(num_control_variates)
        ])

    def forward(self, op_x: torch.Tensor, input_signals: torch.Tensor,
                cur_control_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Causal Model.

        Args:
            op_x (torch.Tensor): Control variable indices (batch size, sequence length).
            input_signals (torch.Tensor): Input signal tensor (batch size, sequence length, input_dim).
            cur_control_values (torch.Tensor): Current control values (batch size, num_control_variates).

        Returns:
            torch.Tensor: Control variate predictions (batch size, sequence length, num_control_variates).
        """
        # Step 1: Embed control variables (Op_x)
        embedded_op_x = self.embedding(op_x)

        # Step 2: Concatenate embedded control variables with input signals
        concatenated_inputs = torch.cat((embedded_op_x, input_signals), dim=-1)

        # Step 3: Pass through the S5 backbone
        s5_output = self.s5_backbone(concatenated_inputs)[:, -1, ...]

        # Expand s5_output to match cur_control_values dimensions and concatenate
        s5_output_expanded = s5_output.unsqueeze(1).expand(-1, cur_control_values.size(1), -1)  # Shape: (B, C, D)
        s5_and_cur_control = torch.cat((s5_output_expanded, cur_control_values.unsqueeze(-1)),
                                       dim=-1)  # Shape: (B, C, D + 1)

        # Step 5: Apply each FC network to the concatenated S5 and cur_control values
        control_variate_outputs = [fc(s5_and_cur_control[:, var_idx]) for var_idx, fc in
                                   enumerate(self.control_variates)]

        # Stack along the last dimension to get (batch size, sequence length, num_control_variates)
        # and return the final value in the sequence
        return torch.cat(control_variate_outputs, dim=-1)


if __name__ == '__main__':

    # Example parameters
    input_dim = 10  # Number of input signals
    num_operating_modes = 5  # Number of operating modes
    sequence_length = 20  # Fixed sequence length
    embedding_dim = 16  # Embedding dimension for control variables
    hidden_dim = 64  # Hidden dimension for S5 backbone
    num_s5_layers = 3  # Number of S5 layers in the backbone
    num_control_variates = 3  # Number of control variates
    s5_dropout = 0.2  # Dropout for S5 backbone
    fc_hidden_dims = [64, 32]  # Hidden layers for FC networks
    fc_dropout = 0.2  # Dropout for FC networks

    # Instantiate the model
    model = CausalModel(
        input_dim=input_dim,
        num_operating_modes=num_operating_modes,
        sequence_length=sequence_length,
        embedding_dim=embedding_dim,
        state_dim=hidden_dim,
        num_s5_layers=num_s5_layers,
        num_control_variates=num_control_variates,
        s5_dropout=s5_dropout,
        fc_hidden_dims=fc_hidden_dims,
        fc_dropout=fc_dropout
    )
    print(model)
    print_num_parameters(model)

    # Dummy data
    op_x = torch.randint(0, num_operating_modes, (32, sequence_length))
    input_signals = torch.randn(32, sequence_length, input_dim)
    cur_control_values = torch.randn(32, num_control_variates)

    # Forward pass
    output = model(op_x, input_signals, cur_control_values)
    # Should be (32, num_control_variates)
    print(output.shape)