import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from typing import List
import numpy as np
from tqdm import tqdm


class SequenceDataset(Dataset):
    def __init__(
            self,
            data_path: Path,
            context_length: int,
            input_feature_col_names: List[str],
            current_value_col_names: List[str],
            next_value_col_names: List[str],
            padding_value: float = 0.0,
    ):
        """
        Dataset for hydropower unit data with sequential sampling and padding support.

        Args:
            data_path (Path): Path to the processed pandas DataFrame (CSV or Parquet).
            context_length (int): Number of past time-steps to include in each sample.
            input_feature_col_names (List[str]): Columns representing input features under the "X" index.
            current_value_col_names (List[str]): Columns representing current target values under the "y_cur" index.
            next_value_col_names (List[str]): Columns representing next target values under the "y_next" index.
            padding_value (float): Value to use for padding.
        """
        # Load the data
        if data_path.suffix == '.csv':
            self.data = pd.read_csv(data_path, header=[0, 1])  # Load with multi-index
        elif data_path.suffix == '.parquet':
            self.data = pd.read_parquet(data_path)
        else:
            raise ValueError("Data file must be a CSV or Parquet file")

        self.context_length = context_length
        self.input_feature_cols = input_feature_col_names
        self.current_value_cols = current_value_col_names
        self.next_value_cols = next_value_col_names
        self.padding_value = padding_value

        # Calculate valid indices where the operating mode is not zero
        # Assumes "operating_mode" is one of the input features under "X"
        self.valid_indices = (np.where(self.data.loc[self.context_length:, ("X", "operating_mode")] != 0)[0]
                              + self.context_length)

    def __len__(self):
        """
        Returns the number of valid samples in the dataset.
        """
        return len(self.valid_indices)

    def __getitem__(self, idx: int):
        """
        Returns a sample from the dataset with padding if necessary.

        Args:
            idx (int): Index in the dataset.

        Returns:
            dict: A dictionary with keys:
                  - "operating_mode": Tensor of operating mode for the sequence.
                  - "input_sequence": Tensor of input features up to the context length.
                  - "current_values": Tensor of current values at the current time step.
                  - "next_values": Tensor of target values for the next time step.
        """
        # Map the dataset index to a valid data index
        valid_idx = self.valid_indices[idx]

        # Define start and end indices for the context window
        end_idx = valid_idx
        start_idx = max(0, end_idx - self.context_length)

        # Extract data within the context window for input features and control values
        input_sequence = self.data.loc[1+start_idx:end_idx, ("X", self.input_feature_cols)].values
        operating_modes = self.data.loc[1+start_idx:end_idx, ("X", "operating_mode")].values
        current_values = self.data.loc[end_idx, ("y_cur", self.current_value_cols)].values
        next_values = self.data.loc[end_idx, ("y_next", self.next_value_cols)].values

        # Apply padding if the sequence is shorter than the context length
        padding_length = self.context_length - len(input_sequence)
        if padding_length > 0:
            input_sequence = np.pad(
                input_sequence, ((padding_length, 0), (0, 0)), mode='constant', constant_values=self.padding_value
            )
            operating_modes = np.pad(
                operating_modes, (padding_length, 0), mode='constant', constant_values=self.padding_value
            )

        # Convert to tensors
        input_sequence = torch.tensor(input_sequence, dtype=torch.float)
        operating_modes = torch.tensor(operating_modes, dtype=torch.long)
        current_values = torch.tensor(current_values, dtype=torch.float)
        next_values = torch.tensor(next_values, dtype=torch.float)

        return {
            "operating_mode": operating_modes,
            "input_sequence": input_sequence,
            "current_values": current_values,
            "next_values": next_values,
        }