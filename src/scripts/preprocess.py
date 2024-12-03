from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import StandardScaler

from alpiq.data.raw_dataset import RawDataset  # Assuming this is the file with RawDataset and Case classes


def normalize_columns(df: pd.DataFrame, continuous_columns: list) -> pd.DataFrame:
    """
    Normalizes continuous columns in the DataFrame using StandardScaler.

    Args:
        df (pd.DataFrame): DataFrame with data to be normalized.
        continuous_columns (list): List of column names to normalize.

    Returns:
        pd.DataFrame: DataFrame with normalized continuous columns.
    """
    scaler = StandardScaler()
    for col in continuous_columns:
        df[col] = scaler.fit_transform(df[[col]])
    return df


def encode_operating_mode(row: pd.Series) -> int:
    """
    Encodes the operating mode based on specific boolean columns.
    If a column is missing, it defaults to False for that mode.

    Args:
        row (pd.Series): A row of the DataFrame.

    Returns:
        int: Encoded operating mode.
    """
    if row.get("dyn_only_on", False):
        return 1
    elif row.get("equilibrium_turbine_mode", False):
        return 2
    elif row.get("equilibrium_pump_mode", False):
        return 3
    elif row.get("equilibrium_short_circuit_mode", False):
        return 4
    else:
        return 0


@hydra.main(config_path="configs", config_name="preprocess_config_vg6")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Load the raw data
    raw_data = RawDataset(
        root=Path(cfg.data.raw_data_root),
        unit=cfg.data.unit,
        load_training=cfg.data.load_training,
        load_synthetic=cfg.data.load_synthetic,
    )

    # Process each case (train, eval, test) into separate DataFrames
    for split_name, case_data in raw_data.data_dict.items():
        # Select measurements DataFrame
        processed_data = case_data.measurements.copy()
        processed_data.index = pd.to_datetime(processed_data.index)

        # Resample the data to ensure regular intervals
        processed_data = processed_data.resample(cfg.data.resample_interval).asfreq()

        # Check for NaN values after resampling
        nan_info = processed_data.isna().sum()
        print("NaN counts per column after interpolation:")
        print(nan_info[nan_info > 0])

        # Interpolate missing values
        processed_data.interpolate(method='linear', inplace=True)

        # Replace any remaining NaNs with 0 (from non-continuous columns)
        processed_data.fillna(0, inplace=True)

        # Normalize continuous columns
        processed_data = normalize_columns(processed_data, cfg.data.continuous_columns)

        # Create an encoded operating mode column
        processed_data["operating_mode"] = processed_data.apply(encode_operating_mode, axis=1)

        # Create shifted y_next columns for next-step prediction
        y_columns = cfg.data.control_value_cols
        processed_data[[f"y_cur_{col}" for col in y_columns]] = processed_data[y_columns]
        processed_data[[f"y_next_{col}" for col in y_columns]] = processed_data[y_columns].shift(-1)

        # Drop the final row, which has NaN in y_next columns after shifting
        processed_data.dropna(inplace=True)

        # Create a multi-index DataFrame for X, y_cur, and y_next columns
        feature_columns = cfg.data.input_feature_cols + ["operating_mode"]
        multi_index_columns = (
            pd.MultiIndex.from_product([["X"], feature_columns])
            .append(pd.MultiIndex.from_product([["y_cur"], y_columns]))
            .append(pd.MultiIndex.from_product([["y_next"], y_columns]))
        )

        # Prepare the DataFrame with X, y_cur, and y_next columns
        processed_data = processed_data[feature_columns + [f"y_cur_{col}" for col in y_columns] + [f"y_next_{col}" for col in y_columns]]
        processed_data.columns = multi_index_columns

        print(processed_data.columns)

        # Split data if specified (e.g., split train data into train and eval)
        if split_name == "train" and cfg.data.split_train_eval:
            train_data, eval_data = split_train_eval(processed_data, cfg.data.train_eval_split_ratio)
        else:
            train_data, eval_data = processed_data, None

        # Save processed files
        output_dir = Path(cfg.data.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if split_name == "train":
            train_data.to_csv(output_dir / cfg.data.train_output_file, index=False)
            print(f"Saved training data to {output_dir / cfg.data.train_output_file}")
            if eval_data is not None:
                eval_data.to_csv(output_dir / cfg.data.eval_output_file, index=False)
                print(f"Saved evaluation data to {output_dir / cfg.data.eval_output_file}")
        else:
            processed_data.to_csv(output_dir / f"processed_{split_name}_{cfg.data.unit}.csv", index=False)
            print(f"Saved test data to {output_dir / f'processed_{split_name}_{cfg.data.unit}.csv'}")


def split_train_eval(data: pd.DataFrame, split_ratio: float):
    """
    Splits data into training and evaluation sets based on split_ratio.

    Args:
        data (pd.DataFrame): The full dataset.
        split_ratio (float): Fraction of data to use for training (e.g., 0.8 for 80% train, 20% eval).

    Returns:
        pd.DataFrame, pd.DataFrame: Training and evaluation datasets.
    """
    train_data = data.sample(frac=split_ratio, random_state=42).sort_index()
    eval_data = data.drop(train_data.index).sort_index()
    return train_data, eval_data


if __name__ == "__main__":
    main()
