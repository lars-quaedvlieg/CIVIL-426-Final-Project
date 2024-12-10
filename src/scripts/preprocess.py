from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import StandardScaler

from alpiq.data.raw_dataset import RawDataset  # Assuming this is the file with RawDataset and Case classes

from joblib import dump, load


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


@hydra.main(config_path="configs", config_name="preprocess_config_vg4")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Attempt to load an existing scaler if specified
    scaler_path = cfg.data.get("scaler_path", None)
    continuous_columns = cfg.data.continuous_columns
    scaler_loaded = False
    scaler = StandardScaler()

    # If a scaler path is provided and it exists, load it
    if scaler_path and Path(scaler_path).exists():
        scaler = load(scaler_path)
        scaler_loaded = True
        print(f"Loaded existing scaler from {scaler_path}")

    # Load the raw data
    raw_data = RawDataset(
        root=Path(cfg.data.raw_data_root),
        unit=cfg.data.unit,
        load_training=cfg.data.load_training,
        load_synthetic=cfg.data.load_synthetic,
        load_anomaly=cfg.data.load_anomaly
    )

    # Extract split names and ensure train comes first if present
    split_names = list(raw_data.data_dict.keys())
    if "train" in split_names:
        split_names.remove("train")
        split_names = ["train"] + split_names

    # Process each case (train, eval, test) into separate DataFrames
    for split_name in split_names:
        case_data = raw_data.data_dict[split_name]

        # Select measurements DataFrame
        processed_data = case_data.measurements.copy()
        processed_data.head()
        processed_data.index = pd.to_datetime(processed_data.index)

        # Add ground_truth column
        if "ground_truth_col" in cfg.data and cfg.data.ground_truth_col in case_data.measurements.columns:
            processed_data["ground_truth"] = case_data.measurements[cfg.data.ground_truth_col]
        else:
            processed_data["ground_truth"] = 0  # Default or computed value

        # Add the injector_opening feature manually
        processed_data["injector_opening"] = processed_data["injector_01_opening"] + processed_data["injector_02_opening"] + processed_data["injector_03_opening"] + processed_data["injector_04_opening"]
        if cfg.data.unit != "VG4":
            processed_data["injector_opening"] = (processed_data["injector_opening"] + processed_data["injector_05_opening"]) / 5
        else:
            processed_data["injector_opening"] = processed_data["injector_opening"] / 4

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

        # Fit or transform with the scaler
        if not scaler_loaded:
            # If we haven't loaded a scaler, we must be on the train data to fit
            if split_name == "train":
                # Fit the scaler on the training data
                scaler.fit(processed_data[continuous_columns])
                for col in continuous_columns:
                    if col in processed_data.columns:
                        if col == 'ground_truth':
                            processed_data[col] = processed_data[[col]]
                        else:
                            processed_data[col] = scaler.fit_transform(processed_data[[col]])
                    else:
                        processed_data[col] = processed_data[[col]]
                if scaler_path is None:
                    scaler_path = f"data/{cfg.data.unit}_scaler"
                    dump(scaler, scaler_path)
                    print(f"Saved new scaler to {scaler_path}")
                scaler_loaded = True
            else:
                # Not train and no scaler loaded - can't proceed
                raise ValueError("No saved scaler found and not processing train set. "
                                 "Please run on the train split first or provide a saved scaler.")
        else:
            # Scaler loaded, just transform
            for col in continuous_columns:
                if col in processed_data.columns:
                    if col == 'ground_truth':
                        processed_data[col] = processed_data[[col]]
                    else:
                        processed_data[col] = scaler.fit_transform(processed_data[[col]])
                else:
                    processed_data[col] = processed_data[[col]]

        # Create an encoded operating mode column
        processed_data["operating_mode"] = processed_data.apply(encode_operating_mode, axis=1)

        # Create shifted y_next columns for next-step prediction
        y_columns = cfg.data.control_value_cols
        processed_data[[f"y_cur_{col}" for col in y_columns]] = processed_data[y_columns]
        processed_data[[f"y_next_{col}" for col in y_columns]] = processed_data[y_columns].shift(-1)

        # Drop the final row, which has NaN in y_next columns after shifting
        processed_data.dropna(inplace=True)

        # Create a multi-index DataFrame for X, y_cur, and y_next columns
        feature_columns = cfg.data.input_feature_cols + ["operating_mode", "ground_truth"]
        multi_index_columns = (
            pd.MultiIndex.from_product([["X"], feature_columns])
            .append(pd.MultiIndex.from_product([["y_cur"], y_columns]))
            .append(pd.MultiIndex.from_product([["y_next"], y_columns]))
        )

        # Prepare the DataFrame with X, y_cur, and y_next columns
        processed_data = processed_data[
            feature_columns + [f"y_cur_{col}" for col in y_columns] + [f"y_next_{col}" for col in y_columns]]
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
    Splits time-series data into training and evaluation sets based on split_ratio.
    Ensures that the split preserves the time order of the data.

    Args:
        data (pd.DataFrame): The full dataset, indexed or sorted by time.
        split_ratio (float): Fraction of data to use for training (e.g., 0.8 for 80% train, 20% eval).

    Returns:
        pd.DataFrame, pd.DataFrame: Training and evaluation datasets.
    """
    # Calculate the split index
    split_index = int(len(data) * split_ratio)

    # Split the data
    train_data = data.iloc[:split_index]
    eval_data = data.iloc[split_index:]

    return train_data, eval_data


if __name__ == "__main__":
    main()
