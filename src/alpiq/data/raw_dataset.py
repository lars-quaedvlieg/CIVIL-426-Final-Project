from dataclasses import dataclass

import pandas as pd
import pyarrow.parquet as pq


@dataclass
class Case:
    info: pd.DataFrame
    measurements: pd.DataFrame


class RawDataset:

    def __init__(self, root, unit="VG4", load_training=False, load_synthetic=False) -> None:

        read_pq_file = lambda f: pq.read_table(root / f).to_pandas()

        cases = {
            "test": [f"{unit}_generator_data_testing_real_measurements.parquet",
                     root / f"{unit}_generator_data_testing_real_info.csv"],
        }

        if load_training:
            cases = {
                **cases,
                "train": [f"{unit}_generator_data_training_measurements.parquet",
                          root / f"{unit}_generator_data_training_info.csv"],
            }

        if load_synthetic:
            cases = {
                **cases,
                "test_s01": [f"{unit}_generator_data_testing_synthetic_01_measurements.parquet",
                             root / f"{unit}_generator_data_testing_synthetic_01_info.csv"],
                "test_s02": [f"{unit}_generator_data_testing_synthetic_02_measurements.parquet",
                             root / f"{unit}_generator_data_testing_synthetic_02_info.csv"]
            }

        self.data_dict = dict()

        for id_c, c in cases.items():
            # if you need to verify the parquet header:
            # pq_rows = RawDataset.read_parquet_schema_df(root / c[0])
            info = pd.read_csv(c[1])
            measurements = read_pq_file(c[0])
            self.data_dict[id_c] = Case(info, measurements)

    @staticmethod
    def read_parquet_schema_df(uri: str) -> pd.DataFrame:
        """Return a Pandas dataframe corresponding to the schema of a local URI of a parquet file.

        The returned dataframe has the columns: column, pa_dtype
        """
        # Ref: https://stackoverflow.com/a/64288036/
        schema = pq.read_schema(uri, memory_map=True)
        schema = pd.DataFrame(
            ({"column": name, "pa_dtype": str(pa_dtype)} for name, pa_dtype in zip(schema.names, schema.types)))
        schema = schema.reindex(columns=["column", "pa_dtype"],
                                fill_value=pd.NA)  # Ensures columns in case the parquet file has an empty dataframe.
        return schema
