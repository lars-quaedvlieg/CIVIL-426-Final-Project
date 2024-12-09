from dataclasses import dataclass

import pandas as pd
import pyarrow.parquet as pq


@dataclass
class Case():
    info: pd.DataFrame
    measurements: pd.DataFrame


class RawDataset():

    def __init__(self, root, unit = "VG5", load_training=False, load_synthetic=False, load_anomaly=False) -> None:
        
        
        read_pq_file = lambda f: pq.read_table(root / f).to_pandas()
        
        if load_anomaly:
            cases = {
                "01_type_a": [f"{unit}_anomaly_01_type_a.parquet"],
                "01_type_b": [f"{unit}_anomaly_01_type_b.parquet"],
                "01_type_c": [f"{unit}_anomaly_01_type_c.parquet"],
                "02_type_a": [f"{unit}_anomaly_02_type_a.parquet"],
                "02_type_b": [f"{unit}_anomaly_02_type_b.parquet"],
                "02_type_c": [f"{unit}_anomaly_02_type_c.parquet"]
            }
        else: 
            cases = {
                "test": [f"{unit}_generator_data_testing_real_measurements.parquet", root / f"{unit}_generator_data_testing_real_info.csv" ], 
            }
            
            if load_training:
                cases = {
                    **cases,
                    "train": [f"{unit}_generator_data_training_measurements.parquet", root / f"{unit}_generator_data_training_info.csv" ], 
                }
            
            if load_synthetic:
                cases = {
                    **cases,
                    "test_s01": [f"{unit}_generator_data_testing_synthetic_01_measurements.parquet", root / f"{unit}_generator_data_testing_synthetic_01_info.csv"], 
                    "test_s02": [f"{unit}_generator_data_testing_synthetic_02_measurements.parquet", root / f"{unit}_generator_data_testing_synthetic_02_info.csv"]
                }
            

        
        
        self.data_dict = dict()
        
        for id_c, c in cases.items():
            # if you need to verify the parquet header:
            # pq_rows = RawDataset.read_parquet_schema_df(root / c[0])
            if id_c.startswith(("01", "02")):  # Anomaly cases
                measurements = read_pq_file(c[0])  # Only read measurements
                self.data_dict[id_c] = Case(info=None, measurements=measurements)
            else:
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
        schema = pd.DataFrame(({"column": name, "pa_dtype": str(pa_dtype)} for name, pa_dtype in zip(schema.names, schema.types)))
        schema = schema.reindex(columns=["column", "pa_dtype"], fill_value=pd.NA)  # Ensures columns in case the parquet file has an empty dataframe.
        return schema
