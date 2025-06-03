import os
import pandas as pd
from pycaret.time_series import *

# Data path
data_path = r'D:\College\databricks_apis\results\ingestion_table_parquet_pandas'

# Check for CSV files in the directory
files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
if not files:
    raise FileNotFoundError("No CSV files found in the specified directory.")
else:
    # Load data; could add options for concatenating multiple CSVs later
    data = pd.read_csv(os.path.join(data_path, files[0]))  
    print(f"Loaded data from {files[0]}")

# Setting up PyCaret
try:
    ts = setup(data, target='your_target_column', fold=3, fh=12, seasonal_period=12, session_id=42)
except Exception as e:
    print(f"An error occurred in setup: {e}")