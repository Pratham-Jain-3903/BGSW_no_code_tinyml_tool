import os
import pandas as pd
from pycaret.time_series import *

def load_data(data_path):
    # Check if the directory exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The directory {data_path} does not exist.")

    # Check for CSV files in the directory
    files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    if not files:
        raise FileNotFoundError("No CSV files found in the specified directory.")
    else:
        # Load data; concatenate if multiple files exist
        data_frames = []
        for file in files:
            df = pd.read_csv(os.path.join(data_path, file))
            data_frames.append(df)
        data = pd.concat(data_frames, ignore_index=True)  # Combine all files into a single DataFrame
        print(f"Loaded data from: {', '.join(files)}")
        return data

def pycaret_setup(data):
    # Setting up PyCaret
    try:
        ts = setup(data, target='your_target_column', fold=3, fh=12, seasonal_period=12, session_id=42)
    except Exception as e:
        print(f"An error occurred in setup: {e}")

# Define the data path
data_path = r'D:\College\databricks_apis\results\ingestion_table_parquet_pandas'

# Load data and run setup
data = load_data(data_path)
pycaret_setup(data)