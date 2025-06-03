import pandas as pd

# Define the path to the raw data
raw_data_path = 'raw_data/all_data/collected_csvs'

# Load the dataset
all_data = pd.read_csv(raw_data_path + '/data.csv')

# Step 1: Find DataTypes
column_data_types = all_data.dtypes.reset_index()
column_data_types.columns = ['Column', 'DataType']

# Show the datatypes of each column
column_data_types