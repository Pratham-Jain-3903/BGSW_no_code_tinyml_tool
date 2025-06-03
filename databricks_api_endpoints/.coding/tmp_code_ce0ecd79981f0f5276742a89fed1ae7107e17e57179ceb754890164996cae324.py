import pandas as pd

# Specify the data path
file_path = 'raw_data/all_data/collected_csvs/data.csv'

try:
    # Load the data
    data = pd.read_csv(file_path)
    data.head()  # Display the first few rows
except FileNotFoundError:
    'File not found. Please check the file name and path.'