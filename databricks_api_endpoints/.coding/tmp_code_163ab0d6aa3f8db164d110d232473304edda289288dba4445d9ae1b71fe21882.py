import os

# Define the path to the expected raw data
raw_data_path = 'raw_data/all_data/collected_csvs/'

# Check if the directory exists and list files in it
file_exists = os.path.exists(raw_data_path)
files_in_directory = os.listdir(raw_data_path) if file_exists else []

file_exists, files_in_directory