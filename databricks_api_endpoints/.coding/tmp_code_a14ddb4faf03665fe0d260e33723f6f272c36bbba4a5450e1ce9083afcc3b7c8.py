import pandas as pd
from pycaret.time_series import *

# Load the mock dataset
data = pd.read_csv('mock_time_series_data.csv', parse_dates=['timestamp'])

data.set_index('timestamp', inplace=True)

# Initialize PyCaret setup for time series
setup(data, fh=24)  # fh is the forecast horizon