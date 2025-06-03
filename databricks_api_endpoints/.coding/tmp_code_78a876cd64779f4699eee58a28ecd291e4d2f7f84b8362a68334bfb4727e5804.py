import pandas as pd
import numpy as np
import random
import datetime

# Create a mock dataset
num_records = 1000
date_rng = pd.date_range(start='2022-01-01', end='2022-01-02', freq='H')

# Generate random values for each variable
data = {
    'timestamp': date_rng,
    'HP_CompE21EnergyIn': np.random.rand(len(date_rng)),
    'HP_EHeatE21EnergyCH': np.random.rand(len(date_rng)),
    'HP_EHeatE21EnergyDHW': np.random.rand(len(date_rng)),
    'HP_EHeatE21EnergyPool': np.random.rand(len(date_rng)),
    'HP_EHeatE21EnergyTotal': np.random.rand(len(date_rng)),
    'HP_EnergyE21InCH': np.random.rand(len(date_rng)),
    'HP_EnergyE21InCool': np.random.rand(len(date_rng)),
    'HP_EnergyE21InDHW': np.random.rand(len(date_rng)),
    'HP_EnergyE21InTotal': np.random.rand(len(date_rng)),
}

mock_df = pd.DataFrame(data)

# Save the mock dataset to a CSV file
mock_df.to_csv('mock_time_series_data.csv', index=False) 

mock_df.head()