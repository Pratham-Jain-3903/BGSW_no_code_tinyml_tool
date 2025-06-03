import pandas as pd
import numpy as np

def create_mock_data(num_samples=1000):
    date_rng = pd.date_range(start='2022-01-01', end='2022-01-02', freq='H')
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
        'HP_EnergyE21InTotal': np.random.rand(len(date_rng))
    }
    return pd.DataFrame(data)

# Create mock data
mock_data = create_mock_data()  
mock_data.to_csv('mock_time_series_data.csv', index=False)