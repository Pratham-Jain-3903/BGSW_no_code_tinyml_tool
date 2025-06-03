from pycaret.classification import *

# Initialize setup with the default configuration
init_config = {
    "raw_data_path": "raw_data/all_data/collected_csvs",
    "output_table": "results/ingestion_table",
    "window_size": 2,
    "filter_columns": [],
    "drop_columns_by_type": [],
    "constant_columns_action": "drop",
    "high_cardinality_threshold": 1000,
    "high_cardinality_action": "keep",
    "missing_values_threshold": 0.95,
    "duplicate_rows_action": "keep",
    "batch_size": 10
}

from pycaret.utils import check_setup
check_setup(init_config)