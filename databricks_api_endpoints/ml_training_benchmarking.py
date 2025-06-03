import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import sys
import warnings
from pathlib import Path
import traceback

# Import Flask
from flask import Flask, request, jsonify

# Suppress warnings
warnings.filterwarnings('ignore')

# Import PyCaret modules
from pycaret.regression import *
from pycaret.classification import *
from pycaret.clustering import *
from pycaret.anomaly import *

# Check for time series module
try:
    from pycaret.time_series import *
    TS_AVAILABLE = True
except ImportError:
    print("Warning: PyCaret time_series module not available. Install with 'pip install pycaret[full]'")
    TS_AVAILABLE = False

# Check for ONNX availability
try:
    import onnxruntime
    import skl2onnx
    import onnxmltools
    ONNX_AVAILABLE = True
except ImportError:
    print("ONNX libraries not available. ONNX conversion will be skipped.")
    ONNX_AVAILABLE = False

# Check for TensorFlow/Keras availability
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, LSTM, GRU, SimpleRNN, Dropout, BatchNormalization
    from tensorflow.keras.layers import Input, Concatenate
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    KERAS_AVAILABLE = True
    
    # Check for tf2onnx availability
    try:
        import tf2onnx
        TF2ONNX_AVAILABLE = True
    except ImportError:
        print("tf2onnx not available. Keras to ONNX conversion will be skipped.")
        TF2ONNX_AVAILABLE = False
except ImportError:
    print("TensorFlow/Keras not available. Neural network models will be skipped.")
    KERAS_AVAILABLE = False
    TF2ONNX_AVAILABLE = False

# Initialize logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Main Configuration
ml_config = {
    "use_case": "time_series",  # Default case: time_series (alternatives: regression, classification, clustering, anomaly)
    "gpu_enabled": False,      # Use GPU acceleration if available
    "k_folds": 5,              # Number of cross-validation folds
    "test_size": 0.2,          # Proportion of data to use for testing
    "logging": "default",      # Logging level
    "top_n_models": 5,         # Number of top models to save
    "save_quantization_data": True,  # Save test data for quantization (default: True)
    "quantization_data_size": 0.1,   # Portion of data to save for quantization (default: 10%)
    "use_pycaret": False,      # Whether to use PyCaret for model training (set to False to save time)
    "target_columns": [        # Target columns to train models for
        "HP_CompE21EnergyIn", 
        "HP_EHeatE21EnergyCH",
        "HP_EHeatE21EnergyDHW",
        "HP_EHeatE21EnergyPool",
        "HP_EHeatE21EnergyTotal",
        "HP_EnergyE21InCH",
        "HP_EnergyE21InCool", 
        "HP_EnergyE21InDHW",
        "HP_EnergyE21InTotal"
    ],
    "primary_target": "HP_CompE21EnergyIn",  # Primary target for reporting
    "output_dir": "results/ml_models",       # Output directory
    "excluded_models": [],     # Models to exclude from training
    
    # Standard ML parameters
    "normalize": True,         # Normalize features
    "fix_imbalance": False,    # Fix class imbalance (for classification)
    "feature_selection": False, # Additional feature selection
    "remove_multicollinearity": True, # Remove multicollinear features
    "pca": False,              # Apply PCA
    "ignore_low_variance": True, # Ignore low variance features
    "polynomial_features": False, # Create polynomial features
    "feature_interaction": False, # Create feature interactions
    "save_formats": ["pkl", "onnx", "tflite", "keras"], # Model formats to save
    
    # Time series specific parameters
    "time_series": {
        "fh": 24,               # Forecast horizon (number of periods to predict)
        "seasonal_period": "D",  # Seasonal period: 'D'=daily, 'W'=weekly, 'M'=monthly, 'Q'=quarterly, 'Y'=yearly
        "seasonality": True,     # Whether to model seasonality
        "exogenous_features": [], # List of exogenous features to use
        "sort_by": None,         # Column to sort by (e.g., date column)
        "seasonal_periods": [7, 30, 365],  # Periods to consider for seasonality
        "transformations": ["detrend", "difference", "log"],  # Data transformations to try
        "decomposition": ["additive", "multiplicative"],  # Decomposition methods to try
        "cross_validation": True, # Whether to use cross-validation
        "fold": 3,              # Number of folds for time series cross-validation
        
        # Neural network specific parameters
        "nn_models": True,      # Whether to include neural network models
        "lstm_layers": [1, 2],  # Number of LSTM layers to try
        "lstm_units": [32, 64], # Number of LSTM units per layer to try
        "epochs": 50,           # Maximum number of training epochs
        "batch_size": 32,       # Batch size for training
        "lookback": 30,         # Number of past time steps to use as input
        "dropout_rate": 0.2,    # Dropout rate for regularization
        "early_stopping": True, # Whether to use early stopping
        "max_lag": 7,           # Maximum lag for feature creation
        "rolling_windows": [7, 14, 30]  # Windows for rolling statistics
    }
}

# --- START API ENDPOINTS ---

@app.route('/api/ml/config', methods=['GET'])
def get_ml_config():
    """API endpoint to retrieve current ML configuration."""
    logger.info("API CALL: GET /api/ml/config")
    return jsonify(ml_config)

@app.route('/api/ml/config', methods=['POST'])
def update_ml_config_api():
    """API endpoint to update ML configuration."""
    global ml_config
    new_config_params = request.json
    if not new_config_params:
        logger.error("API CALL: POST /api/ml/config - Error: No configuration parameters provided")
        return jsonify({"status": "error", "message": "No configuration parameters provided"}), 400
    
    logger.info(f"API CALL: POST /api/ml/config with data: {new_config_params}")
    
    # Simple way to update, for nested dicts like 'time_series', this replaces the whole sub-dict
    # For more granular update of nested dicts, a recursive update function would be needed
    for key, value in new_config_params.items():
        if key in ml_config:
            if isinstance(ml_config[key], dict) and isinstance(value, dict):
                # Update nested dictionaries (e.g., time_series)
                for sub_key, sub_value in value.items():
                    if sub_key in ml_config[key]:
                        ml_config[key][sub_key] = sub_value
                    else:
                        logger.warning(f"API CALL: Unknown sub-parameter '{sub_key}' in '{key}' during config update.")
            else:
                ml_config[key] = value
        else:
            logger.warning(f"API CALL: Unknown configuration parameter '{key}' during config update.")
            
    logger.info(f"API CALL: ML configuration updated. New config: {ml_config}")
    return jsonify({"status": "success", "config": ml_config})

@app.route('/api/ml/run', methods=['POST'])
def run_ml_pipeline_api():
    """API endpoint to run the ML training and benchmarking pipeline."""
    global ml_config
    # Store a deep copy of the original config to restore after the run if overrides are used
    original_config = {k: (v.copy() if isinstance(v, dict) else v) for k, v in ml_config.items()}
    # For nested dicts like 'time_series', ensure its sub-keys are also copied
    if 'time_series' in original_config and isinstance(original_config['time_series'], dict):
        original_config['time_series'] = original_config['time_series'].copy()


    try:
        if request.json:
            temp_override_config = request.json
            logger.info(f"API CALL: POST /api/ml/run with temporary config override: {temp_override_config}")
            
            # Create a temporary config for this run by overriding parts of the global ml_config
            # This is a shallow copy, modify ml_config directly for this run
            for key, value in temp_override_config.items():
                 if key in ml_config:
                    if isinstance(ml_config[key], dict) and isinstance(value, dict):
                        # Update nested dictionaries (e.g., time_series)
                        for sub_key, sub_value in value.items():
                            if sub_key in ml_config[key]:
                                ml_config[key][sub_key] = sub_value
                            else:
                                logger.warning(f"API CALL: Unknown temporary sub-parameter '{sub_key}' in '{key}' for override.")
                    else:
                        ml_config[key] = value
                 else:
                    logger.warning(f"API CALL: Unknown temporary config key '{key}' for override.")
        else:
            logger.info("API CALL: POST /api/ml/run called without temporary override (using current global config).")

        logger.info("API CALL: Starting ML pipeline via API...")
        results = main() # Call the main pipeline function
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"API CALL: Error in /api/ml/run: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        # Restore the original global config if it was temporarily modified
        ml_config = original_config
        logger.info("API CALL: Restored original ml_config after /api/ml/run execution.")

# --- END API ENDPOINTS ---

def load_feature_metadata():
    """Load feature metadata to find the latest artifacts"""
    try:
        metadata_path = "results/feature_metadata.json"
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Feature metadata not found at {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        logger.info("Successfully loaded feature metadata")
        return metadata
    except Exception as e:
        logger.error(f"Error loading feature metadata: {str(e)}")
        return {}

def get_latest_artifact_path(metadata):
    """Get path to latest feature artifacts"""
    if not metadata or "artifacts" not in metadata:
        raise ValueError("Invalid metadata: missing artifacts information")
    
    # Get the latest timestamp
    latest_timestamp = metadata["artifacts"].get("latest")
    if not latest_timestamp:
        raise ValueError("No 'latest' timestamp found in metadata")
    
    logger.info(f"Using artifacts from timestamp: {latest_timestamp}")
    
    # Base path for artifacts
    artifacts_dir = "results/feature_artifacts"
    
    # Path to selected features data
    selected_features_path = os.path.join(artifacts_dir, f"selected_features_data_{latest_timestamp}")
    
    if not os.path.exists(selected_features_path):
        raise FileNotFoundError(f"Selected features data not found at {selected_features_path}")
    
    return selected_features_path, metadata["artifacts"].get(latest_timestamp, {})

def load_feature_data(data_path):
    """Load the feature-engineered data using pandas"""
    try:
        logger.info(f"Loading data from {data_path}")
        
        # Check if directory exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path not found: {data_path}")
        
        # Find parquet files
        if os.path.isdir(data_path):
            # Load all parquet files in directory
            df = pd.read_parquet(data_path)
        else:
            # Load single parquet file
            df = pd.read_parquet(data_path)
        
        logger.info(f"Successfully loaded data with shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading feature data: {str(e)}")
        raise

def check_target_availability(df, target_columns):
    """Check which target columns are available in the dataframe"""
    available_targets = []
    for target in target_columns:
        if target in df.columns:
            available_targets.append(target)
        else:
            logger.warning(f"Target column '{target}' not found in dataframe")
    
    if not available_targets:
        raise ValueError("None of the specified target columns are available in the dataframe")
    
    logger.info(f"Available target columns: {available_targets}")
    return available_targets

def prepare_ml_data(df, target_column):
    """Prepare data for ML training"""
    # Make sure target column exists in the dataframe
    if target_column not in df.columns:
        logger.error(f"Target column '{target_column}' not found in dataframe.")
        logger.info(f"Available columns: {df.columns.tolist()}")
        raise ValueError(f"Target column '{target_column}' not found in dataframe")
    
    logger.info(f"Preparing data for target: {target_column}")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    columns_with_nulls = missing_values[missing_values > 0]
    
    if not columns_with_nulls.empty:
        logger.warning(f"Found columns with missing values: {columns_with_nulls.to_dict()}")
        logger.info("These will be handled by PyCaret's preprocessing pipeline")
    
    # Check for infinite values and replace with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Return the prepared dataframe
    return df

def save_quantization_test_data(df, target_column, output_dir):
    """
    Save a portion of the data as test data for quantization.
    
    Args:
        df: DataFrame containing the data
        target_column: Target column for the model
        output_dir: Directory to save the test data
        
    Returns:
        Path to the saved test data CSV file
    """
    try:
        if not ml_config["save_quantization_data"]:
            logger.info("Skipping quantization test data saving (disabled in config)")
            return None
            
        # Calculate the number of rows to save
        num_rows = int(len(df) * ml_config["quantization_data_size"])
        if num_rows < 10:  # Ensure we have at least a minimum number of samples
            num_rows = min(10, len(df))
            
        logger.info(f"Saving {num_rows} rows ({ml_config['quantization_data_size']*100:.1f}%) as quantization test data")
        
        # Create output directory for test data
        quant_dir = os.path.join(output_dir, "quantization_data")
        os.makedirs(quant_dir, exist_ok=True)
        
        # Take a stratified sample if classification, otherwise random
        if ml_config["use_case"] == "classification":
            try:
                # Try to get a stratified sample
                from sklearn.model_selection import train_test_split
                _, test_data = train_test_split(
                    df, 
                    test_size=ml_config["quantization_data_size"],
                    stratify=df[target_column],
                    random_state=42
                )
            except:
                # Fall back to random sample
                test_data = df.sample(n=num_rows, random_state=42)
        else:
            # Random sample for regression, time series, etc.
            test_data = df.sample(n=num_rows, random_state=42)
        
        # Generate timestamp to avoid overwriting
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as CSV with headers
        csv_path = os.path.join(quant_dir, f"{target_column}_quant_data_{timestamp}.csv")
        test_data.to_csv(csv_path, index=True, header=True)
        
        logger.info(f"Quantization test data saved to: {csv_path}")
        return csv_path
        
    except Exception as e:
        logger.error(f"Error saving quantization test data: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def create_time_series_features(df, target_column, config=None):
    """
    Create advanced features for time series forecasting.
    Includes lags, rolling stats, and Fourier features.
    """
    if config is None:
        config = ml_config["time_series"]
        
    logger.info("Generating advanced time series features...")
    
    # Make sure the DataFrame has a datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("DataFrame does not have DatetimeIndex. Creating time features may be limited.")
        
    # Create lag features
    max_lag = config.get("max_lag", 7)
    if max_lag > 0:
        for lag in range(1, max_lag + 1):
            df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
            
    # Create rolling statistics
    windows = config.get("rolling_windows", [7, 14, 30])
    for window in windows:
        # Rolling mean
        df[f'{target_column}_rolling_mean_{window}'] = df[target_column].rolling(window=window).mean()
        
        # Rolling std
        df[f'{target_column}_rolling_std_{window}'] = df[target_column].rolling(window=window).std()
        
        # Rolling min/max
        df[f'{target_column}_rolling_min_{window}'] = df[target_column].rolling(window=window).min()
        df[f'{target_column}_rolling_max_{window}'] = df[target_column].rolling(window=window).max()
    
    # Create seasonal features using Fourier terms
    if isinstance(df.index, pd.DatetimeIndex):
        for period in config.get("seasonal_periods", [7, 30, 365]):
            for order in [1, 2, 3]:  # Multiple Fourier orders
                # Sine component
                df[f'sin_{period}_{order}'] = np.sin(order * 2 * np.pi * df.index.dayofyear / period)
                # Cosine component
                df[f'cos_{period}_{order}'] = np.cos(order * 2 * np.pi * df.index.dayofyear / period)
    
    # Create additional time-based features
    if isinstance(df.index, pd.DatetimeIndex):
        # Day of week (one-hot encoded)
        for i in range(7):
            df[f'day_of_week_{i}'] = (df.index.dayofweek == i).astype(int)
            
        # Month (one-hot encoded)
        for i in range(1, 13):
            df[f'month_{i}'] = (df.index.month == i).astype(int)
            
        # Hour of day (if data has time component)
        if not (df.index.hour == 0).all():
            for i in range(24):
                df[f'hour_{i}'] = (df.index.hour == i).astype(int)
    
    # Drop NaN values created by lag/rolling features
    initial_rows = len(df)
    df = df.dropna()
    logger.info(f"Dropped {initial_rows - len(df)} rows with NaN values from feature creation")
    
    logger.info(f"Created {len(df.columns) - 1} features for time series forecasting")
    return df

def prepare_time_series_data(df, target_column, config=None):
    """
    Prepare data for time series forecasting.
    
    Args:
        df: Input DataFrame
        target_column: Target column to forecast
        config: Configuration dictionary
    
    Returns:
        Prepared DataFrame and additional metadata
    """
    if config is None:
        config = ml_config["time_series"]
    
    logger.info(f"Preparing time series data for target: {target_column}")
    
    # Check if we have a datetime index or need to create one
    if config["sort_by"] is not None and config["sort_by"] in df.columns:
        # Sort by datetime column
        logger.info(f"Sorting data by {config['sort_by']}")
        df = df.sort_values(by=config["sort_by"])
        
        # Check if the sort column is a datetime
        if pd.api.types.is_datetime64_any_dtype(df[config["sort_by"]]):
            # Set as index
            df = df.set_index(config["sort_by"])
        else:
            # Try to convert to datetime
            try:
                df[config["sort_by"]] = pd.to_datetime(df[config["sort_by"]])
                df = df.set_index(config["sort_by"])
            except:
                logger.warning(f"Could not convert {config['sort_by']} to datetime. Creating artificial time index.")
                # Create an artificial time index
                df = df.reset_index(drop=True)
                df.index = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
    else:
        # No sort column provided, create artificial time index
        logger.info("No sort column provided. Creating artificial time index.")
        df = df.reset_index(drop=True)
        df.index = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
    
    # Handle missing values
    missing_values = df[target_column].isnull().sum()
    if missing_values > 0:
        logger.warning(f"Found {missing_values} missing values in target column. Interpolating...")
        df[target_column] = df[target_column].interpolate(method='time')
    
    # Check if we still have missing values at the start/end
    if df[target_column].isnull().sum() > 0:
        logger.warning("Still have missing values after interpolation. Filling with ffill/bfill...")
        df[target_column] = df[target_column].fillna(method='ffill').fillna(method='bfill')
    
    # Check for stationarity and suggest transformations
    try:
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(df[target_column].values)
        if result[1] > 0.05:
            logger.info("Time series is not stationary. Differencing may be required.")
    except:
        logger.warning("Could not check for stationarity. Continuing anyway.")
    
    # Create features that might be useful for time series
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year
    df['day_of_year'] = df.index.dayofyear
    
    logger.info(f"Time series data preparation complete. Shape: {df.shape}")
    
    return df

def setup_pycaret_time_series(df, target_column, config=None):
    """Set up PyCaret time series forecasting experiment"""
    if not TS_AVAILABLE:
        raise ImportError("PyCaret time_series module is not available. Install with 'pip install pycaret[full]'")
    
    if config is None:
        config = ml_config["time_series"]
    
    logger.info(f"Setting up PyCaret time series forecasting for {target_column}")
    
    # Check if we have a datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("DataFrame does not have DatetimeIndex. This may cause issues with some models.")
    
    # Basic setup parameters
    setup_params = {
        'data': df,
        'target': target_column,
        'fh': config.get('fh', 24),  # Forecast horizon
        'seasonal_period': config.get('seasonal_period', 'D'),  # Seasonal period
        # 'seasonality': config.get('seasonality', True),  # Whether to model seasonality
        'session_id': 42,  # For reproducibility
        'verbose': True,
    }
    
    # Add optional parameters if specified
    if config.get('exogenous_features'):
        setup_params['exogenous_features'] = config['exogenous_features']
    
    if config.get('fold') is not None and config.get('cross_validation', True):
        setup_params['fold'] = config['fold']
        
    if config.get('seasonal_periods'):
        setup_params['seasonal_periods'] = config['seasonal_periods']
    
    # Set up experiment
    experiment = setup(**setup_params)
    
    logger.info("PyCaret time series setup complete")
    return experiment

def setup_pycaret(df, target_column, use_case="time_series", config=None):
    """Set up PyCaret environment based on the use case"""
    if config is None:
        config = ml_config
    
    logger.info(f"Setting up PyCaret for {use_case} with target: {target_column}")
    
    # For time series, use the specialized function
    if use_case == "time_series":
        if TS_AVAILABLE:
            return setup_pycaret_time_series(df, target_column, ml_config["time_series"])
        else:
            logger.warning("Time series module not available, falling back to regression")
            use_case = "regression"
    
    # Common setup parameters
    setup_params = {
        'data': df,
        'target': target_column,
        'session_id': 42,  # For reproducibility
        'log_experiment': config.get('logging', 'default') != 'default',
        'experiment_name': f"{use_case}_{target_column}",
        'fold': config.get('k_folds', 5),
        'train_size': 1 - config.get('test_size', 0.2),
        'use_gpu': config.get('gpu_enabled', False),
        'normalize': config.get('normalize', True),
        'remove_multicollinearity': config.get('remove_multicollinearity', True),
        'polynomial_features': config.get('polynomial_features', False),
        'feature_interaction': config.get('feature_interaction', False),
        'ignore_low_variance': config.get('ignore_low_variance', True),
        'feature_selection': config.get('feature_selection', False),
        'pca': config.get('pca', False),
        'html': False,  # Disable HTML output for headless environments
        'silent': True,  # Suppress setup summary
    }
    
    # Set up PyCaret based on use case
    if use_case == "regression":
        return regression.setup(**setup_params)
    elif use_case == "classification":
        setup_params['fix_imbalance'] = config.get('fix_imbalance', False)
        return classification.setup(**setup_params)
    elif use_case == "clustering":
        # Clustering doesn't use target column
        setup_params.pop('target')
        return clustering.setup(**setup_params)
    elif use_case == "anomaly":
        # Anomaly detection doesn't use target column
        setup_params.pop('target')
        return anomaly.setup(**setup_params)
    else:
        raise ValueError(f"Unsupported use case: {use_case}")

def train_and_evaluate_models(use_case, top_n=5, excluded_models=None):
    """Train and evaluate multiple models using PyCaret"""
    logger.info(f"Training and evaluating models...")
    
    if excluded_models is None:
        excluded_models = []
    
    # Define the appropriate module based on use case
    if use_case == "regression":
        module = regression
    elif use_case == "classification":
        module = classification
    elif use_case == "clustering":
        module = clustering
    elif use_case == "anomaly":
        module = anomaly
    else:
        raise ValueError(f"Unsupported use case: {use_case}")
    
    # Train models
    try:
        # Use PyCaret's compare_models function to train and evaluate multiple models
        top_models = module.compare_models(n_select=top_n, exclude=excluded_models, verbose=True)
        
        # If only one model is returned, convert to list
        if not isinstance(top_models, list):
            top_models = [top_models]
        
        # Get performance metrics for each model
        models_metrics = []
        for i, model in enumerate(top_models):
            # Evaluate model
            try:
                if use_case in ["regression", "classification"]:
                    eval_result = module.pull()
                    model_name = module.get_config('prep_pipe')[0] + ' ' + str(i+1)
                else:
                    # For clustering and anomaly, just store the model type
                    eval_result = pd.DataFrame()
                    model_name = str(type(model).__name__)
                
                models_metrics.append({
                    'model_name': model_name,
                    'metrics': eval_result.to_dict() if not eval_result.empty else {}
                })
            except Exception as e:
                logger.error(f"Error evaluating model {i+1}: {str(e)}")
                models_metrics.append({
                    'model_name': f"model_{i+1}",
                    'metrics': {},
                    'error': str(e)
                })
        
        logger.info(f"Completed training and evaluation of {len(top_models)} models")
        return top_models, models_metrics
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise

def train_and_evaluate_time_series_models(config=None):
    """Train and evaluate time series models using PyCaret"""
    if not TS_AVAILABLE:
        raise ImportError("PyCaret time_series module is not available")
    
    if config is None:
        config = ml_config["time_series"]
    
    logger.info("Training and evaluating time series models with PyCaret...")
    
    # Get top models using PyCaret
    top_models = compare_models(
        n_select=ml_config["top_n_models"],
        sort='MASE',  # Mean Absolute Scaled Error
        verbose=True
    )
    
    # Handle case where a single model is returned
    if not isinstance(top_models, list):
        top_models = [top_models]
    
    # Get metrics for all models
    models_metrics = []
    for i, model in enumerate(top_models):
        try:
            # Get model name
            if hasattr(model, 'get_name'):
                model_name = model.get_name()
            else:
                model_name = str(model.__class__.__name__)
            
            # Finalize the model (train on entire dataset)
            final_model = finalize_model(model)
            
            # Get performance metrics
            metrics = pull()
            
            models_metrics.append({
                'model_name': f"{model_name}_{i+1}",
                'metrics': metrics.to_dict() if isinstance(metrics, pd.DataFrame) else {},
                'final_model': final_model
            })
        except Exception as e:
            logger.error(f"Error evaluating time series model {i+1}: {str(e)}")
            models_metrics.append({
                'model_name': f"ts_model_{i+1}",
                'metrics': {},
                'error': str(e)
            })
    
    logger.info(f"Completed training and evaluation of {len(top_models)} time series models")
    return top_models, models_metrics

def train_neural_network_time_series(df, target_column, config=None):
    """Train neural network models for time series forecasting"""
    if not KERAS_AVAILABLE:
        logger.warning("TensorFlow/Keras not available. Skipping neural network models.")
        return [], []
    
    if config is None:
        config = ml_config["time_series"]
    
    logger.info(f"Training neural network time series models for {target_column}...")
    
    # Prepare data for neural networks
    target_series = df[target_column]
    
    # Normalize the data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(target_series.values.reshape(-1, 1))
    
    # Create sequences for LSTM/RNN input
    def create_sequences(data, lookback):
        X, y = [], []
        for i in range(len(data) - lookback):
            X.append(data[i:i+lookback, 0])
            y.append(data[i+lookback, 0])
        return np.array(X), np.array(y)
    
    # Create sequences
    lookback = config["lookback"]
    X, y = create_sequences(scaled_data, lookback)
    
    # Reshape for LSTM [samples, time steps, features]
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # Train-test split
    train_size = int(len(X) * (1 - ml_config["test_size"]))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Callbacks
    callbacks = []
    if config["early_stopping"]:
        callbacks.append(EarlyStopping(monitor='val_loss', patience=10))
    
    # Model checkpoint to save best model
    model_path = os.path.join(ml_config["output_dir"], "nn_checkpoints")
    os.makedirs(model_path, exist_ok=True)
    callbacks.append(
        ModelCheckpoint(
            os.path.join(model_path, f"{target_column}_best_model.keras"),
            monitor='val_loss',
            save_best_only=True
        )
    )
    
    # Train different neural network architectures
    nn_models = []
    nn_metrics = []
    
    # 1. LSTM model(s)
    for n_layers in config["lstm_layers"]:
        for units in config["lstm_units"]:
            try:
                model_name = f"LSTM_{n_layers}layers_{units}units"
                logger.info(f"Training {model_name}...")
                
                model = Sequential(name=model_name)
                
                # Add layers
                if n_layers == 1:
                    model.add(LSTM(units=units, input_shape=(lookback, 1)))
                else:
                    model.add(LSTM(units=units, return_sequences=True, input_shape=(lookback, 1)))
                    for i in range(n_layers-2):
                        model.add(LSTM(units=units, return_sequences=True))
                    model.add(LSTM(units=units))
                
                model.add(Dropout(config["dropout_rate"]))
                model.add(Dense(1))
                
                # Compile
                model.compile(optimizer=Adam(), loss='mean_squared_error')
                
                # Train
                history = model.fit(
                    X_train, y_train,
                    epochs=config["epochs"],
                    batch_size=config["batch_size"],
                    validation_split=0.1,
                    callbacks=callbacks,
                    verbose=0
                )
                
                # Evaluate
                loss = model.evaluate(X_test, y_test, verbose=0)
                
                # Save model
                model_save_path = os.path.join(model_path, f"{target_column}_{model_name}.keras")
                model.save(model_save_path)
                
                # Make predictions for visualization
                y_pred = model.predict(X_test)
                
                # Inverse transform
                y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
                y_pred_actual = scaler.inverse_transform(y_pred)
                
                # Calculate metrics
                from sklearn.metrics import mean_squared_error, mean_absolute_error
                mse = mean_squared_error(y_test_actual, y_pred_actual)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test_actual, y_pred_actual)
                
                nn_models.append(model)
                nn_metrics.append({
                    'model_name': model_name,
                    'metrics': {
                        'mse': mse,
                        'rmse': rmse, 
                        'mae': mae,
                        'val_loss': min(history.history['val_loss'])
                    },
                    'model_path': model_save_path,
                    'scaler': scaler,
                    'lookback': lookback,
                    'test_predictions': y_pred_actual,
                    'test_actual': y_test_actual
                })
                
                logger.info(f"Finished training {model_name}. RMSE: {rmse:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                logger.error(traceback.format_exc())
    
    # 2. Simple RNN model
    try:
        model_name = "SimpleRNN"
        logger.info(f"Training {model_name}...")
        
        model = Sequential(name=model_name)
        model.add(SimpleRNN(units=64, input_shape=(lookback, 1)))
        model.add(Dropout(config["dropout_rate"]))
        model.add(Dense(1))
        
        model.compile(optimizer=Adam(), loss='mean_squared_error')
        
        history = model.fit(
            X_train, y_train,
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            validation_split=0.1,
            callbacks=callbacks,
            verbose=0
        )
        
        loss = model.evaluate(X_test, y_test, verbose=0)
        
        model_save_path = os.path.join(model_path, f"{target_column}_{model_name}.keras")
        model.save(model_save_path)
        
        y_pred = model.predict(X_test)
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred_actual = scaler.inverse_transform(y_pred)
        
        mse = mean_squared_error(y_test_actual, y_pred_actual)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_actual, y_pred_actual)
        
        nn_models.append(model)
        nn_metrics.append({
            'model_name': model_name,
            'metrics': {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'val_loss': min(history.history['val_loss'])
            },
            'model_path': model_save_path,
            'scaler': scaler,
            'lookback': lookback,
            'test_predictions': y_pred_actual,
            'test_actual': y_test_actual
        })
        
        logger.info(f"Finished training {model_name}. RMSE: {rmse:.4f}")
        
    except Exception as e:
        logger.error(f"Error training SimpleRNN: {str(e)}")
    
    # 3. GRU model
    try:
        model_name = "GRU"
        logger.info(f"Training {model_name}...")
        
        model = Sequential(name=model_name)
        model.add(GRU(units=64, input_shape=(lookback, 1)))
        model.add(Dropout(config["dropout_rate"]))
        model.add(Dense(1))
        
        model.compile(optimizer=Adam(), loss='mean_squared_error')
        
        history = model.fit(
            X_train, y_train,
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            validation_split=0.1,
            callbacks=callbacks,
            verbose=0
        )
        
        loss = model.evaluate(X_test, y_test, verbose=0)
        
        model_save_path = os.path.join(model_path, f"{target_column}_{model_name}.keras")
        model.save(model_save_path)
        
        y_pred = model.predict(X_test)
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred_actual = scaler.inverse_transform(y_pred)
        
        mse = mean_squared_error(y_test_actual, y_pred_actual)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_actual, y_pred_actual)
        
        nn_models.append(model)
        nn_metrics.append({
            'model_name': model_name,
            'metrics': {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'val_loss': min(history.history['val_loss'])
            },
            'model_path': model_save_path,
            'scaler': scaler,
            'lookback': lookback,
            'test_predictions': y_pred_actual,
            'test_actual': y_test_actual
        })
        
        logger.info(f"Finished training {model_name}. RMSE: {rmse:.4f}")
        
    except Exception as e:
        logger.error(f"Error training GRU: {str(e)}")
    
    # 4. Hybrid LSTM model with advanced features
    try:
        # Train hybrid model
        model_name = "HybridLSTM"
        logger.info(f"Training {model_name} with advanced features...")
        
        # Identify potential exogenous features
        exog_features = [col for col in df.columns if col != target_column and 
                         not col.startswith(f"{target_column}_lag_") and 
                         not col.startswith(f"{target_column}_rolling_")]
        
        # Keep at most 10 exogenous features to avoid overfitting
        if len(exog_features) > 10:
            # Calculate correlation with target for feature selection
            correlations = df[exog_features].corrwith(df[target_column]).abs().sort_values(ascending=False)
            exog_features = list(correlations.head(10).index)
        
        logger.info(f"Using {len(exog_features)} exogenous features for hybrid model")
        
        # Prepare exogenous features
        if exog_features:
            exog_scaler = MinMaxScaler(feature_range=(0, 1))
            exog_data = exog_scaler.fit_transform(df[exog_features].values)
        
            # Create sequences with exogenous data
            X_exog = []
            for i in range(len(scaled_data) - lookback):
                X_exog.append(exog_data[i+lookback])
            X_exog = np.array(X_exog)
            
            # Train-test split for exogenous data
            X_exog_train, X_exog_test = X_exog[:train_size], X_exog[train_size:]
            
            # Build hybrid model with exogenous inputs
            target_input = Input(shape=(lookback, 1), name='target_input')
            lstm_out = LSTM(units=64, return_sequences=False)(target_input)
            lstm_out = Dropout(config["dropout_rate"])(lstm_out)
            
            exog_input = Input(shape=(X_exog.shape[1],), name='exog_input')
            exog_out = Dense(32, activation='relu')(exog_input)
            exog_out = BatchNormalization()(exog_out)
            exog_out = Dropout(config["dropout_rate"])(exog_out)
            
            merged = Concatenate()([lstm_out, exog_out])
            output = Dense(32, activation='relu')(merged)
            output = Dropout(config["dropout_rate"]/2)(output)
            output = Dense(1)(output)
            
            hybrid_model = Model(inputs=[target_input, exog_input], outputs=output)
            
            # Compile
            hybrid_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
            
            # Train
            history = hybrid_model.fit(
                [X_train, X_exog_train], y_train,
                epochs=config["epochs"],
                batch_size=config["batch_size"],
                validation_split=0.1,
                callbacks=callbacks,
                verbose=0
            )
            
            # Evaluate
            loss = hybrid_model.evaluate([X_test, X_exog_test], y_test, verbose=0)
            
            # Save model
            model_save_path = os.path.join(model_path, f"{target_column}_{model_name}.keras")
            hybrid_model.save(model_save_path)
            
            # Make predictions
            y_pred = hybrid_model.predict([X_test, X_exog_test])
            y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
            y_pred_actual = scaler.inverse_transform(y_pred)
            
            # Calculate metrics
            mse = mean_squared_error(y_test_actual, y_pred_actual)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test_actual, y_pred_actual)
            
            nn_models.append(hybrid_model)
            nn_metrics.append({
                'model_name': model_name,
                'metrics': {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'val_loss': min(history.history['val_loss'])
                },
                'model_path': model_save_path,
                'scaler': scaler,
                'exog_scaler': exog_scaler,
                'exog_features': exog_features,
                'lookback': lookback,
                'test_predictions': y_pred_actual,
                'test_actual': y_test_actual
            })
            
            logger.info(f"Finished training {model_name}. RMSE: {rmse:.4f}")
        else:
            logger.warning("No exogenous features available for hybrid model")
        
    except Exception as e:
        logger.error(f"Error training hybrid model: {str(e)}")
        logger.error(traceback.format_exc())
    
    logger.info(f"Completed training of {len(nn_models)} neural network models")
    
    return nn_models, nn_metrics

def save_model_to_onnx(model, model_name, output_dir, data_sample):
    """Convert PyCaret model to ONNX format and save"""
    if not ONNX_AVAILABLE:
        logger.warning("ONNX libraries not available. Skipping ONNX conversion.")
        return None
    
    try:
        logger.info(f"Converting {model_name} to ONNX format...")
        
        # Define path for ONNX model
        onnx_path = os.path.join(output_dir, f"{model_name.replace(' ', '_')}.onnx")
        
        # Get the feature columns (assuming data_sample is a pandas DataFrame)
        feature_columns = data_sample.drop(columns=[col for col in data_sample.columns 
                                                   if col.startswith('target_')], 
                                          errors='ignore').columns.tolist()
        
        # Create initial types for ONNX conversion
        from skl2onnx.common.data_types import FloatTensorType
        initial_types = [(feature_name, FloatTensorType([None, 1])) 
                        for feature_name in feature_columns]
        
        # Try to convert using skl2onnx
        try:
            from skl2onnx import convert_sklearn
            
            # Extract the underlying model if it's a pipeline
            if hasattr(model, 'steps'):
                # For sklearn Pipeline
                sklearn_model = model.steps[-1][1]  
            else:
                sklearn_model = model
                
            # Convert to ONNX
            onnx_model = convert_sklearn(
                sklearn_model,
                initial_types=initial_types,
                options={type(sklearn_model): {'zipmap': False}}
            )
            
            # Save the model
            with open(onnx_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
                
            logger.info(f"Saved ONNX model to {onnx_path}")
            return onnx_path
            
        except Exception as e:
            # If skl2onnx fails, try with onnxmltools
            logger.warning(f"skl2onnx conversion failed: {str(e)}")
            logger.info("Attempting conversion with onnxmltools...")
            
            try:
                from onnxmltools.convert import convert_sklearn
                
                # Extract the underlying model if it's a pipeline
                if hasattr(model, 'steps'):
                    sklearn_model = model.steps[-1][1]
                else:
                    sklearn_model = model
                
                # Convert to ONNX
                onnx_model = convert_sklearn(sklearn_model, initial_types=initial_types)
                
                # Save the model
                with open(onnx_path, "wb") as f:
                    f.write(onnx_model.SerializeToString())
                    
                logger.info(f"Saved ONNX model to {onnx_path}")
                return onnx_path
                
            except Exception as e2:
                logger.error(f"onnxmltools conversion also failed: {str(e2)}")
                return None
    
    except Exception as e:
        logger.error(f"Error converting {model_name} to ONNX: {str(e)}")
        return None

def save_keras_model(model, model_name, output_dir, data_sample=None, scaler=None, lookback=None):
    """
    Save Keras model in multiple formats (keras, onnx, tflite)
    """
    results = {'model_name': model_name}
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save in Keras format
        keras_path = os.path.join(output_dir, f"{model_name}.keras")
        model.save(keras_path)
        results['keras_path'] = keras_path
        logger.info(f"Saved {model_name} in Keras format")
        
        # Save in TFLite format
        try:
            tflite_path = os.path.join(output_dir, f"{model_name}.tflite")
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            results['tflite_path'] = tflite_path
            logger.info(f"Saved {model_name} in TFLite format")
        except Exception as e:
            logger.error(f"Error saving {model_name} to TFLite: {str(e)}")
        
        # Save in ONNX format
        try:
            if TF2ONNX_AVAILABLE:
                import tf2onnx
                
                onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
                
                # Convert to ONNX using tf2onnx
                onnx_model, _ = tf2onnx.convert.from_keras(model)
                
                # Save the model
                with open(onnx_path, "wb") as f:
                    f.write(onnx_model.SerializeToString())
                
                results['onnx_path'] = onnx_path
                logger.info(f"Saved {model_name} in ONNX format")
        except Exception as e:
            logger.error(f"Error saving {model_name} to ONNX: {str(e)}")
        
        # Save metadata for preprocessing
        metadata = {}
        if scaler is not None:
            # Save scaler parameters
            metadata['scaler'] = {
                'min': float(scaler.data_min_[0]),
                'max': float(scaler.data_max_[0]),
            }
        if lookback is not None:
            metadata['lookback'] = lookback
        
        if data_sample is not None:
            metadata['data_sample'] = data_sample.tolist() if hasattr(data_sample, 'tolist') else data_sample
        
        metadata_path = os.path.join(output_dir, f"{model_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        results['metadata_path'] = metadata_path
        logger.info(f"Saved metadata for {model_name}")
    
    except Exception as e:
        logger.error(f"Error saving {model_name}: {str(e)}")
        logger.error(traceback.format_exc())
    
    return results

def save_models_and_results(models, metrics, artifacts_info, output_dir, target_column, use_case, df, save_formats=None, quant_data_path=None):
    """Save trained models, metrics, and converted formats"""
    if save_formats is None:
        save_formats = ["pkl", "onnx", "tflite", "keras"]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(output_dir, f"{target_column}_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    logger.info(f"Saving models and results to {model_dir}")
    
    # Save metrics to JSON
    with open(os.path.join(model_dir, "metrics.json"), 'w') as f:
        # Convert DataFrame and numpy objects to serializable format
        serializable_metrics = []
        for metric in metrics:
            serializable_metric = {
                'model_name': metric['model_name'],
                'metrics': {
                    k: {
                        kk: float(vv) if isinstance(vv, np.number) else str(vv) 
                        for kk, vv in v.items()
                    } if isinstance(v, dict) else str(v)
                    for k, v in metric['metrics'].items()
                } if metric.get('metrics') else {}
            }
            if 'error' in metric:
                serializable_metric['error'] = metric['error']
            serializable_metrics.append(serializable_metric)
            
        json.dump(serializable_metrics, f, indent=2)
    
    # Get module based on use case
    if use_case == "regression":
        module = regression
    elif use_case == "classification":
        module = classification
    elif use_case == "clustering":
        module = clustering
    elif use_case == "anomaly":
        module = anomaly
    elif use_case == "time_series" and TS_AVAILABLE:
        module = None  # Handle time series models separately
    else:
        logger.warning(f"Unsupported use case: {use_case}")
        module = None
    
    # Sample data for model conversion
    data_sample = df.head(100)
    
    # Save each model
    model_paths = []
    for i, model in enumerate(models):
        model_name = metrics[i]['model_name'].replace(" ", "_")
        paths = {'model_name': model_name}
        
        try:
            # For Keras models from neural network time series
            if hasattr(model, 'save') and callable(getattr(model, 'save')):
                # This looks like a Keras model
                if "keras" in save_formats or "tflite" in save_formats or "onnx" in save_formats:
                    keras_results = save_keras_model(
                        model, 
                        model_name, 
                        model_dir, 
                        data_sample,
                        metrics[i].get('scaler', None),
                        metrics[i].get('lookback', None)
                    )
                    paths.update(keras_results)
            # For PyCaret models
            elif module is not None:
                # Save as pickle if requested
                if "pkl" in save_formats:
                    pickle_path = os.path.join(model_dir, f"{model_name}.pkl")
                    module.save_model(model, pickle_path)
                    logger.info(f"Saved {model_name} as pickle to {pickle_path}")
                    paths['pickle_path'] = pickle_path
                
                # Save as ONNX if requested
                if "onnx" in save_formats:
                    onnx_path = save_model_to_onnx(model, model_name, model_dir, data_sample)
                    if onnx_path:
                        paths['onnx_path'] = onnx_path
            
            model_paths.append(paths)
            
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {str(e)}")
            logger.error(traceback.format_exc())
            paths['error'] = str(e)
            model_paths.append(paths)
    
    # Save configuration and paths
    config_info = {
        "timestamp": timestamp,
        "target_column": target_column,
        "use_case": use_case,
        "models": model_paths,
        "feature_artifacts": artifacts_info,
        "quantization_data_path": quant_data_path
    }
    
    with open(os.path.join(model_dir, "model_info.json"), 'w') as f:
        json.dump(config_info, f, indent=2)
    
    # Update main ML registry
    ml_registry_path = os.path.join(output_dir, "ml_registry.json")
    
    if os.path.exists(ml_registry_path):
        with open(ml_registry_path, 'r') as f:
            ml_registry = json.load(f)
    else:
        ml_registry = {"models": {}}
    
    # Add this run to the registry
    if "models" not in ml_registry:
        ml_registry["models"] = {}
        
    if target_column not in ml_registry["models"]:
        ml_registry["models"][target_column] = {}
    
    ml_registry["models"][target_column][timestamp] = {
        "path": model_dir,
        "metrics": serializable_metrics,
        "feature_artifacts": artifacts_info
    }
    
    # Update the latest timestamp
    ml_registry["latest"] = timestamp
    ml_registry["latest_" + target_column] = timestamp
    
    with open(ml_registry_path, 'w') as f:
        json.dump(ml_registry, f, indent=2)
    
    logger.info(f"Updated ML registry at {ml_registry_path}")
    
    return model_dir, ml_registry

def visualize_time_series_forecasts(df, target_column, model, model_info, output_path):
    """Create and save visualizations of time series forecasts."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.dates import DateFormatter
        import seaborn as sns
        
        # Create the visualization directory
        os.makedirs(output_path, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("muted")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot actual values
        if isinstance(df.index, pd.DatetimeIndex):
            ax.plot(df.index, df[target_column], label='Actual', color='blue', alpha=0.7)
        else:
            ax.plot(df[target_column], label='Actual', color='blue', alpha=0.7)
            
        # For neural network models, plot test predictions if available
        if 'test_predictions' in model_info and 'test_actual' in model_info:
            test_preds = model_info['test_predictions']
            test_actual = model_info['test_actual']
            
            # Create time index for predictions that matches the end of the dataframe
            test_size = len(test_preds)
            if isinstance(df.index, pd.DatetimeIndex):
                pred_index = df.index[-test_size:]
                ax.plot(pred_index, test_preds, label='Predictions', color='red', linestyle='--')
            else:
                # Use numerical index
                offset = len(df) - test_size
                ax.plot(range(offset, len(df)), test_preds, label='Predictions', color='red', linestyle='--')
        
        # Customize plot
        model_name = model_info.get('model_name', 'Unknown Model')
        ax.set_title(f'Time Series Forecast for {target_column} using {model_name}', fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(target_column, fontsize=12)
        ax.legend(loc='best')
        
        # Format x-axis dates
        if isinstance(df.index, pd.DatetimeIndex):
            date_format = DateFormatter('%Y-%m-%d')
            ax.xaxis.set_major_formatter(date_format)
            plt.xticks(rotation=45)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure
        model_name = model_info.get('model_name', 'unknown_model').replace(' ', '_')
        fig_path = os.path.join(output_path, f"{target_column}_{model_name}_forecast.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved forecast visualization to {fig_path}")
        
        # Close figure to avoid memory issues
        plt.close(fig)
        
        return fig_path
    
    except ImportError:
        logger.warning("Matplotlib or seaborn not available. Skipping visualization.")
        return None
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def main():
    """Main function to execute the ML training and benchmarking pipeline"""
    try:
        logger.info("\n=== Starting ML Training & Benchmarking Pipeline ===\n")
        
        # Create output directory
        os.makedirs(ml_config["output_dir"], exist_ok=True)
        
        # Load metadata
        metadata = load_feature_metadata()
        
        # Get path to latest feature data
        data_path, artifacts_info = get_latest_artifact_path(metadata)
        
        # Load feature data
        df = load_feature_data(data_path)
        
        # Check which target columns are available
        available_targets = check_target_availability(df, ml_config["target_columns"])
        
        # Create a directory to store quantization test data
        quant_data_paths = {}

        # Process each target variable
        all_results = {}
        for target_column in available_targets:
            logger.info(f"\n--- Processing target: {target_column} ---\n")
            
            try:
                # Save quantization test data early in the pipeline
                # This happens before any preprocessing to ensure raw data is available for quantization
                quant_data_path = save_quantization_test_data(
                    df.copy(), 
                    target_column, 
                    ml_config["output_dir"]
                )
                
                if quant_data_path:
                    quant_data_paths[target_column] = quant_data_path
                
                # Handle based on use case
                if ml_config["use_case"] == "time_series":
                    logger.info("Using time series forecasting approach")
                    
                    # Prepare time series data
                    ts_df = prepare_time_series_data(df.copy(), target_column, ml_config["time_series"])
                    
                    # Create advanced time series features
                    ts_df_enhanced = create_time_series_features(ts_df.copy(), target_column, ml_config["time_series"])
                    
                    # Process with either PyCaret time series or neural networks
                    ts_models = []
                    ts_metrics = []
                    
                    # Train PyCaret time series models if enabled
                    if ml_config["use_pycaret"] and TS_AVAILABLE:
                        try:
                            logger.info("Using PyCaret for time series modeling")
                            setup_pycaret(ts_df, target_column, use_case="time_series", config=ml_config)
                            ts_models, ts_metrics = train_and_evaluate_models(
                                "time_series", 
                                top_n=ml_config["top_n_models"],
                                excluded_models=ml_config["excluded_models"]
                            )
                            logger.info(f"Trained {len(ts_models)} PyCaret time series models")
                        except Exception as e:
                            logger.error(f"Error training PyCaret time series models: {str(e)}")
                            logger.error(traceback.format_exc())
                    else:
                        if not ml_config["use_pycaret"]:
                            logger.info("PyCaret is disabled in configuration. Skipping PyCaret models.")
                        else:
                            logger.info("PyCaret time series module not available. Skipping PyCaret models.")
                    
                    # No models trained from PyCaret
                    if len(ts_models) == 0:
                        logger.info(f"No PyCaret models trained for {target_column}. Using neural network models only.")
                        
                        # Train neural network models with enhanced features
                        if ml_config["time_series"]["nn_models"] and KERAS_AVAILABLE:
                            try:
                                nn_models, nn_metrics = train_neural_network_time_series(
                                    ts_df_enhanced, target_column, ml_config["time_series"]
                                )
                                
                                # Create visualizations for neural network models
                                viz_dir = os.path.join(ml_config["output_dir"], "visualizations")
                                for i, model in enumerate(nn_models):
                                    viz_path = visualize_time_series_forecasts(
                                        ts_df_enhanced, target_column, model, nn_metrics[i], viz_dir
                                    )
                                    if viz_path:
                                        nn_metrics[i]['visualization'] = viz_path
                                
                                logger.info(f"Trained {len(nn_models)} neural network time series models")
                                
                                # Use neural network models as the main models
                                ts_models = nn_models
                                ts_metrics = nn_metrics
                                
                            except Exception as e:
                                logger.error(f"Error training neural network time series models: {str(e)}")
                                logger.error(traceback.format_exc())
                        
                        if len(ts_models) == 0:
                            logger.error(f"No models could be trained for {target_column}.")
                            raise ValueError(f"Failed to train any models for {target_column}")
                    
                    # Sort all models by metric (RMSE for regression/time series)
                    model_dir, ml_registry = save_models_and_results(
                        ts_models,
                        ts_metrics,
                        artifacts_info,
                        ml_config["output_dir"],
                        target_column,
                        "time_series",
                        ts_df_enhanced,
                        save_formats=ml_config["save_formats"],
                        quant_data_path=quant_data_paths.get(target_column)  # Pass the quantization data path
                    )
                    
                    # Store results
                    all_results[target_column] = {
                        "status": "success",
                        "model_dir": model_dir,
                        "top_model": ts_metrics[0]["model_name"] if ts_metrics else None,
                    }
                
                else:
                    # Standard ML approach (regression, classification, etc.)
                    # Prepare data for target
                    target_df = prepare_ml_data(df.copy(), target_column)
                    
                    # Use PyCaret if enabled
                    if ml_config["use_pycaret"]:
                        logger.info("Using PyCaret for standard ML approach")
                        # Set up PyCaret
                        setup_pycaret(
                            target_df, 
                            target_column, 
                            use_case=ml_config["use_case"],
                            config=ml_config
                        )
                        
                        # Train and evaluate models
                        top_models, model_metrics = train_and_evaluate_models(
                            ml_config["use_case"],
                            top_n=ml_config["top_n_models"],
                            excluded_models=ml_config["excluded_models"]
                        )
                    else:
                        logger.info("PyCaret is disabled. Using only neural network models.")
                        # Create empty lists since we're skipping PyCaret
                        top_models = []
                        model_metrics = []
                    
                    # Save models and results
                    model_dir, ml_registry = save_models_and_results(
                        top_models,
                        model_metrics,
                        artifacts_info,
                        ml_config["output_dir"],
                        target_column,
                        ml_config["use_case"],
                        target_df,
                        save_formats=ml_config["save_formats"],
                        quant_data_path=quant_data_paths.get(target_column)  # Pass the quantization data path
                    )
                    
                    # Store results
                    all_results[target_column] = {
                        "status": "success",
                        "model_dir": model_dir,
                        "top_model": model_metrics[0]["model_name"] if model_metrics else None,
                    }
                
            except Exception as e:
                logger.error(f"Error processing target {target_column}: {str(e)}")
                logger.error(traceback.format_exc())
                
                all_results[target_column] = {
                    "status": "error",
                    "error": str(e)
                }
        
        # Print summary
        logger.info("\n=== ML Pipeline Summary ===")
        for target, result in all_results.items():
            status = result["status"]
            if status == "success":
                logger.info(f"✓ {target}: Success - Top model: {result['top_model']}")
            else:
                logger.info(f"✗ {target}: Failed - {result.get('error', 'Unknown error')}")
        
        return all_results
    
    except Exception as e:
        logger.error(f"Critical error in ML pipeline: {str(e)}")
        logger.error(traceback.format_exc())
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5002, debug=True)
    # main()