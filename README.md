# Cloud2Stm : ML Training and Benchmarking Pipeline on Databricks

## Description
This project provides a robust pipeline for machine learning model training, benchmarking, and deployment, with a focus on integration with Databricks. It supports various ML use cases, including multi-target modeling and time series forecasting, and facilitates model conversion and quantization for optimized deployment. The system is designed for compatibility with both Windows and Windows Subsystem for Linux (WSL) environments.

## Key Features
- **Multi-target Modeling**: Train models for multiple target variables efficiently in a single run.
- **Time Series Forecasting**: Specialized functionalities for advanced time series prediction tasks.
- **Comprehensive Model Benchmarking**: Leverages PyCaret (optional) for automatic comparison and selection of top-performing models.
- **Multi-format Model Export**: Save trained models in various formats including pickle, ONNX, TFLite, and Keras.
- **Quantization Support**: Automatically extracts test data to facilitate model quantization for optimized inference.
- **Performance Visualization**: Generates insightful charts and visualizations of model performance.
- **Feature Engineering Integration**: Seamlessly integrates with a feature engineering pipeline to use engineered features for modeling.
- **GPU Acceleration**: Supports GPU usage (configurable) for faster model training, leveraging technologies like CUDA and cuDNN.
- **Cross-Platform Compatibility**: Full support for Windows and Windows Subsystem for Linux (WSL).
- **Databricks Integration**: Designed for interaction with Databricks environments (as implied by `databricks_connector_linux.py`).

## Use Cases

- **Multi-target modeling**: Train models for multiple target variables in one run
- **Time series forecasting**: Advanced features for time series prediction
- **Model benchmarking**: Automatic comparison and selection of top-performing models
- **Model format conversion**: Save models in multiple formats (pickle, ONNX, TFLite, Keras)
- **Quantization support**: Automatic test data extraction for model quantization- **WSL compatibility**: Full support for Windows Subsystem for Linux
- **Result visualization**: Generate performance visualizations for trained models
- **Feature engineering integration**: Seamless integration with feature engineering pipeline## Directory Structure
- **WSL compatibility**: Full support for Windows Subsystem for Linux

## Technical Stack Highlights
This project utilizes a range of powerful libraries and technologies, including:
- **Python 3.9+**
- **PyCaret**: For automated machine learning and model benchmarking.
- **TensorFlow/Keras**: For building and training deep learning models.
- **ONNX (Open Neural Network Exchange)**: For model interoperability and deployment.
- **Pandas & NumPy**: For data manipulation and numerical operations.│   ├── feature_artifacts/            # Feature engineering outputs
- **Matplotlib & Seaborn**: For data visualization.eatures_data_*  # Feature datasets
- **Protobuf**: For efficient data serialization (used internally by TensorFlow).│   │   └── ...
- **CUDA/cuDNN**: For GPU acceleration in TensorFlow. ├── ml_models/                    # ML model outputs
ompE21EnergyIn_*/     # Models for specific target variables

## Directory Structure/        
```
databricks_apis/v/                         # Python virtual environment for WSL
├── databricks_api_endpoints/
│   ├── ml_training_benchmarking.py   # Main ML pipeline
│   ├── feature_engineering_linux.py  # Feature engineering pipeline for Linux
│   └── ...
├── results/
│   ├── feature_artifacts/            # Feature engineering outputs
│   │   ├── selected_features_data_*  # Feature datasets
│   │   └── ...
│   ├── ml_models/                    # ML model outputs
│   │   ├── HP_CompE21EnergyIn_*/     # Models for specific target variables
│   │   ├── quantization_data/        # Data for model quantization```
│   │   └── visualizations/           # Performance charts and visualizations
│   └── feature_metadata.json         # Feature engineering metadatanvironment and install dependencies:
└── wsl_venv/                         # Python virtual environment for WSL   ```bash
``` venv
   venv\Scripts\activate
## Setup Instructionsements.txt

### Windows Setup

1. Clone the repository:bash
   ```bash   python databricks_api_endpoints/ml_training_benchmarking.py
   git clone https://github.com/yourusername/databricks_apis.git
   cd databricks_apis
   ```stem for Linux) Setup

2. Create a virtual environment and install dependencies:alled:
   ```bashbash
   python -m venv venv   # On Windows PowerShell as Administrator
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
nch WSL and navigate to your project directory:
3. Run the pipeline:   ```bash
   ```bash
   python databricks_api_endpoints/ml_training_benchmarking.py   cd /mnt/d/College/databricks_apis
   ```

### WSL (Windows Subsystem for Linux) SetupSL:

1. Install WSL if not already installed:reate virtual environment
   ```bash   python3 -m venv wsl_venv
   # On Windows PowerShell as Administrator
   wsl --install
   ```

2. Launch WSL and navigate to your project directory:
   ```bash   # Install additional packages for full functionality
   # If your project is on D: drivennxruntime onnxmltools skl2onnx tf2onnx matplotlib seaborn
   cd /mnt/d/College/databricks_apis
   ```
line in WSL:
3. Create a Python virtual environment in WSL:
   ```bashpython databricks_api_endpoints/ml_training_benchmarking.py
   # Create virtual environment
   python3 -m venv wsl_venv
   source wsl_venv/bin/activateConfiguration
   
   # Install dependenciess include:
   pip install -r requirements.txt
   - `use_case`: Type of ML problem (regression, classification, time_series, etc.)
   # Install additional packages for full functionalitytion (set to False to save time during development)
   pip install pycaret[full] tensorflow onnx onnxruntime onnxmltools skl2onnx tf2onnx matplotlib seaborncolumns`: List of target variables to model
   ```ion
e_series`: Configuration for time series models
4. Run the ML training pipeline in WSL:
   ```bashation adjustment:
   python databricks_api_endpoints/ml_training_benchmarking.py```python
   ```
ml_config["gpu_enabled"] = True  # Enable GPU acceleration if available
## Configuration

The pipeline is configured through the `ml_config` dictionary in ml_training_benchmarking.py. Key options include:

- `use_case`: Type of ML problem (regression, classification, time_series, etc.)s of key functionality:
- `use_pycaret`: Toggle PyCaret integration (set to False to save time during development)
- `target_columns`: List of target variables to model Feature engineering pipeline optimized for Linux/WSL
- `save_quantization_data`: Save test data for model quantizationicks_connector_linux.py`: Databricks API connector for Linux environments
- `time_series`: Configuration for time series models
tions and parallel processing optimizations for Linux environments.
Example configuration adjustment:
```python## Results
ml_config["use_pycaret"] = True  # Enable PyCaret for comprehensive model comparison
ml_config["gpu_enabled"] = True  # Enable GPU acceleration if availableAfter running the pipeline, results are organized in the results directory:
```
- Model files in various formats (pickle, ONNX, TFLite, Keras)

## Linux-specific Files

The repository includes Linux-specific implementations of key functionality:- Quantization test data for deployment optimization

- `feature_engineering_linux.py`: Feature engineering pipeline optimized for Linux/WSL## Dependencies
- `databricks_connector_linux.py`: Databricks API connector for Linux environments
- Python 3.9+
These files handle platform-specific operations like filesystem interactions and parallel processing optimizations for Linux environments.
- TensorFlow/Keras

## Results

After running the pipeline, results are organized in the results directory:

- Model files in various formats (pickle, ONNX, TFLite, Keras)## License
- Performance metrics in JSON format
- Visualizations of model performanceThis project is licensed under the MIT License - see the LICENSE file for details.- Quantization test data for deployment optimization## Dependencies- Python 3.9+- PyCaret- TensorFlow/Keras- ONNX Runtime- Pandas/NumPy- Matplotlib/Seaborn

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## run command

pm2 start ecosystem.config.js

or 

autogenstudio ui

for local demos today