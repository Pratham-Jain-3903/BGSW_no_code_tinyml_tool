# 🚀 Cloud2Stm : ML Training and Benchmarking Pipeline on Databricks

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Platform-Windows%20%7C%20WSL-0078D6?logo=windows" alt="Platform">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/TensorFlow-2.0+-orange?logo=tensorflow" alt="TensorFlow">
  <img src="https://img.shields.io/badge/PyCaret-3.0+-blue" alt="PyCaret">
</div>

## 📚 Resources
1. [Proposed Wireframes for Cloud Deployment](https://claude.ai/public/artifacts/71389c69-a5c3-4668-b89b-fb91c72363e7)
2. [Video Demo](https://drive.google.com/drive/folders/1cip_Qbi3IvU_WVg14flQM8Pn28oHKXIC?usp=sharing)
3. [Project Presentation](https://www.canva.com/design/DAGmvIGKaPs/yIqU9jLeg0LYEhrwnd7ETg/edit?utm_content=DAGmvIGKaPs&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

## 🧠 Description
This project provides a robust pipeline for machine learning model training, benchmarking, and deployment, with a focus on integration with Databricks. It supports various ML use cases, including multi-target modeling and time series forecasting, and facilitates model conversion and quantization for optimized deployment. The system is designed for compatibility with both Windows and Windows Subsystem for Linux (WSL) environments.

## ⭐ Key Features
- **🧩 Multi-target Modeling**: Train models for multiple target variables efficiently in a single run.
- **📈 Time Series Forecasting**: Specialized functionalities for advanced time series prediction tasks.
- **🏆 Comprehensive Model Benchmarking**: Leverages PyCaret (optional) for automatic comparison and selection of top-performing models.
- **📤 Multi-format Model Export**: Save trained models in various formats including pickle, ONNX, TFLite, and Keras.
- **⚡ Quantization Support**: Automatically extracts test data to facilitate model quantization for optimized inference.
- **📊 Performance Visualization**: Generates insightful charts and visualizations of model performance.
- **⚙️ Feature Engineering Integration**: Seamlessly integrates with a feature engineering pipeline to use engineered features for modeling.
- **🎮 GPU Acceleration**: Supports GPU usage (configurable) for faster model training, leveraging technologies like CUDA and cuDNN.
- **🖥️ Cross-Platform Compatibility**: Full support for Windows and Windows Subsystem for Linux (WSL).
- **☁️ Databricks Integration**: Designed for interaction with Databricks environments (as implied by `databricks_connector_linux.py`).

## 💼 Use Cases
- **Multi-target modeling**: Train models for multiple target variables in one run
- **Time series forecasting**: Advanced features for time series prediction
- **Model benchmarking**: Automatic comparison and selection of top-performing models
- **Model format conversion**: Save models in multiple formats (pickle, ONNX, TFLite, Keras)
- **Quantization support**: Automatic test data extraction for model quantization
- **WSL compatibility**: Full support for Windows Subsystem for Linux
- **Result visualization**: Generate performance visualizations for trained models
- **Feature engineering integration**: Seamless integration with feature engineering pipeline

## 🧱 Technical Stack Highlights
This project utilizes a range of powerful libraries and technologies, including:
- <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python" height="20"> **Python 3.9+**
- <img src="https://img.shields.io/badge/PyCaret-3.0+-blue" height="20"> **PyCaret**: For automated machine learning and model benchmarking.
- <img src="https://img.shields.io/badge/TensorFlow-2.0+-orange?logo=tensorflow" height="20"> **TensorFlow/Keras**: For building and training deep learning models.
- <img src="https://img.shields.io/badge/ONNX-1.0+-green" height="20"> **ONNX (Open Neural Network Exchange)**: For model interoperability and deployment.
- <img src="https://img.shields.io/badge/Pandas-1.0+-blue?logo=pandas" height="20"> **Pandas & NumPy**: For data manipulation and numerical operations.
- <img src="https://img.shields.io/badge/Matplotlib-3.0+-blue?logo=matplotlib" height="20"> **Matplotlib & Seaborn**: For data visualization.
- <img src="https://img.shields.io/badge/Protobuf-3.0+-green" height="20"> **Protobuf**: For efficient data serialization (used internally by TensorFlow).
- <img src="https://img.shields.io/badge/CUDA-11.0+-green?logo=nvidia" height="20"> **CUDA/cuDNN**: For GPU acceleration in TensorFlow.

## 📂 Directory Structure
```
databricks_apis/                         
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
│   │   ├── quantization_data/        # Data for model quantization
│   │   └── visualizations/           # Performance charts and visualizations
│   └── feature_metadata.json         # Feature engineering metadata
└── wsl_venv/                         # Python virtual environment for WSL
```

## ⚙️ Setup Instructions

### 🪟 Windows Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/databricks_apis.git
   cd databricks_apis
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Run the pipeline:
   ```bash
   python databricks_api_endpoints/ml_training_benchmarking.py
   ```

### 🐧 WSL (Windows Subsystem for Linux) Setup
1. Install WSL if not already installed:
   ```powershell
   # On Windows PowerShell as Administrator
   wsl --install
   ```

2. Launch WSL and navigate to your project directory:
   ```bash
   # If your project is on D: drive
   cd /mnt/d/College/databricks_apis
   ```

3. Create a Python virtual environment in WSL:
   ```bash
   python3 -m venv wsl_venv
   source wsl_venv/bin/activate
   pip install -r requirements.txt
   # Install additional packages for full functionality
   pip install pycaret[full] tensorflow onnx onnxruntime onnxmltools skl2onnx tf2onnx matplotlib seaborn
   ```

4. Run the ML training pipeline in WSL:
   ```bash
   python databricks_api_endpoints/ml_training_benchmarking.py
   ```

## ⚙️ Configuration
The pipeline is configured through the `ml_config` dictionary in `ml_training_benchmarking.py`. Key options include:
- `use_case`: Type of ML problem (regression, classification, time_series, etc.)
- `use_pycaret`: Toggle PyCaret integration (set to False to save time during development)
- `target_columns`: List of target variables to model
- `save_quantization_data`: Save test data for model quantization
- `time_series`: Configuration for time series models

Example configuration adjustment:
```python
ml_config["use_pycaret"] = True  # Enable PyCaret for comprehensive model comparison
ml_config["gpu_enabled"] = True  # Enable GPU acceleration if available
```

## 🐧 Linux-specific Files
The repository includes Linux-specific implementations of key functionality:
- `feature_engineering_linux.py`: Feature engineering pipeline optimized for Linux/WSL
- `databricks_connector_linux.py`: Databricks API connector for Linux environments

These files handle platform-specific operations like filesystem interactions and parallel processing optimizations for Linux environments.

## 📊 Results
After running the pipeline, results are organized in the results directory:
- Model files in various formats (pickle, ONNX, TFLite, Keras)
- Performance metrics in JSON format
- Visualizations of model performance
- Quantization test data for deployment optimization

## 📦 Dependencies
- Python 3.9+
- TensorFlow/Keras
- PyCaret
- ONNX Runtime
- Pandas/NumPy
- Matplotlib/Seaborn

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

## ▶️ Run Command
```bash
pm2 start ecosystem.config.js
```
or 
```bash
autogenstudio ui
```
(for local demos today)
```
