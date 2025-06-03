from flask import Flask, request, jsonify
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnan, mean as spark_mean, lit, expr, monotonically_increasing_id
from pyspark.sql.types import IntegerType, DoubleType, StringType, FloatType, BooleanType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, MinMaxScaler
from pyspark.ml.feature import Imputer, PCA
from pyspark.ml.regression import LinearRegression
from pyspark.ml.stat import Correlation
from pyspark.ml import Pipeline, PipelineModel
import numpy as np
import pandas as pd
import os
import json
from functools import reduce
from typing import List, Dict, Any, Optional, Union
import ast
import math
import pyarrow.parquet as pq
from glob import glob
import pickle
from datetime import datetime

# Set JAVA_HOME environment variable
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"  # Use Linux Java path

# Explicitly set HADOOP_HOME in the script environment as well
# os.environ["HADOOP_HOME"] = r"C:\hadoop\winutils\hadoop-3.3.6"

# Add this before creating the SparkSession
temp_dir = "/tmp/spark_temp"
os.makedirs(temp_dir, exist_ok=True)

# Initialize Flask app for API endpoints
app = Flask(__name__)

import sys
import os
# Add PYTHONPATH for Spark
os.environ["PYTHONPATH"] = os.path.join(os.environ.get("SPARK_HOME", ""), "python") + ":" + os.environ.get("PYTHONPATH", "")

# Use current Python executable
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# Alternative approach using individual config calls

builder = SparkSession.builder \
    .appName("FeatureEngineeringAPI") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.memory.offHeap.enabled", "true") \
    .config("spark.memory.offHeap.size", "2g") \
    .config("spark.driver.maxResultSize", "2g") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.1.0") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.python.worker.memory", "1g") \
    .config("spark.executor.cores", "2") \
    .config("spark.task.maxFailures", "10") \
    .config("spark.python.worker.reuse", "true") \
    .config("spark.localdir", "/tmp/spark_temp") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
    .config("spark.sql.execution.arrow.maxRecordsPerBatch", "10000") \
    .config("spark.executor.heartbeatInterval", "30s") \
    .config("spark.network.timeout", "800s") \
    .master("local[2]")

# Add Windows-specific configurations
if False:
    print("Windows environment detected: applying optimized settings")
    builder = builder.config("spark.default.parallelism", "2")
    builder = builder.config("spark.sql.shuffle.partitions", "2")

# Now create the session
spark = builder.getOrCreate()

# Set log level
spark.sparkContext.setLogLevel("ERROR")

# Default configuration - UPDATED to use ingestion outputs
default_config = {
    # Target the pandas parquet output from ingest_data.py
    "input_path": "results/ingestion_table_parquet_pandas",
    # Data quality report from ingest_data.py
    "data_quality_report_path": "results/ingestion_table_data_quality_report.json",
    "output_path": "results/feature_engineered_data",
    # Change from single target to multiple targets
    "target_columns": [
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
    "primary_target": "HP_CompE21EnergyIn",  # The main target for reporting
    "imputation_method": "ffill",  # Options: "mean", "zero", "ffill"
    "categorical_columns": [],  # List of columns to one-hot encode
    "numerical_columns": [],  # List of numerical columns to scale
    "auto_detect_categorical": True,  # Automatically detect categorical columns
    "categorical_threshold": 10,  # Maximum unique values for automatic categorical detection
    "scaling_method": "standard",  # Options: "standard", "minmax"
    "selection_methods": ["correlation", "lasso"],  # Feature selection methods to use
    "correlation_method": "pearson",  # Options: "pearson", "spearman", "kendall"
    "correlation_threshold": 0.1,  # Minimum correlation to keep a feature
    "lasso_alpha": 0.01,  # L1 regularization strength
    "use_pca": False,  # Whether to apply PCA
    "pca_components": 0.95,  # Number of PCA components or variance to retain (0-1)
    "use_rfe": False,  # Whether to use Recursive Feature Elimination (expensive for large data)
    "rfe_n_features": 10,  # Number of features to select using RFE
    "selection_ratio": {  # Ratio of features to keep from each method (0-1)
        "correlation": 1.0,
        "lasso": 0.0,
        "rfe": 0.0,
        "pca": 1.0
    },
    "batch_size": 10000,  # Number of rows to process at once for incremental methods
    "reuse_data_quality_insights": True  # Flag to reuse data quality report insights
}

# Global configuration that will be modified via API
config = default_config.copy()

def load_data_quality_report() -> Dict[str, Any]:
    """Load the data quality report from ingest_data.py if available."""
    try:
        report_path = config["data_quality_report_path"]
        if not os.path.isabs(report_path):
            # Convert to absolute path if not already
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            report_path = os.path.join(base_dir, report_path)
        
        print(f"Loading data quality report from: {report_path}")
        
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                report = json.load(f)
            print(f"Successfully loaded data quality report with {len(report.get('dtypes', {}))} columns")
            return report
        else:
            print(f"Data quality report not found at {report_path}")
            return {}
    except Exception as e:
        print(f"Error loading data quality report: {str(e)}")
        return {}

def load_data(input_path: str) -> Any:
    """Load data from parquet, delta or csv files with pandas fallback."""
    print(f"Loading data from: {input_path}")
    
    # Convert to absolute path if not already
    if not os.path.isabs(input_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        input_path = os.path.join(base_dir, input_path)
    
    try:
        # Check if this is the pandas parquet output from ingest_data.py
        if "parquet_pandas" in input_path and os.path.exists(input_path):
            # This is likely the pandas parquet output, which is optimized for direct loading
            pandas_file = os.path.join(input_path, "data.parquet")
            if os.path.exists(pandas_file):
                print(f"Found pandas parquet file: {pandas_file}")
                try:
                    # Load with pandas for better performance
                    pandas_df = pd.read_parquet(pandas_file)
                    # Convert to Spark DataFrame
                    return spark.createDataFrame(pandas_df)
                except Exception as pandas_error:
                    print(f"Error loading pandas parquet file: {str(pandas_error)}")
                    # Fall through to general loading approach
        
        # If not the specific pandas output or pandas loading failed, try the standard approach
        if os.path.exists(input_path):
            try:
                if os.path.isdir(input_path):
                    if "parquet" in input_path:
                        return spark.read.parquet(input_path)
                    elif "delta" in input_path:
                        return spark.read.format("delta").load(input_path)
                    elif "csv" in input_path:
                        return spark.read.format("csv").option("header", "true").load(input_path)
                    else:
                        # Try to infer format from directory content
                        files = os.listdir(input_path)
                        if any(f.endswith(".parquet") for f in files):
                            return spark.read.parquet(input_path)
                        elif any(f == "_delta_log" for f in files):
                            return spark.read.format("delta").load(input_path)
                        else:
                            return spark.read.format("csv").option("header", "true").load(input_path)
                else:
                    # Single file
                    if input_path.endswith(".parquet"):
                        return spark.read.parquet(input_path)
                    elif input_path.endswith(".csv"):
                        return spark.read.format("csv").option("header", "true").load(input_path)
                    else:
                        raise ValueError(f"Unsupported file format for {input_path}")
            except Exception as spark_error:
                print(f"Spark reader failed with error: {str(spark_error)}")
                print("Falling back to pandas for data loading...")
                
                # FALLBACK: Use pandas + pyarrow instead of Spark
                if os.path.isdir(input_path):
                    if "parquet" in input_path or any(f.endswith(".parquet") for f in os.listdir(input_path)):
                        # Read Parquet files with pyarrow
                        parquet_files = glob(os.path.join(input_path, "*.parquet"))
                        if not parquet_files:
                            # Look in part files
                            parquet_files = glob(os.path.join(input_path, "part-*.parquet"))
                        
                        if not parquet_files:
                            raise FileNotFoundError(f"No parquet files found in {input_path}")
                        
                        # Read and concatenate all parquet files
                        pandas_dfs = []
                        for file in parquet_files[:10]:  # Limit to first 10 files to avoid memory issues
                            try:
                                pandas_dfs.append(pd.read_parquet(file))
                            except Exception as e:
                                print(f"Warning: Could not read {file}: {str(e)}")
                        
                        if not pandas_dfs:
                            raise ValueError("Could not read any parquet files")
                        
                        # Combine all dataframes
                        combined_df = pd.concat(pandas_dfs, ignore_index=True)
                        
                        # Convert pandas DataFrame to Spark DataFrame
                        return spark.createDataFrame(combined_df)
                    
                    elif "csv" in input_path or any(f.endswith(".csv") for f in os.listdir(input_path)):
                        # Read CSV files with pandas
                        csv_files = glob(os.path.join(input_path, "*.csv"))
                        if not csv_files:
                            # Look in part files
                            csv_files = glob(os.path.join(input_path, "part-*.csv"))
                        
                        if not csv_files:
                            raise FileNotFoundError(f"No CSV files found in {input_path}")
                        
                        # Read and concatenate all CSV files
                        pandas_dfs = []
                        for file in csv_files[:10]:  # Limit to first 10 files to avoid memory issues
                            try:
                                pandas_dfs.append(pd.read_csv(file))
                            except Exception as e:
                                print(f"Warning: Could not read {file}: {str(e)}")
                        
                        if not pandas_dfs:
                            raise ValueError("Could not read any CSV files")
                        
                        # Combine all dataframes
                        combined_df = pd.concat(pandas_dfs, ignore_index=True)
                        
                        # Convert pandas DataFrame to Spark DataFrame
                        return spark.createDataFrame(combined_df)
                    else:
                        raise ValueError(f"Unsupported directory format for pandas fallback: {input_path}")
                else:
                    # Single file
                    if input_path.endswith(".parquet"):
                        pandas_df = pd.read_parquet(input_path)
                        return spark.createDataFrame(pandas_df)
                    elif input_path.endswith(".csv"):
                        pandas_df = pd.read_csv(input_path)
                        return spark.createDataFrame(pandas_df)
                    else:
                        raise ValueError(f"Unsupported file format for pandas fallback: {input_path}")
        else:
            raise FileNotFoundError(f"Input path not found: {input_path}")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def sanitize_column_names(df: Any) -> tuple:
    """
    Sanitize column names by replacing dots and other problematic characters with underscores.
    Returns: (sanitized dataframe, mapping of old to new column names)
    """
    # Create a mapping of original column names to sanitized ones
    original_to_sanitized = {}
    for column in df.columns:
        # Replace dots and other problematic characters with underscores
        sanitized = column.replace(".", "_").replace(" ", "_").replace("-", "_")
        original_to_sanitized[column] = sanitized
    
    # Only rename columns that actually changed
    columns_to_rename = {col: sanitized 
                        for col, sanitized in original_to_sanitized.items() 
                        if col != sanitized}
    
    if not columns_to_rename:
        print("No column names need sanitizing")
        return df, {}
    
    print(f"Sanitizing {len(columns_to_rename)} column names with special characters")
    
    # Create a new dataframe with sanitized column names
    sanitized_df = df
    for original, sanitized in columns_to_rename.items():
        sanitized_df = sanitized_df.withColumnRenamed(original, sanitized)
    
    return sanitized_df, original_to_sanitized

def analyze_columns(df: Any, data_quality_report: Dict[str, Any] = None) -> Dict[str, List[str]]:
    """Analyze dataframe columns to categorize them as categorical or numerical."""
    # First check if we can use the data quality report from ingest_data.py
    if config["reuse_data_quality_insights"] and data_quality_report and "dtypes" in data_quality_report:
        print("Using data types from data quality report to optimize column analysis")
        dtypes = data_quality_report["dtypes"]
        
        # Get cardinality from the report if available (helps identify categorical columns)
        cardinality = data_quality_report.get("cardinality", {})
        
        # Only analyze the columns that actually exist in the dataframe
        existing_columns = set(df.columns)
        dtypes = {col: dtype for col, dtype in dtypes.items() if col in existing_columns}
    else:
        # Otherwise get data types directly from the dataframe
        print("Data quality report not available or reuse disabled, analyzing columns directly")
        dtypes = dict(df.dtypes)
        cardinality = {}
    
    categorical_columns = []
    numerical_columns = []
    
    # Explicitly specified columns take precedence
    if config["categorical_columns"]:
        categorical_columns = [col for col in config["categorical_columns"] if col in df.columns]
    
    if config["numerical_columns"]:
        numerical_columns = [col for col in config["numerical_columns"] if col in df.columns]
    
    # Auto-detect categorical and numerical columns if enabled
    if config["auto_detect_categorical"]:
        for column, dtype in dtypes.items():
            # Skip already categorized columns
            if column in categorical_columns or column in numerical_columns:
                continue
            
            # Skip target column 
            if column in config["target_columns"]:
                continue
                
            if dtype in ["int", "bigint", "short", "tinyint"]:
                # If we have cardinality from data quality report, use it
                if column in cardinality:
                    is_categorical = cardinality[column] <= config["categorical_threshold"]
                else:
                    # MODIFICATION: Instead of using df.select().distinct().count() which causes worker crashes,
                    # Let's use a safer approach by assuming numeric columns are continuous
                    # This is a compromise for stability - in production you'd want better cardinality detection
                    is_categorical = False  # Default to treating integers as numerical
                
                if is_categorical:
                    categorical_columns.append(column)
                else:
                    numerical_columns.append(column)
            
            elif dtype in ["double", "float", "decimal"]:
                numerical_columns.append(column)
            
            elif dtype in ["string"]:
                categorical_columns.append(column)
    
    print(f"Detected {len(categorical_columns)} categorical columns and {len(numerical_columns)} numerical columns")
    
    return {
        "categorical": categorical_columns,
        "numerical": numerical_columns
    }

def handle_one_hot_encoding(df: Any, categorical_columns: List[str]) -> tuple:
    """One-hot encode categorical columns."""
    if not categorical_columns:
        print("No categorical columns to encode")
        return df, [], {}
    
    print(f"One-hot encoding {len(categorical_columns)} categorical columns")
    
    # Keep track of categorical to encoded column mapping
    encoded_feature_mapping = {}
    
    # Create pipeline stages
    string_indexer_stages = []
    one_hot_encoder_stages = []
    
    # Add string indexers for each categorical column
    for column in categorical_columns:
        indexer = StringIndexer(
            inputCol=column,
            outputCol=f"{column}_indexed",
            handleInvalid="keep"  # Keep unknown categories
        )
        string_indexer_stages.append(indexer)
        
        encoder = OneHotEncoder(
            inputCol=f"{column}_indexed",
            outputCol=f"{column}_encoded",
            dropLast=True  # Drop the last category to avoid redundancy
        )
        one_hot_encoder_stages.append(encoder)
        
        # Store mapping for later reference
        encoded_feature_mapping[column] = f"{column}_encoded"
    
    # Create and fit the pipeline
    pipeline = Pipeline(stages=string_indexer_stages + one_hot_encoder_stages)
    
    try:
        model = pipeline.fit(df)
        encoded_df = model.transform(df)
        
        # Get all encoded column names
        encoded_columns = [f"{col}_encoded" for col in categorical_columns]
        
        # Drop intermediate indexed columns to save memory
        for col in categorical_columns:
            encoded_df = encoded_df.drop(f"{col}_indexed")
        
        return encoded_df, encoded_columns, encoded_feature_mapping
    except Exception as e:
        print(f"Error during one-hot encoding: {str(e)}")
        # If encoding fails, return original dataframe
        return df, [], {}

def handle_imputation(df: Any, columns: List[str], data_quality_report: Dict[str, Any] = None) -> Any:
    """Impute missing values in specified columns with pandas fallback for Windows stability."""
    if not columns:
        print("No columns to impute")
        return df
    
    print(f"Imputing missing values using method: {config['imputation_method']}")
    
    # Filter out columns that don't need imputation based on data quality report
    if config["reuse_data_quality_insights"] and data_quality_report and "missing_values" in data_quality_report:
        missing_values_info = data_quality_report["missing_values"]
        columns_to_impute = []
        
        for col in columns:
            if col in missing_values_info and missing_values_info[col]["percentage"] > 0:
                columns_to_impute.append(col)
        
        if columns_to_impute:
            print(f"Optimized: Imputing only {len(columns_to_impute)} out of {len(columns)} columns with missing values")
            columns = columns_to_impute
        else:
            print("Optimized: No columns need imputation based on data quality report")
            return df
    
    imputation_method = config["imputation_method"].lower()
    
    # Try Spark imputation first
    try:
        print("Attempting Spark-based imputation...")
        if imputation_method == "mean":
            # Try with a small batch of columns at a time to reduce memory pressure
            max_cols_per_batch = 10
            remaining_columns = columns.copy()
            imputed_df = df
            
            # Process columns in small batches
            while remaining_columns:
                batch_columns = remaining_columns[:max_cols_per_batch]
                remaining_columns = remaining_columns[max_cols_per_batch:]
                
                print(f"Processing batch of {len(batch_columns)} columns...")
                imputer = Imputer(
                    inputCols=batch_columns,
                    outputCols=[f"{c}_imputed" for c in batch_columns],
                    strategy="mean"
                )
                
                # Fit and transform
                imputer_model = imputer.fit(imputed_df)
                imputed_df = imputer_model.transform(imputed_df)
                
                # Replace original columns with imputed ones
                for col in batch_columns:
                    imputed_df = imputed_df.withColumn(col, col(f"{col}_imputed")).drop(f"{col}_imputed")
            
            return imputed_df
        elif imputation_method == "zero":
            # Fill all missing values with 0
            return df.fillna(0, columns)
        elif imputation_method == "ffill":
            # Approximate ffill with means
            mean_values = {}
            for column in columns:
                mean_val = df.select(spark_mean(col(column))).collect()[0][0]
                if mean_val is not None:
                    mean_values[column] = mean_val
                else:
                    mean_values[column] = 0
            
            # Fill with mean values
            return df.fillna(mean_values)
        else:
            print(f"Unknown imputation method: {imputation_method}. No imputation applied.")
            return df
    
    except Exception as spark_error:
        # If Spark imputation fails, fall back to pandas
        print(f"Spark imputation failed with error: {str(spark_error)}")
        print("Falling back to pandas-based imputation (may be slower but more stable)...")
        
        try:
            # Convert to pandas
            print("Converting to pandas DataFrame...")
            pandas_df = df.toPandas()
            
            if imputation_method == "mean":
                # Use pandas fillna with mean
                print("Imputing with pandas mean method...")
                for col in columns:
                    pandas_df[col] = pandas_df[col].fillna(pandas_df[col].mean())
            elif imputation_method == "zero":
                # Fill with zeros
                pandas_df[columns] = pandas_df[columns].fillna(0)
            elif imputation_method == "ffill":
                # Forward fill (pandas makes this easy)
                pandas_df[columns] = pandas_df[columns].fillna(method='ffill')
                # Fill any remaining NAs with mean
                for col in columns:
                    pandas_df[col] = pandas_df[col].fillna(pandas_df[col].mean())
            
            # Convert back to Spark
            print("Converting back to Spark DataFrame...")
            return spark.createDataFrame(pandas_df)
        
        except Exception as pandas_error:
            print(f"Pandas imputation also failed: {str(pandas_error)}")
            print("Returning original DataFrame without imputation.")
            return df
        
def handle_scaling(df: Any, numerical_columns: List[str]) -> tuple:
    """Scale numerical features using the specified method."""
    if not numerical_columns:
        print("No numerical columns to scale")
        return df, {}, None
    
    print(f"Scaling {len(numerical_columns)} numerical columns using {config['scaling_method']}")
    
    # Assemble features into a vector
    assembler = VectorAssembler(
        inputCols=numerical_columns, 
        outputCol="numerical_features",
        handleInvalid="skip"  # Skip rows with null values
    )
    
    # Apply the appropriate scaler
    if config["scaling_method"].lower() == "standard":
        scaler = StandardScaler(
            inputCol="numerical_features", 
            outputCol="scaled_features",
            withStd=True,
            withMean=True
        )
    else:  # Use minmax by default
        scaler = MinMaxScaler(
            inputCol="numerical_features", 
            outputCol="scaled_features",
            min=0.0,
            max=1.0
        )
    
    # Create and fit pipeline
    pipeline = Pipeline(stages=[assembler, scaler])
    
    try:
        # Fit the pipeline to get a model
        model = pipeline.fit(df)
        scaled_df = model.transform(df)
        
        # Create a mapping of original to scaled features
        feature_mapping = {col: "scaled_features" for col in numerical_columns}
        
        # Return the transformed dataframe, feature mapping, and the fitted model
        return scaled_df, feature_mapping, model
    except Exception as e:
        print(f"Error during scaling: {str(e)}")
        # If scaling fails, return original dataframe
        return df, {}, None
    

def calculate_correlation(df: Any, target_column: str, feature_columns: List[str]) -> Dict[str, float]:
    """Calculate correlation between each feature and the target using pure pandas approach."""
    if not target_column or target_column not in df.columns:
        print(f"Target column '{target_column}' not found")
        return {}
    
    # Correlation method
    corr_method = config["correlation_method"].lower()
    print(f"Calculating {corr_method} correlation with target '{target_column}'")
    
    # Initialize correlation dict
    correlations = {}
    
    # For large datasets, we'll use pandas for more correlation methods
    # Convert to pandas for correlation calculation (sample if necessary)
    if df.count() > config["batch_size"]:
        # Sample to avoid memory issues
        sample_size = min(config["batch_size"], df.count())
        sample_df = df.sample(False, sample_size / df.count()).toPandas()
    else:
        sample_df = df.toPandas()
    
    # Calculate correlations for each feature
    for feature in feature_columns:
        try:
            if corr_method == "pearson":
                corr = sample_df[feature].corr(sample_df[target_column], method='pearson')
            elif corr_method == "spearman":
                corr = sample_df[feature].corr(sample_df[target_column], method='spearman')
            elif corr_method == "kendall":
                corr = sample_df[feature].corr(sample_df[target_column], method='kendall')
            else:
                corr = sample_df[feature].corr(sample_df[target_column], method='pearson')
                
            # Store the absolute correlation
            correlations[feature] = abs(corr) if not math.isnan(corr) else 0
                
        except Exception as e:
            print(f"Error calculating correlation for {feature}: {str(e)}")
            correlations[feature] = 0
    
    return correlations

def perform_lasso_selection(df: Any, target_column: str, feature_columns: List[str]) -> Dict[str, float]:
    """Use Lasso regression for feature selection."""
    if not target_column or target_column not in df.columns:
        print(f"Target column '{target_column}' not found")
        return {}
    
    print(f"Performing Lasso-based feature selection (alpha={config['lasso_alpha']})")
    
    # Assemble features into a vector
    assembler = VectorAssembler(
        inputCols=feature_columns, 
        outputCol="features",
        handleInvalid="skip"  # Skip rows with null values
    )
    
    # Create and fit the pipeline with Lasso regression
    lr = LinearRegression(
        featuresCol="features",
        labelCol=target_column,
        standardization=True,
        elasticNetParam=1.0,  # L1 regularization (Lasso)
        regParam=config['lasso_alpha'],
        maxIter=100
    )
    
    try:
        # Prepare the data
        assembled_df = assembler.transform(df)
        
        # Fit the model
        model = lr.fit(assembled_df)
        
        # Get coefficients
        coefficients = model.coefficients.toArray()
        
        # Create dictionary mapping features to their coefficients
        feature_importance = {}
        for i, feature in enumerate(feature_columns):
            feature_importance[feature] = abs(coefficients[i])
        
        return feature_importance
        
    except Exception as e:
        print(f"Error during Lasso feature selection: {str(e)}")
        return {}

def perform_pca(df: Any, feature_columns: List[str]) -> tuple:
    """Apply Incremental PCA to the features."""
    if not feature_columns or not config["use_pca"]:
        return df, [], None
    
    print(f"Applying PCA with {config['pca_components']} components/variance")
    
    # Assemble features into a vector
    assembler = VectorAssembler(
        inputCols=feature_columns, 
        outputCol="pca_features",
        handleInvalid="skip"  # Skip rows with null values
    )
    
    # Determine number of components
    pca_k = config["pca_components"]
    if isinstance(pca_k, float) and 0 < pca_k < 1:
        # User specified variance to retain, calculate max components
        pca_k = min(len(feature_columns), int(len(feature_columns) * pca_k))
    else:
        # User specified exact number of components
        pca_k = min(len(feature_columns), int(pca_k))
    
    # Create PCA transformer
    pca = PCA(
        inputCol="pca_features",
        outputCol="pca_result",
        k=pca_k
    )
    
    # Create and fit pipeline
    pipeline = Pipeline(stages=[assembler, pca])
    
    try:
        # Fit the model
        model = pipeline.fit(df)
        transformed_df = model.transform(df)
        
        # Get PCA components (for debugging/analysis)
        pca_model = model.stages[1]
        explained_variance = pca_model.explainedVariance.toArray()
        
        print(f"PCA explained variance: {sum(explained_variance):.4f}")
        
        # Create column names for PCA components
        pca_columns = [f"pca_{i}" for i in range(pca_k)]
        
        # Return the transformed dataframe, PCA column names, and the fitted model
        return transformed_df, pca_columns, model
        
    except Exception as e:
        print(f"Error during PCA transformation: {str(e)}")
        return df, [], None
    

def combine_selection_methods(correlation=None, lasso=None, rfe=None, weights=None) -> List[str]:
    """
    Combine feature selection results from different methods with weights.
    
    Args:
        correlation: List of features selected by correlation
        lasso: List of features selected by lasso
        rfe: List of features selected by RFE
        weights: Dictionary of weights for each method

    Returns:
        List of selected features sorted by combined score
    """
    if weights is None:
        weights = {"correlation": 0.8, "lasso": 0.5, "rfe": 0.3}
    
    # Initialize combined scores
    feature_scores = {}
    
    # Add correlation-based features
    if correlation:
        for feature in correlation:
            feature_scores[feature] = weights.get("correlation", 0.8)
    
    # Add lasso-based features
    if lasso:
        for feature in lasso:
            feature_scores[feature] = feature_scores.get(feature, 0) + weights.get("lasso", 0.5)
    
    # Add RFE-based features
    if rfe:
        for feature in rfe:
            feature_scores[feature] = feature_scores.get(feature, 0) + weights.get("rfe", 0.3)
    
    # Sort features by score (higher is better)
    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return features sorted by score
    return [feature for feature, _ in sorted_features]

def select_features(correlations: Dict[str, float], lasso_importances: Dict[str, float]) -> List[str]:
    """Select final features based on correlations and importances."""
    # Combine all feature selection methods
    feature_scores = {}
    
    # Normalize and weight correlation scores
    if correlations and "correlation" in config["selection_methods"]:
        max_corr = max(correlations.values()) if correlations.values() else 1
        for feature, score in correlations.items():
            if feature not in feature_scores:
                feature_scores[feature] = 0
            # Add weighted correlation score
            feature_scores[feature] += (score / max_corr) * config["selection_ratio"]["correlation"]
    
    # Normalize and weight Lasso scores
    if lasso_importances and "lasso" in config["selection_methods"]:
        max_imp = max(lasso_importances.values()) if lasso_importances.values() else 1
        for feature, score in lasso_importances.items():
            if feature not in feature_scores:
                feature_scores[feature] = 0
            # Add weighted Lasso score
            feature_scores[feature] += (score / max_imp) * config["selection_ratio"]["lasso"]
    
    # Sort features by combined score
    ranked_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Determine how many features to keep
    # Use the minimum ratio across methods as the final ratio
    selection_ratios = [ratio for method, ratio in config["selection_ratio"].items() 
                        if method in config["selection_methods"]]
    if not selection_ratios:
        final_ratio = 0.5  # Default to keeping 50%
    else:
        final_ratio = min(selection_ratios)
    
    num_features_to_keep = max(1, int(len(ranked_features) * final_ratio))
    
    # Get the top features
    selected_features = [feature for feature, score in ranked_features[:num_features_to_keep]]
    
    print(f"Selected {len(selected_features)} features out of {len(feature_scores)}")
    return selected_features

def save_spark_ml_models(model, prefix, artifacts_dir, timestamp):
    """Save Spark ML models to disk."""
    try:
        # Create a specialized path for Spark ML models
        model_path = os.path.join(artifacts_dir, f"{prefix}_{timestamp}_sparkml")
        
        # Save the Spark ML model
        model.save(model_path)
        print(f"Saved {prefix} Spark ML model to: {model_path}")
        
        return model_path
    except Exception as e:
        print(f"Error saving {prefix} Spark ML model: {str(e)}")
        return None

def save_artifacts(df, selected_features, sanitized_targets, per_target_results, scaler=None, pca_model=None, metadata=None):
    """
    Save feature engineering artifacts for later use in testing or production.
    
    Args:
        df: Original dataframe
        selected_features: List of selected feature names
        sanitized_targets: List of sanitized target column names
        per_target_results: Dictionary of per-target feature selection results
        scaler: Trained scaler model (if used)
        pca_model: Trained PCA model (if used)
        metadata: Feature metadata dictionary
    """
    # Create artifacts directory
    artifacts_dir = os.path.join("results", "feature_artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Add timestamp to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save selected features table - both with schema and actual data
    print(f"Saving selected features table...")
    
    # Combine selected features and target columns
    columns_to_save = list(set(selected_features + sanitized_targets))
    
    if columns_to_save:
        try:
            # Try to select columns from the dataframe using Spark
            selected_df = df.select(*columns_to_save)
            
            # Save schema in JSON format for reference
            schema_file = os.path.join(artifacts_dir, f"selected_features_schema_{timestamp}.json")
            with open(schema_file, 'w') as f:
                schema_json = {col: str(dtype) for col, dtype in selected_df.dtypes}
                json.dump(schema_json, f, indent=2)
            print(f"Saved selected features schema to: {schema_file}")
            
            # Save the selected features data in parquet format
            selected_data_path = os.path.join(artifacts_dir, f"selected_features_data_{timestamp}")
            selected_df.write.parquet(selected_data_path, mode="overwrite")
            print(f"Saved selected features data to: {selected_data_path}")
            
            # Save per-target feature selections
            for target, target_results in per_target_results.items():
                target_features = target_results["selected_features"]
                if target_features:
                    try:
                        # Get target name for file naming
                        target_name = target.split('.')[-1].replace(' ', '_')
                        
                        # Save target-specific selected features
                        target_cols = list(set(target_features + [target]))
                        target_df = df.select(*target_cols)
                        
                        # Save to parquet
                        target_data_path = os.path.join(artifacts_dir, f"target_{target_name}_data_{timestamp}")
                        target_df.write.parquet(target_data_path, mode="overwrite")
                        print(f"Saved {target_name} specific features to: {target_data_path}")
                    except Exception as e:
                        print(f"Error saving target-specific features for {target}: {str(e)}")
        except Exception as e:
            print(f"Error saving selected features table: {str(e)}")
                        # Ensure selected_data_path is not defined if saving fails
            selected_data_path = None
    else:
        print("No features or targets to save.")
        selected_data_path = None

    # Save scaler model if used
    if scaler is not None:
        try:
            scaler_path = os.path.join(artifacts_dir, f"scaler_{timestamp}.pkl")
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            print(f"Saved scaler model to: {scaler_path}")
        except Exception as e:
            print(f"Error saving scaler model: {str(e)}")
            scaler_path = None # Ensure path is None if saving fails
    else:
        scaler_path = None
        print("No scaler model to save.")
    
    # Save PCA model if used
    if pca_model is not None:
        try:
            pca_path = os.path.join(artifacts_dir, f"pca_{timestamp}.pkl")
            with open(pca_path, 'wb') as f:
                pickle.dump(pca_model, f)
            print(f"Saved PCA model to: {pca_path}")
        except Exception as e:
            print(f"Error saving PCA model: {str(e)}")
            pca_path = None # Ensure path is None if saving fails
    else:
        pca_path = None
        print("No PCA model to save.")
    
    # Save feature metadata with reference to artifacts
    if metadata is not None:
        try:
            # Add artifact references to metadata
            metadata["artifacts"] = {
                "timestamp": timestamp,
                "selected_features_schema": f"selected_features_schema_{timestamp}.json" if selected_data_path else None,
                "selected_features_data": f"selected_features_data_{timestamp}" if selected_data_path else None,
                "per_target_data": {
                    target.split('.')[-1].replace(' ', '_'): f"target_{target.split('.')[-1].replace(' ', '_')}_data_{timestamp}"
                    for target in per_target_results.keys() if target_results["selected_features"]
                },
                "scaler_model": f"scaler_{timestamp}.pkl" if scaler else None,
                "pca_model": f"pca_{timestamp}.pkl" if pca_model else None
            }
            
            # Save updated metadata with artifact references
            metadata_path = os.path.join(artifacts_dir, f"feature_metadata_{timestamp}.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Saved feature metadata with artifact references to: {metadata_path}")
            
            # Also update the main metadata file
            main_metadata_path = os.path.join("results", f"feature_metadata.json")
            if os.path.exists(main_metadata_path):
                try:
                    with open(main_metadata_path, 'r') as f:
                        main_metadata = json.load(f)
                    
                    # Add artifacts reference to main metadata
                    if "artifacts" not in main_metadata:
                        main_metadata["artifacts"] = {}
                    
                    main_metadata["artifacts"]["latest"] = timestamp
                    main_metadata["artifacts"][timestamp] = metadata["artifacts"]
                    
                    # Save updated main metadata
                    with open(main_metadata_path, 'w') as f:
                        json.dump(main_metadata, f, indent=2)
                    print(f"Updated main metadata file with artifact references")
                except Exception as e:
                    print(f"Error updating main metadata file: {str(e)}")
            else:
                # If main metadata doesn't exist, create it
                with open(main_metadata_path, 'w') as f:
                    json.dump({"artifacts": {
                        "latest": timestamp,
                        timestamp: metadata["artifacts"]
                    }}, f, indent=2)
                print(f"Created new main metadata file")
        except Exception as e:
            print(f"Error saving metadata with artifact references: {str(e)}")
    
    return {
        "timestamp": timestamp,
        "artifacts_dir": artifacts_dir,
        "selected_data_path": selected_data_path,
        "per_target_paths": {
            target.split('.')[-1].replace(' ', '_'): f"target_{target.split('.')[-1].replace(' ', '_')}_data_{timestamp}"
            for target in per_target_results.keys()
        } if per_target_results else {}
    }

def load_artifacts(timestamp=None):
    """
    Load feature engineering artifacts for testing.
    
    Args:
        timestamp: Specific timestamp of artifacts to load (if None, latest is used)
    """
    # Load main metadata to find latest artifacts
    main_metadata_path = os.path.join("results", "feature_metadata.json")
    with open(main_metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # If timestamp not provided, use latest
    if timestamp is None and "artifacts" in metadata and "latest" in metadata["artifacts"]:
        timestamp = metadata["artifacts"]["latest"]
    
    if not timestamp:
        raise ValueError("No timestamp provided and no latest artifacts found in metadata")
    
    artifacts_dir = os.path.join("results", "feature_artifacts")
    
    # Load artifacts
    artifacts = {}
    
    # Load selected features data
    selected_data_path = os.path.join(artifacts_dir, f"selected_features_data_{timestamp}")
    if os.path.exists(selected_data_path):
        artifacts["selected_features_data"] = spark.read.parquet(selected_data_path)
    
    # Load scaler model
    scaler_path = os.path.join(artifacts_dir, f"scaler_{timestamp}.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            artifacts["scaler"] = pickle.load(f)
    
    # Load PCA model
    pca_path = os.path.join(artifacts_dir, f"pca_{timestamp}.pkl")
    if os.path.exists(pca_path):
        with open(pca_path, 'rb') as f:
            artifacts["pca_model"] = pickle.load(f)
    
    return artifacts

def run_feature_engineering_pipeline() -> Dict[str, Any]:
    """Run the complete feature engineering pipeline with current configuration."""
    try:
        # Load data quality report first to optimize subsequent operations
        data_quality_report = load_data_quality_report() if config["reuse_data_quality_insights"] else {}
        
        # Load the data
        input_path = config["input_path"]
        df = load_data(input_path)
        
        # Sanitize column names to handle dots and special characters
        df, column_name_mapping = sanitize_column_names(df)
        
        # Create reverse mapping to restore original column names in metadata
        reverse_mapping = {v: k for k, v in column_name_mapping.items()}
        
        # Validate which target columns are actually in the data
        target_columns = config["target_columns"]
        sanitized_targets = []
        
        # Map original target names to sanitized names
        for target in target_columns:
            sanitized_target = column_name_mapping.get(target, target)
            if sanitized_target in df.columns:
                sanitized_targets.append(sanitized_target)
            else:
                print(f"Warning: Target column '{target}' not found in dataset and will be skipped")
        
        if not sanitized_targets:
            raise ValueError("None of the specified target columns were found in the dataset")
            
        # Set primary target if it exists, otherwise use the first available target
        primary_target = column_name_mapping.get(config["primary_target"], config["primary_target"])
        if primary_target not in sanitized_targets:
            primary_target = sanitized_targets[0]
            print(f"Primary target not found. Using '{reverse_mapping.get(primary_target, primary_target)}' as primary target")
            
        # Update data quality report column names if needed
        if data_quality_report and "dtypes" in data_quality_report:
            sanitized_dtypes = {}
            for col, dtype in data_quality_report["dtypes"].items():
                if col in column_name_mapping:
                    sanitized_dtypes[column_name_mapping[col]] = dtype
                else:
                    sanitized_dtypes[col] = dtype
            data_quality_report["dtypes"] = sanitized_dtypes
            
            # Update other relevant sections of the data quality report
            if "missing_values" in data_quality_report:
                sanitized_missing = {}
                for col, info in data_quality_report["missing_values"].items():
                    if col in column_name_mapping:
                        sanitized_missing[column_name_mapping[col]] = info
                    else:
                        sanitized_missing[col] = info
                data_quality_report["missing_values"] = sanitized_missing
            
            if "cardinality" in data_quality_report:
                sanitized_cardinality = {}
                for col, value in data_quality_report["cardinality"].items():
                    if col in column_name_mapping:
                        sanitized_cardinality[column_name_mapping[col]] = value
                    else:
                        sanitized_cardinality[col] = value
                data_quality_report["cardinality"] = sanitized_cardinality
            
            if "constant_columns" in data_quality_report:
                sanitized_constant = {}
                for col, value in data_quality_report["constant_columns"].items():
                    if col in column_name_mapping:
                        sanitized_constant[column_name_mapping[col]] = value
                    else:
                        sanitized_constant[col] = value
                data_quality_report["constant_columns"] = sanitized_constant

        # Analyze columns with data quality report assistance
        column_analysis = analyze_columns(df, data_quality_report)
        categorical_columns = column_analysis["categorical"]
        numerical_columns = column_analysis["numerical"]
        
        # Store the original columns before any transformations
        original_columns = df.columns
        
        # Initialize variables to store models
        scaler_model = None
        pca_model = None
        
        # Process categorical columns if any
        if categorical_columns:
            print(f"Processing {len(categorical_columns)} categorical columns")
            try:
                # Apply one-hot encoding
                df, encoded_columns, encoded_feature_mapping = handle_one_hot_encoding(df, categorical_columns)
                
                # Add encoded columns to numerical columns for subsequent processing
                if encoded_columns:
                    print(f"Added {len(encoded_columns)} encoded columns to numerical features")
                    numerical_columns.extend(encoded_columns)
            except Exception as e:
                print(f"Warning: Error during categorical processing: {str(e)}")
                print("Continuing with only numerical features")
        
        # Handle numerical columns (missing value imputation)
        if numerical_columns:
            df = handle_imputation(df, numerical_columns, data_quality_report)
            
            # Feature scaling if configured
            if config["scaling_method"] != "none":
                # Capture both the transformed dataframe and the scaler model
                scaling_result = handle_scaling(df, numerical_columns)
                df = scaling_result[0]  # First return value is the transformed dataframe
                scaled_columns = scaling_result[1]  # Second return value is the feature mapping
                
                # Extract the actual scaler model from the pipeline for saving
                try:
                    scaler_model = {
                        "type": config["scaling_method"],
                        "columns": numerical_columns,
                        "pipeline_model": scaling_result[2] if len(scaling_result) > 2 else None
                    }
                    print(f"Captured {config['scaling_method']} scaler model for saving")
                except Exception as e:
                    print(f"Warning: Could not extract scaler model: {str(e)}")
        
        # Multi-target feature selection
        per_target_results = {}
        
        # Store combined selected features across all targets
        all_selected_features = set()
        
        # Process each target separately
        for target_column in sanitized_targets:
            print(f"\nProcessing feature selection for target: {reverse_mapping.get(target_column, target_column)}")
            
            # Calculate correlations for this target
            feature_columns = [col for col in numerical_columns if col != target_column]
            correlations = calculate_correlation(df, target_column, feature_columns)
            
            # Get features by correlation for this target
            correlation_selected = select_features(correlations, {})
            
            # Lasso-based selection for this target if enabled
            lasso_selected = []
            lasso_importances = {}
            if "lasso" in config["selection_methods"] and correlation_selected:
                lasso_importances = perform_lasso_selection(df, target_column, correlation_selected)
                lasso_selected = select_features({}, lasso_importances)
            
            # Combine and rank features for this target
            target_selected_features = list(set(correlation_selected + lasso_selected))
            
            # Update the combined set
            all_selected_features.update(target_selected_features)
            
            # Store per-target results
            per_target_results[target_column] = {
                "correlations": correlations,
                "lasso_importances": lasso_importances,
                "selected_features": target_selected_features
            }
        
        # Convert to list for consistent ordering
        selected_features = list(all_selected_features)
        
        # If no features were selected, use a default set of numerical columns
        if not selected_features:
            print("Warning: No features were selected by any selection method. Using default feature set.")
            selected_features = numerical_columns[:10] if len(numerical_columns) > 10 else numerical_columns
        
        # Apply PCA if configured
        if config["use_pca"] and selected_features:
            try:
                # Apply PCA
                pca_result = perform_pca(df, selected_features)
                df = pca_result[0]  # First return value is the transformed dataframe
                pca_columns = pca_result[1]  # Second return value is the feature column names
                
                # Extract PCA model if it was returned
                if len(pca_result) > 2:
                    pca_model = pca_result[2]  # Third return value would be the actual model
                    print(f"Captured PCA model with {len(pca_columns)} components")
            except Exception as e:
                print(f"Warning: PCA transformation failed: {str(e)}")
        
        # Create feature mapping
        feature_mapping = {}
        for col in original_columns:
            if col in categorical_columns:
                feature_mapping[col] = "categorical_features"
            elif col in numerical_columns:
                feature_mapping[col] = "scaled_features" if config["scaling_method"] != "none" else "numerical_features"
            else:
                feature_mapping[col] = "other"
        
        # Generate metadata with multi-target information
        metadata = {
            "original_columns": [reverse_mapping.get(str(col), str(col)) for col in original_columns],
            "categorical_columns": [reverse_mapping.get(col, col) for col in categorical_columns],
            "numerical_columns": [reverse_mapping.get(col, col) for col in numerical_columns],
            "selected_features": [reverse_mapping.get(col, col) for col in selected_features],
            "targets": {
                reverse_mapping.get(target, target): {
                    "correlations": {reverse_mapping.get(f, f): c 
                                  for f, c in per_target_results[target]["correlations"].items()
                                  if f in per_target_results[target]["selected_features"]},
                    "selected_features": [reverse_mapping.get(f, f) for f in per_target_results[target]["selected_features"]],
                    "feature_importance": {reverse_mapping.get(f, f): float(i) if not isinstance(i, (int, float)) else i 
                                         for f, i in per_target_results[target].get("lasso_importances", {}).items() 
                                         if f in per_target_results[target]["selected_features"]}
                }
                for target in sanitized_targets
            },
            "primary_target": reverse_mapping.get(primary_target, primary_target),
            "feature_mapping": {reverse_mapping.get(k, k): str(v) for k, v in feature_mapping.items()},
            "column_name_mapping": {k: v for k, v in column_name_mapping.items() if k in original_columns},
            "data_quality_report_used": bool(data_quality_report)
        }
        
        # Save the processed data
        output_path = config["output_path"]
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        output_format = "parquet"

        try:
            # Save the processed DataFrame
            processed_data_path = os.path.join(output_path, f"processed_data_{primary_target}")
            
            # Include only selected features and target columns in the output
            output_columns = selected_features + sanitized_targets
            output_df = df.select(*output_columns)
            
            # Save as parquet
            output_df.write.mode("overwrite").parquet(processed_data_path)
            print(f"Saved processed data to {processed_data_path}")
            
            # Save metadata to separate file
            metadata_path = os.path.join(output_path, f"metadata_{primary_target}.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Saved metadata to {metadata_path}")
            
        except Exception as e:
            print(f"Error saving processed data: {str(e)}")
            # Create minimal metadata file even if saving fails
            try:
                minimal_metadata_path = os.path.join(output_path, "minimal_metadata.json")
                with open(minimal_metadata_path, 'w') as f:
                    json.dump({
                        "error": str(e),
                        "selected_features": metadata["selected_features"],
                        "column_name_mapping": metadata["column_name_mapping"]
                    }, f, indent=2)
                print(f"Saved minimal metadata to {minimal_metadata_path}")
                metadata_path = minimal_metadata_path
            except:
                print("Failed to save even minimal metadata")
                metadata_path = None
        
        # Save artifacts including per-target models and target columns
        artifacts_info = save_artifacts(
            df=df,
            selected_features=selected_features,
            sanitized_targets=sanitized_targets,
            per_target_results=per_target_results,
            scaler=scaler_model,  # Now passing the actual scaler model
            pca_model=pca_model,  # Now passing the actual PCA model
            metadata=metadata
        )
        
        # Return success result with multi-target information
        return {
            "status": "success",
            "message": "Feature engineering completed successfully",
            "input_path": input_path,
            "input_rows": df.count(),
            "input_columns": len(original_columns),
            "categorical_columns": len(categorical_columns),
            "numerical_columns": len(numerical_columns),
            "selected_features": len(selected_features),
            "target_columns": len(sanitized_targets),
            "primary_target": reverse_mapping.get(primary_target, primary_target),
            "output_path": output_path,
            "metadata_path": metadata_path,
            "artifacts": artifacts_info
        }
    except Exception as e:
        print(f"Error in feature engineering pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e)
        }

# API Endpoints
@app.route('/api/config', methods=['GET'])
def get_config():
    """API endpoint to retrieve current configuration."""
    return jsonify(config)

@app.route('/api/config', methods=['POST'])
def update_config():
    """API endpoint to update configuration."""
    new_config = request.json
    if not new_config:
        return jsonify({"status": "error", "message": "No configuration provided"}), 400
    
    for key, value in new_config.items():
        if key in config:
            config[key] = value
        else:
            return jsonify({"status": "error", "message": f"Unknown configuration parameter: {key}"}), 400
    
    return jsonify({"status": "success", "config": config})

@app.route('/api/config/reset', methods=['POST'])
def reset_config():
    """API endpoint to reset configuration to defaults."""
    global config
    config = default_config.copy()
    return jsonify({"status": "success", "config": config})

@app.route('/api/run', methods=['POST'])
def run_pipeline():
    """API endpoint to run the feature engineering pipeline."""
    global config
    try:
        if request.json:
            temp_config = config.copy()
            for key, value in request.json.items():
                if key in temp_config:
                    temp_config[key] = value
            
            old_config = config.copy()
            config = temp_config
            
            result = run_feature_engineering_pipeline()
            
            config = old_config
        else:
            result = run_feature_engineering_pipeline()
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/analyze', methods=['GET'])
def analyze_data():
    """API endpoint to analyze data without running the full pipeline."""
    try:
        # Load the data
        input_path = request.args.get('input_path', config["input_path"])
        df = load_data(input_path)
        
        # Sanitize column names
        df, column_name_mapping = sanitize_column_names(df)
        
        # Load data quality report if available
        data_quality_report = load_data_quality_report() if config["reuse_data_quality_insights"] else {}
        
        # Analyze columns
        column_analysis = analyze_columns(df, data_quality_report)
        
        # Get sample correlations if target is specified
        primary_target = request.args.get('primary_target', config["primary_target"])
        correlations = {}
        if primary_target in df.columns:
            # Sample a few numerical columns
            sample_columns = column_analysis["numerical"][:10]  # Limit to 10 for quick analysis
            correlations = calculate_correlation(df, primary_target, sample_columns)
        
        return jsonify({
            "status": "success",
            "row_count": df.count(),
            "column_count": len(df.columns),
            "column_analysis": column_analysis,
            "sample_correlations": correlations,
            "column_name_mapping": column_name_mapping
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

def main():
    """Main function to run the feature engineering pipeline directly."""
    try:
        print("Starting feature engineering pipeline...")
        result = run_feature_engineering_pipeline()

        if result["status"] == "success":
            print(f" Feature engineering completed successfully:")
            print(f"- Input: {result['input_rows']} rows, {result['input_columns']} columns")
            print(f"- Processed: {result['categorical_columns']} categorical, {result['numerical_columns']} numerical columns")
            print(f"- Selected: {result['selected_features']} features")
            print(f"- Output: {result['output_path']}")
            return True
        else:
            print(f" Feature engineering failed: {result.get('message', 'Unknown error')}")
            return False
    except Exception as e:
        print(f" Critical error in feature engineering pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()