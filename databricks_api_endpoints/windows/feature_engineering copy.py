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

# Set JAVA_HOME environment variable
os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jdk-11"

# Explicitly set HADOOP_HOME in the script environment as well
os.environ["HADOOP_HOME"] = r"C:\hadoop\winutils\hadoop-3.3.6"

# Initialize Flask app for API endpoints
app = Flask(__name__)

# Initialize Spark session WITH MORE MEMORY
spark = SparkSession.builder \
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
    .config("hadoop.home.dir", os.environ["HADOOP_HOME"]) \
    .getOrCreate()

# Set log level to reduce noise
spark.sparkContext.setLogLevel("ERROR")

# Default configuration
default_config = {
    "input_path": "results/ingestion_table_delta",  # Path to parquet files from ingest_data.py
    "output_path": "results/feature_engineered_data",  # Path for output
    "target_column": "target",  # Name of the target column
    "imputation_method": "mean",  # Options: "mean", "zero", "ffill"
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
        "correlation": 0.8,
        "lasso": 0.5,
        "rfe": 0.3,
        "pca": 1.0
    },
    "batch_size": 10000  # Number of rows to process at once for incremental methods
}

# Global configuration that will be modified via API
config = default_config.copy()

def load_data(input_path: str) -> Any:
    """Load data from parquet, delta or csv files with pandas fallback."""
    print(f"Loading data from: {input_path}")
    try:
        # First attempt - try with Spark (may fail with Hadoop native library error)
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

def analyze_columns(df: Any) -> Dict[str, List[str]]:
    """Analyze dataframe columns to categorize them as categorical or numerical."""
    # Get data types and statistics
    dtypes = dict(df.dtypes)
    
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
            if column == config["target_column"]:
                continue
                
            if dtype in ["int", "bigint", "short", "tinyint"]:
                # Check if this is likely a categorical column by counting distinct values
                if df.select(column).distinct().count() <= config["categorical_threshold"]:
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

def handle_imputation(df: Any, columns: List[str]) -> Any:
    """Impute missing values in specified columns."""
    if not columns:
        print("No columns to impute")
        return df
    
    print(f"Imputing missing values using method: {config['imputation_method']}")
    
    imputation_method = config["imputation_method"].lower()
    
    if imputation_method == "mean":
        # Use Spark ML Imputer for mean imputation
        imputer = Imputer(
            inputCols=columns,
            outputCols=[f"{c}_imputed" for c in columns],
            strategy="mean"
        )
        
        # Fit and transform
        imputer_model = imputer.fit(df)
        imputed_df = imputer_model.transform(df)
        
        # Replace original columns with imputed ones
        for col in columns:
            imputed_df = imputed_df.withColumn(col, col(f"{col}_imputed")).drop(f"{col}_imputed")
        
        return imputed_df
        
    elif imputation_method == "zero":
        # Fill all missing values with 0
        return df.fillna(0, columns)
        
    elif imputation_method == "ffill":
        # Forward fill requires window functions and is complex in Spark
        # We'll use a simpler approximation by filling with column means first
        mean_values = {}
        for column in columns:
            mean_val = df.select(spark_mean(col(column))).collect()[0][0]
            if mean_val is not None:
                mean_values[column] = mean_val
            else:
                mean_values[column] = 0
        
        # Fill with mean values (approximate ffill)
        return df.fillna(mean_values)
        
    else:
        print(f"Unknown imputation method: {imputation_method}. No imputation applied.")
        return df

def handle_scaling(df: Any, numerical_columns: List[str]) -> tuple:
    """Scale numerical features using the specified method."""
    if not numerical_columns:
        print("No numerical columns to scale")
        return df, {}
    
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
        model = pipeline.fit(df)
        scaled_df = model.transform(df)
        
        # Create a mapping of original to scaled features
        feature_mapping = {col: "scaled_features" for col in numerical_columns}
        
        return scaled_df, feature_mapping
    except Exception as e:
        print(f"Error during scaling: {str(e)}")
        # If scaling fails, return original dataframe
        return df, {}

def calculate_correlation(df: Any, target_column: str, feature_columns: List[str]) -> Dict[str, float]:
    """Calculate correlation between each feature and the target."""
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
        return df, []
    
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
        
        return transformed_df, pca_columns
        
    except Exception as e:
        print(f"Error during PCA transformation: {str(e)}")
        return df, []

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

def run_feature_engineering_pipeline() -> Dict[str, Any]:
    """Run the complete feature engineering pipeline with current configuration."""
    try:
        # Load the data
        input_path = config["input_path"]
        df = load_data(input_path)
        
        # Analyze columns
        column_analysis = analyze_columns(df)
        categorical_columns = column_analysis["categorical"]
        numerical_columns = column_analysis["numerical"]
        
        # Track original columns and engineered features
        original_columns = df.columns
        feature_mapping = {}
        
        # Handle categorical features (one-hot encoding)
        df, encoded_columns, categorical_mapping = handle_one_hot_encoding(df, categorical_columns)
        feature_mapping.update(categorical_mapping)
        
        # Impute missing values in numerical columns
        df = handle_imputation(df, numerical_columns)
        
        # Scale numerical features
        df, numerical_mapping = handle_scaling(df, numerical_columns)
        feature_mapping.update(numerical_mapping)
        
        # Prepare for feature selection
        target_column = config["target_column"]
        
        # Calculate correlations between features and target
        correlations = calculate_correlation(df, target_column, numerical_columns + encoded_columns)
        
        # Perform Lasso-based feature selection
        lasso_importances = {}
        if "lasso" in config["selection_methods"]:
            lasso_importances = perform_lasso_selection(df, target_column, numerical_columns + encoded_columns)
        
        # Select the final feature set
        selected_features = select_features(correlations, lasso_importances)
        
        # Apply PCA if configured
        if config["use_pca"]:
            df, pca_columns = perform_pca(df, selected_features)
            # If PCA was successful, update selected features
            if pca_columns:
                selected_features = pca_columns
        
        # Create a final feature set with all engineered features
        final_df = df.select([target_column] + selected_features)
        
        print(f"Writing engineered features to: {output_path}")
        output_format = None
        try:
            # Try to write with Spark first
            final_df.write.mode("overwrite").parquet(output_path)
            output_format = "parquet"
        except Exception as spark_write_error:
            print(f"Error writing with Spark: {str(spark_write_error)}")
            print("Falling back to pandas for data writing...")
            
            try:
                # Convert to pandas for writing
                pandas_df = final_df.toPandas()
                
                # Create output directory
                os.makedirs(output_path, exist_ok=True)
                
                # Write as parquet with pandas+pyarrow
                pandas_output_file = os.path.join(output_path, "data.parquet")
                pandas_df.to_parquet(pandas_output_file, index=False)
                output_format = "parquet_pandas"
                print(f"Successfully wrote data using pandas to {pandas_output_file}")
            except Exception as pandas_write_error:
                print(f"Pandas parquet write failed: {str(pandas_write_error)}")
                
                # Last resort: CSV
                try:
                    csv_output_path = f"{output_path}_csv"
                    os.makedirs(csv_output_path, exist_ok=True)
                    csv_output_file = os.path.join(csv_output_path, "data.csv")
                    pandas_df.to_csv(csv_output_file, index=False)
                    output_path = csv_output_path
                    output_format = "csv_pandas"
                    print(f"Successfully wrote data using pandas CSV to {csv_output_file}")
                except Exception as csv_write_error:
                    print(f"All write methods failed. Last error: {str(csv_write_error)}")
                    raise Exception("Failed to write output data in any format")
        
        # Save feature metadata for reference
        metadata_path = os.path.join(os.path.dirname(output_path), "feature_metadata.json")
        metadata = {
            "original_columns": original_columns,
            "categorical_columns": categorical_columns,
            "numerical_columns": numerical_columns,
            "selected_features": selected_features,
            "correlations": {f: float(c) if not isinstance(c, (int, float)) else c 
                            for f, c in correlations.items() if f in selected_features},
            "feature_importance": {f: float(i) if not isinstance(i, (int, float)) else i 
                                  for f, i in lasso_importances.items() if f in selected_features},
            "feature_mapping": {k: str(v) for k, v in feature_mapping.items()}
        }
        
        # Ensure the metadata is JSON serializable
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        return {
            "status": "success",
            "input_rows": df.count(),
            "input_columns": len(original_columns),
            "categorical_columns": len(categorical_columns),
            "numerical_columns": len(numerical_columns),
            "selected_features": len(selected_features),
            "output_path": output_path,
            "output_format": output_format,
            "metadata_path": metadata_path
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
        
        # Analyze columns
        column_analysis = analyze_columns(df)
        
        # Get sample correlations if target is specified
        target_column = request.args.get('target_column', config["target_column"])
        correlations = {}
        if target_column in df.columns:
            # Sample a few numerical columns
            sample_columns = column_analysis["numerical"][:10]  # Limit to 10 for quick analysis
            correlations = calculate_correlation(df, target_column, sample_columns)
        
        return jsonify({
            "status": "success",
            "row_count": df.count(),
            "column_count": len(df.columns),
            "column_analysis": column_analysis,
            "sample_correlations": correlations
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

def main():
    """Main function to run the feature engineering pipeline directly."""
    try:
        print("Starting feature engineering pipeline...")
        result = run_feature_engineering_pipeline()

        if result["status"] == "success":
            print(f"\nFeature engineering completed successfully:")
            print(f"- Input: {result['input_rows']} rows, {result['input_columns']} columns")
            print(f"- Processed: {result['categorical_columns']} categorical, {result['numerical_columns']} numerical columns")
            print(f"- Selected: {result['selected_features']} features")
            print(f"- Output: {result['output_path']} ({result['output_format']})")
            print(f"- Metadata: {result['metadata_path']}")
            return True
        else:
            print(f"\nFeature engineering failed: {result.get('message', 'Unknown error')}")
            return False
    except Exception as e:
        print(f"\nCritical error in feature engineering pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()