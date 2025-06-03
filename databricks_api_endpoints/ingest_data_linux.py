from flask import Flask, request, jsonify
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, col, count, when, isnan, countDistinct
import pandas as pd
import numpy as np
from glob import glob
import pyarrow.parquet as pq
import json
import os
import sys
from functools import reduce
from typing import List, Dict, Any, Optional, Union

# Set JAVA_HOME environment variable for Linux
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"  # Linux Java path

# Create temp directory for Spark
temp_dir = "/tmp/spark_temp"
os.makedirs(temp_dir, exist_ok=True)

# Set Python executable for Spark
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# Initialize Flask app for API endpoints
app = Flask(__name__)

# Initialize Spark session with Linux-compatible configurations
spark = SparkSession.builder \
    .appName("ConfigurableETLPipeline") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.memory.offHeap.enabled", "true") \
    .config("spark.memory.offHeap.size", "2g") \
    .config("spark.driver.maxResultSize", "2g") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.1.0") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.localdir", temp_dir) \
    .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
    .config("spark.sql.execution.arrow.maxRecordsPerBatch", "10000") \
    .config("spark.network.timeout", "800s") \
    .master("local[2]") \
    .getOrCreate()

# Set log level to reduce noise
spark.sparkContext.setLogLevel("ERROR")

# Default configuration - Use Linux-style paths
default_config = {
    "raw_data_path": "raw_data/all_data/collected_csvs",  # Linux path format
    "output_table": "results/ingestion_table",  # Linux path format
    "window_size": 2,
    "filter_columns": [],  # Empty list means keep all columns
    "drop_columns_by_type": [],  # E.g., ["string", "int"]
    "constant_columns_action": "drop",  # Options: "keep", "drop"
    "high_cardinality_threshold": 1000,  # Threshold for high cardinality
    "high_cardinality_action": "keep",  # Options: "keep", "drop"
    "missing_values_threshold": 0.95,  # Drop columns with more than 90% missing values
    "duplicate_rows_action": "keep",  # Options: "keep", "drop"
    "batch_size": 10  # Number of columns to process at once during data quality analysis
}

# Global configuration that will be modified via API
config = default_config.copy()

def get_file_list(path: str) -> list:
    """Get list of files from the specified path."""
    return [
        f"{path}/1sk8DoNzAKNjTg4tM4ula7tN_2024-01-01.csv",
        f"{path}/1sk8DoNzAKNjTg4tM4ula7tN_2024-01-02.csv",
    ]

# Process files with sliding window (unchanged logic)
def process_files_with_sliding_window(files: List[str], window_size: int) -> Any:
    """Process files using a sliding window approach."""
    all_columns = set()
    
    if not files:
        return None, all_columns
    
    # Process one file at a time instead of batches (more memory-efficient)
    df = None
    
    for i, file in enumerate(files):
        try:
            print(f"Processing file {i+1}/{len(files)}: {file}")
            
            # Read the current file
            current_df = spark.read.format("csv") \
                          .option("header", "true") \
                          .option("inferSchema", "true") \
                          .option("mode", "PERMISSIVE") \
                          .option("nullValue", "") \
                          .load(file)
            
            # Update all columns set
            all_columns.update(current_df.columns)
            
            # Create or union with the accumulated dataframe
            if df is None:
                df = current_df
            else:
                df = df.unionByName(current_df, allowMissingColumns=True)
                
            # Optional: checkpoint to prevent lineage from growing too large
            if (i+1) % 5 == 0:  # Every 5 files
                print(f"Checkpointing after {i+1} files...")
                # Enable checkpointing directory - Use Linux-style path
                if not spark.sparkContext._jsc.sc().getCheckpointDir().isDefined():
                    checkpoint_dir = "/tmp/spark_checkpoint"
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    spark.sparkContext.setCheckpointDir(checkpoint_dir)
                
                df = df.checkpoint(eager=True)
        
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
            # Continue with other files
    
    return df, all_columns

# The remaining functions remain largely unchanged, just ensure paths use forward slashes

def apply_column_filters(df: Any, all_columns: set) -> Any:
    """Apply column filtering based on configuration."""
    if config["filter_columns"]:
        columns_to_keep = [col for col in df.columns if col in config["filter_columns"]]
        df = df.select(*columns_to_keep)
    
    if config["drop_columns_by_type"]:
        for col_name, dtype in df.dtypes:
            if dtype in config["drop_columns_by_type"]:
                df = df.drop(col_name)
    
    return df

def analyze_data_quality(df: Any, files: List[str], all_columns: set) -> Dict[str, Any]:
    """Analyze data quality metrics based on the combined DataFrame (Optimized)."""
    dtypes = dict(df.dtypes)

    try:
        # Cache the dataframe to avoid recomputing it multiple times
        df.cache()
        
        # Check if DataFrame is empty - handle safely
        row_count = df.count()
        if row_count == 0:
            df.unpersist()  # Release cache if dataframe is empty
            return {
                "dtypes": dtypes,
                "total_rows": 0,
                "missing_values": {col: {"count": 0, "percentage": 0} for col in df.columns},
                "constant_columns": {col: False for col in df.columns},
                "cardinality": {col: 0 for col, dtype in dtypes.items() if dtype in ['string', 'int'] and col in df.columns},
                "duplicate_row_count": "N/A (skipped for performance)"
            }
        
        # Process columns in smaller batches to reduce memory pressure
        batch_size = config["batch_size"] 
        column_batches = [df.columns[i:i + batch_size] for i in range(0, len(df.columns), batch_size)]
        
        # Initial base stats
        stats_exprs = [count(lit(1)).alias("total_rows")]
        results = df.agg(*stats_exprs).collect()[0]
        total_rows = results["total_rows"]
        
        # Process column batches
        missing_values = {}
        constant_columns = {}
        cardinality = {}
        
        for batch in column_batches:
            print(f"Analyzing batch of {len(batch)} columns...")
            batch_stats = []
            
            for col_name in batch:
                # Missing count
                batch_stats.append(
                    count(when(col(f"`{col_name}`").isNull() | isnan(col(f"`{col_name}`")), col_name))
                    .alias(f"{col_name}_missing_count")
                )
                # Distinct count
                batch_stats.append(
                    countDistinct(col(f"`{col_name}`")).alias(f"{col_name}_distinct_count")
                )
                # Non-null count
                batch_stats.append(
                    count(col(f"`{col_name}`")).alias(f"{col_name}_non_null_count")
                )
            
            # Execute aggregation for this batch
            batch_results = df.agg(*batch_stats).collect()[0]
            
            # Process results for this batch
            for col_name in batch:
                missing_count = batch_results[f"{col_name}_missing_count"]
                distinct_count = batch_results[f"{col_name}_distinct_count"] 
                non_null_count = batch_results[f"{col_name}_non_null_count"]
                
                # Missing values
                missing_percentage = missing_count / total_rows if total_rows > 0 else 0
                missing_values[col_name] = {"count": missing_count, "percentage": missing_percentage}
                
                # Constant columns
                is_constant = distinct_count == 1 and non_null_count > 0
                constant_columns[col_name] = is_constant
                
                # Cardinality
                if dtypes.get(col_name) in ['string', 'int']:
                    cardinality[col_name] = distinct_count
        
        # Release the cached dataframe
        df.unpersist()
        
        data_quality = {
            "dtypes": dtypes,
            "total_rows": total_rows,
            "missing_values": missing_values,
            "constant_columns": constant_columns,
            "cardinality": cardinality,
            "duplicate_row_count": "N/A (skipped for performance)"
        }
        
        return data_quality
    
    except Exception as e:
        # Make sure to release cache even if there's an error
        if df is not None:
            try:
                df.unpersist()
            except:
                pass
        
        print(f"Error analyzing data quality: {str(e)}")
        return {
            "dtypes": dtypes,
            "error": str(e),
            "total_rows": 0,
            "missing_values": {},
            "constant_columns": {},
            "cardinality": {}
        }

def apply_data_quality_filters(df: Any, data_quality: Dict[str, Any]) -> Any:
    """Apply filters based on data quality analysis."""
    try:
        if config["missing_values_threshold"] < 1.0:
            for col_name, info in data_quality["missing_values"].items():
                if col_name in df.columns and info["percentage"] > config["missing_values_threshold"]:
                    print(f"Dropping column '{col_name}' due to high missing values ({info['percentage']:.2%})")
                    df = df.drop(col_name)
    
        if config["constant_columns_action"] == "drop":
            for col_name, is_constant in data_quality["constant_columns"].items():
                if col_name in df.columns and is_constant:
                    print(f"Dropping constant column '{col_name}'")
                    df = df.drop(col_name)
    
        if config["high_cardinality_action"] == "drop":
            for col_name, card_value in data_quality["cardinality"].items():
                if col_name in df.columns and card_value > config["high_cardinality_threshold"]:
                    print(f"Dropping high cardinality column '{col_name}' (cardinality: {card_value})")
                    df = df.drop(col_name)
    
        if config["duplicate_rows_action"] == "drop":
            print(f"Dropping duplicate rows (based on all columns)")
            df = df.dropDuplicates()
    
        return df
    except Exception as e:
        print(f"Error applying data quality filters: {str(e)}")
        # Return original DataFrame in case of error
        return df

def run_etl_pipeline() -> Dict[str, Any]:
    """Run the complete ETL pipeline with current configuration."""
    files = get_file_list(config["raw_data_path"])

    df, all_columns = process_files_with_sliding_window(files, config["window_size"])

    if df is None:
        return {"status": "error", "message": "No files found to process"}

    initial_columns = set(df.columns)

    df = apply_column_filters(df, all_columns)
    columns_after_initial_filter = set(df.columns)
    dropped_by_type_or_explicit = initial_columns - columns_after_initial_filter

    data_quality = analyze_data_quality(df, files, columns_after_initial_filter)

    df = apply_data_quality_filters(df, data_quality)
    final_columns = set(df.columns)

    dropped_total = len(initial_columns) - len(final_columns)

    output_path = config["output_table"]
    output_dir = os.path.dirname(output_path)
    output_base_name = os.path.basename(output_path)
    print(f"Writing final DataFrame to directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    write_format_used = None
    final_output_location = None

    try:
        # Attempt 1: Delta with file URI (Delta requires Spark)
        delta_output_location = os.path.join(output_dir, output_base_name + "_delta")
        # Use Linux-style paths (no need for file://)
        print(f"Attempting Delta write to: {delta_output_location}")
        try:
            df.write.format("delta").mode("overwrite").save(delta_output_location)
            print("Successfully wrote data using Delta format.")
            write_format_used = "delta"
            final_output_location = delta_output_location
        except Exception as delta_error:
            print(f"Delta write failed: {delta_error}")
            
            # Attempt 2a: Parquet with Spark
            parquet_output_location = os.path.join(output_dir, output_base_name + "_parquet")
            print(f"Falling back to Parquet format (via Spark) at: {parquet_output_location}")
            try:
                df.write.format("parquet").mode("overwrite").save(parquet_output_location)
                print("Successfully wrote data using Parquet format (via Spark).")
                write_format_used = "parquet"
                final_output_location = parquet_output_location
            except Exception as parquet_error:
                print(f"Parquet write (via Spark) failed: {parquet_error}")
                
                # Attempt 2b: Parquet with Pandas+PyArrow
                pandas_parquet_location = os.path.join(output_dir, output_base_name + "_parquet_pandas")
                print(f"Falling back to Parquet format (via Pandas) at: {pandas_parquet_location}. WARNING: May cause memory issues on large datasets!")
                try:
                    # Create the directory
                    os.makedirs(pandas_parquet_location, exist_ok=True)
                    
                    # Convert to pandas DataFrame - MEMORY WARNING!
                    print("Converting to pandas DataFrame...")
                    pandas_df = df.toPandas()
                    
                    # Write as parquet file
                    pandas_parquet_file = os.path.join(pandas_parquet_location, "data.parquet")
                    print(f"Writing pandas DataFrame to {pandas_parquet_file}...")
                    pandas_df.to_parquet(pandas_parquet_file, index=False)
                    print("Successfully wrote data using Parquet format (via Pandas).")
                    write_format_used = "parquet_pandas"
                    final_output_location = pandas_parquet_location
                except Exception as pandas_parquet_error:
                    print(f"Parquet write (via Pandas) failed: {pandas_parquet_error}")
                    
                    # Attempt 3a: CSV with Spark
                    csv_output_location = os.path.join(output_dir, output_base_name + "_csv")
                    print(f"Falling back to CSV format (via Spark) at: {csv_output_location}")
                    try:
                        # Using coalesce to write to a single file for easier viewing
                        df.coalesce(1).write.format("csv").option("header", "true").mode("overwrite").save(csv_output_location)
                        print("Successfully wrote data using CSV format (via Spark).")
                        write_format_used = "csv"
                        final_output_location = csv_output_location
                    except Exception as csv_spark_error:
                        print(f"CSV write (via Spark) failed: {csv_spark_error}")
                        
                        # Attempt 3b: CSV with Pandas
                        pandas_csv_location = os.path.join(output_dir, output_base_name + "_csv_pandas")
                        print(f"Falling back to CSV format (via Pandas) at: {pandas_csv_location}")
                        try:
                            # Create the directory
                            os.makedirs(pandas_csv_location, exist_ok=True)
                            
                            # If we already converted to pandas above, use that DataFrame
                            if 'pandas_df' not in locals():
                                print("Converting to pandas DataFrame...")
                                pandas_df = df.toPandas()
                            
                            # Write as CSV file
                            pandas_csv_file = os.path.join(pandas_csv_location, "data.csv")
                            print(f"Writing pandas DataFrame to {pandas_csv_file}...")
                            pandas_df.to_csv(pandas_csv_file, index=False)
                            print("Successfully wrote data using CSV format (via Pandas).")
                            write_format_used = "csv_pandas"
                            final_output_location = pandas_csv_location
                        except Exception as pandas_csv_error:
                            print(f"CSV write (via Pandas) failed: {pandas_csv_error}")
                            
                            # If we've tried everything and nothing worked
                            print("All write attempts failed.")
                            raise Exception("Failed to write data in any format (Delta, Parquet, CSV via both Spark and Pandas).")

        if write_format_used is None or final_output_location is None:
            raise Exception("All write attempts (Delta, Parquet, CSV) failed.")

        # Save data quality report
        dq_report_path = os.path.join(output_dir, output_base_name + "_data_quality_report.json")
        print(f"Saving data quality report to: {dq_report_path}")
        try:
            serializable_dq = data_quality.copy()
            
            # Ensure JSON serialization won't fail
            for key in serializable_dq:
                if key == "dtypes":
                    # dtypes are already strings, should be fine
                    pass
                elif isinstance(serializable_dq[key], dict):
                    # Handle nested dictionaries with non-serializable values
                    for subkey in list(serializable_dq[key]):
                        if not isinstance(serializable_dq[key][subkey], (dict, list, str, int, float, bool, type(None))):
                            serializable_dq[key][subkey] = str(serializable_dq[key][subkey])
                elif not isinstance(serializable_dq[key], (dict, list, str, int, float, bool, type(None))):
                    serializable_dq[key] = str(serializable_dq[key])
            
            with open(dq_report_path, 'w') as f:
                json.dump(serializable_dq, f, indent=4)
            print("Successfully saved data quality report.")
        except Exception as dq_save_error:
            print(f"Warning: Failed to save data quality report: {dq_save_error}")

    except Exception as write_error:
        print(f"Error during write process: {str(write_error)}")
        try:
            rows = df.count()
        except:
            rows = "N/A (Error after processing)"

        return {
            "status": "partial_success",
            "files_processed": len(files),
            "rows_processed": rows,
            "initial_columns": len(initial_columns),
            "columns_kept": len(final_columns),
            "columns_dropped_total": dropped_total,
            "data_quality_analysis": data_quality,
            "write_error": str(write_error),
            "note": "Data processing completed successfully but writing to output location failed."
        }

    return {
        "status": "success",
        "files_processed": len(files),
        "rows_processed": df.count(),
        "initial_columns": len(initial_columns),
        "columns_kept": len(final_columns),
        "columns_dropped_total": dropped_total,
        "data_quality_analysis": data_quality,
        "format_written": write_format_used,
        "output_location": final_output_location,
        "data_quality_report_location": dq_report_path if 'dq_report_path' in locals() else None
    }

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
    """API endpoint to run the ETL pipeline."""
    global config
    try:
        if request.json:
            temp_config = config.copy()
            for key, value in request.json.items():
                if key in temp_config:
                    temp_config[key] = value
            
            old_config = config.copy()
            config = temp_config
            
            result = run_etl_pipeline()
            
            config = old_config
        else:
            result = run_etl_pipeline()
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_data():
    """API endpoint to analyze data without running the full pipeline.
    Accepts JSON body with optional 'user_files' list.
    """
    try:
        # Parse request and get files
        req_json = request.get_json(silent=True) or {}
        user_files = req_json.get("user_files")
        
        # Make sure user_files is a list of file paths
        if user_files is None:
            user_files = get_file_list(config["raw_data_path"])
        elif isinstance(user_files, str):
            # If a single string was provided, wrap it in a list
            user_files = [user_files]
        
        # Process files - this is part 1
        df, all_columns = process_files_with_sliding_window(user_files, config["window_size"])
        
        if df is None:
            return jsonify({"status": "error", "message": "No files found to analyze"}), 404
        
        # Get basic info before potentially expensive operations
        file_count = len(user_files)
        try:
            column_count = len(df.columns)
            row_count = df.count()
            
            # Successfully completed part 1 - return partial result if part 2 fails
            base_result = {
                "status": "partial_success",
                "file_count": file_count,
                "row_count": row_count,
                "column_count": column_count,
                "message": "Data loaded successfully. Basic analysis completed."
            }
            
            # Try to perform data quality analysis
            data_quality = analyze_data_quality(df, user_files, all_columns)
            
            # Full success with data quality
            return jsonify({
                "status": "success",
                "file_count": file_count,
                "row_count": row_count,
                "column_count": column_count,
                "data_quality": data_quality
            })
        except Exception as e:
            # Part 1 completed but part 2 failed
            base_result["message"] = f"Data loaded but quality analysis failed: {str(e)}"
            return jsonify(base_result)
            
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

def main():
    """Main function to run the ETL pipeline directly without using the API."""
    try:
        print("Starting ETL pipeline...")
        result = run_etl_pipeline()

        if result["status"] == "success":
            print(f"\nETL pipeline completed successfully:")
            print(f"- Processed {result['files_processed']} files with {result['rows_processed']} rows")
            print(f"- Kept {result['columns_kept']} columns, dropped {result['columns_dropped_total']} columns")
            print(f"- Data saved to: {result.get('output_location', 'N/A')} (Format: {result.get('format_written', 'N/A')})")
            if result.get('data_quality_report_location'):
                print(f"- Data quality report saved to: {result['data_quality_report_location']}")
            return True
        elif result["status"] == "partial_success":
             print(f"\nETL pipeline completed processing but failed to write data:")
             print(f"- Processed {result['files_processed']} files")
             print(f"- Write Error: {result.get('write_error', 'Unknown')}")
             return False
        else:
            print(f"\nETL pipeline failed: {result.get('message', 'Unknown error')}")
            return False
    except Exception as e:
        print(f"\nCritical error running ETL pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # main()
    app.run(host='0.0.0.0', port=5000, debug=True)
