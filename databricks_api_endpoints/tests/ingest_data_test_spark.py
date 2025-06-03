from pyspark.sql import SparkSession
import os

# Set environment variables
os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jdk-11"
os.environ["HADOOP_HOME"] = r"C:\Program Files\hadoop\winutils\hadoop-3.3.6"

# Create SparkSession with significantly more memory
spark = SparkSession.builder \
    .appName("SimplifiedETL") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.memory.offHeap.enabled", "true") \
    .config("spark.memory.offHeap.size", "2g") \
    .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.1.0") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("hadoop.home.dir", os.environ["HADOOP_HOME"]) \
    .getOrCreate()

# Set log level to reduce noise
spark.sparkContext.setLogLevel("ERROR")

def test_load_csv():
    """Test loading a single CSV file to diagnose issues"""
    try:
        # Path to your data
        data_path = "raw_data/all_data/collected_csvs"
        
        # List files to attempt to load
        files = [
            f"{data_path}/1sk8DoNzAKNjTg4tM4ula7tN_2024-01-01.csv",
            f"{data_path}/1sk8DoNzAKNjTg4tM4ula7tN_2024-01-02.csv",
        ]
        
        print(f"Attempting to load files: {files}")
        
        # Try loading each file individually
        for file in files:
            try:
                print(f"\nLoading file: {file}")
                
                # Read with minimal options, allowing Spark to handle headers automatically 
                df = spark.read.format("csv") \
                    .option("header", "true") \
                    .option("inferSchema", "true") \
                    .option("mode", "PERMISSIVE") \
                    .option("nullValue", "") \
                    .load(file)
                
                # Print info about the file
                print(f"Successfully loaded file with schema:")
                df.printSchema()
                print(f"Row count: {df.count()}")
                
                # Show a sample of the data (only head to avoid memory issues)
                print("Sample data:")
                df.show(5, truncate=False)
                
                print("=" * 50)
            except Exception as e:
                print(f"Error processing file {file}: {str(e)}")
                print("=" * 50)
        
        print("\nTest completed")
        
    except Exception as e:
        print(f"Overall test error: {str(e)}")
    finally:
        # Cleanup - important to prevent resource leaks
        spark.stop()

if __name__ == "__main__":
    test_load_csv()