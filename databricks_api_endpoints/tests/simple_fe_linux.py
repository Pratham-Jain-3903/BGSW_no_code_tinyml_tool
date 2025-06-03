from pyspark.sql import SparkSession
import os
import sys

# Set up environment variables for Java and Spark
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"  # Use Linux Java path
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# Create temp directory
temp_dir = "/tmp/spark_temp"
os.makedirs(temp_dir, exist_ok=True)

# Initialize Spark session
spark = SparkSession.builder \
    .appName("SimpleFeatureEngineering") \
    .config("spark.driver.memory", "4g") \
    .config("spark.localdir", temp_dir) \
    .master("local[2]") \
    .getOrCreate()

# Simple test
print("Successfully created Spark session")
df = spark.createDataFrame([(1, "a"), (2, "b"), (3, "c")], ["id", "value"])
df.show()

print("Spark test completed successfully")
