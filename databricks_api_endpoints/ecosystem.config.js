module.exports = {
  apps: [
    {
      name: 'ingest-api',
      script: '/mnt/d/College/databricks_apis/wsl_venv/bin/gunicorn', // Explicit path to Gunicorn
      interpreter: '/mnt/d/College/databricks_apis/wsl_venv/bin/python3', // Explicit path to python
      args: '--bind 0.0.0.0:5000 --workers 1 --timeout 3600 --log-level info ingest_data_linux:app',
      cwd: '/mnt/d/College/databricks_apis/databricks_api_endpoints',
      instances: 1,
      autorestart: true,
      watch: false,
      env: {
        PORT: 5000,
        PYTHONUNBUFFERED: "1", // Useful for seeing logs immediately
        JAVA_HOME: "/usr/lib/jvm/java-11-openjdk-amd64", // Ensure this is correct
        // SPARK_HOME: "/opt/spark", // Example: Set your SPARK_HOME if needed by apps
        // PYTHONPATH: "/mnt/d/College/databricks_apis/databricks_api_endpoints" // If needed
      }
    },
    {
      name: 'feature-api',
      script: '/mnt/d/College/databricks_apis/wsl_venv/bin/gunicorn',
      interpreter: '/mnt/d/College/databricks_apis/wsl_venv/bin/python3',
      args: '--bind 0.0.0.0:5001 --workers 1 --timeout 3600 --log-level info feature_engineering_linux:app',
      cwd: '/mnt/d/College/databricks_apis/databricks_api_endpoints',
      instances: 1,
      autorestart: true,
      watch: false,
      env: {
        PORT: 5001,
        PYTHONUNBUFFERED: "1",
        JAVA_HOME: "/usr/lib/jvm/java-11-openjdk-amd64",
        // SPARK_HOME: "/opt/spark",
        // PYTHONPATH: "/mnt/d/College/databricks_apis/databricks_api_endpoints"
      }
    },
    {
      name: 'ml-api',
      script: '/mnt/d/College/databricks_apis/wsl_venv/bin/gunicorn',
      interpreter: '/mnt/d/College/databricks_apis/wsl_venv/bin/python3',
      args: '--bind 0.0.0.0:5002 --workers 1 --timeout 3600 --log-level info ml_training_benchmarking:app',
      cwd: '/mnt/d/College/databricks_apis/databricks_api_endpoints',
      instances: 1,
      autorestart: true,
      watch: false,
      env: {
        PORT: 5002,
        PYTHONUNBUFFERED: "1",
        JAVA_HOME: "/usr/lib/jvm/java-11-openjdk-amd64",
        // SPARK_HOME: "/opt/spark",
        // PYTHONPATH: "/mnt/d/College/databricks_apis/databricks_api_endpoints"
      }
    }
  ]
};