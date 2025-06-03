FROM ubuntu:22.04

# LABEL maintainer="Your Name <your.email@example.com>"
# LABEL description="Container for Databricks API endpoints and AutoGen Studio"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    python3-venv \
    openjdk-11-jdk \
    curl wget \
    nodejs npm \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install PM2 globally
RUN npm install -g pm2

# Create directories
RUN mkdir -p /app/databricks_api_endpoints /app/autogenstudio
WORKDIR /app

# Create Python virtual environment
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Copy requirements file (assuming you have one)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir fastapi uvicorn autogenstudio

# Copy the API endpoints code and configuration
COPY databricks_api_endpoints /app/databricks_api_endpoints/
COPY autogenstudio /app/autogenstudio/
COPY spark_env.sh /app/
COPY raw_data /app/raw_data/

# Make scripts executable
RUN chmod +x /app/spark_env.sh

# Set up environment variables for Spark
RUN . /app/spark_env.sh

# Modify ecosystem.config.js to work in Docker
COPY databricks_api_endpoints/ecosystem.config.js /app/ecosystem.docker.js
RUN sed -i 's|/mnt/d/College/databricks_apis/wsl_venv/bin/|/app/venv/bin/|g' /app/ecosystem.docker.js \
    && sed -i 's|/mnt/d/College/databricks_apis/databricks_api_endpoints|/app/databricks_api_endpoints|g' /app/ecosystem.docker.js

# Add AutoGen Studio API to ecosystem config
RUN sed -i '/  ]/{s/}/},\n    {\n      name: '\''autogen-api'\'',\n      script: '\''\/app\/venv\/bin\/uvicorn'\'',\n      args: '\''app_with_mgmt_Fastapi_no_cache:app --host 0.0.0.0 --port 5003'\'',\n      cwd: '\''\/app\/autogenstudio'\'',\n      instances: 1,\n      autorestart: true,\n      watch: false,\n      env: {\n        PORT: 5003,\n        PYTHONUNBUFFERED: "1"\n      }\n    }/}' /app/ecosystem.docker.js

# Expose the API ports
EXPOSE 5000 5001 5002 5003

# Start PM2 with our configuration
CMD ["pm2-runtime", "start", "/app/ecosystem.docker.js"]