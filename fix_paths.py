import os

# Fix path separators for Linux
with open('databricks_api_endpoints/feature_engineering_dev.py', 'r') as file:
    content = file.read()

# Replace Windows-specific paths/configurations
content = content.replace(r'C:\\Program Files\\Java\\jdk-11', os.environ.get('JAVA_HOME', '/usr/lib/jvm/java-11-openjdk-amd64'))
content = content.replace(r'C:\\hadoop\\winutils\\hadoop-3.3.6', '/tmp/hadoop')
content = content.replace('D:/temp', '/tmp/spark_temp')
content = content.replace('os.name == \'nt\'', 'False')  # Disable Windows-specific code

# Save the modified file
with open('databricks_api_endpoints/feature_engineering_linux.py', 'w') as file:
    file.write(content)
