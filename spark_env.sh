#!/bin/bash
export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java))))
export SPARK_HOME=$(python3 -c "import pyspark; print(pyspark.__path__[0])")
export PYTHONPATH=$SPARK_HOME/python:$PYTHONPATH
export PYSPARK_PYTHON=python3
export PYSPARK_DRIVER_PYTHON=python3
