#!/bin/bash

# Use the current directory as the project root.
PROJECT_ROOT=$(pwd)

# Adding the project directory to PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
export PYTHONPATH="${PROJECT_ROOT}/pyflex:${PYTHONPATH}"

# Set up pyflex
cd pyflex
source ./prepare.sh

echo "PYTHONPATH set to ${PYTHONPATH}"
cd ..

echo "Environment setup complete."
