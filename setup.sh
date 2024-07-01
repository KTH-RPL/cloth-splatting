#!/bin/bash

# Use the current directory as the project root.
PROJECT_ROOT=$(pwd)
PATH_TO_BLENDER="/home/omniverse/workspace/synthetic-cloth-data/airo-blender/blender"

# Adding the project directory to PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
export PYTHONPATH="${PROJECT_ROOT}/pyflex:${PYTHONPATH}"

export PATH="$PATH_TO_BLENDER/blender-3.4.1-linux-x64:$PATH"

# Set up pyflex
cd pyflex
source ./prepare.sh

echo "PYTHONPATH set to ${PYTHONPATH}"
cd ..

echo "Environment setup complete."
