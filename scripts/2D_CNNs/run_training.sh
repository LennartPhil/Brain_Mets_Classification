#!/bin/bash

# --- Configuration: SET YOUR ABSOLUTE PATHS HERE ---
# Set the absolute path to the project's root directory.
# This is the directory that contains your 'logs', 'TFRecords', and 'scripts' folders.
export PROJECT_BASE_DIR="/home/lennart"

# --- Environment Setup (Optional but good practice) ---
# If you were using Conda or venv, you would activate it here.
# For example: source /path/to/your/miniconda/bin/activate your_env_name
echo "Using Python executable: $(which python)"
echo "Base directory set to: $PROJECT_BASE_DIR"
echo "------------------------------------------------"

# --- Execution ---
# Run your training script. It will now inherit the PROJECT_BASE_DIR variable.
# The 'exec' command replaces the shell process with the python process.
# The "$@" allows you to pass arguments from the command line to the python script if needed.
exec python 2D_CNN_conv.py "$@"