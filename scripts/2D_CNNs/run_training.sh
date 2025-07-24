#!/bin/bash

# --- Configuration: SET YOUR ABSOLUTE PATHS HERE ---
# Set the absolute path to the project's root directory.
# This is the directory that contains your 'logs', 'TFRecords', and 'scripts' folders.
export PROJECT_BASE_DIR="/home/lennart/work"

# This script runs multiple TensorFlow training scripts sequentially.
# The '&&' operator ensures that a script only runs if the previous one succeeded.
# All output (both standard output and errors) will be redirected
# by the nohup command when this script is executed.

echo "======================================================================"
echo "Starting experiment run at $(date)"
echo "======================================================================"

echo "--- Starting training for: 2D_CNN_conv.py ---"
python3 2D_CNN_conv.py && \
echo "--- Finished training for: 2D_CNN_conv.py ---"

echo "--- Starting training for: 2D_CNN_resnet34.py ---"
python3 2D_CNN_resnet34.py && \
echo "--- Finished training for: 22D_CNN_resnet34.py ---"

echo "--- Starting training for: 2D_CNN_resnet152.py ---"
python3 2D_CNN_resnet152.py && \
echo "--- Finished training for: 2D_CNN_resnet152.py ---"

echo "--- Starting training for: 2D_CNN_resnext50.py ---"
python3 2D_CNN_resnext50.py && \
echo "--- Finished training for: 2D_CNN_resnext50.py ---"

echo "--- Starting training for: 2D_CNN_resnext101.py ---"
python3 2D_CNN_resnext101.py && \
echo "--- Finished training for: 2D_CNN_resnext101.py ---"

# Add your other 7+ model training scripts here following the same pattern
# echo "--- Starting training for: [next_model_script.py] ---"
# python3 [next_model_script.py] && \
# echo "--- Finished training for: [next_model_script.py] ---"

echo "======================================================================"
echo "All experiments finished at $(date)"
echo "======================================================================"