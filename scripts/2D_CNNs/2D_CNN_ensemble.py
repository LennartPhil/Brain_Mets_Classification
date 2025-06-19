import tensorflow as tf
import tensorflow_hub as hub
import helper_funcs as hf
import constants
from pathlib import Path
import os
from time import strftime
from functools import partial
import numpy as np

# --- GPU setup ---
gpus = tf.config.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
print("tensorflow_setup successful")

# --- Configuration ---
dataset_type = constants.Dataset.NORMAL # PRETRAIN_ROUGH, PRETRAIN_FINE, NORMAL
training_mode = constants.Training.NORMAL # LEARNING_RATE_TUNING, NORMAL, K_FOLD, UPPER_LAYER

cutout = False
rgb_images = True # using gray scale images as input
contrast_DA = False # data augmentation with contrast
clinical_data = True
use_layer = True
num_classes = 2

# --- Paths to Pre-trained Base Model Weights ---
PATH_TO_EFFICENTNET_WEIGHTS = Path("PATH")
PATH_TO_RESNET_WEIGHTS = Path("PATH")
PATH_TO_MODEL3_WEIGHTS = Path("PATH")

# Basic to-do:
# build models
# load weights
# get output of each model
# add deep layers on top
# train ensemble model