import tensorflow as tf
import helper_funcs as hf
from pathlib import Path
import os
from time import strftime
from functools import partial
import numpy as np

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

cutout = False
rgb_images = True # using gray scale images as input
contrast_DA = False # data augmentation with contrast
clinical_data = True
use_layer = True
num_classes = 2
use_k_fold = False
learning_rate_tuning = True


def build_ensemble_model(clinical_data, use_layer):

    image_input = tf.keras.layers.Input(shape=(240, 240, 4))
    sex_input = tf.keras.layers.Input(shape=(1,))
    age_input = tf.keras.layers.Input(shape=(1,))
    layer_input = tf.keras.layers.Input(shape=(1,))

    t1, t1c, t2, flair = tf.split(image_input, num_or_size_splits=4, axis=-1)

    def build_subnetwork(name):
        inp = tf.keras.layers.Input(shape=(240, 240, 1), name=f"{name}_input")
        
        # insert proper model here

        # return model