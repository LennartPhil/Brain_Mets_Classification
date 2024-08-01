import tensorflow as tf
import os
import random
from time import strftime
import numpy as np

from pathlib import Path

path_to_tfrs = "/tfrs"
path_to_logs = "/logs"

num_classes = 2

## train / val / test split
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# can be either "hp", "lr" or "train"
# hp = hyperparameter tuning
# lr = learning rate
# train = train the model
mode = "hp"

use_k_fold = False

batch_size = 4
epochs = 500 #1000
early_stopping_patience = 150
shuffle_buffer_size = 100
repeat_count = 1
starting_lr = 0.00001

activation_func = "mish"
optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=starting_lr, momentum=0.9, nesterov=True)

time = strftime("run_%Y_%m_%d_%H_%M_%S")
class_directory = f"lr_{num_classes}_classes_{time}"

# create callbacks directory
path_to_callbacks = Path(path_to_logs) / Path(class_directory)
os.makedirs(path_to_callbacks, exist_ok=True)

def train_ai():

    tensorflow_setup()

    # Data setup
    tfr_paths = get_tfr_paths()
    train_paths, val_paths, test_paths = split_data(tfr_paths)
    train_data, val_data, test_data = read_data(train_paths, val_paths, test_paths)

    # callbacks
    callbacks = get_callbacks()

    #model = build_simple_model()
    #model = buil_resnext_model()
    #model = build_resnet_model()
    model = build_simple_model_30_06_24()
    history = model.fit(train_data,
              validation_data = val_data,
              epochs = epochs,
              batch_size = batch_size,
              callbacks = callbacks)
    
    history_dict = history.history

    path_to_np_file = path_to_callbacks / "history.npy"
    np.save(path_to_np_file, history_dict)

if __name__ == "__main__":
    train_ai()