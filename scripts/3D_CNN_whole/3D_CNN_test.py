# To-do:
# - add class weights
# - add ResNexT structure, doesn't work yet :/
# - test nvidia docker igpu

import tensorflow as tf
import os
import random
from time import strftime

from pathlib import Path

import helper_funcs as hf

#import matplotlib.pyplot as plt

from functools import partial

random.seed(42)

# Variables
path_to_tfrs = "/tfrs"
path_to_logs = "/logs"

num_classes = 2

## train / val / test split
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

batch_size = 1
epochs = 20
shuffle_buffer_size = 400

activation_func = "mish"
optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)

time = strftime("run_%Y_%m_%d_%H_%M_%S")
class_directory = f"{num_classes}_classes_{time}"

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

    model = build_model()
    model.fit(train_data,
              validation_data = val_data,
              epochs = epochs,
              batch_size = batch_size,
              callbacks = callbacks)
    
    score = model.evaluate(test_data)

def tensorflow_setup():
    """
    Set up the TensorFlow environment for training.

    This function sets the random seed for TensorFlow and restricts TensorFlow to use only the first GPU.
    It also prints the number of physical and logical GPUs available.
    """

    tf.keras.utils.set_random_seed(42)

    # copied directly from: https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
    gpus = tf.config.list_physical_devices('GPU')
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

def verify_tfrecord(file_path):
    try:
        for _ in tf.data.TFRecordDataset(file_path, compression_type="GZIP"):
            pass
    except tf.errors.DataLossError:
        print(f"Corrupted TFRecord file: {file_path}")

def get_tfr_paths():
    tfr_file_names = [file for file in os.listdir(path_to_tfrs) if file.endswith(".tfrecord")]
    random.shuffle(tfr_file_names)

    tfr_paths = [str(path_to_tfrs) + "/" + file for file in tfr_file_names]

    print(f"total tfrs: {len(tfr_paths)}")

    for path in tfr_paths:
        verify_tfrecord(path)

    return tfr_paths

def split_data(tfr_paths):

    train_size = int(len(tfr_paths) * train_ratio)
    val_size = int(len(tfr_paths) * val_ratio)
    test_size = int(len(tfr_paths) * test_ratio)

    train_paths = tfr_paths[:train_size]
    val_paths = tfr_paths[train_size:train_size + val_size]
    test_paths = tfr_paths[train_size + val_size:]

    print(f"train: {len(train_paths)} | val: {len(val_paths)} | test: {len(test_paths)}")

    # save train / val / test patients to txt file
    hf.save_paths_to_txt(train_paths, "train", path_to_callbacks)
    hf.save_paths_to_txt(val_paths, "val", path_to_callbacks)
    hf.save_paths_to_txt(test_paths, "test", path_to_callbacks)

    sum = len(train_paths) + len(val_paths) + len(test_paths)
    if sum != len(tfr_paths):
        print("WARNING: error occured in train / val / test split!")

    return train_paths, val_paths, test_paths

def read_data(train_paths, val_paths, test_paths):

    train_data = tf.data.TFRecordDataset([train_paths], compression_type="GZIP")
    val_data = tf.data.TFRecordDataset([val_paths], compression_type="GZIP")
    test_data = tf.data.TFRecordDataset([test_paths], compression_type="GZIP")

    train_data = train_data.map(partial(parse_record, labeled = True))
    val_data = val_data.map(partial(parse_record, labeled = True))
    test_data = test_data.map(partial(parse_record, labeled = True))

    train_data = train_data.shuffle(buffer_size=shuffle_buffer_size)
    val_data = val_data.shuffle(buffer_size=shuffle_buffer_size)
    test_data = test_data.shuffle(buffer_size=shuffle_buffer_size)

    train_data = train_data.repeat(count = epochs)
    val_data = val_data.repeat(count = epochs)
    test_data = test_data.repeat(count = epochs)

    train_data = train_data.batch(batch_size)
    val_data = val_data.batch(batch_size)
    test_data = test_data.batch(batch_size)

    train_data = train_data.prefetch(buffer_size=1)
    val_data = val_data.prefetch(buffer_size=1)
    test_data = test_data.prefetch(buffer_size=1)

    return train_data, val_data, test_data


data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip(mode = "horizontal"),
    tf.keras.layers.RandomBrightness(factor = (-0.2, 0.2), value_range=(0, 1)),
    #tf.keras.layers.RandomContrast(0.5), # consider removing the random contrast layer as that causes pixel values to go beyond 1
    tf.keras.layers.RandomRotation(factor = (-0.07, 0.07), fill_mode = "nearest"),
    tf.keras.layers.RandomTranslation(
        height_factor = 0.05,
        width_factor = 0.05,
        fill_mode = "nearest"
    )
])

def parse_record(record, labeled = False):

    feature_description = {
        "image": tf.io.FixedLenFeature([155, 240, 240, 4], tf.float32),
        "sex": tf.io.FixedLenFeature([2], tf.int64, default_value=[0,0]),
        "age": tf.io.FixedLenFeature([], tf.int64, default_value=0),
        "primary": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    }

    example = tf.io.parse_single_example(record, feature_description)
    image = example["image"]
    image = tf.reshape(image, [155, 240, 240, 4])
    image = data_augmentation(image)
    if labeled:
        return (image, example["sex"], example["age"]), example["primary"]
        #return image, example["primary"] 
    else:
        return image

def get_callbacks():

    def get_run_logdir(root_logdir= path_to_callbacks / "tensorboard"):
        return Path(root_logdir) / strftime("run_%Y_%m_%d_%H_%M_%S")

    run_logdir = get_run_logdir()

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath = path_to_callbacks / "saved_weights.weights.h5",
                                                    monitor = "val_accuracy",
                                                    mode = "max",
                                                    save_best_only = True,
                                                    save_weights_only = True)

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10,
                                                        restore_best_weights = True)

    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir = run_logdir,
                                                    histogram_freq = 1)
    
    print("get_callbacks successful")

    return [checkpoint_cb, early_stopping_cb, tensorboard_cb]

# MCDropout
# https://arxiv.org/abs/1506.02142
class MCDropout(tf.keras.layers.Dropout):
    def call(self, inputs, training=False):
        return super().call(inputs, training=True)

def build_model():

    # Define inputs
    image_input = tf.keras.layers.Input(shape=(155, 240, 240, 4))
    sex_input = tf.keras.layers.Input(shape=(2,))
    age_input = tf.keras.layers.Input(shape=(1,))

    batch_norm_layer = tf.keras.layers.BatchNormalization()
    conv_1_layer = tf.keras.layers.Conv3D(filters = 64, kernel_size = 7, input_shape = [155, 240, 240, 4], strides=(2,2,2), activation=activation_func, kernel_initializer=tf.keras.initializers.HeNormal())
    max_pool_1_layer = tf.keras.layers.MaxPooling3D(pool_size = (2,2,2))
    conv_2_layer = tf.keras.layers.Conv3D(filters = 64, kernel_size = 7, strides=(2,2,2), activation=activation_func, kernel_initializer=tf.keras.initializers.HeNormal())
    max_pool_2_layer = tf.keras.layers.MaxPooling3D(pool_size = (2,2,2))
    dense_1_layer = tf.keras.layers.Dense(100, activation=activation_func, kernel_initializer=tf.keras.initializers.HeNormal())
    dropout_1_layer = tf.keras.layers.Dropout(0.5)
    dense_2_layer = tf.keras.layers.Dense(100, activation=activation_func, kernel_initializer=tf.keras.initializers.HeNormal())
    dropout_2_layer = tf.keras.layers.Dropout(0.5)
    output_layer = tf.keras.layers.Dense(2, activation="softmax")

    batch_norm = batch_norm_layer(image_input)

    conv_1 = conv_1_layer(batch_norm)
    max_pool_1 = max_pool_1_layer(conv_1)

    conv_2 = conv_2_layer(max_pool_1)
    max_pool_2 = max_pool_2_layer(conv_2)

    flatten = tf.keras.layers.Flatten()(max_pool_2)

    dense_1 = dense_1_layer(flatten)
    dropout_1 = dropout_1_layer(dense_1)

    dense_2 = dense_2_layer(dropout_1)
    dropout_2 = dropout_2_layer(dense_2)

    output = output_layer(dropout_2)

    flattened_images = tf.keras.layers.Flatten()(output)
    flattened_sex_input = tf.keras.layers.Flatten()(sex_input)
    age_input_reshaped = tf.keras.layers.Reshape((1,))(age_input)  # Reshape age_input to have 2 dimensions
    concatenated_inputs = tf.keras.layers.Concatenate()([flattened_images, age_input_reshaped, flattened_sex_input])

    x = MCDropout(0.4)(concatenated_inputs)
    x = tf.keras.layers.Dense(200, activation="mish")(x)
    x = MCDropout(0.4)(x)
    x = tf.keras.layers.Dense(200, activation="mish")(x)
    x = MCDropout(0.4)(x)
    x = tf.keras.layers.Dense(200, activation="mish")(x)

    match num_classes:
        case 2:
            output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        case 3:
            output = tf.keras.layers.Dense(3, activation='softmax')(x)
        case 4:
            output = tf.keras.layers.Dense(4, activation='softmax')(x)
        case 5:
            output = tf.keras.layers.Dense(5, activation='softmax')(x)
        case 6:
            output = tf.keras.layers.Dense(6, activation='softmax')(x)
        case _:
            print("Wrong num classes set in the buil_ai func, please pick a number between 2 and 6")

    model = tf.keras.Model(inputs = [image_input, sex_input, age_input], outputs = [output])
    model.compile(loss="mse", optimizer=optimizer, metrics = ["RootMeanSquaredError", "accuracy"])
    model.summary()

    return model

if __name__ == "__main__":
    train_ai()