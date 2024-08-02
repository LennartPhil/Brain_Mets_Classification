import tensorflow as tf
import keras_tuner as kt
import os
import random
from time import strftime
import numpy as np
import glob
from functools import partial

from pathlib import Path

import helper_funcs as hf

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
hyperparameter_tuning = False

batch_size = 4
epochs = 500 #1000
early_stopping_patience = 150
shuffle_buffer_size = 100
repeat_count = 1
starting_lr = 0.00001

activation_func = "mish"
optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=starting_lr, momentum=0.9, nesterov=True)

training_codename = "001"

time = strftime("run_%Y_%m_%d_%H_%M_%S")
if hyperparameter_tuning:
    class_directory = f"hptuning_{num_classes}_classes_{time}"
else:
    class_directory = f"{training_codename}_{num_classes}_classes_{time}"

# create callbacks directory
path_to_callbacks = Path(path_to_logs) / Path(class_directory)
os.makedirs(path_to_callbacks, exist_ok=True)

def train_ai():

    tensorflow_setup()

    # Data setup
    # get list of all patients
    # then split patients into train / val / test
    # then get the tfr paths for each split

    patients = get_patient_paths()

    if use_k_fold:
        pass
    elif hyperparameter_tuning:

        train_patients, val_patients, test_patients = split_patients(patients, fraction_to_use = 0.1)

        train_paths = get_tfr_paths_for_patients(train_patients)
        val_paths = get_tfr_paths_for_patients(val_patients)
        test_paths = get_tfr_paths_for_patients(test_patients)
        train_data, val_data, test_data = read_data(train_paths, val_paths, test_paths)

        callbacks = get_callbacks(
            use_checkpoint = False,
            early_stopping_patience = 5,
            use_csv_logger = False
        )

        hyperband_tuner = kt.Hyperband(
            hypermodel = build_hp_model,
            objective = "val_accuracy",
            max_epochs = 100,
            factor = 4,
            #hyperband_iterations = 2,
            directory = path_to_callbacks,
            project_name = "3D_CNN_hyperband",
            overwrite = True,
            seed = 42
        )

    else:
        pass
        # regular training

    # callbacks
    callbacks = get_callbacks()

    #model = build_simple_model()
    #model = buil_resnext_model()
    #model = build_resnet_model()
    # history = model.fit(train_data,
    #           validation_data = val_data,
    #           epochs = epochs,
    #           batch_size = batch_size,
    #           callbacks = callbacks)
    
    # history_dict = history.history

    # path_to_np_file = path_to_callbacks / "history.npy"
    # np.save(path_to_np_file, history_dict)

def tensorflow_setup():

    tf.keras.mixed_precision.set_global_policy('mixed_float16')

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

def get_patient_paths():
    patients = [f for f in os.listdir(path_to_tfrs) if os.path.isdir(os.path.join(path_to_tfrs, f))]

    patient_paths = [str(path_to_tfrs) + "/" + patient for patient in patients]

    print(f"total patients: {len(patient_paths)}")

    return patient_paths

def split_patients(patient_paths, fraction_to_use = 1):

    random.shuffle(patient_paths)

    patient_paths = patient_paths[:int(len(patient_paths) * fraction_to_use)]

    if fraction_to_use != 1:
        print(f"actual tfrs length: {len(patient_paths)}")

    train_size = int(len(patient_paths) * train_ratio)
    val_size = int(len(patient_paths) * val_ratio)

    train_patients_paths = patient_paths[:train_size]
    val_patients_paths = patient_paths[train_size:train_size + val_size]
    test_patients_paths = patient_paths[train_size + val_size:]

    print(f"train: {len(train_patients_paths)} | val: {len(val_patients_paths)} | test: {len(test_patients_paths)}")

    # save train / val / test patients to txt file
    hf.save_paths_to_txt(train_patients_paths, "train", path_to_callbacks)
    hf.save_paths_to_txt(val_patients_paths, "val", path_to_callbacks)
    hf.save_paths_to_txt(test_patients_paths, "test", path_to_callbacks)

    sum = len(train_patients_paths) + len(val_patients_paths) + len(test_patients_paths)
    if sum != len(patient_paths):
        print("WARNING: error occured in train / val / test split!")

    return train_patients_paths, val_patients_paths, test_patients_paths

def get_tfr_paths_for_patients(patient_paths):

    tfr_paths = []

    for patient in patient_paths:
        tfr_paths.extend(glob.glob(patient + "/*.tfrecords"))
    
    for path in tfr_paths:
        verify_tfrecord(path)

    print(f"total tfrs: {len(tfr_paths)}")

    return tfr_paths

def read_data(train_paths, val_paths, test_paths = None):

    train_data = tf.data.Dataset.from_tensor_slices(train_paths)
    val_data = tf.data.Dataset.from_tensor_slices(val_paths)

    train_data = train_data.interleave(
        lambda x: tf.data.TFRecordDataset([x], compression_type="GZIP"),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )
    val_data = val_data.interleave(
        lambda x: tf.data.TFRecordDataset([x], compression_type="GZIP"),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )

    train_data = train_data.map(partial(parse_record, labeled = True), num_parallel_calls=tf.data.AUTOTUNE)
    val_data = val_data.map(partial(parse_record, labeled = True), num_parallel_calls=tf.data.AUTOTUNE)

    train_data = train_data.shuffle(buffer_size=shuffle_buffer_size)
    val_data = val_data.shuffle(buffer_size=shuffle_buffer_size)

    train_data = train_data.repeat(count = repeat_count)
    val_data = val_data.repeat(count = repeat_count)

    train_data = train_data.batch(batch_size)
    val_data = val_data.batch(batch_size)

    train_data = train_data.prefetch(buffer_size=1)
    val_data = val_data.prefetch(buffer_size=1)

    if test_paths is not None:
        test_data = tf.data.Dataset.from_tensor_slices(test_paths)
        test_data = test_data.interleave(
            lambda x: tf.data.TFRecordDataset([x], compression_type="GZIP"),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False
        )
        test_data = test_data.map(partial(parse_record, labeled = True), num_parallel_calls=tf.data.AUTOTUNE)
        test_data = test_data.batch(batch_size)
        test_data = test_data.prefetch(buffer_size=1)

        return train_data, val_data, test_data

    return train_data, val_data

def parse_record(record, labeled = False):

    feature_description = {
        "image": tf.io.FixedLenFeature([240, 240, 4], tf.float32),
        "sex": tf.io.FixedLenFeature([2], tf.int64, default_value=[0,0]),
        "age": tf.io.FixedLenFeature([], tf.int64, default_value=0),
        "primary": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    }

    example = tf.io.parse_single_example(record, feature_description)
    image = example["image"]
    image = tf.reshape(image, [240, 240, 4])
    image = data_augmentation(image)
    if labeled:
        return (image, example["sex"], example["age"]), example["primary"]
    else:
        return image
    
class image_normalize(tf.keras.layers.Layer):
    def __init__(self):
        super(image_normalize, self).__init__()

    def call(self, inputs):
        min_val = tf.reduce_min(inputs)
        max_val = tf.reduce_max(inputs)
        normalized = (inputs - min_val) / (max_val - min_val)
        return normalized
    
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip(mode = "horizontal"),
    tf.keras.layers.Rescaling(1/255),
    tf.keras.layers.RandomContrast(0.5), # consider removing the random contrast layer as that causes pixel values to go beyond 1
    tf.keras.layers.RandomBrightness(factor = (0, 0.4), value_range=(0, 1)),
    tf.keras.layers.RandomRotation(factor = (-0.1, 0.1), fill_mode = "nearest"),
    image_normalize(),
    tf.keras.layers.RandomTranslation(
        height_factor = 0.05,
        width_factor = 0.05,
        fill_mode = "nearest"
    )
])

def verify_tfrecord(file_path):
    try:
        for _ in tf.data.TFRecordDataset(file_path, compression_type="GZIP"):
            pass
    except tf.errors.DataLossError:
        print(f"Corrupted TFRecord file: {file_path}")

def get_callbacks(fold_num = 0,
                  use_checkpoint = True,
                  use_early_stopping = True,
                  early_stopping_patience = early_stopping_patience,
                  use_tensorboard = True,
                  use_csv_logger = True):

    callbacks = []

    path_to_fold_callbacks = path_to_callbacks / f"fold_{fold_num}"

    def get_run_logdir(root_logdir = path_to_fold_callbacks / "tensorboard"):
        return Path(root_logdir) / strftime("run_%Y_%m_%d_%H_%M_%S")

    run_logdir = get_run_logdir()

    # model checkpoint
    if use_checkpoint:
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath = path_to_fold_callbacks / "saved_weights.weights.h5",
            monitor = "val_accuracy",
            mode = "max",
            save_best_only = True,
            save_weights_only = True,
        )
        callbacks.append(checkpoint_cb)

    # early stopping
    if use_early_stopping:
        early_stopping_cb = tf.keras.callbacks.EarlyStopping(
            patience = early_stopping_patience,
            restore_best_weights = True,
            verbose = 1
        )
        callbacks.append(early_stopping_cb)

    # tensorboard, doesn't really work yet
    if use_tensorboard:
        tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir = run_logdir,
                                                    histogram_freq = 1)
        callbacks.append(tensorboard_cb)
    
    # csv logger
    if use_csv_logger:
        csv_logger_cb = tf.keras.callbacks.CSVLogger(path_to_fold_callbacks / "training.csv", separator = ",", append = True)
        callbacks.append(csv_logger_cb)

    print("get_callbacks successful")

    return callbacks

def build_hp_model(hp):

    n_conv_levels = hp.Int("n_conv_levels", min_value=1, max_value=5, default=3)
    n_kernel_size = hp.Int("n_kernel_size", min_value=2, max_value=7, default=3)
    n_filters = hp.Int("n_filters", min_value=32, max_value=256, default=64, step=32)
    n_pooling = hp.Int("n_pooling", min_value=1, max_value=4, default=2)
    n_strides = hp.Int("n_strides", min_value=1, max_value=4, default=1)
    n_img_dense_layers = hp.Int("n_img_dense_layers", min_value=1, max_value=3, default=2)
    n_img_dense_neurons = hp.Int("n_img_dense_neurons", min_value=32, max_value=200, default=64)
    n_end_dense_layers = hp.Int("n_end_dense_layers", min_value=1, max_value=3, default=2)
    n_end_dense_neurons = hp.Int("n_end_dense_neurons", min_value=32, max_value=200, default=64)
    img_dropout = hp.Boolean("img_dropout")
    end_dropout = hp.Boolean("end_dropout")
    dropout_rate = hp.Float("dropout_rate", min_value=0.0, max_value=0.5, default=0.3)
    activation = hp.Choice("activation", values=["relu", "mish"], default="relu")
    learning_rate = hp.Float("learning_rate", min_value=1e-5, max_value=1e-1, sampling="log")
    optimizer = hp.Choice("optimizer", values=["adam", "sgd"], default="adam")

    if optimizer == "adam":
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    elif optimizer == "sgd":
        optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
    
    # Define inputs
    image_input = tf.keras.layers.Input(shape=(240, 240, 4))
    sex_input = tf.keras.layers.Input(shape=(2,))
    age_input = tf.keras.layers.Input(shape=(1,))

    x = tf.keras.layers.BatchNormalization()(image_input)

    for _ in range(n_conv_levels):
        if n_strides > 1:
            x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=n_kernel_size, strides=n_strides, activation=activation, padding="same")(x)
        else:
            x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=n_kernel_size, strides=n_strides, activation=activation, padding="valid")(x)
        x = tf.keras.layers.MaxPool3D(pool_size=n_pooling)(x)

    x = tf.keras.layers.Flatten()(x)
    for _ in range(n_img_dense_layers):
        x = tf.keras.layers.Dense(n_img_dense_neurons, activation=activation)(x)
        if img_dropout:
            x = tf.keras.layers.Dropout(dropout_rate)(x)

    flattened_sex_input = tf.keras.layers.Flatten()(sex_input)
    age_input_reshaped = tf.keras.layers.Reshape((1,))(age_input)
    x = tf.keras.layers.Concatenate()([x, age_input_reshaped, flattened_sex_input])

    for _ in range(n_end_dense_layers):
        x = tf.keras.layers.Dense(n_end_dense_neurons, activation=activation)(x)
        if end_dropout:
            x = tf.keras.layers.Dropout(dropout_rate)(x)

    x = tf.keras.layers.Dense(1)(x)
    output = tf.keras.layers.Activation('sigmoid', dtype='float32', name='predictions')(x)

    model = tf.keras.Model(inputs=[image_input, sex_input, age_input], outputs=output)

    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy", "RootMeanSquaredError", "AUC"])

    return model

if __name__ == "__main__":
    train_ai()