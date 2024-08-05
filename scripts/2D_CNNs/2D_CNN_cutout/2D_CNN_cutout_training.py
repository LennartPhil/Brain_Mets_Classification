import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
import os
import random
from time import strftime
import numpy as np
import glob
from functools import partial

os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = "false"

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
learning_rate_tuning = False

batch_size = 50
epochs = 400 #1000
early_stopping_patience = 150
shuffle_buffer_size = 100
repeat_count = 1
starting_lr = 1e-8 #0.00001
learning_rate = 0.0005

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

    #tensorflow_setup()

    # Data setup
    # get list of all patients
    # then split patients into train / val / test
    # then get the tfr paths for each split

    patients = get_patient_paths()

    if learning_rate_tuning:
        train_patients, val_patients, test_patients = split_patients(patients)

        train_paths = get_tfr_paths_for_patients(train_patients)
        val_paths = get_tfr_paths_for_patients(val_patients)
        test_paths = get_tfr_paths_for_patients(test_patients)
        train_data, val_data, test_data = read_data(train_paths, val_paths, test_paths)
        
        callbacks = get_callbacks(
            use_lrscheduler=True,
            use_early_stopping=False
        )

        model = build_simple_model()

        history = model.fit(
            train_data,
            validation_data = val_data,
            epochs = epochs,
            batch_size = batch_size,
            callbacks = callbacks
        )

        history_dict = history.history

        # save history
        history_file_name = f"history.npy"
        path_to_np_file = path_to_callbacks / history_file_name
        np.save(path_to_np_file, history_dict)

    elif use_k_fold:
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
            factor = 3,
            #hyperband_iterations = 2,
            directory = path_to_callbacks,
            project_name = "3D_CNN_hyperband",
            overwrite = True,
            seed = 42
        )

        hyperband_tuner.search(train_data,
                               epochs=5,
                               validation_data=val_data,
                               callbacks=callbacks)

    else:
        # regular training

        train_patients, val_patients, test_patients = split_patients(patients, fraction_to_use = 0.1)

        train_paths = get_tfr_paths_for_patients(train_patients)
        val_paths = get_tfr_paths_for_patients(val_patients)
        test_paths = get_tfr_paths_for_patients(test_patients)
        train_data, val_data, test_data = read_data(train_paths, val_paths, test_paths)
        
        callbacks = get_callbacks(0)

        #model = build_simple_model()
        model = build_hypertuned_model()

        history = model.fit(
            train_data,
            validation_data = val_data,
            epochs = epochs,
            batch_size = batch_size,
            callbacks = callbacks
        )

        history_dict = history.history

        # save history
        history_file_name = f"history.npy"
        path_to_np_file = path_to_callbacks / history_file_name
        np.save(path_to_np_file, history_dict)

def tensorflow_setup():

    #tf.keras.mixed_precision.set_global_policy('mixed_float16')

    # copied directly from: https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
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

def get_patient_paths():
    patients = [f for f in os.listdir(path_to_tfrs) if os.path.isdir(os.path.join(path_to_tfrs, f))]

    patient_paths = [str(path_to_tfrs) + "/" + patient for patient in patients]

    print(f"total patients: {len(patient_paths)}")

    for path in patient_paths:
        patient_not_empty = False
        patient_files = os.listdir(path)
        for file in patient_files:
            if file.endswith(".tfrecord"):
                patient_not_empty = True
        
        if patient_not_empty == False:
            patient_paths.remove(path)

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
        tfr_paths.extend(glob.glob(patient + "/*.tfrecord"))
    
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

    train_data = train_data.map(partial(parse_record, labeled = True, num_classes = num_classes), num_parallel_calls=tf.data.AUTOTUNE)
    val_data = val_data.map(partial(parse_record, labeled = True, num_classes = num_classes), num_parallel_calls=tf.data.AUTOTUNE)

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
        test_data = test_data.map(partial(parse_record, labeled = True, num_classes = num_classes), num_parallel_calls=tf.data.AUTOTUNE)
        test_data = test_data.batch(batch_size)
        test_data = test_data.prefetch(buffer_size=1)

        return train_data, val_data, test_data

    return train_data, val_data

def parse_record(record, labeled = False, num_classes = 2):

    feature_description = {
        "image": tf.io.FixedLenFeature([240, 240, 4], tf.float32),
        "sex": tf.io.FixedLenFeature([], tf.int64, default_value=[0]),
        "age": tf.io.FixedLenFeature([], tf.int64, default_value=0),
        "primary": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    }

    example = tf.io.parse_single_example(record, feature_description)
    image = example["image"]
    image = tf.reshape(image, [240, 240, 4])
    #image = data_augmentation(image)
    primary_to_return = tf.constant(0, dtype=tf.int64)

    if num_classes == 2:
        if example["primary"] == tf.constant(1, dtype=tf.int64):
            primary_to_return = example["primary"]
        else:
            primary_to_return = tf.constant(0, dtype=tf.int64)
    elif num_classes == 3:
        if example["primary"] == tf.constant(1, dtype=tf.int64) or example["primary"] == tf.constant(2, dtype=tf.int64):
            primary_to_return = example["primary"]
        else:
            primary_to_return = tf.constant(0, dtype=tf.int64)
    elif num_classes == 4:
        if example["primary"] == tf.constant(1, dtype=tf.int64) or example["primary"] == tf.constant(2, dtype=tf.int64) or example["primary"] == tf.constant(3, dtype=tf.int64):
            primary_to_return = example["primary"]
        else:
            primary_to_return = tf.constant(0, dtype=tf.int64)
    elif num_classes == 5:
        if example["primary"] == tf.constant(1, dtype=tf.int64) or example["primary"] == tf.constant(2, dtype=tf.int64) or example["primary"] == tf.constant(3, dtype=tf.int64) or example["primary"] == tf.constant(4, dtype=tf.int64):
            primary_to_return = example["primary"]
        else:
            primary_to_return = tf.constant(0, dtype=tf.int64)
    elif num_classes == 6:
        if example["primary"] == tf.constant(1, dtype=tf.int64) or example["primary"] == tf.constant(2, dtype=tf.int64) or example["primary"] == tf.constant(3, dtype=tf.int64) or example["primary"] == tf.constant(4, dtype=tf.int64) or example["primary"] == tf.constant(5, dtype=tf.int64):
            primary_to_return = example["primary"]
        else:
            primary_to_return = tf.constant(0, dtype=tf.int64)
    else:
            print("ERROR")
            print("num classes not supported")
            print("Check parse_record function")
            print("____________________________")

    if labeled:
        return (image, example["sex"], example["age"]), primary_to_return #example["primary"]
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
                  use_csv_logger = True,
                  use_lrscheduler = False):

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
    
    if use_lrscheduler:
        lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch * 0.0175))
        callbacks.append(lr_schedule)

    print("get_callbacks successful")

    return callbacks

# MCDropout
# https://arxiv.org/abs/1506.02142
class MCDropout(tf.keras.layers.Dropout):
    def call(self, inputs, training=False):
        return super().call(inputs, training=True)

def build_simpler_hp_model(hp):
    n_conv_levels = hp.Int("n_conv_levels", min_value=1, max_value=3, default=2)

    # Define inputs
    image_input = tf.keras.layers.Input(shape=(240, 240, 4))

    model = tf.keras.Sequential()
    model.add(image_input)
    model.add(tf.keras.layers.BatchNormalization())

    for i in range(n_conv_levels):
        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation=activation_func, padding="same"))
        # if x.shape[1] >= n_pooling and x.shape[2] >= n_pooling:
        #     x = tf.keras.layers.MaxPool2D(pool_size=n_pooling)(x)
        #     print(f"Shape after conv and pool level {i+1}:", x.shape)

    model.add(tf.keras.layers.Flatten())
    for i in range(4):
        model.add(tf.keras.layers.Dense(20, activation=activation_func))
        model.add(tf.keras.layers.Dropout(0.4))
        #print(f"Shape after dense layer {i+1}:", x.shape)

    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    #model = tf.keras.Model(inputs=image_input, outputs=output)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model


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
    sex_input = tf.keras.layers.Input(shape=(1,))
    age_input = tf.keras.layers.Input(shape=(1,))

    x = tf.keras.layers.BatchNormalization()(image_input)
    print("Input shape:", x.shape)

    for i in range(n_conv_levels):
        if n_strides > 1:
            x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=n_kernel_size, strides=n_strides, activation=activation, padding="same")(x)
        else:
            x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=n_kernel_size, strides=n_strides, activation=activation, padding="valid")(x)

        if x.shape[1] >= n_pooling and x.shape[2] >= n_pooling:
            x = tf.keras.layers.MaxPool2D(pool_size=n_pooling)(x)
            print(f"Shape after conv and pool level {i+1}:", x.shape)
        else:
            print(f"Skipping pooling at level {i+1} due to small dimensions: {x.shape}")

        print(f"Shape after conv and pool level {i+1}:", x.shape)

    x = tf.keras.layers.Flatten()(x)
    for i in range(n_img_dense_layers):
        x = tf.keras.layers.Dense(n_img_dense_neurons, activation=activation)(x)
        if img_dropout:
            x = tf.keras.layers.Dropout(dropout_rate)(x)
        print(f"Shape after dense layer {i+1}:", x.shape)

    flattened_sex_input = tf.keras.layers.Flatten()(sex_input)
    age_input_reshaped = tf.keras.layers.Reshape((1,))(age_input)
    x = tf.keras.layers.Concatenate()([x, age_input_reshaped, flattened_sex_input])

    for i in range(n_end_dense_layers):
        x = tf.keras.layers.Dense(n_end_dense_neurons, activation=activation)(x)
        if end_dropout:
            x = tf.keras.layers.Dropout(dropout_rate)(x)
        print(f"Shape after end dense layer {i+1}:", x.shape)

    x = tf.keras.layers.Dense(1)(x)
    output = tf.keras.layers.Activation('sigmoid', dtype='float32', name='predictions')(x)

    model = tf.keras.Model(inputs=[image_input, sex_input, age_input], outputs=output)

    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy", "RootMeanSquaredError", "AUC"])

    return model

def build_hypertuned_model():
    # Define inputs
    image_input = tf.keras.layers.Input(shape=(240, 240, 4))
    sex_input = tf.keras.layers.Input(shape=(1,))
    age_input = tf.keras.layers.Input(shape=(1,))

    n_conv_levels = 4
    n_kernel_size = 2
    n_filters = 128
    n_pooling = 2
    n_strides = 4
    n_img_dense_layers = 2
    n_img_dense_neurons = 124
    n_end_dense_layers = 1
    n_end_dense_neurons = 116
    img_dropout = False
    end_dropout = True
    dropout_rate = 0.3
    activation = "mish"
    learning_rate = 2e-5
    optimizer = "sgd"

    optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)

    x = tf.keras.layers.BatchNormalization()(image_input)
    print("Input shape:", x.shape)

    for i in range(n_conv_levels):
        if n_strides > 1:
            x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=n_kernel_size, strides=n_strides, activation=activation, padding="same")(x)
        else:
            x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=n_kernel_size, strides=n_strides, activation=activation, padding="valid")(x)

        if x.shape[1] >= n_pooling and x.shape[2] >= n_pooling:
            x = tf.keras.layers.MaxPool2D(pool_size=n_pooling)(x)
            print(f"Shape after conv and pool level {i+1}:", x.shape)
        else:
            print(f"Skipping pooling at level {i+1} due to small dimensions: {x.shape}")

        print(f"Shape after conv and pool level {i+1}:", x.shape)

    x = tf.keras.layers.Flatten()(x)
    for i in range(n_img_dense_layers):
        x = tf.keras.layers.Dense(n_img_dense_neurons, activation=activation)(x)
        if img_dropout:
            x = tf.keras.layers.Dropout(dropout_rate)(x)
        print(f"Shape after dense layer {i+1}:", x.shape)

    flattened_sex_input = tf.keras.layers.Flatten()(sex_input)
    age_input_reshaped = tf.keras.layers.Reshape((1,))(age_input)
    x = tf.keras.layers.Concatenate()([x, age_input_reshaped, flattened_sex_input])

    for i in range(n_end_dense_layers):
        x = tf.keras.layers.Dense(n_end_dense_neurons, activation=activation)(x)
        if end_dropout:
            x = tf.keras.layers.Dropout(dropout_rate)(x)
        print(f"Shape after end dense layer {i+1}:", x.shape)

    x = tf.keras.layers.Dense(1)(x)
    output = tf.keras.layers.Activation('sigmoid', dtype='float32', name='predictions')(x)

    model = tf.keras.Model(inputs=[image_input, sex_input, age_input], outputs=output)

    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy", "RootMeanSquaredError", "AUC"])

    return model

def build_simple_model():
    optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)

    # Define inputs
    image_input = tf.keras.layers.Input(shape=(240, 240, 4))
    sex_input = tf.keras.layers.Input(shape=(1,))
    age_input = tf.keras.layers.Input(shape=(1,))

    batch_norm_layer = tf.keras.layers.BatchNormalization()
    conv_1_layer = tf.keras.layers.Conv2D(filters = 64, kernel_size = 3, input_shape = [240, 240, 4], strides=(2,2), activation=activation_func, kernel_initializer=tf.keras.initializers.HeNormal())
    max_pool_1_layer = tf.keras.layers.MaxPool2D(pool_size = (2,2))

    conv_2_layer = tf.keras.layers.Conv2D(filters = 64, kernel_size = 3, strides=(1,1), activation=activation_func, kernel_initializer=tf.keras.initializers.HeNormal())
    max_pool_2_layer = tf.keras.layers.MaxPool2D(pool_size = (2,2))

    conv_3_layer = tf.keras.layers.Conv2D(filters = 128, kernel_size = 3, strides=(1,1), activation=activation_func, kernel_initializer=tf.keras.initializers.HeNormal())
    max_pool_3_layer = tf.keras.layers.MaxPool2D(pool_size = (2,2))
    
    # conv_4_layer = tf.keras.layers.Conv2D(filters = 256, kernel_size = 3, strides=(1,1,1), activation=activation_func, kernel_initializer=tf.keras.initializers.HeNormal())
    # max_pool_4_layer = tf.keras.layers.MaxPool2D(pool_size = (2,2,2))

    dense_1_layer = tf.keras.layers.Dense(100, activation=activation_func, kernel_initializer=tf.keras.initializers.HeNormal())
    dropout_1_layer = tf.keras.layers.Dropout(0.5)
    dense_2_layer = tf.keras.layers.Dense(100, activation=activation_func, kernel_initializer=tf.keras.initializers.HeNormal())
    dropout_2_layer = tf.keras.layers.Dropout(0.5)
    # output_layer = tf.keras.layers.Dense(2, activation="softmax")

    batch_norm = batch_norm_layer(image_input)

    conv_1 = conv_1_layer(batch_norm)
    max_pool_1 = max_pool_1_layer(conv_1)

    conv_2 = conv_2_layer(max_pool_1)
    max_pool_2 = max_pool_2_layer(conv_2)

    conv_3 = conv_3_layer(max_pool_2)
    max_pool_3 = max_pool_3_layer(conv_3)

    # conv_4 = conv_4_layer(max_pool_3)
    # max_pool_4 = max_pool_4_layer(conv_4)

    flatten = tf.keras.layers.Flatten()(max_pool_3)

    # dense_1 = dense_1_layer(flatten)
    # dropout_1 = dropout_1_layer(dense_1)

    # dense_2 = dense_2_layer(dropout_1)
    # dropout_2 = dropout_2_layer(dense_2)

    # output = output_layer(dropout_2)

    #flattened_images = tf.keras.layers.Flatten()(output)
    flattened_sex_input = tf.keras.layers.Flatten()(sex_input)
    age_input_reshaped = tf.keras.layers.Reshape((1,))(age_input)  # Reshape age_input to have 2 dimensions
    concatenated_inputs = tf.keras.layers.Concatenate()([flatten, age_input_reshaped, flattened_sex_input])

    x = MCDropout(0.4)(concatenated_inputs)
    x = dense_1_layer(x)
    x = MCDropout(0.4)(x)
    x = dense_2_layer(x)
    # x = MCDropout(0.4)(x)
    # x = tf.keras.layers.Dense(200, activation="mish")(x)

    match num_classes:
        case 2:
            x = tf.keras.layers.Dense(1)(x)
            output = tf.keras.layers.Activation('sigmoid', dtype='float32', name='predictions')(x)
        case 3:
            x = tf.keras.layers.Dense(3)(x)
            output = tf.keras.layers.Activation('softmax', dtype='float32', name='predictions')(x)
        case 4:
            x = tf.keras.layers.Dense(4)(x)
            output = tf.keras.layers.Activation('softmax', dtype='float32', name='predictions')(x)
        case 5:
            x = tf.keras.layers.Dense(5)(x)
            output = tf.keras.layers.Activation('softmax', dtype='float32', name='predictions')(x)
        case 6:
            x = tf.keras.layers.Dense(6)(x)
            output = tf.keras.layers.Activation('softmax', dtype='float32', name='predictions')(x)
        case _:
            print("Wrong num classes set in the buil_ai func, please pick a number between 2 and 6")

    model = tf.keras.Model(inputs = [image_input, sex_input, age_input], outputs = [output])
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics = ["RootMeanSquaredError", "accuracy"])
    model.summary()

    return model

if __name__ == "__main__":
    train_ai()