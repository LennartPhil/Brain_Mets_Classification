# Checklist:
# Initialization: He
# Activation: Mish

# Import libraries
import tensorflow as tf
import keras_tuner as kt
from keras import backend as backend
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from time import strftime

from pathlib import Path

import helper_funcs as hf

from functools import partial

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

# Variables
path_to_tfrs = "/tfrs"
path_to_logs = "/logs"

num_classes = 2

use_k_fold = False
hyperparameter_tuning = True

## train / val / test split
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1
fold_num = 10

batch_size = 2
epochs = 1000
early_stopping_patience = 150
shuffle_buffer_size = 100
repeat_count = 1
learning_rate = 0.001 #previously 0.0001

activation_func = "mish"

time = strftime("run_%Y_%m_%d_%H_%M_%S")
if hyperparameter_tuning:
    class_directory = f"hptuning_{num_classes}_classes_{time}"
else:
    class_directory = f"{num_classes}_classes_{time}"

# create callbacks directory
path_to_callbacks = Path(path_to_logs) / Path(class_directory)
os.makedirs(path_to_callbacks, exist_ok=True)

def train_ai():

    tensorflow_setup()

    # Data setup
    tfr_paths = np.array(get_tfr_paths())

    if use_k_fold:
        train_paths, test_paths = train_test_split(tfr_paths, test_size=test_ratio, random_state=42)


        print(f"total tfrs: {len(tfr_paths)}")
        print(f"Training set: {len(train_paths)} | Test set: {len(test_paths)}")

        kf = KFold(n_splits=fold_num, shuffle=True, random_state=42)

        for fold, (train_index, val_index) in enumerate(kf.split(train_paths)):
            train_fold = train_paths[train_index]
            val_fold = train_paths[val_index]
            
            print(f"*** Fold {fold + 1} ***")
            print("Training Fold: ", len(train_fold))
            print("Validation Fold: ", len(val_fold))

            # check for any errors during k-fold splitting
            for val_path in val_fold:
                if val_path in train_fold:
                    print("WARNING: Duplicate path in train and val fold!")

            # Data setup
            train_data, val_data = read_data(train_fold, val_fold)

            # callbacks
            callbacks = get_callbacks(fold + 1)

            model = build_simple_model()
            #model = build_resnet_model()
            history = model.fit(
                train_data,
                validation_data = val_data,
                epochs = epochs,
                batch_size = batch_size,
                callbacks = callbacks
            )
            
            history_dict = history.history

            # save history
            history_file_name = f"history_{fold + 1}.npy"
            path_to_np_file = path_to_callbacks / history_file_name
            np.save(path_to_np_file, history_dict)

    elif hyperparameter_tuning:
        
        train_paths, val_paths, test_paths = split_data(tfr_paths, fraction_to_use = 0.1)

        train_data, val_data, test_data = read_data(train_paths, val_paths, test_paths)

        callbacks = get_callbacks(0,
                                  use_checkpoint=False,
                                  early_stopping_patience=5,
                                  use_csv_logger=False)
        
        # random_search_tuner = kt.RandomSearch(
        #     hypermodel = build_hp_model,
        #     objective = "val_accuracy",
        #     max_trials = 10,
        #     seed = 42
        # )

        # random_search_tuner.search(train_data,
        #                             epochs=15,
        #                             validation_data=val_data,
        #                             callbacks=callbacks)

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
        train_paths, val_paths, test_paths = split_data(tfr_paths)

        train_data, val_data, test_data = read_data(train_paths, val_paths, test_paths)

        callbacks = get_callbacks(0)

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

    #train_paths, test_paths = train_test_split(tfr_paths, test_size=test_ratio, random_state=42)

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

def split_data(tfr_paths, fraction_to_use = 1):

    random.shuffle(tfr_paths)

    tfr_paths = tfr_paths[:int(len(tfr_paths) * fraction_to_use)]

    if fraction_to_use != 1:
        print(f"actual tfrs length: {len(tfr_paths)}")

    train_size = int(len(tfr_paths) * train_ratio)
    val_size = int(len(tfr_paths) * val_ratio)
    #test_size = int(len(tfr_paths) * test_ratio)

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


data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip(mode = "horizontal"),
    tf.keras.layers.RandomBrightness(factor = (-0.2, 0.2), value_range=(0, 1)),
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

def get_callbacks(fold_num,
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

# MCDropout
# https://arxiv.org/abs/1506.02142
class MCDropout(tf.keras.layers.Dropout):
    def call(self, inputs, training=False):
        return super().call(inputs, training=True)

DefaultConv3D = partial(tf.keras.layers.Conv3D, kernel_size = 3, strides = 1, padding = "same", kernel_initializer = "he_normal", use_bias = False)

class ResidualUnit(tf.keras.layers.Layer):
    def __init__(self, filters, strides = 1, activation = "relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)

        self.main_layers = [
            DefaultConv3D(filters, strides = strides),
            tf.keras.layers.BatchNormalization(),
            self.activation,
            DefaultConv3D(filters),
            tf.keras.layers.BatchNormalization()
        ]

        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                DefaultConv3D(filters, kernel_size = 1, strides = strides),
                tf.keras.layers.BatchNormalization()
            ]
        
    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)

def build_hp_model(hp):

    backend.clear_session()

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
    image_input = tf.keras.layers.Input(shape=(155, 240, 240, 4))
    sex_input = tf.keras.layers.Input(shape=(2,))
    age_input = tf.keras.layers.Input(shape=(1,))

    x = tf.keras.layers.BatchNormalization()(image_input)

    for _ in range(n_conv_levels):
        if n_strides > 1:
            x = tf.keras.layers.Conv3D(filters=n_filters, kernel_size=n_kernel_size, strides=n_strides, activation=activation, padding="same")(x)
        else:
            x = tf.keras.layers.Conv3D(filters=n_filters, kernel_size=n_kernel_size, strides=n_strides, activation=activation, padding="valid")(x)
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


def build_simple_model():

    optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)

    # Define inputs
    image_input = tf.keras.layers.Input(shape=(155, 240, 240, 4))
    sex_input = tf.keras.layers.Input(shape=(2,))
    age_input = tf.keras.layers.Input(shape=(1,))

    batch_norm_layer = tf.keras.layers.BatchNormalization()
    conv_1_layer = tf.keras.layers.Conv3D(filters = 64, kernel_size = 3, input_shape = [155, 240, 240, 4], strides=(2,2,2), activation=activation_func, kernel_initializer=tf.keras.initializers.HeNormal())
    max_pool_1_layer = tf.keras.layers.MaxPooling3D(pool_size = (2,2,2))

    conv_2_layer = tf.keras.layers.Conv3D(filters = 64, kernel_size = 3, strides=(1,1,1), activation=activation_func, kernel_initializer=tf.keras.initializers.HeNormal())
    max_pool_2_layer = tf.keras.layers.MaxPooling3D(pool_size = (2,2,2))

    conv_3_layer = tf.keras.layers.Conv3D(filters = 128, kernel_size = 3, strides=(1,1,1), activation=activation_func, kernel_initializer=tf.keras.initializers.HeNormal())
    max_pool_3_layer = tf.keras.layers.MaxPooling3D(pool_size = (2,2,2))
    
    # conv_4_layer = tf.keras.layers.Conv3D(filters = 256, kernel_size = 3, strides=(1,1,1), activation=activation_func, kernel_initializer=tf.keras.initializers.HeNormal())
    # max_pool_4_layer = tf.keras.layers.MaxPooling3D(pool_size = (2,2,2))

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

    conv_3 = conv_3_layer(max_pool_2)
    max_pool_3 = max_pool_3_layer(conv_3)

    # conv_4 = conv_4_layer(max_pool_3)
    # max_pool_4 = max_pool_4_layer(conv_4)

    flatten = tf.keras.layers.Flatten()(max_pool_3)

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
    x = tf.keras.layers.Dense(50, activation="mish")(x)
    x = MCDropout(0.4)(x)
    x = tf.keras.layers.Dense(50, activation="mish")(x)
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

def build_resnet_model():

    optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)

    # Define inputs
    image_input = tf.keras.layers.Input(shape=(155, 240, 240, 4))
    sex_input = tf.keras.layers.Input(shape=(2,))
    age_input = tf.keras.layers.Input(shape=(1,))

    resnet_model = tf.keras.Sequential([
        DefaultConv3D(filters = 64, kernel_size = 7, strides = 2, input_shape = (155, 240, 240, 4)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(activation_func),
        tf.keras.layers.MaxPool3D(pool_size = 3, strides = 2, padding = "same"),
    ])

    prev_filters = 64
    for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
        strides = 1 if filters == prev_filters else 2
        resnet_model.add(ResidualUnit(filters, strides = strides))
        prev_filters = filters
    
    resnet_model.add(tf.keras.layers.GlobalAvgPool3D())

    flattened_images = tf.keras.layers.Flatten()(resnet_model(image_input))
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