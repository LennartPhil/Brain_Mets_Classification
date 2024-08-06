import tensorflow as tf
import random
import os
from functools import partial
from pathlib import Path
from time import strftime
import glob

kernel_initializer = "he_normal"
activation_func = "mish"

train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

shuffle_buffer_size = 200
repeat_count = 1

early_stopping_patience = 200

def setup_data(path_to_tfrs, path_to_callbacks, num_classes, batch_size, rgb = False):
    patients = get_patient_paths(path_to_tfrs)

    train_paths, val_paths, test_paths = split_patients(patients, path_to_callbacks=path_to_callbacks, fraction_to_use=1)
    train_paths = get_tfr_paths_for_patients(train_paths)
    val_paths = get_tfr_paths_for_patients(val_paths)
    test_paths = get_tfr_paths_for_patients(test_paths)

    train_data, val_data, test_data = read_data(train_paths, val_paths, num_classes, batch_size, test_paths, rgb = rgb)

    return train_data, val_data, test_data


def save_paths_to_txt(paths, type, path_to_callbacks):
    f = open(f"{path_to_callbacks}/{type}.txt", "w")

    # get patient id from path
    paths = [path.split("/")[-1] for path in paths]

    for path in paths:
        f.write(f"{path}\n")

    f.close()

    print(f"Saved {type} paths to txt file")


def get_patient_paths(path_to_tfrs):
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


def split_patients(patient_paths, path_to_callbacks, fraction_to_use = 1):

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
    save_paths_to_txt(train_patients_paths, "train", path_to_callbacks)
    save_paths_to_txt(val_patients_paths, "val", path_to_callbacks)
    save_paths_to_txt(test_patients_paths, "test", path_to_callbacks)

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

    #print(f"total tfrs: {len(tfr_paths)}")

    return tfr_paths

def read_data(train_paths, val_paths, num_classes, batch_size, test_paths = None, rgb = False):

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

    train_data = train_data.map(partial(parse_record, image_only = False, labeled = True, num_classes = num_classes, rgb = rgb), num_parallel_calls=tf.data.AUTOTUNE)
    val_data = val_data.map(partial(parse_record, image_only = False, labeled = True, num_classes = num_classes, rgb = rgb), num_parallel_calls=tf.data.AUTOTUNE)

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
        test_data = test_data.map(partial(parse_record, image_only = False, labeled = True, num_classes = num_classes, rgb = rgb), num_parallel_calls=tf.data.AUTOTUNE)
        test_data = test_data.batch(batch_size)
        test_data = test_data.prefetch(buffer_size=1)

        return train_data, val_data, test_data

    return train_data, val_data

def parse_record(record, image_only = False, labeled = False, num_classes = 2, rgb = False, sequence = "t1c"):

    image_shape = []

    if rgb: # rgb images need three channels
        image_shape = [240, 240, 3, 4]
    else: # gray scale images don't
        image_shape = [240, 240, 4]

    feature_description = {
        "image": tf.io.FixedLenFeature(image_shape, tf.float32),
        "sex": tf.io.FixedLenFeature([], tf.int64, default_value=[0]),
        "age": tf.io.FixedLenFeature([], tf.int64, default_value=0),
        "primary": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    }

    example = tf.io.parse_single_example(record, feature_description)
    image = example["image"]
    image = tf.reshape(image, image_shape)

    # primary should have a value between 0 and 5
    # depending on num classes return different values
    # if num_classes = 2, return 1 if primary is 1, else 0
    # if num_classes = 3, return primaries 1 and 2, else 0
    # if num_classes = 4, return primaries 1, 2 and 3, else 0
    # if num_classes = 5, return primaries 1, 2, 3 and 4, else 0
    # if num_classes = 6, return primaries 1, 2, 3, 4 and 5, else 0

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

    if rgb: # select the right sequence to return
        if sequence == "t1":
            image = image[:, :, :, 0]
        elif sequence == "t1c":
            image = image[:, :, :, 1]
        elif sequence == "t2":
            image = image[:, :, :, 2]
        elif sequence == "flair":
            image = image[:, :, :, 3]

    if image_only:
        return image, primary_to_return
    elif labeled:
        return (image, example["sex"], example["age"]), primary_to_return #example["primary"]
    else:
        return image
    
def verify_tfrecord(file_path):
    try:
        for _ in tf.data.TFRecordDataset(file_path, compression_type="GZIP"):
            pass
    except tf.errors.DataLossError:
        print(f"Corrupted TFRecord file: {file_path}")


def get_callbacks(path_to_callbacks,
                  fold_num = 0,
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

def __initial_conv_block(input, weight_decay = 5e-4):
    ''' Adds an initial convolution block, with batch normalization and relu activation
    Args:
        input: input tensor
        weight_decay: weight decay factor
    Returns: a keras tensor
    '''

    x = tf.keras.layers.Conv3D(filters = 64,
                               kernel_size = 3,
                               padding = "same",
                               kernel_initializer = kernel_initializer,
                               kernel_regularizer = tf.keras.regularizers.l2(weight_decay))(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation_func)(x)

    return x

def __grouped_convolution_block(input, grouped_channels, cardinality, strides, weight_decay = 5e-4):
    ''' Adds a grouped convolution block. It is an equivalent block from the paper
    Args:
        input: input tensor
        grouped_channels: grouped number of filters
        cardinality: cardinality factor describing the number of groups
        strides: performs strided convolution for downscaling if > 1
        weight_decay: weight decay term
    Returns: a keras tensor
    '''

    group_list = []

    if cardinality == 1:
        # with cardinality 1, it is a standard convolution
        x = tf.keras.layers.Conv3D(filters = grouped_channels,
                                   kernel_size = 3,
                                   padding = "same",
                                   use_bias = False,
                                   strides = (strides, strides, strides),
                                   kernel_initializer = kernel_initializer,
                                   kernel_regularizer = tf.keras.regularizers.l2(weight_decay))(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation_func)(x)

        return x
    
    # cardinality loop
    for c in range(cardinality):
        x = tf.keras.layers.Lambda(lambda x: x[:, :, :, :, c * grouped_channels : (c + 1) * grouped_channels])(input)

        x = tf.keras.layers.Conv3D(filters = grouped_channels,
                                   kernel_size = 3,
                                   padding = "same",
                                   use_bias = False,
                                   strides = (strides, strides, strides),
                                   kernel_initializer = kernel_initializer,
                                   kernel_regularizer = tf.keras.regularizers.l2(weight_decay))(x)
        
        group_list.append(x)
    
    group_merge = tf.keras.layers.Concatenate(axis=-1)(group_list)
    x = tf.keras.layers.BatchNormalization()(group_merge)
    x = tf.keras.layers.Activation(activation_func)(x)

    return x


def __bottleneck_block(input, filters, cardinality, strides, weight_decay):
    init = input

    # Determine if the shortcut path needs a convolution for matching dimensions
    needs_conv = strides > 1 or input.shape[-1] != filters * 2

    grouped_channels = filters // cardinality
    
    if needs_conv:
        # Apply convolution to shortcut path to match the main path's dimensions
        init = tf.keras.layers.Conv3D(filters * 2, 1, strides=strides, padding="same", use_bias=False,
                                      kernel_initializer=kernel_initializer,
                                      kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(init)
        init = tf.keras.layers.BatchNormalization()(init)

    # Main path
    x = tf.keras.layers.Conv3D(filters, 1, padding="same", use_bias=False,
                               kernel_initializer=kernel_initializer,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation_func)(x)

    x = __grouped_convolution_block(x, grouped_channels, cardinality, strides, weight_decay)

    x = tf.keras.layers.Conv3D(filters * 2, 1, padding="same", use_bias=False,
                               kernel_initializer=kernel_initializer,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Addition - ensuring init and x have compatible shapes
    x = tf.keras.layers.Add()([init, x])
    x = tf.keras.layers.Activation(activation_func)(x)

    return x
 

def create_res_next(img_input, depth = 29, cardinality = 8, width = 4,
                      weight_decay = 5e-4, pooling = None):
    ''' Creates a ResNeXt model with specified parameters
    Args:
        nb_classes: Number of output classes
        img_input: Input tensor or layer
        include_top: Flag to include the last dense layer
        depth: Depth of the network. Can be an positive integer or a list
               Compute N = (n - 2) / 9.
               For a depth of 56, n = 56, N = (56 - 2) / 9 = 6
               For a depth of 101, n = 101, N = (101 - 2) / 9 = 11
        cardinality: the size of the set of transformations.
               Increasing cardinality improves classification accuracy,
        width: Width of the network.
        weight_decay: weight_decay (l2 norm)
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
    Returns: a Keras Model
    '''

    if type(depth) is list or type(depth) is tuple:
        # if a list is provided, defer to user how many blocks are present
        N = list(depth)
    else:
        # otherwise, default to 3 blocks each of default number of group convolution blocks
        N = [(depth - 2) // 9 for _ in range(3)]
    
    filters = cardinality * width
    filters_list = []

    for _ in range(len(N)):
        filters_list.append(filters)
        filters *= 2
    
    x = __initial_conv_block(img_input, weight_decay)

    # block 1 (no pooling)
    for _ in range(N[0]):
        x = __bottleneck_block(x, filters_list[0], cardinality, strides=1, weight_decay=weight_decay)
    
    N = N[1:] # remove the first block from block definition list
    filters_list = filters_list[1:] # remove the first filter from the filter list

    # block 2 to N
    for block_idx, n_i in enumerate(N):
        for i in range(n_i):
            if i == 0:
                x = __bottleneck_block(x, filters_list[block_idx], cardinality, strides = 2, weight_decay=weight_decay)
            else:
                x = __bottleneck_block(x, filters_list[block_idx], cardinality, strides = 1, weight_decay=weight_decay)
        
    if pooling == "avg":
        x = tf.keras.layers.GlobalAveragePooling3D()(x)
    elif pooling == "max":
        x = tf.keras.layers.GlobalMaxPooling3D()(x)
    
    return x
