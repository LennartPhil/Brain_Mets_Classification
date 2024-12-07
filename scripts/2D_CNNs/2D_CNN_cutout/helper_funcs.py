import tensorflow as tf
import random
import os
from functools import partial
from pathlib import Path
from time import strftime
import glob
import datetime
import numpy as np

kernel_initializer = "he_normal"
activation_func = "mish"

train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

shuffle_buffer_size = 200
repeat_count = 1

early_stopping_patience = 300 #200

#two_class_weights = {1: 0.92156863, 0 :1.09302326}
two_class_weights = {0: 1.09302326, 1: 0.92156863}

def setup_data(path_to_tfrs, path_to_callbacks, path_to_splits, num_classes, batch_size, rgb = False, current_fold = 0):
    #patients = get_patient_paths(path_to_tfrs)

    #train_paths, val_paths, test_paths = split_patients(patients, path_to_callbacks=path_to_callbacks, fraction_to_use=1)

    train_paths, val_paths = get_patient_paths_for_fold(current_fold, path_to_splits, path_to_tfrs)
    test_paths = get_test_paths(path_to_splits, path_to_tfrs)
    train_paths = get_tfr_paths_for_patients(train_paths)
    val_paths = get_tfr_paths_for_patients(val_paths)
    test_paths = get_tfr_paths_for_patients(test_paths)

    train_data, val_data, test_data = read_data(train_paths, val_paths, num_classes, batch_size, test_paths, rgb = rgb)

    return train_data, val_data, test_data

def get_patient_paths_for_fold(fold, path_to_splits, path_to_tfrs):
    # read .txt file
    txt_train_file_name = f"fold_{fold}_train_ids.txt"
    txt_val_file_name = f"fold_{fold}_val_ids.txt"

    with open(f"{path_to_splits}/{txt_train_file_name}", "r") as f:
        train_patients = [line.strip() for line in f]
        train_patients = [f"{path_to_tfrs}/{pat}" for pat in train_patients]

    with open(f"{path_to_splits}/{txt_val_file_name}", "r") as f:
        val_patients = [line.strip() for line in f]
        val_patients = [f"{path_to_tfrs}/{pat}" for pat in val_patients]

    return train_patients, val_patients

def get_test_paths(path_to_splits, path_to_tfrs):
    # read .txt file
    txt_test_file_name = f"test_ids.txt"

    with open(f"{path_to_splits}/{txt_test_file_name}", "r") as f:
        test_patients = [line.strip() for line in f]
        test_patients = [f"{path_to_tfrs}/{pat}" for pat in test_patients]

    return test_patients

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
                  use_lrscheduler = False,
                  stop_training = False):

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

    if stop_training:
        unfreeze = UnfreezeCallback()
        callbacks.append(unfreeze)

    print("get_callbacks successful")

    return callbacks


#Custom Weighted Cross Entropy Loss
class WeightedCrossEntropyLoss(tf.keras.losses.Loss):
    def __init__(self, class_weights):
        super().__init__()
        self.class_weights = tf.constant(class_weights, dtype=tf.float32)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int64)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        y_true_one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[1])
        cross_entropy = -tf.reduce_sum(y_true_one_hot * tf.math.log(y_pred), axis=-1)
        weights = tf.gather(self.class_weights, y_true)
        weighted_cross_entropy = weights * cross_entropy
        return tf.reduce_mean(weighted_cross_entropy)
        

class UnfreezeCallback(tf.keras.callbacks.Callback):
    def __init__(self, patience=3, monitor='val_accuracy', min_delta=0.01):
        super(UnfreezeCallback, self).__init__()
        self.patience = patience
        self.monitor = monitor
        self.min_delta = min_delta
        self.wait = 0
        self.best = -float('inf')
        self.unfreeze = False

    def on_epoch_end(self, epoch, logs=None):
        #print("Epoch ended")

        current = logs.get(self.monitor)
        if current is None:
            raise ValueError(f"Monitor {self.monitor} is not available in logs.")
        
        if current > self.best + self.min_delta:
            self.best = current
            self.wait = 0
            print("\nnot gonna unfreeze")
        else:
            self.wait += 1
            if self.wait >= self.patience and not self.unfreeze:
                print(f"\nStopping Tranining at epoch {epoch + 1}")

                self.model.stop_training = True

                self.unfreeze = True
                self.wait = 0

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


def print_training_timestamps(isStart, training_codename):
    if isStart:
        print()
        print("_______________________________________________________________________________")
        print()
        print("Starting training: " + training_codename)
        print("at " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print()
        print("_______________________________________________________________________________")
        print()
    else:
        print()
        print("_______________________________________________________________________________")
        print()
        print("Finishing training: " + training_codename)
        print("at " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print()
        print("_______________________________________________________________________________")
        print()



def get_training_codename(code_name, num_classes, clinical_data, is_cutout, is_rgb_images, contrast_DA, is_learning_rate_tuning, is_k_fold, is_upper_layer_training = False):

    training_codename = code_name

    training_codename += f"_{num_classes}_cls"

    if clinical_data:
        training_codename = training_codename + "_with_clin"
    else:
        training_codename = training_codename + "_no_clin"

    if is_cutout:
        training_codename = training_codename + "_cutout"
    else:
        training_codename = training_codename + "_slice"

    if is_rgb_images:
        training_codename += "_rgb"
    else:
        training_codename += "_gray"

    if contrast_DA:
        training_codename = training_codename + "_contrast_DA"
    else:
        training_codename = training_codename + "_normal_DA"

    if is_learning_rate_tuning:
        training_codename = training_codename + "_lr"
    elif is_upper_layer_training:
        training_codename = training_codename + "_upper_layer"
    elif is_k_fold:
        training_codename = training_codename + "_kfold"
    else:
        training_codename = training_codename + "_normal"

    return training_codename

def get_path_to_tfrs(is_cutout, is_rgb_images):
    if is_cutout:
        if is_rgb_images:
            # is cutout with color images
            path_to_tfrs = "/tfrs/all_pats_single_cutout_rgb"
        else:
            # is cutout with gray images
            path_to_tfrs = "/tfrs/all_pats_single_cutout_gray"
    else:
        if is_rgb_images:
            # is brain slice with color images
            path_to_tfrs = "/tfrs/all_pats_single_slice_rgb"
        else:
            # is brain slice with gray images
            path_to_tfrs = "/tfrs/all_pats_single_slice_gray"
    
    return path_to_tfrs

def clear_tf_session():
    tf.keras.backend.clear_session()
    print("Clearing session...")

def print_fold_info(fold, is_start):
    if is_start:
        print()
        print("Starting fold: " + str(fold))
        print("at " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    else:
        print()
        print("Finishing fold: " + str(fold))
        print("at " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def save_training_history(history, training_codename, time, path_to_callbacks, fold = -1):
    history_dict = history.history

    if fold == -1:
        history_file_name = f"history_{training_codename}_{time}.npy"
    else:
        history_file_name = f"history_{training_codename}_fold_{fold}_{time}.npy"

    path_to_np_file = path_to_callbacks / history_file_name
    np.save(path_to_np_file, history_dict)