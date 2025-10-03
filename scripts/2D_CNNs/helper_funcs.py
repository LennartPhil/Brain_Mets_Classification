import tensorflow as tf
import random
import os
from functools import partial
from pathlib import Path
from time import strftime
import glob
import datetime
import numpy as np
import constants
import sys

# --- Data Setup Functions ---

def setup_data(path_to_tfrs, path_to_callbacks, path_to_splits, num_classes, batch_size, selected_indices, rgb = False, use_clinical_data = True, use_layer = True, current_fold = 0):
    #patients = get_patient_paths(path_to_tfrs)

    #train_paths, val_paths, test_paths = split_patients(patients, path_to_callbacks=path_to_callbacks, fraction_to_use=1)

    train_paths, val_paths = get_patient_paths_for_fold(current_fold, path_to_tfrs) #path_to_splits
    test_paths = get_test_paths(path_to_splits, path_to_tfrs)

    # covnert patient directories to list of .tfrecord files
    train_paths = get_tfr_paths_for_patients(train_paths)
    val_paths = get_tfr_paths_for_patients(val_paths)
    test_paths = get_tfr_paths_for_patients(test_paths)

    print(f"Fold {current_fold}: Train {len(train_paths)}, Val {len(val_paths)}, Test {len(test_paths)}")


    train_data, val_data, test_data = read_data(
        train_paths = train_paths,
        val_paths = val_paths,
        selected_indices = selected_indices,
        num_classes = num_classes,
        batch_size = batch_size,
        use_clinical_data = use_clinical_data,
        use_layer = use_layer,
        test_paths = test_paths,
        rgb = rgb,
    )

    return train_data, val_data, test_data


def setup_pretraining_data(path_to_tfrs, batch_size, selected_indices, dataset_type, rgb = False):

    if dataset_type == constants.Dataset.PRETRAIN_FINE:
        # execute code if no pretraining is needed
        train_paths, val_paths = get_patient_paths_for_fold(0, path_to_tfrs, dataset_type = dataset_type)
        train_data, val_data = read_data(train_paths, val_paths, selected_indices, batch_size, dataset_type = dataset_type, rgb = rgb)

    elif dataset_type == constants.Dataset.PRETRAIN_ROUGH:
        # execute code if rough pretraining is needed
        train_path, val_path = constants.paths_to_rough_pretraining
        train_data, val_data = read_data(train_path, val_path, selected_indices, batch_size, dataset_type = dataset_type)

    return train_data, val_data

def get_patient_paths_for_fold(fold, path_to_tfrs, dataset_type = constants.Dataset.NORMAL):
    # read .txt files
    if dataset_type == constants.Dataset.NORMAL:

        txt_train_file_name = f"fold_{fold}_train_ids.txt"
        txt_val_file_name = f"fold_{fold}_val_ids.txt"

        with open(f"{constants.path_to_splits}/{txt_train_file_name}", "r") as f:
            train_patients = [line.strip() for line in f]
            train_patients = [f"{path_to_tfrs}/{pat}" for pat in train_patients]

        with open(f"{constants.path_to_splits}/{txt_val_file_name}", "r") as f:
            val_patients = [line.strip() for line in f]
            val_patients = [f"{path_to_tfrs}/{pat}" for pat in val_patients]

        return train_patients, val_patients

    elif dataset_type == constants.Dataset.PRETRAIN_FINE:

        txt_train_file_name = "pretraining_fine_train_2_classes.txt"
        txt_val_file_name = "pretraining_fine_val_2_classes.txt"

        image_shape = [constants.IMG_SIZE, constants.IMG_SIZE, 5]
        feature_description = {
            "image": tf.io.FixedLenFeature(image_shape, tf.float32),
            "label": tf.io.FixedLenFeature([], tf.int64, default_value=0),
        }

        with open(f"{str(constants.path_to_splits)}/{txt_train_file_name}", "r") as f:
            train_patients = [line.strip() for line in f]
            train_patients = [f"{path_to_tfrs}/{pat}" for pat in train_patients]
            # only keep patients that end with .tfrecord
            train_patients = [pat for pat in train_patients if pat.endswith(".tfrecord")]

            # check if tfrecord is valid
            for pat in train_patients:
                verify_tfrecord(pat, feature_description)

        with open(f"{str(constants.path_to_splits)}/{txt_val_file_name}", "r") as f:
            val_patients = [line.strip() for line in f]
            val_patients = [f"{path_to_tfrs}/{pat}" for pat in val_patients]
            # only keep patients that end with .tfrecord
            val_patients = [pat for pat in val_patients if pat.endswith(".tfrecord")]

            # check if tfrecord is valid
            for pat in val_patients:
                verify_tfrecord(pat, feature_description)

        return train_patients, val_patients
    
    else:
        # raise error
        raise ValueError(f"Invalid pretraining type: {dataset_type}")

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


# def get_tfr_paths_for_patients(patient_paths):

#     tfr_paths = []

#     for patient in patient_paths:
#         tfr_paths.extend(glob.glob(patient + "/*.tfrecord"))
    
#     for path in tfr_paths:
#         verify_tfrecord(path)

#     #print(f"total tfrs: {len(tfr_paths)}")

#     return tfr_paths

def get_tfr_paths_for_patients(patient_paths):
    """
    Gathers all .tfrecord paths for a list of patient directories and verifies
    each one, returning only the list of valid, uncorrupted file paths.
    """
    # Define the full feature description that all files are expected to have.
    # This must match the schema the files were created with.
    image_shape = [constants.IMG_SIZE, constants.IMG_SIZE, 5]
    feature_description = {
        "image": tf.io.FixedLenFeature(image_shape, tf.float32),
        "sex": tf.io.FixedLenFeature([], tf.int64, default_value=0),
        "age": tf.io.FixedLenFeature([], tf.int64, default_value=constants.AGE_MIN),
        "layer": tf.io.FixedLenFeature([], tf.int64, default_value=constants.LAYER_MIN),
        "primary": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    }

    all_tfr_paths = []
    for patient in patient_paths:
        all_tfr_paths.extend(glob.glob(patient + "/*.tfrecord"))

    print(f"Found {len(all_tfr_paths)} total .tfrecord files. Verifying integrity and content schema...")
    
    # Pass the feature_description to the new verification function
    verified_tfr_paths = [
        path for path in all_tfr_paths if verify_tfrecord(path, feature_description)
    ]

    num_removed = len(all_tfr_paths) - len(verified_tfr_paths)
    if num_removed > 0:
        print(f"WARNING: Removed {num_removed} invalid or corrupted .tfrecord file(s) from the dataset.")

    return verified_tfr_paths


def read_data(train_paths, val_paths, selected_indices, batch_size, num_classes = None, test_paths = None, rgb = False, use_clinical_data = True, use_layer = True, dataset_type = constants.Dataset.NORMAL):

    if dataset_type != constants.Dataset.PRETRAIN_ROUGH:

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
            deterministic=True
        )

        if dataset_type == constants.Dataset.NORMAL:
            train_parse_partial = partial(parse_record,
                                          selected_indices = selected_indices,
                                          dataset_type = dataset_type,
                                          use_clinical_data = use_clinical_data,
                                          use_layer = use_layer,
                                          labeled = True,
                                          num_classes = num_classes,
                                          rgb = rgb)
            
            val_parse_partial = partial(parse_record,
                                        selected_indices = selected_indices,
                                        dataset_type = dataset_type,
                                        use_clinical_data = use_clinical_data,
                                        use_layer = use_layer,
                                        labeled = True,
                                        num_classes = num_classes,
                                        rgb = rgb)

            train_data = train_data.map(train_parse_partial, num_parallel_calls=tf.data.AUTOTUNE)
            val_data = val_data.map(val_parse_partial, num_parallel_calls=tf.data.AUTOTUNE)

        elif dataset_type == constants.Dataset.PRETRAIN_FINE:
            # train_data = train_data.map(partial(parse_record, selected_indices, dataset_type = dataset_type, use_clinical_data = False, use_layer = False, labeled = True, num_classes = num_classes, rgb = rgb), num_parallel_calls=tf.data.AUTOTUNE)
            # val_data = val_data.map(partial(parse_record, selected_indices, dataset_type = dataset_type, use_clinical_data = False, use_layer = False, labeled = True, num_classes = num_classes, rgb = rgb), num_parallel_calls=tf.data.AUTOTUNE)
            train_parse_partial = partial(parse_fine_pretraining_record,
                                          selected_indices = selected_indices,
                                          labeled = True,
                                          num_classes = num_classes,
                                          rgb = rgb)
            
            val_parse_partial = partial(parse_fine_pretraining_record,
                                        selected_indices = selected_indices,
                                        labeled = True,
                                        num_classes = num_classes,
                                        rgb = rgb)

            train_data = train_data.map(train_parse_partial, num_parallel_calls=tf.data.AUTOTUNE)
            val_data = val_data.map(val_parse_partial, num_parallel_calls=tf.data.AUTOTUNE)

    else: # ROUGH pretraining
        
        train_data = tf.data.TFRecordDataset([train_paths], compression_type="GZIP")
        val_data = tf.data.TFRecordDataset([val_paths], compression_type="GZIP")

        feature_description = {
            "image": tf.io.FixedLenFeature([constants.IMG_SIZE, constants.IMG_SIZE, constants.ROUGH_NUM_IMAGES], tf.float32),
            "label": tf.io.FixedLenFeature([], tf.int64, default_value=0),
        }

        def rough_parse(serialize_tumor):
            example = tf.io.parse_single_example(serialize_tumor, feature_description)
            image = example["image"]
            #image = tf.reshape(image, [constants.IMG_SIZE, constants.IMG_SIZE, rough_num_images])
            
            return image, example["label"]
        
        train_data = train_data.map(rough_parse, num_parallel_calls=tf.data.AUTOTUNE)
        val_data = val_data.map(rough_parse, num_parallel_calls=tf.data.AUTOTUNE)


    train_data = train_data.shuffle(buffer_size=constants.shuffle_buffer_size)
    #val_data = val_data.shuffle(buffer_size=constants.shuffle_buffer_size)

    train_data = train_data.repeat(count = constants.repeat_count)
    #val_data = val_data.repeat(count = constants.repeat_count)

    train_data = train_data.batch(batch_size)
    val_data = val_data.batch(batch_size)

    train_data = train_data.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_data = val_data.prefetch(buffer_size=tf.data.AUTOTUNE)

    if test_paths is not None:
        test_data = tf.data.Dataset.from_tensor_slices(test_paths)
        test_data = test_data.interleave(
            lambda x: tf.data.TFRecordDataset([x], compression_type="GZIP"),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True
        )

        test_parse_partial = partial(parse_record,
                                    selected_indices = selected_indices,
                                    dataset_type = dataset_type,
                                    use_clinical_data = use_clinical_data,
                                    use_layer = use_layer,
                                    labeled = True,
                                    num_classes = num_classes,
                                    rgb = rgb)


        test_data = test_data.map(test_parse_partial, num_parallel_calls=tf.data.AUTOTUNE)
        test_data = test_data.batch(batch_size)
        test_data = test_data.prefetch(buffer_size=1)

        return train_data, val_data, test_data

    return train_data, val_data


def parse_record(record, selected_indices = [0, 1, 2, 3, 4], dataset_type = constants.Dataset.NORMAL, use_clinical_data = True, use_layer = False, labeled = False, num_classes = 2, rgb = False):

    image_shape = [constants.IMG_SIZE, constants.IMG_SIZE, 5]

    feature_description = {
        "image": tf.io.FixedLenFeature(image_shape, tf.float32),
        "sex": tf.io.FixedLenFeature([], tf.int64, default_value=[0]),
        "age": tf.io.FixedLenFeature([], tf.int64, default_value=constants.AGE_MIN),
        "layer": tf.io.FixedLenFeature([], tf.int64, default_value=constants.LAYER_MIN),
        "primary": tf.io.FixedLenFeature([], tf.int64, default_value=0), # actual label
    }

    example = tf.io.parse_single_example(record, feature_description)

    # --- Image Channel Selection ---
    full_image = example["image"]
    full_image = tf.reshape(full_image, image_shape)

    if rgb:
        # RGB mode: Select one channel and tile to 3 channels
        # selected_indices should contain exactly one index (validated in main script)
        tf.debugging.assert_equal(tf.shape(selected_indices)[0], 1,
                                  message = "RGB mode requires exactly one selected index in the parser.")
        single_channel_image = tf.gather(full_image, selected_indices, axis=-1)
        # tile the single channel 3 times along the last axis (shape becomes [H, W, 3])
        image = tf.tile(single_channel_image, [1, 1, 3])
    else:
        # Grayscale mode: Gather the specified channels (shape becomes [H, W, num_selected_channels])
        image = tf.gather(full_image, selected_indices, axis=-1)

    # scale age and layer
    # the values also get clipped to [0, 1]
    scaled_age = min_max_scale(example["age"], constants.AGE_MIN, constants.AGE_MAX)
    scaled_layer = min_max_scale(example["layer"], constants.LAYER_MIN, constants.LAYER_MAX)

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
        raise ValueError(f"num_classes must have a value between 2 and 6, got {num_classes}")


    # cast to float32 if num_classes is 2, else int64
    if num_classes == 2:
        primary_to_return = tf.cast(primary_to_return, tf.float32)
        # primary_to_return = tf.expand_dims(primary_to_return, axis=-1) # for binary classification with sigmoid output

    sex_float = tf.cast(example["sex"], tf.float32)

    # if use_clinical_data is False and use_layer is False, return only the image and the primary
    if use_clinical_data == False and use_layer == False:
        return image, primary_to_return
    
    # if use_clinical_data is False and use_layer is True, return only the image, the layer and the primary
    elif use_clinical_data == False and use_layer:
        return (image, scaled_layer), primary_to_return
    
    # if use_clinical_data is True and use_layer is False, return the image, the sex, the age and the primary
    elif use_clinical_data and use_layer == False and labeled:
        return (image, sex_float, scaled_age), primary_to_return #example["primary"]
    
    # if use_clinical_data is True and use_layer is True and labeled, return the image, the sex, the age, the layer and the primary
    elif use_clinical_data and use_layer and labeled:
        return (image, sex_float, scaled_age, scaled_layer), primary_to_return

    # if use_clinical_data is True and use_layer is True and not labeled, return the image, the sex, the age and the layer, not the primary!
    elif use_clinical_data and use_layer and not labeled:
        return image, sex_float, scaled_age, scaled_layer

    else:
        return image


def parse_fine_pretraining_record(record, selected_indices = [0, 1, 2, 3, 4], labeled = False, num_classes = 4, rgb = False):
    
    # # --- Add Debugging ---
    # tf.print("--- Debug parse_record ---", output_stream=sys.stderr)
    # tf.print("Record input type:", type(record), output_stream=sys.stderr)
    # # Check if it's even a tensor before accessing properties
    # if isinstance(record, tf.Tensor):
    #      tf.print("Record shape:", tf.shape(record), output_stream=sys.stderr)
    #      tf.print("Record rank:", tf.rank(record), output_stream=sys.stderr)
    #      tf.print("Record dtype:", record.dtype, output_stream=sys.stderr)
    #      # Try to assert properties if it is a Tensor
    #      tf.debugging.assert_scalar(record, message="Assertion Failure: Record is not scalar!")
    #      tf.debugging.assert_equal(record.dtype, tf.string, message="Assertion Failure: Record is not string!")
    # else:
    #      tf.print("Record is not a Tensor!", output_stream=sys.stderr)
    # tf.print("--- End Debug ---", output_stream=sys.stderr)


    image_shape = [constants.IMG_SIZE, constants.IMG_SIZE, 5]

    feature_description = {
        "image": tf.io.FixedLenFeature(image_shape, tf.float32),
        "label": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    }

    example = tf.io.parse_single_example(record, feature_description)

    # --- Image Channel Selection ---
    full_image = example["image"]
    full_image = tf.reshape(full_image, image_shape)

    if rgb:
        # RGB mode: Select one channel and tile to 3 channels
        # selected_indices should contain exactly one index (validated in main script)
        tf.debugging.assert_equal(tf.shape(selected_indices)[0], 1,
                                  message = "RGB mode requires exactly one selected index in the parser.")
        single_channel_image = tf.gather(full_image, selected_indices, axis=-1)
        # tile the single channel 3 times along the last axis (shape becomes [H, W, 3])
        image = tf.tile(single_channel_image, [1, 1, 3])
    else:
        # Grayscale mode: Gather the specified channels (shape becomes [H, W, num_selected_channels])
        image = tf.gather(full_image, selected_indices, axis=-1)

    original_label = example["label"]

    # Map original labels {0, 4} to new labels {0, 1}
    if original_label == tf.constant(4, dtype=tf.int64):
        label = tf.constant(1, dtype=tf.int64)
    else:
        label = original_label

    primary_to_return = label


    if labeled:
        primary_to_return = tf.cast(primary_to_return, tf.float32)
        # primary_to_return = tf.expand_dims(primary_to_return, axis=-1) # for binary classification with sigmoid output
        return image, primary_to_return
    else:
        return image


# def verify_tfrecord(file_path):
#     try:
#         for _ in tf.data.TFRecordDataset(file_path, compression_type="GZIP"):
#             pass
#         return True
#     except tf.errors.DataLossError as e:
#         print(f"Corrupted TFRecord file: {file_path}\n{e}")
#         return False
#     except Exception as e:
#         print(f"Error verifying TFRecord file: {file_path}\nError: {e}")
#         return False

def verify_tfrecord(file_path, feature_description):
    """
    Verifies a TFRecord file for both structural integrity (DataLossError)
    and content validity (InvalidArgumentError by attempting to parse).
    """
    try:
        # Create a dataset from the single file
        dataset = tf.data.TFRecordDataset(str(file_path), compression_type="GZIP")

        # Define a simple parsing function to use for verification
        def _parse_function(proto):
            return tf.io.parse_single_example(proto, feature_description)

        # Map the parsing function. This will fail if content is invalid.
        parsed_dataset = dataset.map(_parse_function)

        # Iterate through the parsed dataset to trigger the execution
        for _ in parsed_dataset:
            pass
        
        # If the loop completes, the file is valid in structure and content
        return True

    except tf.errors.DataLossError as e:
        print(f"\n!!! CORRUPTION DETECTED (Structural Error) !!!")
        print(f"File: {file_path}")
        print(f"Error: {e}\n")
        return False
    except tf.errors.InvalidArgumentError as e:
        print(f"\n!!! CORRUPTION DETECTED (Content/Schema Error) !!!")
        print(f"File: {file_path}")
        print(f"Error: This file is structurally valid but its content does not match the expected schema. {e}\n")
        return False
    except Exception as e:
        # Catch other potential errors
        print(f"\n!!! UNEXPECTED VERIFICATION ERROR !!!")
        print(f"File: {file_path}")
        print(f"Error: {e}\n")
        return False


def get_callbacks(path_to_callbacks,
                  fold_num = 0,
                  use_checkpoint = True,
                  use_early_stopping = True,
                  early_stopping_patience = constants.early_stopping_patience,
                  use_tensorboard = True,
                  use_csv_logger = True,
                  use_lrscheduler = False):

    callbacks = []

    path_to_fold_callbacks = path_to_callbacks / f"fold_{fold_num}"
    os.makedirs(path_to_fold_callbacks, exist_ok=True)

    # def get_run_logdir(root_logdir = path_to_fold_callbacks / "tensorboard"):
    #     return Path(root_logdir) / strftime("run_%Y_%m_%d_%H_%M_%S")

    # run_logdir = get_run_logdir()

    # model checkpoint
    if use_checkpoint:
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath = path_to_fold_callbacks / "saved_weights.weights.h5",
            monitor = "val_accuracy",
            mode = "max",
            save_best_only = True,
            save_weights_only = True
        )
        callbacks.append(checkpoint_cb)

    # early stopping
    if use_early_stopping:
        early_stopping_cb = tf.keras.callbacks.EarlyStopping(
            monitor = "val_loss",
            patience = early_stopping_patience,
            restore_best_weights = True,
            verbose = 1
        )
        callbacks.append(early_stopping_cb)

    # tensorboard, doesn't really work yet
    if use_tensorboard:
        # The log directory is structured as follows:
        # .../logs/<run_name>/fold_0/tensorboard/
        # .../logs/<run_name>/fold_1/tensorboard/
        # etc.
        tensorboard_log_dir = path_to_fold_callbacks / "tensorboard"
        tensorboard_cb = tf.keras.callbacks.TensorBoard(
            log_dir = tensorboard_log_dir,
            histogram_freq = 1
        )
        callbacks.append(tensorboard_cb)
    
    # csv logger
    if use_csv_logger:
        csv_logger_cb = tf.keras.callbacks.CSVLogger(
            path_to_fold_callbacks / "training.csv",
            separator = ",",
            append = False
        )
        callbacks.append(csv_logger_cb)
    
    # Learning Rate Scheduler
    if use_lrscheduler:
        lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch * 0.0175))
        callbacks.append(lr_schedule)

    print("get_callbacks successful")

    return callbacks


# --- Utility functions and layers ---

def min_max_scale(data, min_val, max_val):
    """
    Min-Max scaling with fixed min and max values.
    Args:
        data: Input tensor to be scaled.
        min_val: Fixed minimum value.
        max_val: Fixed maximum value.
    Returns:
        scaled_data: Tensor scaled to the range [0, 1].
    """
    data = tf.cast(data, tf.float32)
    min_val = tf.cast(min_val, tf.float32)
    max_val = tf.cast(max_val, tf.float32)

    scaled_data = (data - min_val) / (max_val - min_val)
    clipped_data = tf.clip_by_value(scaled_data, 0.0, 1.0)

    return clipped_data


class NormalizeToRange(tf.keras.layers.Layer):
    """Layer to normalize input tensor values to [0, 1] or [-1, 1]."""
    def __init__(self, zero_to_one=True, epsilon=1e-7, **kwargs):
        super().__init__(**kwargs) #super(NormalizeToRange, self).__init__()
        self.zero_to_one = zero_to_one
        self.epsilon = epsilon

    def call(self, inputs):
        min_val = tf.reduce_min(inputs)
        max_val = tf.reduce_max(inputs)

        range_val = max_val - min_val
        range_val = tf.maximum(range_val, self.epsilon)

        if self.zero_to_one:
            # Normalize to [0, 1]
            normalized = (inputs - min_val) / range_val
        else:
            # Normalize to [-1, 1]
            normalized = 2 * (inputs - min_val) / range_val - 1
        return normalized

# Custom Data Augmentation Layers
class RandomRescale(tf.keras.layers.Layer):
    def __init__(self, scale_range=(0.8, 1.2), **kwargs):
        """
        Custom layer for random rescaling of images.
        Args:
            scale_range (tuple): A tuple specifying the minimum and maximum scaling factors.
                                 Values < 1.0 zoom out, and > 1.0 zoom in.
        """
        super(RandomRescale, self).__init__(**kwargs)
        self.scale_range = scale_range

    def call(self, inputs, training=None):
        if training:
            # Randomly choose a scaling factor
            scale = tf.random.uniform([], self.scale_range[0], self.scale_range[1])
            
            # Get image dimensions
            input_shape = tf.shape(inputs)
            height, width = input_shape[1], input_shape[2]

            # For testing without the batch size
            #height, width = input_shape[0], input_shape[1]
            
            # Compute new dimensions
            new_height = tf.cast(tf.cast(height, tf.float32) * scale, tf.int32)
            new_width = tf.cast(tf.cast(width, tf.float32) * scale, tf.int32)
            
            # Resize image to new dimensions
            scaled_image = tf.image.resize(inputs, [new_height, new_width])
            
            # Crop or pad to original size
            scaled_image = tf.image.resize_with_crop_or_pad(scaled_image, height, width)
            
            return scaled_image
        else:
            return inputs

    def get_config(self):
        config = super(RandomRescale, self).get_config()
        config.update({"scale_range": self.scale_range})
        return config

# Data Augmentation
# one with brightness and contrast adjustments, one without

contrast_data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip(mode = "horizontal"),
        tf.keras.layers.RandomContrast(0.5), # consider removing the random contrast layer as that causes pixel values to go beyond 1
        tf.keras.layers.RandomBrightness(factor = (-0.2, 0.4)), #, value_range=(0, 1)
        tf.keras.layers.RandomRotation(factor = (-0.1, 0.1), fill_mode = "nearest"),
        NormalizeToRange(zero_to_one=True),
        tf.keras.layers.RandomTranslation(
            height_factor = 0.05,
            width_factor = 0.05,
            fill_mode = "nearest",
            interpolation = "bilinear"
        ),
        RandomRescale(scale_range=(0.7, 1.2))
    ], name = "contrast_data_augmentation")

normal_data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip(mode = "horizontal"),
        tf.keras.layers.RandomRotation(factor = (-0.14, 0.14), fill_mode = "nearest"),
        NormalizeToRange(zero_to_one=True),
        tf.keras.layers.RandomTranslation(
            height_factor = 0.05,
            width_factor = 0.05,
            fill_mode = "nearest",
            interpolation = "bilinear"
        ),
        RandomRescale(scale_range=(0.7, 1.2))
    ], name = "normal_data_augmentation")


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
                               kernel_initializer = constants.kernel_initializer,
                               kernel_regularizer = tf.keras.regularizers.l2(weight_decay))(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(constants.activation_func)(x)

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
                                   kernel_initializer = constants.kernel_initializer,
                                   kernel_regularizer = tf.keras.regularizers.l2(weight_decay))(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(constants.activation_func)(x)

        return x
    
    # cardinality loop
    for c in range(cardinality):
        x = tf.keras.layers.Lambda(lambda x: x[:, :, :, :, c * grouped_channels : (c + 1) * grouped_channels])(input)

        x = tf.keras.layers.Conv3D(filters = grouped_channels,
                                   kernel_size = 3,
                                   padding = "same",
                                   use_bias = False,
                                   strides = (strides, strides, strides),
                                   kernel_initializer = constants.kernel_initializer,
                                   kernel_regularizer = tf.keras.regularizers.l2(weight_decay))(x)
        
        group_list.append(x)
    
    group_merge = tf.keras.layers.Concatenate(axis=-1)(group_list)
    x = tf.keras.layers.BatchNormalization()(group_merge)
    x = tf.keras.layers.Activation(constants.activation_func)(x)

    return x


def __bottleneck_block(input, filters, cardinality, strides, weight_decay):
    init = input

    # Determine if the shortcut path needs a convolution for matching dimensions
    needs_conv = strides > 1 or input.shape[-1] != filters * 2

    grouped_channels = filters // cardinality
    
    if needs_conv:
        # Apply convolution to shortcut path to match the main path's dimensions
        init = tf.keras.layers.Conv3D(filters * 2, 1, strides=strides, padding="same", use_bias=False,
                                      kernel_initializer=constants.kernel_initializer,
                                      kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(init)
        init = tf.keras.layers.BatchNormalization()(init)

    # Main path
    x = tf.keras.layers.Conv3D(filters, 1, padding="same", use_bias=False,
                               kernel_initializer=constants.kernel_initializer,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(constants.activation_func)(x)

    x = __grouped_convolution_block(x, grouped_channels, cardinality, strides, weight_decay)

    x = tf.keras.layers.Conv3D(filters * 2, 1, padding="same", use_bias=False,
                               kernel_initializer=constants.kernel_initializer,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Addition - ensuring init and x have compatible shapes
    x = tf.keras.layers.Add()([init, x])
    x = tf.keras.layers.Activation(constants.activation_func)(x)

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



def get_training_codename(code_name, num_classes, clinical_data, use_layer, is_cutout, is_rgb_images, selected_sequences_str,contrast_DA, dataset_type, training_mode):
    
    training_codename = f"{code_name}_{num_classes}cls"

    training_codename += "_cuout" if is_cutout else "_slice"
    training_codename += "_clin" if clinical_data else "_no_clin"
    training_codename += "_layer" if use_layer else "_no_layer"
    training_codename += "_rgb" if is_rgb_images else "_gray"

    training_codename += f"_seq[{selected_sequences_str}]" # e.g., _seq[t1c-flair-mask]

    training_codename += "_contrast_DA" if contrast_DA else "_normal_DA"

    if dataset_type == constants.Dataset.PRETRAIN_FINE:
        training_codename = training_codename + "_pretrain_fine"
    elif dataset_type == constants.Dataset.PRETRAIN_ROUGH:
        training_codename = training_codename + "_pretrain_rough"

    if training_mode == constants.Training.LEARNING_RATE_TUNING:
        training_codename = training_codename + "_lr"
    elif training_mode == constants.Training.UPPER_LAYER:
        training_codename = training_codename + "_upper_layer"
    elif training_mode == constants.Training.K_FOLD:
        training_codename = training_codename + "_kfold"
    elif training_mode == constants.Training.NORMAL:
        training_codename = training_codename + "_normal"
    else:
        raise(ValueError(f"Invalid training mode for codename: {training_mode}"))

    return training_codename

def get_path_to_tfrs(is_rgb_images, is_cutout = False, dataset_type = constants.Dataset.NORMAL):
    if dataset_type == constants.Dataset.NORMAL:
        if is_cutout:
            if is_rgb_images:
                # is cutout with color images
                path_to_tfrs = constants.path_to_tfr_dirs / "all_pats_single_cutout_rgb"
            else:
                # is cutout with gray images
                path_to_tfrs = constants.path_to_tfr_dirs / "all_pats_single_cutout_gray"
        else:
            path_to_tfrs = constants.path_to_tfr_dirs / "all_pats_single_slice_gray"
            # if is_rgb_images:
            #     # is brain slice with color images
            #     path_to_tfrs = constants.path_to_tfr_dirs / "all_pats_single_slice_rgb"
            # else:
            #     # is brain slice with gray images
            #     path_to_tfrs = constants.path_to_tfr_dirs / "all_pats_single_slice_gray"
    
    elif dataset_type == constants.Dataset.PRETRAIN_FINE:
        path_to_tfrs = constants.path_to_fine_pretraining / "pretraining_fine_gray_2_classes"
    
    else:
        return None
    
    return path_to_tfrs

def clear_tf_session():
    tf.keras.backend.clear_session()

    import gc
    gc.collect()

    print("Clearing session and ran garbage collection")

def print_fold_info(fold, is_start):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if is_start:
        print(f"\n--- Starting Fold: {fold} at {now} ---")
    else:
        print(f"\n--- Finishing Fold: {fold} at {now} ---")


def save_training_history(history, training_codename, time, path_to_callbacks, fold = -1):
    history_dict = history.history

    if fold == -1:
        history_file_name = f"history_{training_codename}_{time}.npy"
    else:
        history_file_name = f"history_{training_codename}_fold_{fold}_{time}.npy"

    path_to_np_file = path_to_callbacks / history_file_name
    np.save(path_to_np_file, history_dict)





def check_dataset(dataset, dataset_name, batch_size, expected_input_shape,
                  num_classes, use_clinical_data, use_layer, dataset_type,
                  num_batches_to_check=1):
    """
    Checks the structure, shapes, and types of batches yielded by a tf.data.Dataset.

    Args:
        dataset: The tf.data.Dataset object to check (e.g., train_data, val_data).
        dataset_name: A string name for the dataset (e.g., "Training", "Validation").
        batch_size: The expected batch size.
        expected_input_shape: The expected shape of the image tensor (H, W, C).
        num_classes: The expected number of classes.
        use_clinical_data: Boolean flag indicating if clinical data is expected.
        use_layer: Boolean flag indicating if layer data is expected.
        dataset_type: The constants.Dataset enum value.
        num_batches_to_check: How many batches to fetch and inspect.
    """
    print(f"\n--- Checking Dataset: {dataset_name} ---")
    print(f"Expected Config: Batch={batch_size}, ImgShape={expected_input_shape}, Classes={num_classes}, Clinical={use_clinical_data}, Layer={use_layer}, Type={dataset_type.name}")

    iterator = iter(dataset.take(num_batches_to_check))
    batch_count = 0

    try:
        for batch in iterator:
            batch_count += 1
            print(f"\n--- Inspecting Batch {batch_count} ---")

            # 1. Check Overall Batch Structure (Inputs, Labels, [Weights])
            batch_len = len(batch)
            print(f"Batch tuple length: {batch_len}")
            if batch_len < 2 or batch_len > 3:
                print(f"ERROR: Unexpected batch tuple length. Expected 2 or 3, got {batch_len}.")
                # Attempt to print elements anyway for debugging
                for i, elem in enumerate(batch):
                   print(f"  Element {i} type: {type(elem)}")
                   if hasattr(elem, 'shape'):
                       print(f"  Element {i} shape: {elem.shape}")
                   if hasattr(elem, 'dtype'):
                       print(f"  Element {i} dtype: {elem.dtype}")
                break # Stop checking after structural error

            inputs_batch = batch[0]
            labels_batch = batch[1]
            weights_batch = batch[2] if batch_len == 3 else None

            # 2. Check Inputs Structure
            is_input_tuple = isinstance(inputs_batch, (tuple, list))
            print(f"Inputs is a tuple/list: {is_input_tuple}")

            expected_num_inputs = 1
            if use_clinical_data:
                expected_num_inputs += 2 # sex, age
            if use_layer:
                expected_num_inputs += 1 # layer

            # For pretraining, inputs might be simpler
            if dataset_type == constants.Dataset.PRETRAIN_FINE or dataset_type == constants.Dataset.PRETRAIN_ROUGH:
                 is_input_tuple = False # Expecting only image for these types usually
                 expected_num_inputs = 1

            image_batch = None
            if is_input_tuple:
                actual_num_inputs = len(inputs_batch)
                print(f"  Number of elements in inputs tuple: {actual_num_inputs}")
                if actual_num_inputs != expected_num_inputs:
                    print(f"  ERROR: Unexpected number of input elements. Expected {expected_num_inputs}, got {actual_num_inputs}.")
                    # Print details of tuple elements
                    for i, elem in enumerate(inputs_batch):
                        print(f"    Input Element {i} type: {type(elem)}")
                        if hasattr(elem, 'shape'): print(f"    Input Element {i} shape: {elem.shape}")
                        if hasattr(elem, 'dtype'): print(f"    Input Element {i} dtype: {elem.dtype}")

                if actual_num_inputs > 0:
                     image_batch = inputs_batch[0] # Image should always be first

                # Check other inputs if expected
                input_idx = 1
                if use_clinical_data:
                    if actual_num_inputs > input_idx:
                        print(f"  Clinical Sex Shape: {inputs_batch[input_idx].shape}, Dtype: {inputs_batch[input_idx].dtype}") # Expect (batch_size,) or (batch_size, 1)
                    input_idx += 1
                    if actual_num_inputs > input_idx:
                        print(f"  Clinical Age Shape: {inputs_batch[input_idx].shape}, Dtype: {inputs_batch[input_idx].dtype}") # Expect (batch_size,) or (batch_size, 1), float32
                    input_idx += 1
                if use_layer:
                     if actual_num_inputs > input_idx:
                        print(f"  Layer Shape: {inputs_batch[input_idx].shape}, Dtype: {inputs_batch[input_idx].dtype}") # Expect (batch_size,) or (batch_size, 1), float32
                     input_idx += 1

            else: # Inputs should be a single tensor (the image)
                if use_clinical_data or use_layer and dataset_type == constants.Dataset.NORMAL:
                    print(f"  WARNING: Inputs is a single tensor, but clinical/layer data was expected for NORMAL dataset type.")
                image_batch = inputs_batch

            # 3. Check Image Batch
            if image_batch is None:
                 print("  ERROR: Could not identify image batch.")
            else:
                print(f"Image Batch Shape: {image_batch.shape}") # Expect (batch_size, H, W, C)
                print(f"Image Batch Dtype: {image_batch.dtype}")  # Expect float32
                if len(image_batch.shape) != 4:
                    print(f"  ERROR: Image batch should be 4D (Batch, H, W, C), but got {len(image_batch.shape)}D.")
                elif image_batch.shape[0] != batch_size and image_batch.shape[0] is not None: # Allow for last partial batch
                     print(f"  WARNING: Image batch size ({image_batch.shape[0]}) doesn't match expected ({batch_size}). (May be last partial batch)")
                # Check image dimensions H, W, C
                if image_batch.shape[1] != expected_input_shape[0] or \
                   image_batch.shape[2] != expected_input_shape[1] or \
                   image_batch.shape[3] != expected_input_shape[2]:
                   print(f"  ERROR: Image batch dimensions {image_batch.shape[1:]} don't match expected {expected_input_shape}.")

                # Optional: Check value range (if augmentation/normalization applied)
                img_min = tf.reduce_min(image_batch)
                img_max = tf.reduce_max(image_batch)
                print(f"Image Batch Value Range: [{img_min:.4f}, {img_max:.4f}]")

            # 4. Check Labels Batch
            print(f"Labels Batch Shape: {labels_batch.shape}") # Expect (batch_size,)
            print(f"Labels Batch Dtype: {labels_batch.dtype}")  # Expect int64 or int32
            if len(labels_batch.shape) != 1:
                 print(f"  ERROR: Labels batch should be 1D (Batch,), but got {len(labels_batch.shape)}D.")
            elif labels_batch.shape[0] != batch_size and labels_batch.shape[0] is not None:
                 print(f"  WARNING: Labels batch size ({labels_batch.shape[0]}) doesn't match expected ({batch_size}). (May be last partial batch)")

            # Optional: Check label values
            unique_labels, _ = tf.unique(tf.reshape(labels_batch, [-1]))
            print(f"Unique labels in batch: {unique_labels.numpy()}")
            if tf.reduce_max(unique_labels) >= num_classes or tf.reduce_min(unique_labels) < 0:
                 print(f"  ERROR: Labels found outside expected range [0, {num_classes-1}].")

            # 5. Check Weights Batch (if present)
            if weights_batch is not None:
                print(f"Weights Batch Shape: {weights_batch.shape}") # Expect (batch_size,)
                print(f"Weights Batch Dtype: {weights_batch.dtype}")  # Expect float32
                if len(weights_batch.shape) != 1:
                    print(f"  ERROR: Weights batch should be 1D (Batch,), but got {len(weights_batch.shape)}D.")
                elif weights_batch.shape[0] != batch_size and weights_batch.shape[0] is not None:
                     print(f"  WARNING: Weights batch size ({weights_batch.shape[0]}) doesn't match expected ({batch_size}). (May be last partial batch)")
                # Optional: Check weight values
                w_min = tf.reduce_min(weights_batch)
                w_max = tf.reduce_max(weights_batch)
                print(f"Weights Batch Value Range: [{w_min:.4f}, {w_max:.4f}]")
            elif batch_len == 3:
                 print("  WARNING: Batch tuple length was 3, but weights_batch is None.")


    except tf.errors.OutOfRangeError:
        print("\nDataset exhausted before checking specified number of batches.")
    except Exception as e:
        print(f"\n--- ERROR encountered during dataset checking ---")
        import traceback
        traceback.print_exc() # Print detailed exception traceback

    if batch_count == 0:
         print("\nERROR: Could not iterate through any batches. Dataset might be empty or setup failed.")

    print(f"\n--- Finished Checking Dataset: {dataset_name} ---")