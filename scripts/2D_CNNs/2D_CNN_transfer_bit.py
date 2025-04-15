import tensorflow as tf
import tensorflow_hub as hub
import helper_funcs as hf
from pathlib import Path
import os
from time import strftime
from functools import partial
import numpy as np
import constants

# So the idea with the pretrained models is the following:
# 1. Run: adjust the upper layers with trainable turned off
# !before starting 2nd run: define the path_to_weights below!
# 2. Run: find best learning rate with exported weights
# 3. Run: train of the best learning rate

# IMPORTANT NOTE: this script only accepts 3 channels as input!!!

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
dataset_type = constants.Dataset.PRETRAIN_ROUGH # PRETRAIN_ROUGH, PRETRAIN_FINE, NORMAL
training_mode = constants.Training.UPPER_LAYER # LEARNING_RATE_TUNING, NORMAL, K_FOLD, UPPER_LAYER

cutout = False
rgb_images = False # using gray scale images as input
contrast_DA = False # data augmentation with contrast
clinical_data = False
use_layer = False
num_classes = 2

# --- Select Sequences ---
selected_sequences = ["t1", "t1c", "t2", "flair", "mask"]

if dataset_type == constants.Dataset.PRETRAIN_ROUGH:
    num_classes = 3
    cutout = False
    clinical_data = False
    use_layer = False
    rgb_images = True
    selected_sequences = ["t1c"]
    if training_mode == constants.Training.K_FOLD:
        raise ValueError(f"For PRETRAIN_ROUGH dataset, only UPPER_LAYER, LEARNING_RATE_TUNING and NORMAL training modes are supported. Current mode: {training_mode}")
        

elif dataset_type == constants.Dataset.PRETRAIN_FINE:
    num_classes = 2 # we only fine train on the glioblastoma and brain metatases, meaning we'll only need 2 classes
    cutout = False
    clinical_data = False
    use_layer = False
    if training_mode == constants.Training.K_FOLD:
        raise ValueError(f"For PRETRAIN_ROUGH dataset, only UPPER_LAYER, LEARNING_RATE_TUNING and NORMAL training modes are supported. Current mode: {training_mode}")

if rgb_images == True and len(selected_sequences) > 1:
    raise ValueError(f"RGB images cannot be used when multiple sequences are selected, selected sequences: {selected_sequences}. Please select only 1 sequence.")


try:
    selected_indices = [constants.SEQUENCE_TO_INDEX[name] for name in selected_sequences]
    num_selected_channels = len(selected_indices)
    if num_selected_channels == 0:
        raise ValueError("selected_sequences cannot be empty.")
    if num_selected_channels == 1 and rgb_images == True:
        input_shape = (constants.IMG_SIZE, constants.IMG_SIZE, 3)
    else:
        if rgb_images == True:
            raise ValueError(f"RGB images cannot be used when multiple sequences are selected, selected sequences: {selected_sequences}")
        input_shape = (constants.IMG_SIZE, constants.IMG_SIZE, num_selected_channels)
    print(f"Using sequences: {selected_sequences} -> Indices: {selected_indices}")
    print(f"Derived input shape: {input_shape}, using RGB images: {rgb_images}")
except KeyError as e:
    raise ValueError(f"Invalid sequence name in selected_sequences: {e}. Available keys: {list(constants.SEQUENCE_TO_INDEX.keys())}")



batch_size = 20
if training_mode == constants.Training.LEARNING_RATE_TUNING:
    training_epochs = 400
else:
    training_epochs = 1500

learning_rate = 0.0003
if training_mode == constants.Training.UPPER_LAYER:
    learning_rate = 0.001

# Regularization
dropout_rate = 0.4
l2_regularization = 0.0001

image_size = 384

codename = "transfer_bit_00"
training_codename = hf.get_training_codename(
    code_name = codename,
    num_classes = num_classes,
    clinical_data = clinical_data,
    use_layer = use_layer,
    is_cutout = cutout,
    is_rgb_images = rgb_images,
    selected_sequences_str = "-".join(selected_sequences),
    contrast_DA = contrast_DA,
    dataset_type = dataset_type,
    training_mode = training_mode,
)


path_to_tfrs = hf.get_path_to_tfrs(
    is_rgb_images = rgb_images,
    is_cutout = cutout,
    dataset_type = dataset_type,
)

# slice + clinical data + contrast weights:
#path_to_weights = path_to_logs + "/transfer_bit_00_2_cls_slice_rgb_contrast_DA_upper_layer_run_2024_11_18_23_23_54/fold_0/saved_weights.weights.h5"

# slice + clinical data - contrast weights:
#path_to_weights = path_to_logs + "/transfer_bit_00_2_cls_slice_rgb_normal_DA_upper_layer_run_2024_11_22_21_34_15/fold_0/saved_weights.weights.h5"

# slice - cinical data - contrast weights:
path_to_weights = constants.path_to_logs / "transfer_bit_00_2_cls_no_clin_slice_rgb_normal_DA_upper_layer_run_2024_12_08_03_16_11/fold_0/saved_weights.weights.h5"

if path_to_tfrs is None and dataset_type != constants.Dataset.PRETRAIN_ROUGH:
    raise ValueError(f"Could not determine path to TFRecords for dataset type {dataset_type}")
print(f"Using TFRecords from: {path_to_tfrs}")


time = strftime("run_%Y_%m_%d_%H_%M_%S")
class_directory = f"{training_codename}_{time}"
path_to_callbacks = Path(constants.path_to_logs) / Path(class_directory)
os.makedirs(path_to_callbacks, exist_ok=True)

def train_ai():

    hf.print_training_timestamps(isStart = True, training_codename = training_codename)

    if dataset_type == constants.Dataset.PRETRAIN_ROUGH:
        # Rough pretraining data setup (uses its own parsing logic in helper_funcs.py)

        train_data, val_data = hf.setup_pretraining_data(
            path_to_tfrs,
            batch_size,
            selected_indices,
            dataset_type
        )

        # get callbacks
        callbacks = hf.get_callbacks(
            path_to_callbacks = path_to_callbacks,
            fold_num = 0,
            use_lrscheduler = True if training_mode == constants.Training.LEARNING_RATE_TUNING else False,
            use_early_stopping = False if training_mode == constants.Training.LEARNING_RATE_TUNING else True,
            early_stopping_patience = constants.early_stopping_patience_upper_layer if training_mode == constants.Training.UPPER_LAYER else None
        )

        # build model
        model = build_transfer_bit_model(
            trainable = False if training_mode == constants.Training.UPPER_LAYER else True
        )

        # load weights
        if training_mode != constants.Training.UPPER_LAYER:
            model.load_weights(path_to_weights)

        # traing model
        history = model.fit(
            train_data,
            validation_data = val_data,
            epochs = training_epochs,
            batch_size = batch_size,
            callbacks = callbacks,
            class_weight = constants.rough_class_weights
        )        

        # save history
        hf.save_training_history(
            history = history,
            training_codename = training_codename,
            time = time,
            path_to_callbacks = path_to_callbacks
        )
    
    elif dataset_type == constants.Dataset.PRETRAIN_FINE:

        train_data, val_data = hf.setup_pretraining_data(
            path_to_tfrs,
            batch_size,
            selected_indices,
            dataset_type,
            rgb = rgb_images
        )

        # get callbacks
        callbacks = hf.get_callbacks(
            path_to_callbacks = path_to_callbacks,
            fold_num = 0,
            use_lrscheduler = True if training_mode == constants.Training.LEARNING_RATE_TUNING else False,
            use_early_stopping = False if training_mode == constants.Training.LEARNING_RATE_TUNING else True,
            early_stopping_patience = constants.early_stopping_patience_upper_layer if training_mode == constants.Training.UPPER_LAYER else None
        )

        # build model
        model = build_transfer_bit_model(
            trainable = False if training_mode == constants.Training.UPPER_LAYER else True
        )

        # load weights
        if training_mode != constants.Training.UPPER_LAYER:
            model.load_weights(path_to_weights)

        # traing model
        history = model.fit(
            train_data,
            validation_data = val_data,
            epochs = training_epochs,
            batch_size = batch_size,
            callbacks = callbacks,
            class_weight = constants.fine_two_class_weights
        )        

        # save history
        hf.save_training_history(
            history = history,
            training_codename = training_codename,
            time = time,
            path_to_callbacks = path_to_callbacks
        )
    
    elif dataset_type == constants.Dataset.NORMAL:

        k_fold_amount = 10 if training_mode == constants.Training.K_FOLD else 1

        for fold in range(k_fold_amount):

            if training_mode == constants.Training.K_FOLD:
                hf.print_fold_info(fold, is_start = True)

            train_data, val_data, test_data = hf.setup_data(
                path_to_tfrs = path_to_tfrs,
                path_to_callbacks = path_to_callbacks,
                path_to_splits = constants.path_to_splits,
                num_classes = num_classes,
                batch_size = batch_size,
                selected_indices = selected_indices,
                use_clinical_data = clinical_data,
                use_layer = use_layer,
                rgb = rgb_images,
                current_fold = fold
            )
            
            callbacks = hf.get_callbacks(
                path_to_callbacks = path_to_callbacks,
                fold_num = 0,
                use_lrscheduler = True if training_mode == constants.Training.LEARNING_RATE_TUNING else False,
                use_early_stopping = False if training_mode == constants.Training.LEARNING_RATE_TUNING else True,
                early_stopping_patience = constants.early_stopping_patience_upper_layer if training_mode == constants.Training.UPPER_LAYER else None
            )
            
            # hf.check_dataset(train_data, "Training", batch_size, input_shape,
            #                num_classes, clinical_data, use_layer, dataset_type, num_batches_to_check=2)
            # hf.check_dataset(val_data, "Validation", batch_size, input_shape,
            #                num_classes, clinical_data, use_layer, dataset_type, num_batches_to_check=2)
            # if test_data is not None:
            #     hf.check_dataset(test_data, "Test", batch_size, input_shape,
            #         num_classes, clinical_data, use_layer, dataset_type, num_batches_to_check=2)

            # build model
            model = build_transfer_bit_model(
                trainable = False if training_mode == constants.Training.UPPER_LAYER else True
            )

            # load weights
            if training_mode != constants.Training.UPPER_LAYER:
                model.load_weights(path_to_weights)

            #training model
            history = model.fit(
                train_data,
                validation_data = val_data,
                epochs = training_epochs,
                callbacks = callbacks,
            )      

            # save history
            hf.save_training_history(
                history = history,
                training_codename = training_codename,
                time = time,
                path_to_callbacks = path_to_callbacks,
                fold = fold if training_mode == constants.Training.K_FOLD else -1
            )

            if training_mode == constants.Training.K_FOLD:
                hf.clear_tf_session()

                hf.print_fold_info(fold, is_start = False)
    
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}. Supported types: {constants.Dataset.NORMAL}, {constants.Dataset.PRETRAIN_ROUGH}, {constants.Dataset.PRETRAIN_FINE}")
    
    hf.clear_tf_session()

    hf.print_training_timestamps(isStart = False, training_codename = training_codename)

def build_transfer_bit_model(trainable = True):

    DefaultDenseLayer = partial(
        tf.keras.layers.Dense,
        activation = constants.activation_func,
        kernel_initializer = "he_normal",
        kernel_regularizer = tf.keras.regularizers.l2(l2_regularization)
    )

    optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)

    # Define inputs
    image_input = tf.keras.layers.Input(shape=input_shape, name = "image_input")
    sex_input = tf.keras.layers.Input(shape=(1,), name = "sex_input")
    age_input = tf.keras.layers.Input(shape=(1,), name = "age_input")
    layer_input = tf.keras.layers.Input(shape=(1,), name = "layer_input")

    # Choose Data Augmentation pipeline
    augment_layer = hf.contrast_data_augmentation if contrast_DA else hf.normal_data_augmentation

    # --- Model Architecture ---
    x = augment_layer(image_input)
    x = tf.keras.layers.Resizing(image_size, image_size)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # Use the pretrained base model
    # this is the R152x4 architecture, which unfortunately doesn't fit into memory, so I went down with the size
    x = hub.KerasLayer("https://www.kaggle.com/models/google/bit/TensorFlow2/m-r152x4/1", trainable=trainable)(x)
    # R101x3 architecture also didn't fit into memory
    # x = hub.KerasLayer("https://www.kaggle.com/models/google/bit/TensorFlow2/m-r101x3/1", trainable=trainable)(batch_normed_augment)
    # x = hub.KerasLayer("https://www.kaggle.com/models/google/bit/TensorFlow2/m-r101x1/1", trainable=trainable)(batch_normed_augment)
    #x = tf.keras.layers.GlobalMaxPool2D()(x)

    bit_image_features = tf.keras.layers.Flatten()(x)

    # --- Feature Concatenation ---
    # use 'clinical_data' and 'use_layer'

    inputs_to_concat = [bit_image_features]

    if clinical_data:
        inputs_to_concat.extend([sex_input, age_input])
        if use_layer:
            inputs_to_concat.append(layer_input)
    elif use_layer:
        inputs_to_concat.append(layer_input)

    if len(inputs_to_concat) > 1:
        concatenated_features = tf.keras.layers.Concatenate()(inputs_to_concat)
    else:
        concatenated_features = bit_image_features # No concatenation needed

    # --- Dense Layers ---
    x = tf.keras.layers.BatchNormalization()(concatenated_features)
    x = DefaultDenseLayer(units=512)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = DefaultDenseLayer(units=256)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    # --- Output Layer ---
    if num_classes == 2:
        # Binary Classification
        x = tf.keras.layers.Dense(1)(x)
        output = tf.keras.layers.Activation('sigmoid', dtype='float32', name='predictions')(x)
        loss = "binary_crossentropy"
        metrics = ["accuracy",
                   tf.keras.metrics.AUC(name = "auc"),
                   tf.keras.metrics.Precision(name = "precision"),
                   tf.keras.metrics.Recall(name = "recall")]
    elif num_classes > 2 and num_classes <= 6:
        x = tf.keras.layers.Dense(num_classes)(x)
        output = tf.keras.layers.Activation('softmax', dtype='float32', name='predictions')(x)
        loss = "sparse_categorical_crossentropy"
        metrics = ["accuracy"]
    else:
        raise ValueError("num_classes must have a value between 2 and 6")

    # --- Create and compile model ---
    if dataset_type == constants.Dataset.NORMAL:
        if clinical_data == True:
            if use_layer == True:
                model = tf.keras.Model(inputs = [image_input, sex_input, age_input, layer_input], outputs = [output])
            else:
                model = tf.keras.Model(inputs = [image_input, sex_input, age_input], outputs = [output])
        else:
            if use_layer == True:
                model = tf.keras.Model(inputs = [image_input, layer_input], outputs = [output])
            else:
                model = tf.keras.Model(inputs = [image_input], outputs = [output])
    else:
        model = tf.keras.Model(inputs = [image_input], outputs = [output])

    model.compile(
        loss = loss,
        optimizer = optimizer,
        metrics = metrics
    )
    
    model.summary()

    return model


if __name__ == "__main__":
    train_ai()