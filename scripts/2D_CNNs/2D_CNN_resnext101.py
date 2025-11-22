import tensorflow as tf
import helper_funcs as hf
from pathlib import Path
import os
from time import strftime
from functools import partial
import numpy as np
import constants

# mixed precision setup
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# --- GPU setup ---
gpus = tf.config.list_physical_devices('GPU')
print(f"{len(gpus)} GPU(s) detected.")
# if gpus:
#     try:
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)

# print("tensorflow_setup successful")

# --- Configuration ---
dataset_type = constants.Dataset.PRETRAIN_ROUGH # PRETRAIN_ROUGH, PRETRAIN_FINE, NORMAL
training_mode = constants.Training.NORMAL # LEARNING_RATE_TUNING, NORMAL, K_FOLD, UPPER_LAYER

START_FOLD = 0

cutout = False
rgb_images = False # using gray scale images as input
contrast_DA = False # data augmentation with contrast
clinical_data = False
use_layer = False
num_classes = 2

use_pretrained_weights = True # if True, will load weights from path_to_weights if it exists
weight_folder = "/home/lennart/work/weights/pretrain_fine/ResNeXt101/resnext101_00_2cls_slice_no_clin_no_layer_gray_seq[t1c]_normal_DA_pretrain_fine_normal_run_2025_10_19_15_17_45" + "/saved_weights.weights.h5"
path_to_weights = constants.path_to_logs / weight_folder

# --- Select Sequences ---
selected_sequences = ["t1c"] #["t1", "t1c", "t2", "flair", "mask"]

if dataset_type == constants.Dataset.PRETRAIN_ROUGH:
    num_classes = 3
    cutout = False
    clinical_data = False
    use_layer = False
    rgb_images = False
    selected_sequences = ["t1c"]
    if training_mode != constants.Training.LEARNING_RATE_TUNING and training_mode != constants.Training.NORMAL:
        raise ValueError(f"For PRETRAIN_ROUGH dataset, only LEARNING_RATE_TUNING and NORMAL training modes are supported. Current mode: {training_mode}")
        

elif dataset_type == constants.Dataset.PRETRAIN_FINE:
    num_classes = 2 # we only fine train on the glioblastoma and brain metatases, meaning we'll only need 2 classes
    cutout = False
    clinical_data = False
    use_layer = False
    if training_mode != constants.Training.LEARNING_RATE_TUNING and training_mode != constants.Training.NORMAL:
        raise ValueError(f"For PRETRAIN_ROUGH dataset, only LEARNING_RATE_TUNING and NORMAL training modes are supported. Current mode: {training_mode}")

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



batch_size = 50#20 #50
if training_mode == constants.Training.LEARNING_RATE_TUNING:
    training_epochs = constants.LEARNING_RATE_EPOCHS #400
else:
    training_epochs = constants.MAX_TRAINING_EPOCHS #1500
learning_rate = 0.01

# Regularization
dropout_rate = 0.45 #constants.REGULAR_DROPOUT_RATE #0.4
l2_regularization = constants.REGULAR_L2_REGULARIZATION

codename = "resnext101_00"
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
            use_early_stopping = False if training_mode == constants.Training.LEARNING_RATE_TUNING else True
        )

        # build model
        model = build_resnext_model(architecture = "ResNeXt101")

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
            use_early_stopping = False if training_mode == constants.Training.LEARNING_RATE_TUNING else True
        )

        # build model
        model = build_resnext_model(architecture = "ResNeXt101")

        if use_pretrained_weights == True and path_to_weights is not None and Path(path_to_weights).exists(): #'path_to_weights' in locals() and 
            try:
                print(f"Loading weights from: {path_to_weights}")
                # Use by_name=True and skip_mismatch=True for flexibility
                model.load_weights(str(path_to_weights), by_name=True, skip_mismatch=True)
                print("Weights loaded successfully.")
            except Exception as e:
                print(f"ERROR: Could not load weights from {path_to_weights}. Training from scratch. Error: {e}")
                raise e
        else:
            if path_to_weights is not None:
                print(f"Weight file not found at {path_to_weights}. Training from scratch.")
            else:
                print("No path_to_weights specified or it's None. Training from scratch (expected for Stage 1).")


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

        for fold in range(START_FOLD, k_fold_amount):

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
                fold_num = fold,
                use_lrscheduler = True if training_mode == constants.Training.LEARNING_RATE_TUNING else False,
                use_early_stopping = False if training_mode == constants.Training.LEARNING_RATE_TUNING else True
            )
            
            # hf.check_dataset(train_data, "Training", batch_size, input_shape,
            #                num_classes, clinical_data, use_layer, dataset_type, num_batches_to_check=2)
            # hf.check_dataset(val_data, "Validation", batch_size, input_shape,
            #                num_classes, clinical_data, use_layer, dataset_type, num_batches_to_check=2)
            # if test_data is not None:
            #     hf.check_dataset(test_data, "Test", batch_size, input_shape,
            #         num_classes, clinical_data, use_layer, dataset_type, num_batches_to_check=2)

            # build model
            model = build_resnext_model(architecture = "ResNeXt101")

            if use_pretrained_weights == True and path_to_weights is not None and Path(path_to_weights).exists():
                try:
                    print(f"Loading weights from: {path_to_weights}")
                    # Use by_name=True and skip_mismatch=True for flexibility
                    model.load_weights(str(path_to_weights), by_name=True, skip_mismatch=True)
                    print("Weights loaded successfully.")
                except Exception as e:
                    print(f"ERROR: Could not load weights from {path_to_weights}. Training from scratch. Error: {e}")
                    raise e
            else:
                if path_to_weights is not None:
                    print(f"Weight file not found at {path_to_weights}. Training from scratch.")
                else:
                    print("No path_to_weights specified or it's None. Training from scratch (expected for Stage 1).")


            #training model
            history = model.fit(
                train_data,
                validation_data = val_data,
                epochs = training_epochs,
                callbacks = callbacks,
                class_weight = constants.class_weights_dict[num_classes]
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


def build_resnext_model(architecture="ResNeXt50"):

    architectures = {
        "ResNeXt50": [3, 4, 6, 3],
        "ResNeXt101": [3, 4, 23, 3],
    }

    if architecture not in architectures:
        raise ValueError(f"Architecture {architecture} not recognized. Available architectures: {list(architectures.keys())}")

    repetitions = architectures[architecture]

    DefaultConv2D = partial(
        tf.keras.layers.Conv2D,
        kernel_size = 3,
        strides = 1,
        padding = "same",
        activation = None,
        kernel_initializer = "he_normal",
        use_bias = False,
        kernel_regularizer = tf.keras.regularizers.l2(l2_regularization)
    )

    DefaultDenseLayer = partial(
        tf.keras.layers.Dense,
        activation = constants.activation_func,
        kernel_initializer = "he_normal",
        kernel_regularizer = tf.keras.regularizers.l2(l2_regularization)
    )
    
    class ResNeXtBlock(tf.keras.layers.Layer):
        def __init__(self, filters, cardinality, strides=1, input_filters = None, activation="relu", **kwargs):
            super().__init__(**kwargs)
            self.activation = tf.keras.activations.get(activation)
            self.main_layers = [
                DefaultConv2D(filters // 2, kernel_size=1, strides=1),
                tf.keras.layers.BatchNormalization(),
                self.activation,
                DefaultConv2D(filters // 2, kernel_size=3, strides=strides, groups=cardinality),
                tf.keras.layers.BatchNormalization(),
                self.activation,
                DefaultConv2D(filters, kernel_size=1, strides=1),
                tf.keras.layers.BatchNormalization()
            ]
            self.skip_layers = []
            if strides > 1 or filters != input_filters:
                self.skip_layers = [
                    DefaultConv2D(filters, kernel_size=1, strides=strides),
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

    optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)

    # Define inputs
    image_input = tf.keras.layers.Input(shape=input_shape, name = "image_input")
    sex_input = tf.keras.layers.Input(shape=(1,), name = "sex_input")
    age_input = tf.keras.layers.Input(shape=(1,), name = "age_input")
    layer_input = tf.keras.layers.Input(shape=(1,), name = "layer_input")

    # Choose Data Augmentation pipeline
    augment_layer = hf.contrast_data_augmentation if contrast_DA else hf.normal_data_augmentation

    # --- Model Architecture ---
    x = augment_layer(image_input) # Apply augmentation first

    x = tf.keras.layers.BatchNormalization(name = "b_norm_1")(x)
    x = DefaultConv2D(filters=64, kernel_size=7, strides=2, name = "conv_1")(x)
    x = tf.keras.layers.BatchNormalization(name = "b_norm_2")(x)
    x = tf.keras.layers.Activation(constants.activation_func)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same", name = "pool_1")(x)

    cardinality = 32
    filters = 256 #128
    #repetitions = [3, 4, 6, 3]
    input_filters = x.shape[-1]
    for i, reps in enumerate(repetitions):
        for j in range(reps):
            strides = 2 if i > 0 and j == 0 else 1
            x = ResNeXtBlock(filters, cardinality, strides=strides, input_filters=input_filters, name = f"resnext_stage_{i}_block_{j}")(x)
            input_filters = x.shape[-1] #filters
        filters *= 2

    x = tf.keras.layers.GlobalAveragePooling2D(name = "gap")(x)
    resnext_image_features = tf.keras.layers.Flatten(name = "flatten")(x)

    # --- Feature Concatenation ---
    # use 'clinical_data' and 'use_layer'

    inputs_to_concat = [resnext_image_features]

    if clinical_data:
        inputs_to_concat.extend([sex_input, age_input])
        if use_layer:
            inputs_to_concat.append(layer_input)
    elif use_layer:
        inputs_to_concat.append(layer_input)

    if len(inputs_to_concat) > 1:
        concatenated_features = tf.keras.layers.Concatenate(name = "concatenated_features")(inputs_to_concat)
    else:
        concatenated_features = resnext_image_features # No concatenation needed


    # --- Dense Layers ---
    x = tf.keras.layers.BatchNormalization(name = "b_norm_dense_1")(concatenated_features)
    x = DefaultDenseLayer(units=512, name = "dense_1")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name = "dropout_1")(x)

    x = tf.keras.layers.BatchNormalization(name = "b_norm_dense_2")(x)
    x = DefaultDenseLayer(units=256, name = "dense_2")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name = "dropout_2")(x)

    # --- Output Layer ---

    if num_classes == 2:
        # Binary Classification
        x = tf.keras.layers.Dense(1, name = f"dense_output_{num_classes}cls")(x)
        output = tf.keras.layers.Activation('sigmoid', dtype='float32', name='predictions')(x)
        loss = "binary_crossentropy"
        metrics = ["accuracy",
                   tf.keras.metrics.AUC(name = "auc"),
                   tf.keras.metrics.Precision(name = "precision", thresholds = 0.5),
                   tf.keras.metrics.Recall(name = "recall", thresholds = 0.5),
                   tf.keras.metrics.F1Score(name = "f1_score", threshold = 0.5, average="micro")]
    elif num_classes > 2 and num_classes <= 6:
        x = tf.keras.layers.Dense(num_classes, name = f"dense_output_{num_classes}cls")(x)
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