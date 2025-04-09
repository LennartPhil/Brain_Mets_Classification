import tensorflow as tf
import helper_funcs as hf
from pathlib import Path
import os
from time import strftime
from functools import partial
import numpy as np
import constants

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
training_mode = constants.Training.NORMAL # LEARNING_RATE_TUNING, NORMAL, K_FOLD, UPPER_LAYER

cutout = False
rgb_images = False # using gray scale images as input
#include_mask = True
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
    #include_mask = False
    rgb_images = True
    selected_sequences = ["t1c"]

elif dataset_type == constants.Dataset.PRETRAIN_FINE:
    num_classes = 5
    cutout = False
    clinical_data = False
    use_layer = False


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



batch_size = 75 #50
if training_mode == constants.Training.LEARNING_RATE_TUNING:
    training_epochs = 400
else:
    training_epochs = 1500
learning_rate = 0.001

# Regularization
dropout_rate = 0.4
l2_regularization = 0.0001

codename = "conv_00"
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


path_to_tfrs = hf.get_path_to_tfrs(cutout, rgb_images, dataset_type)

if path_to_tfrs is None and dataset_type != constants.Dataset.PRETRAIN_ROUGH:
    raise ValueError(f"Could not determine path to TFRecords for dataset type {dataset_type}")
print(f"Using TFRecords from: {path_to_tfrs}")


time = strftime("run_%Y_%m_%d_%H_%M_%S")
class_directory = f"{training_codename}_{time}"
path_to_callbacks = Path(constants.path_to_logs) / Path(class_directory)
os.makedirs(path_to_callbacks, exist_ok=True)

def train_ai():

    hf.print_training_timestamps(isStart = True, training_codename = training_codename)

    # Prepare partial function for parsing data with selected sequences
    #parse_fn = partial(hf.parse_record, selected_indices = selected_indices)

    if dataset_type == constants.Dataset.PRETRAIN_ROUGH:
        # Rough pretraining data setup (uses its own parsing logic in helper_funcs)
        # regular training
        train_data, val_data = hf.setup_pretraining_data(
            path_to_tfrs,
            batch_size,
            selected_indices,
            dataset_type
        )

        # get callbacks
        callbacks = hf.get_callbacks(path_to_callbacks, 0)

        # build model
        model = build_conv_model()

        # traing model
        history = model.fit(
            train_data,
            validation_data = val_data,
            epochs = training_epochs,
            batch_size = batch_size,
            callbacks = callbacks
        )        

        # save history
        hf.save_training_history(
            history = history,
            training_codename = training_codename,
            time = time,
            path_to_callbacks = path_to_callbacks
        )

    elif dataset_type == constants.Dataset.PRETRAIN_FINE:

        # regular training
        train_data, val_data = hf.setup_pretraining_data(
            path_to_tfrs,
            batch_size,
            selected_indices,
            dataset_type,
            #parse_fn = parse_fn # Pass the parse function
        )

        # get callbacks
        callbacks = hf.get_callbacks(path_to_callbacks, 0)

        # build model
        model = build_conv_model()

        # traing model
        history = model.fit(
            train_data,
            validation_data = val_data,
            epochs = training_epochs,
            batch_size = batch_size,
            callbacks = callbacks
        )        

        # save history
        hf.save_training_history(
            history = history,
            training_codename = training_codename,
            time = time,
            path_to_callbacks = path_to_callbacks
        )

    elif training_mode == constants.Training.K_FOLD:
        
        for fold in range(10):

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
                current_fold = fold,
            )

            callbacks = hf.get_callbacks(path_to_callbacks, fold)
            
            # build model
            model = build_conv_model()

            #training model
            history = model.fit(
                train_data,
                validation_data = val_data,
                epochs = training_epochs,
                #batch_size = batch_size,
                callbacks = callbacks,
                #class_weight = constants.normal_two_class_weights if num_classes == 2 else None
            )

            # save history
            hf.save_training_history(
                history = history,
                training_codename = training_codename,
                time = time,
                fold = fold,
                path_to_callbacks = path_to_callbacks
            )

            hf.clear_tf_session()

            hf.print_fold_info(fold, is_start = False)

    elif training_mode == constants.Training.LEARNING_RATE_TUNING:

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
        )
        
        callbacks = hf.get_callbacks(path_to_callbacks, 0,
                                     use_lrscheduler = True,
                                     use_early_stopping = False)
        
        hf.check_dataset(train_data, "Training", batch_size, input_shape,
                       num_classes, clinical_data, use_layer, dataset_type, num_batches_to_check=2)
        hf.check_dataset(val_data, "Validation", batch_size, input_shape,
                       num_classes, clinical_data, use_layer, dataset_type, num_batches_to_check=2)
        if test_data is not None:
            hf.check_dataset(test_data, "Test", batch_size, input_shape,
                num_classes, clinical_data, use_layer, dataset_type, num_batches_to_check=2)

        # build model
        model = build_conv_model()

        #training model
        history = model.fit(
            train_data,
            validation_data = val_data,
            epochs = training_epochs,
            #batch_size = batch_size,
            callbacks = callbacks,
            #class_weight = constants.normal_two_class_weights if num_classes == 2 else None
        )        

        # save history
        hf.save_training_history(
            history = history,
            training_codename = training_codename,
            time = time,
            path_to_callbacks = path_to_callbacks
        )

    elif training_mode == constants.Training.NORMAL:
        # regular training
        train_data, val_data, test_data = hf.setup_data(
            path_to_tfrs,
            path_to_callbacks,
            constants.path_to_splits,
            num_classes,
            batch_size = batch_size,
            selected_indices = selected_indices,
            use_clinical_data = clinical_data,
            use_layer = use_layer,
            rgb = rgb_images,
        )
        

        # get callbacks
        callbacks = hf.get_callbacks(path_to_callbacks, 0)

        # build model
        model = build_conv_model()

        # traing model
        history = model.fit(
            train_data,
            validation_data = val_data,
            epochs = training_epochs,
            #batch_size = batch_size,
            callbacks = callbacks,
            #class_weight = constants.normal_two_class_weights if num_classes == 2 else None
        )        

        # save history
        hf.save_training_history(
            history = history,
            training_codename = training_codename,
            time = time,
            path_to_callbacks = path_to_callbacks
        )
    
    else:
        raise ValueError("Wrong training mode selected, please pick a training mode")

    hf.clear_tf_session()

    hf.print_training_timestamps(isStart = False, training_codename = training_codename)

def build_conv_model():
  
    DefaultConv2D = partial(
        tf.keras.layers.Conv2D,
        kernel_size = 3,
        padding = "same",
        activation = constants.activation_func,
        kernel_initializer = "he_normal",
        kernel_regularizer = tf.keras.regularizers.l2(l2_regularization)  # L2 Regularization
    )

    DefaultDenseLayer = partial(
        tf.keras.layers.Dense,
        activation = constants.activation_func,
        kernel_initializer = "he_normal",
        kernel_regularizer = tf.keras.regularizers.l2(l2_regularization)
    )

    optimizer = tf.keras.optimizers.legacy.SGD(learning_rate = learning_rate, momentum = 0.9, nesterov = True)

    # Define inputs
    image_input = tf.keras.layers.Input(shape=input_shape, name="image_input")
    sex_input = tf.keras.layers.Input(shape=(1,), name="sex_input") 
    age_input = tf.keras.layers.Input(shape=(1,), name="age_input")
    layer_input = tf.keras.layers.Input(shape=(1,), name="layer_input")

    # Choose Data Augmentation pipeline
    augment_layer = hf.contrast_data_augmentation if contrast_DA else hf.normal_data_augmentation

    # --- Model Architecture ---
    x = augment_layer(image_input) # Apply augmentation first

    x = tf.keras.layers.BatchNormalization()(x) # BN before first conv
    x = DefaultConv2D(filters = 64, kernel_size = 7, strides = 2)(x)
    x = tf.keras.layers.MaxPool2D(pool_size = (2,2))(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = DefaultConv2D(filters = 128)(x)
    x = DefaultConv2D(filters = 128)(x)
    x = tf.keras.layers.MaxPool2D(pool_size = (2,2))(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = DefaultConv2D(filters = 256)(x)
    x = DefaultConv2D(filters = 256)(x)
    x = tf.keras.layers.MaxPool2D(pool_size = (2,2))(x)

    image_features = tf.keras.layers.Flatten()(x)

    # --- Feature Concatenation ---
    # use 'clincal_data' and 'use_layer'

    inputs_to_concat = [image_features]

    if clinical_data:
        inputs_to_concat.extend([sex_input, age_input])
        if use_layer:
            inputs_to_concat.append(layer_input)
    elif use_layer:
        inputs_to_concat.append(layer_input)

    if len(inputs_to_concat) > 1:
        concatenated_features = tf.keras.layers.Concatenate()(inputs_to_concat)
    else:
        concatenated_features = image_features # No concatenation needed
        

    # --- Dense Layers ---
    x = tf.keras.layers.BatchNormalization()(concatenated_features)
    x = DefaultDenseLayer(units = 512)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = DefaultDenseLayer(units = 256)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    # batch_norm_1_layer = tf.keras.layers.BatchNormalization()
    # conv_1_layer = DefaultConv2D(filters = 64, kernel_size = 7, strides = 2) # , input_shape = [240, 240, 4]
    # max_pool_1_layer = tf.keras.layers.MaxPool2D(pool_size = (2,2))

    # batch_norm_2_layer = tf.keras.layers.BatchNormalization()
    # conv_2_layer = DefaultConv2D(filters = 128)
    # conv_3_layer = DefaultConv2D(filters = 128)
    # max_pool_2_layer = tf.keras.layers.MaxPool2D(pool_size = (2,2))

    # batch_norm_3_layer = tf.keras.layers.BatchNormalization()
    # conv_4_layer = DefaultConv2D(filters = 256)
    # conv_5_layer = DefaultConv2D(filters = 256)
    # max_pool_3_layer = tf.keras.layers.MaxPool2D(pool_size = (2,2))

    # batch_norm_4_layer = tf.keras.layers.BatchNormalization()
    # dense_1_layer = DefaultDenseLayer(units = 512)
    # dropout_1_layer = tf.keras.layers.Dropout(dropout_rate)

    # batch_norm_5_layer = tf.keras.layers.BatchNormalization()
    # dense_2_layer = DefaultDenseLayer(units = 256)
    # dropout_2_layer = tf.keras.layers.Dropout(dropout_rate)

    # augment = augment_layer(image_input)
    # batch_norm_1 = batch_norm_1_layer(augment)

    # conv_1 = conv_1_layer(batch_norm_1)
    # max_pool_1 = max_pool_1_layer(conv_1)

    # batch_norm_2 = batch_norm_2_layer(max_pool_1)
    # conv_2 = conv_2_layer(batch_norm_2)
    # conv_3 = conv_3_layer(conv_2)
    # max_pool_2 = max_pool_2_layer(conv_3)

    # batch_norm_3 = batch_norm_3_layer(max_pool_2)
    # conv_4 = conv_4_layer(batch_norm_3)
    # conv_5 = conv_5_layer(conv_4)
    # max_pool_3 = max_pool_3_layer(conv_5)

    # flatten = tf.keras.layers.Flatten()(max_pool_3)

    # Clinical Data Usage
    # if clinical_data == True and use_layer == True:
    #     concatenated_inputs = tf.keras.layers.Concatenate()([
    #         flatten,
    #         age_input,
    #         sex_input,
    #         layer_input
    #     ])
    # elif clinical_data == True and use_layer == False:
    #     concatenated_inputs = tf.keras.layers.Concatenate()([
    #         flatten,
    #         age_input,
    #         sex_input,
    #     ])
    # elif clinical_data == False and use_layer == True:
    #     concatenated_inputs = tf.keras.layers.Concatenate()([
    #         flatten,
    #         layer_input
    #     ])
    # else:
    #     # if clinical data is not wanted, then only the image is used
    #     concatenated_inputs = flatten

    # x = batch_norm_4_layer(concatenated_inputs)
    # x = dense_1_layer(x)
    # x = dropout_1_layer(x)
    # x = batch_norm_5_layer(x)
    # x = dense_2_layer(x)
    # x = dropout_2_layer(x)


    # --- Output Layer ---

    if num_classes == 2:
        # Binary Classification
        x = tf.keras.layers.Dense(1)(x)
        output = tf.keras.layers.Activation('sigmoid', dtype='float32', name='predictions')(x)
        loss = "binary_crossentropy"
    elif num_classes > 2 and num_classes <= 6:
        x = tf.keras.layers.Dense(num_classes)(x)
        output = tf.keras.layers.Activation('softmax', dtype='float32', name='predictions')(x)
        loss = "sparse_categorical_crossentropy"
    else:
        raise ValueError("numm_classes must have a value between 2 and 6")

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
        metrics = ["accuracy"] #"RootMeanSquaredError", "AUC", "Precision", "Recall" 
    )
    
    model.summary()

    return model


if __name__ == "__main__":
    train_ai()