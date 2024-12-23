import tensorflow as tf
import tensorflow_hub as hub
import helper_funcs as hf
from pathlib import Path
import os
from time import strftime
from functools import partial
import numpy as np

# So the idea with the pretrained models is the following:
# 1. Run: adjust the upper layers with trainable turned off
# !before starting 2nd run: define the path_to_weights below!
# 2. Run: find best learning rate with exported weights
# 3. Run: train of the best learning rate

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

cutout = False
rgb_images = True # using gray scale images as input
contrast_DA = False
clinical_data = False
num_classes = 2
train_upper_layers = False
use_k_fold = False
learning_rate_tuning = False

batch_size = 20
if learning_rate_tuning:
    training_epochs = 400
else:
    training_epochs = 1500

learning_rate = 0.0003
if train_upper_layers:
    learning_rate = 0.001

dropout_rate = 0.6

image_size = 384

codename = "transfer_bit_00"
training_codename = hf.get_training_codename(
    code_name = codename,
    num_classes = num_classes,
    clinical_data = clinical_data,
    is_cutout = cutout,
    is_rgb_images = rgb_images,
    contrast_DA = contrast_DA,
    is_learning_rate_tuning = learning_rate_tuning,
    is_k_fold = use_k_fold,
    is_upper_layer_training = train_upper_layers
)


path_to_tfrs = hf.get_path_to_tfrs(cutout, rgb_images)
path_to_logs = "/logs"
path_to_splits = "/tfrs/split_text_files"

activation_func = "mish"

# slice + clinical data + contrast weights:
#path_to_weights = path_to_logs + "/transfer_bit_00_2_cls_slice_rgb_contrast_DA_upper_layer_run_2024_11_18_23_23_54/fold_0/saved_weights.weights.h5"

# slice + clinical data - contrast weights:
#path_to_weights = path_to_logs + "/transfer_bit_00_2_cls_slice_rgb_normal_DA_upper_layer_run_2024_11_22_21_34_15/fold_0/saved_weights.weights.h5"

# slice - cinical data - contrast weights:
path_to_weights = path_to_logs + "/transfer_bit_00_2_cls_no_clin_slice_rgb_normal_DA_upper_layer_run_2024_12_08_03_16_11/fold_0/saved_weights.weights.h5"


time = strftime("run_%Y_%m_%d_%H_%M_%S")
class_directory = f"{training_codename}_{time}"
path_to_callbacks = Path(path_to_logs) / Path(class_directory)
os.makedirs(path_to_callbacks, exist_ok=True)

def train_ai():

    hf.print_training_timestamps(isStart = True, training_codename = training_codename)

    if use_k_fold:
        
        for fold in range(10):

            hf.print_fold_info(fold, is_start = True)

            train_data, val_data, test_data = hf.setup_data(path_to_tfrs, path_to_callbacks, path_to_splits, num_classes, batch_size = batch_size, rgb = rgb_images, current_fold = fold)

            callbacks = hf.get_callbacks(path_to_callbacks, fold)
            
            # build model
            model = build_transfer_bit_model(clinical_data = clinical_data)

            # load weights
            model.load_weights(path_to_weights)

            #training model
            history = model.fit(
                train_data,
                validation_data = val_data,
                epochs = training_epochs,
                batch_size = batch_size,
                callbacks = callbacks,
                class_weight = hf.two_class_weights
            )

            # save history
            hf.save_training_history(
                history = history,
                training_codename = training_codename,
                time = time,
                path_to_callbacks = path_to_callbacks,
                fold = fold
            )

            hf.clear_tf_session()

            hf.print_fold_info(fold, is_start = False)

    elif train_upper_layers:

        train_data, val_data, test_data = hf.setup_data(path_to_tfrs, path_to_callbacks, path_to_splits, num_classes, batch_size = batch_size, rgb = rgb_images)
        
        callbacks = hf.get_callbacks(path_to_callbacks, 0,
                                     use_early_stopping = True,
                                     stop_training = False,
                                     early_stopping_patience = 20)
        
        model = build_transfer_bit_model(clinical_data = clinical_data, trainable = False)

        # traing model
        history = model.fit(
            train_data,
            validation_data = val_data,
            epochs = training_epochs,
            batch_size = batch_size,
            callbacks = callbacks,
            class_weight = hf.two_class_weights
        )        

        # save history
        hf.save_training_history(
            history = history,
            training_codename = training_codename,
            time = time,
            path_to_callbacks = path_to_callbacks
        )

    elif learning_rate_tuning:

        train_data, val_data, test_data = hf.setup_data(path_to_tfrs, path_to_callbacks, path_to_splits, num_classes, batch_size = batch_size, rgb = rgb_images)
        
        callbacks = hf.get_callbacks(path_to_callbacks, 0,
                                     use_lrscheduler=True,
                                     use_early_stopping=False)

        # build model
        model = build_transfer_bit_model(clinical_data = clinical_data)

        # load weights
        model.load_weights(path_to_weights)

        # traing model
        history = model.fit(
            train_data,
            validation_data = val_data,
            epochs = training_epochs,
            batch_size = batch_size,
            callbacks = callbacks,
            class_weight = hf.two_class_weights
        )        

        # save history
        hf.save_training_history(
            history = history,
            training_codename = training_codename,
            time = time,
            path_to_callbacks = path_to_callbacks
        )

    else:
        # regular training
        train_data, val_data, test_data = hf.setup_data(path_to_tfrs, path_to_callbacks, path_to_splits, num_classes, batch_size = batch_size, rgb = rgb_images)

        # get callbacks
        callbacks = hf.get_callbacks(path_to_callbacks, 0)

        # build model
        model = build_transfer_bit_model(clinical_data = clinical_data)

        # load weights
        model.load_weights(path_to_weights)

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
    
    hf.clear_tf_session()

    hf.print_training_timestamps(isStart = False, training_codename = training_codename)

def build_transfer_bit_model(clinical_data, trainable = True):

    optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)

    # Define inputs
    image_input = tf.keras.layers.Input(shape=(240, 240, 3))
    sex_input = tf.keras.layers.Input(shape=(1,))
    age_input = tf.keras.layers.Input(shape=(1,))

    augmented = data_augmentation(image_input)
    
    tf.print("Augmented shape:", augmented.shape)

    # Use the pretrained base model
    # this is the R152x4 architecture, which unfortunately doesn't fit into memory, so I went down with the size
    # x = hub.KerasLayer("https://www.kaggle.com/models/google/bit/TensorFlow2/m-r152x4/1", trainable=trainable)(augmented)
    # R101x3 architecture also didn't fit into memory
    # x = hub.KerasLayer("https://www.kaggle.com/models/google/bit/TensorFlow2/m-r101x3/1", trainable=trainable)(augmented)
    x = hub.KerasLayer("https://www.kaggle.com/models/google/bit/TensorFlow2/m-r101x1/1", trainable=trainable)(augmented)
    #x = tf.keras.layers.GlobalMaxPool2D()(x)

    output = tf.keras.layers.Flatten()(x)

    # Clinical Data
    if clinical_data == True:
        # Process sex and age inputs
        flattened_sex_input = tf.keras.layers.Flatten()(sex_input)
        age_input_reshaped = tf.keras.layers.Reshape((1,))(age_input)
        concatenated_inputs = tf.keras.layers.Concatenate()([output, age_input_reshaped, flattened_sex_input])
    else:
        concatenated_inputs = output

    # Define dense and dropout layers
    dense_1_layer = tf.keras.layers.Dense(512, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())
    dropout_1_layer = tf.keras.layers.Dropout(dropout_rate)
    dense_2_layer = tf.keras.layers.Dense(256, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())
    dropout_2_layer = tf.keras.layers.Dropout(dropout_rate)

    # Fully connected layers
    x = dense_1_layer(concatenated_inputs)
    x = dropout_1_layer(x)
    x = dense_2_layer(x)
    x = dropout_2_layer(x)

    # Output layer
    match num_classes:
        case 2:
            x = tf.keras.layers.Dense(1)(x)
            output = tf.keras.layers.Activation('sigmoid', dtype='float32', name='predictions')(x)
        case 3 | 4 | 5 | 6:
            x = tf.keras.layers.Dense(num_classes)(x)
            output = tf.keras.layers.Activation('softmax', dtype='float32', name='predictions')(x)
        case _:
            raise ValueError("num_classes must be 2, 3, 4, 5 or 6.")

    model = tf.keras.Model(inputs = [image_input, sex_input, age_input], outputs = [output], name = "transfer_bit_model")

    if num_classes > 2:
        model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics = ["RootMeanSquaredError", "accuracy"])
    else:
        model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics = ["RootMeanSquaredError", "accuracy"])
    
    model.summary()

    return model


class NormalizeToRange(tf.keras.layers.Layer):
    def __init__(self, zero_to_one=True):
        super(NormalizeToRange, self).__init__()
        self.zero_to_one = zero_to_one

    def call(self, inputs):
        min_val = tf.reduce_min(inputs)
        max_val = tf.reduce_max(inputs)
        if self.zero_to_one:
            # Normalize to [0, 1]
            normalized = (inputs - min_val) / (max_val - min_val)
        else:
            # Normalize to [-1, 1]
            normalized = 2 * (inputs - min_val) / (max_val - min_val) - 1
        return normalized
    
if contrast_DA:
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip(mode = "horizontal"),
        #tf.keras.layers.Rescaling(1/255),
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
    ])
else:
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip(mode = "horizontal"),
        #tf.keras.layers.Rescaling(1/255),
        #tf.keras.layers.RandomContrast(0.5), # consider removing the random contrast layer as that causes pixel values to go beyond 1
        #tf.keras.layers.RandomBrightness(factor = (-0.2, 0.4)), #, value_range=(0, 1)
        tf.keras.layers.RandomRotation(factor = (-0.1, 0.1), fill_mode = "nearest"),
        NormalizeToRange(zero_to_one=True),
        tf.keras.layers.RandomTranslation(
            height_factor = 0.05,
            width_factor = 0.05,
            fill_mode = "nearest",
            interpolation = "bilinear"
        ),
    ])

if __name__ == "__main__":
    train_ai()