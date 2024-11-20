import tensorflow as tf
import helper_funcs as hf
from pathlib import Path
import os
from time import strftime
from functools import partial
import numpy as np

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
rgb_images = False # using gray scale images as input
contrast_DA = True # data augmentation with contrast
num_classes = 2
use_k_fold = False
learning_rate_tuning = False


batch_size = 75 #50
if learning_rate_tuning:
    training_epochs = 400
else:
    training_epochs = 1000
learning_rate = 0.001 #0.01 is apparently too large

dropout_rate = 0.5

codename = "conv_01"
training_codename = hf.get_training_codename(
    code_name = codename,
    num_classes = num_classes,
    is_cutout = cutout,
    is_rgb_images = rgb_images,
    contrast_DA = contrast_DA,
    is_learning_rate_tuning = learning_rate_tuning,
    is_k_fold = use_k_fold,
)
# cutout = True #if true, the metastasis is simpley cutout, if false the entire slice of the brain is used

# if learning_rate_tuning:
#     training_codename = training_codename + "_lr"

# training_codename += f"_{num_classes}_cls"

# if rgb_images:
#     training_codename += "_rgb"
# else:
#     training_codename += "_gray"

# if cutout:
#     training_codename = training_codename + "_cutout"
# else:
#     training_codename = training_codename + "_slice"



# if cutout:
#     if rgb_images:
#         # is cuout color images
#         path_to_tfrs = ""
#     else:
#         # is cutout, gray images
#         path_to_tfrs = "/tfrs/all_pats_single_cutout_gray"
# else:
#     # is brain slice
#     if rgb_images
#     path_to_tfrs = ""
path_to_tfrs = hf.get_path_to_tfrs(cutout, rgb_images)
path_to_logs = "/logs"
path_to_splits = "/tfrs/split_text_files"

activation_func = "mish"


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
            model = build_conv_model()

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
                fold = fold,
                path_to_callbacks = path_to_callbacks
            )
            # history_dict = history.history
            # history_file_name = f"history_{training_codename}_fold_{fold}.npy"
            # path_to_np_file = path_to_callbacks / history_file_name
            # np.save(path_to_np_file, history_dict)

            hf.clear_tf_session()

            hf.print_fold_info(fold, is_start = False)

    elif learning_rate_tuning:

        train_data, val_data, test_data = hf.setup_data(path_to_tfrs, path_to_callbacks, path_to_splits, num_classes, batch_size = batch_size,rgb = rgb_images)

        
        callbacks = hf.get_callbacks(path_to_callbacks, 0,
                                     use_lrscheduler=True,
                                     use_early_stopping=False)

        # build model
        model = build_conv_model()

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
        # history_dict = history.history
        # history_file_name = f"history_{training_codename}.npy"
        # path_to_np_file = path_to_callbacks / history_file_name
        # np.save(path_to_np_file, history_dict)
        hf.save_training_history(
            history = history,
            training_codename = training_codename,
            time = time,
            path_to_callbacks = path_to_callbacks
        )

    else:
        # regular training
        train_data, val_data, test_data = hf.setup_data(path_to_tfrs, path_to_callbacks, path_to_splits, num_classes, batch_size = batch_size,rgb = rgb_images)


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
        # history_dict = history.history
        # history_file_name = f"history_{training_codename}.npy"
        # path_to_np_file = path_to_callbacks / history_file_name
        # np.save(path_to_np_file, history_dict)
        hf.save_training_history(
            history = history,
            training_codename = training_codename,
            time = time,
            path_to_callbacks = path_to_callbacks
        )

    hf.clear_tf_session()

    hf.print_training_timestamps(isStart = False, training_codename = training_codename)

def build_conv_model():

    DefaultConv2D = partial(tf.keras.layers.Conv2D, kernel_size=3, padding="same", activation = activation_func, kernel_initializer="he_normal")

    optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)

    # Define inputs
    image_input = tf.keras.layers.Input(shape=(240, 240, 4))
    sex_input = tf.keras.layers.Input(shape=(1,))
    age_input = tf.keras.layers.Input(shape=(1,))

    batch_norm_layer = tf.keras.layers.BatchNormalization()
    conv_1_layer = DefaultConv2D(filters = 64, kernel_size = 7, strides = 2, input_shape = [240, 240, 4])
    max_pool_1_layer = tf.keras.layers.MaxPool2D(pool_size = (2,2))

    conv_2_layer = DefaultConv2D(filters = 128)
    conv_3_layer = DefaultConv2D(filters = 128)
    max_pool_2_layer = tf.keras.layers.MaxPool2D(pool_size = (2,2))

    conv_4_layer = DefaultConv2D(filters = 256)
    conv_5_layer = DefaultConv2D(filters = 256)
    max_pool_3_layer = tf.keras.layers.MaxPool2D(pool_size = (2,2))
    
    # conv_4_layer = tf.keras.layers.Conv2D(filters = 256, kernel_size = 3, strides=(1,1,1), activation=activation_func, kernel_initializer=tf.keras.initializers.HeNormal())
    # max_pool_4_layer = tf.keras.layers.MaxPool2D(pool_size = (2,2,2))

    dense_1_layer = tf.keras.layers.Dense(512, activation=activation_func, kernel_initializer=tf.keras.initializers.HeNormal())
    dropout_1_layer = tf.keras.layers.Dropout(dropout_rate)
    dense_2_layer = tf.keras.layers.Dense(256, activation=activation_func, kernel_initializer=tf.keras.initializers.HeNormal())
    dropout_2_layer = tf.keras.layers.Dropout(dropout_rate)

    augment = data_augmentation(image_input)
    batch_norm = batch_norm_layer(augment)

    conv_1 = conv_1_layer(batch_norm)
    max_pool_1 = max_pool_1_layer(conv_1)

    conv_2 = conv_2_layer(max_pool_1)
    conv_3 = conv_3_layer(conv_2)
    max_pool_2 = max_pool_2_layer(conv_3)

    conv_4 = conv_4_layer(max_pool_2)
    conv_5 = conv_5_layer(conv_4)
    max_pool_3 = max_pool_3_layer(conv_5)

    flatten = tf.keras.layers.Flatten()(max_pool_3)

    flattened_sex_input = tf.keras.layers.Flatten()(sex_input)
    age_input_reshaped = tf.keras.layers.Reshape((1,))(age_input)  # Reshape age_input to have 2 dimensions
    concatenated_inputs = tf.keras.layers.Concatenate()([flatten, age_input_reshaped, flattened_sex_input])

    x = dense_1_layer(concatenated_inputs)
    x = dropout_1_layer(x)
    x = dense_2_layer(x)
    x = dropout_2_layer(x)

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