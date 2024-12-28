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
contrast_DA = True
clinical_data = True
use_layer = True
num_classes = 2
use_k_fold = False
learning_rate_tuning = True


batch_size = 20 #50
if learning_rate_tuning:
    training_epochs = 400
else:
    training_epochs = 1500
learning_rate = 0.001

# Regularization
dropout_rate = 0.4
l2_regularization = 0.0001

codename = "resnext101_00"
training_codename = hf.get_training_codename(
    code_name = codename,
    num_classes = num_classes,
    clinical_data = clinical_data,
    use_layer = use_layer,
    is_cutout = cutout,
    is_rgb_images = rgb_images,
    contrast_DA = contrast_DA,
    is_learning_rate_tuning = learning_rate_tuning,
    is_k_fold = use_k_fold
)


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
            model = build_resnext_model(clinical_data = clinical_data, use_layer = use_layer, architecture = "ResNeXt101")

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

    elif learning_rate_tuning:

        train_data, val_data, test_data = hf.setup_data(path_to_tfrs, path_to_callbacks, path_to_splits, num_classes, batch_size = batch_size,rgb = rgb_images)
        
        callbacks = hf.get_callbacks(path_to_callbacks, 0,
                                     use_lrscheduler = True,
                                     use_early_stopping = False)

        # build model
        model = build_resnext_model(clinical_data = clinical_data, use_layer = use_layer, architecture = "ResNeXt101")

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
        train_data, val_data, test_data = hf.setup_data(path_to_tfrs, path_to_callbacks, path_to_splits, num_classes, batch_size = batch_size,rgb = rgb_images)

        # get callbacks
        callbacks = hf.get_callbacks(path_to_callbacks, 0)

        # build model
        model = build_resnext_model(clinical_data = clinical_data, use_layer = use_layer, architecture = "ResNeXt101")

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


def build_resnext_model(clinical_data, use_layer, architecture="ResNeXt50"):

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
        activation = activation_func,
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
    image_input = tf.keras.layers.Input(shape=(240, 240, 4))
    sex_input = tf.keras.layers.Input(shape=(1,))
    age_input = tf.keras.layers.Input(shape=(1,))
    layer_input = tf.keras.layers.Input(shape=(1,))

    augment = data_augmentation(image_input)
    batch_normed_augment = tf.keras.layers.BatchNormalization()(augment)

    x = DefaultConv2D(filters=64, kernel_size=7, strides=2, input_shape=[240, 240, 4])(batch_normed_augment)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation_func)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(x)

    cardinality = 32
    filters = 128
    #repetitions = [3, 4, 6, 3]
    input_filters = x.shape[-1]
    for i, reps in enumerate(repetitions):
        for j in range(reps):
            strides = 1 if j != 0 else 2 #if i == 0 or
            x = ResNeXtBlock(filters, cardinality, strides=strides, input_filters=input_filters)(x)
            input_filters = filters
        filters *= 2

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    resnext = tf.keras.layers.Flatten()(x)

    # Clinical Data Usage
    if clinical_data == True and use_layer == True:
        concatenated_inputs = tf.keras.layers.Concatenate()([
            resnext,
            sex_input,
            age_input,
            layer_input
        ])
    elif clinical_data == True and use_layer == False:
        concatenated_inputs = tf.keras.layers.Concatenate()([
            resnext,
            sex_input,
            age_input
        ])
    elif clinical_data == False and use_layer == True:
        concatenated_inputs = tf.keras.layers.Concatenate()([
            resnext,
            layer_input
        ])
    else:
        # if clinical data is not wanted, then only the image is used
        concatenated_inputs = resnext

    dense_1_layer = DefaultDenseLayer(units = 512)
    dropout_1_layer = tf.keras.layers.Dropout(dropout_rate)
    dense_2_layer = DefaultDenseLayer(units = 256)
    dropout_2_layer = tf.keras.layers.Dropout(dropout_rate)

    x = tf.keras.layers.BatchNormalization()(concatenated_inputs)
    x = dense_1_layer(x)
    x = dropout_1_layer(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = dense_2_layer(x)
    x = dropout_2_layer(x)

    match num_classes:
        case 2:
            x = tf.keras.layers.Dense(1)(x)
            output = tf.keras.layers.Activation('sigmoid', dtype='float32', name='predictions')(x)
        case 3 | 4 | 5 | 6:
            x = tf.keras.layers.Dense(num_classes)(x)
            output = tf.keras.layers.Activation('softmax', dtype='float32', name='predictions')(x)
        case _:
            raise ValueError("num_classes must be 2, 3, 4, 5 or 6.")

    model = tf.keras.Model(inputs=[image_input, sex_input, age_input, layer_input], outputs=[output], name=architecture)

    if num_classes > 2:
        model.compile(
            loss = "sparse_categorical_crossentropy",
            optimizer = optimizer,
            metrics = ["RootMeanSquaredError", "accuracy"]
        )
    else:
        model.compile(
            loss = "binary_crossentropy",
            optimizer = optimizer,
            metrics = ["RootMeanSquaredError", "accuracy"]
        )
    model.summary()

    return model

    
if contrast_DA:
    data_augmentation = hf.contrast_data_augmentation
else:
    data_augmentation = hf.normal_data_augmentation

if __name__ == "__main__":
    train_ai()