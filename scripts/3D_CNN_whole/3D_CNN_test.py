import tensorflow as tf
import os
import random

from pathlib import Path

#import matplotlib.pyplot as plt

from functools import partial

random.seed(42)

# Variables
path_to_tfrs = "/tfrs"

## train / val / test split
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

batch_size = 1

activation_func = "relu"
optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)


def train_ai():
    tfr_paths = get_tfr_paths()
    train_paths, val_paths, test_paths = split_data(tfr_paths)
    train_data, val_data, test_data = read_data(train_paths, val_paths, test_paths)
    model = build_model()
    model.fit(train_data, validation_data = val_data, epochs=20, batch_size=1,)
    score = model.evaluate(test_data)


def get_tfr_paths():
    tfr_file_names = [file for file in os.listdir(path_to_tfrs) if file.endswith(".tfrecord")]
    tfr_file_names = random.shuffle(tfr_file_names)

    tfr_paths = [str(path_to_tfrs) + "/" + file for file in tfr_file_names]

    print(f"total tfrs: {len(tfr_paths)}")

    return tfr_paths

def split_data(tfr_paths):

    train_size = int(len(tfr_paths) * train_ratio)
    val_size = int(len(tfr_paths) * val_ratio)
    test_size = int(len(tfr_paths) * test_ratio)

    train_paths = tfr_paths[:train_size]
    val_paths = tfr_paths[train_size:train_size + val_size]
    test_paths = tfr_paths[train_size + val_size:]

    sum = len(train_paths) + len(val_paths) + len(test_paths)
    if sum != len(tfr_paths):
        print("WARNING: error occured in train / val / test split!")

    return train_paths, val_paths, test_paths

def read_data(train_paths, val_paths, test_paths):

    train_data = tf.data.TFRecordDataset([train_paths], compression_type="GZIP")
    val_data = tf.data.TFRecordDataset([val_paths], compression_type="GZIP")
    test_data = tf.data.TFRecordDataset([test_paths], compression_type="GZIP")

    train_data = train_data.map(partial(parse_record, labeled = True))
    val_data = val_data.map(partial(parse_record, labeled = True))
    test_data = test_data.map(partial(parse_record, labeled = True))

    train_data = train_data.shuffle(buffer_size=100)
    val_data = val_data.shuffle(buffer_size=100)
    test_data = test_data.shuffle(buffer_size=100)

    train_data = train_data.batch(batch_size)
    val_data = val_data.batch(batch_size)
    test_data = test_data.batch(batch_size)

    train_data = train_data.prefetch(buffer_size=1)
    val_data = val_data.prefetch(buffer_size=1)
    test_data = test_data.prefetch(buffer_size=1)

    return train_data, val_data, test_data


data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip(mode = "horizontal"),
    tf.keras.layers.RandomBrightness(factor = (-0.2, 0.2), value_range=(0, 1)),
    #tf.keras.layers.RandomContrast(0.5), # consider removing the random contrast layer as that causes pixel values to go beyond 1
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


def build_model():
    batch_norm_layer = tf.keras.layers.BatchNormalization()
    conv_1_layer = tf.keras.layers.Conv3D(filters = 64, kernel_size = 7, input_shape = [155, 240, 240, 4], strides=(2,2,2), activation=activation_func, kernel_initializer=tf.keras.initializers.HeNormal())
    max_pool_1_layer = tf.keras.layers.MaxPooling3D(pool_size = (2,2,2))
    conv_2_layer = tf.keras.layers.Conv3D(filters = 64, kernel_size = 7, strides=(2,2,2), activation=activation_func, kernel_initializer=tf.keras.initializers.HeNormal())
    max_pool_2_layer = tf.keras.layers.MaxPooling3D(pool_size = (2,2,2))
    dense_1_layer = tf.keras.layers.Dense(100, activation=activation_func, kernel_initializer=tf.keras.initializers.HeNormal())
    dropout_1_layer = tf.keras.layers.Dropout(0.5)
    dense_2_layer = tf.keras.layers.Dense(100, activation=activation_func, kernel_initializer=tf.keras.initializers.HeNormal())
    dropout_2_layer = tf.keras.layers.Dropout(0.5)
    output_layer = tf.keras.layers.Dense(2, activation="softmax")

    # Define inputs
    input_image = tf.keras.layers.Input(shape=(155, 240, 240, 4))
    sex_input = tf.keras.layers.Input(shape=(2,))
    age_input = tf.keras.layers.Input(shape=(1,))

    # concatenate input sex and input age

    batch_norm = batch_norm_layer(input_image)
    conv_1 = conv_1_layer(batch_norm)
    max_pool_1 = max_pool_1_layer(conv_1)
    conv_2 = conv_2_layer(max_pool_1)
    max_pool_2 = max_pool_2_layer(conv_2)

    flattened_images = tf.keras.layers.Flatten()(max_pool_2)
    flattened_sex_input = tf.keras.layers.Flatten()(sex_input)
    age_input_reshaped = tf.keras.layers.Reshape((1,))(age_input)  # Reshape age_input to have 2 dimensions
    concatenated_inputs = tf.keras.layers.Concatenate()([flattened_images, age_input_reshaped, flattened_sex_input])

    dense_1 = dense_1_layer(concatenated_inputs)
    dropout_1 = dropout_1_layer(dense_1)
    dense_2 = dense_2_layer(dropout_1)
    dropout_2 = dropout_2_layer(dense_2)
    output = output_layer(dropout_2)

    model = tf.keras.Model(inputs = [input_image, sex_input, age_input], outputs = [output])
    model.compile(loss="mse", optimizer=optimizer, metrics = ["RootMeanSquaredError", "accuracy"])
    model.summary()

    return model

if __name__ == "__main__":
    train_ai()