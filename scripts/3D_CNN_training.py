# This script executes the code that trains the 3D CNN network
# For more information about the code used in this script please see the 3D_CNN.ipynb file

# To-do:
# - Let user choose how many classes are going to get trained
# - let user choose the folder for the training data
# - let user choose the export folder for the trained model
# - let user choose the tensorboard folder
# - insert code from 3D CNN

import argparse
import tensorflow as tf

import sys
sys.path.append(r"/Users/LennartPhilipp/Desktop/Uni/Prowiss/Code/Brain_Mets_Classification")

import brain_mets_classification.ai_funcs as ai_funcs

def train_ai():

    tf.keras.utils.set_random_seed(42)

    training_set, validation_set, testing_set = load_data()

    train_images, train_set, traing_ages, train_primaries = training_set
    val_images, val_sex, val_ages, val_primaries = validation_set
    test_images, test_sex, test_ages, test_primaries = validation_set

    model = build_ai()


def load_data(path_to_tfr_folder, num_classes):

    match num_classes:
        case 2:
            path_to_tfr = ""
        case 2:
            path_to_tfr = ""
        case 2:
            path_to_tfr = ""
        case 2:
            path_to_tfr = ""
        case 2:
            path_to_tfr = ""
        case _:
            print("there's no dataset matching this amount of classes")

    feature_description = {
        "image": tf.io.FixedLenFeature([149, 185, 155, 4], tf.float32),
        "sex": tf.io.FixedLenFeature([2], tf.int64, default_value=[0,0]),
        "age": tf.io.FixedLenFeature([], tf.int64, default_value=0),
        "primary": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    }

    def parse(serialize_patient):
        example = tf.io.parse_single_example(serialize_patient, feature_description)
        image = example["image"]
        image = tf.reshape(image, [149, 185, 155, 4])
        return image, example["sex"], example["age"], example["primary"]

    dataset = tf.data.TFRecordDataset([path_to_tfr], compression_type="GZIP")
    parsed_dataset = dataset.map(parse)

    # split the dataset into 80% training, 10% validation and 10% testing
    total_samples = sum(1 for _ in parsed_dataset)
    train_size = int(0.8 * total_samples)
    val_size = int(0.1 * total_samples)
    test_size = total_samples - train_size - val_size

    print(f"Training size: {train_size}")
    print(f"Validation size: {val_size}")
    print(f"Testing size: {test_size}")

    # Shuffle dataset
    dataset = parsed_dataset.shuffle(buffer_size=200)
    train_dataset = dataset.take(train_size).prefetch(buffer_size = tf.data.AUTOTUNE)
    remainder_dataset = dataset.skip(train_size).prefetch(buffer_size = tf.data.AUTOTUNE)
    val_dataset = remainder_dataset.take(val_size).prefetch(buffer_size = tf.data.AUTOTUNE)
    test_dataset = remainder_dataset.skip(val_size).prefetch(buffer_size = tf.data.AUTOTUNE)

    def split_dataset(dataset):
        images = []
        ages = []
        sexes = []
        primaries = []
        for image, sex, age, primary in dataset:
            images.append(image)
            ages.append(age)
            sexes.append(sex)
            primaries.append(primary)
        return tf.stack(images), tf.stack(sexes), tf.stack(ages), tf.stack(primaries)

    # split each dataset into the MRI images, the sex, age and primary for each patient
    training_set = split_dataset(train_dataset)
    validation_set = split_dataset(val_dataset)
    testing_set = split_dataset(test_dataset)

    return training_set, validation_set, testing_set


def build_ai(train_ages, train_sex, num_classes):

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip(mode = "horizontal"),
        tf.keras.layers.RandomBrightness(factor = (-0.2, 0.5), value_range=(0, 1)), # consider adding later
        tf.keras.layers.RandomContrast(0.5), # consider adding later
        tf.keras.layers.RandomRotation(factor = (-0.07, 0.07), fill_mode = "nearest"),
        tf.keras.layers.RandomTranslation(
           height_factor = 0.025,
           width_factor = 0.05,
           fill_mode = "nearest"
        )
    ])

    input_shape = (149,185,155,4)
    nb_classes = 2

    img_input = tf.keras.layers.Input(shape=input_shape)
    age_input = tf.keras.layers.Input(shape=train_ages.shape[1:])
    sex_input = tf.keras.layers.Input(shape=train_sex.shape[1:])

    augmented_images = data_augmentation(img_input)

    output_tensor = ai_funcs.create_res_next(nb_classes = nb_classes,
                                      img_input = augmented_images,
                                      depth = [3,4,6,3],
                                      cardinality = 32,
                                      width = 4,
                                      weight_decay = 5e-4,
                                      pooling = "avg")

    flattened_images = tf.keras.layers.Flatten()(output_tensor)
    flattened_sex_input = tf.keras.layers.Flatten()(sex_input)
    # EDIT START
    age_input_reshaped = tf.keras.layers.Reshape((1,))(age_input)  # Reshape age_input to have 2 dimensions
    # EDIT END
    concatenated_inputs = tf.keras.layers.Concatenate()([flattened_images, age_input_reshaped, flattened_sex_input])

    x = MCDropout(0.4)(concatenated_inputs)
    x = tf.keras.layers.Dense(200, activation="mish")(x)
    x = MCDropout(0.4)(x)
    x = tf.keras.layers.Dense(200, activation="mish")(x)
    x = MCDropout(0.4)(x)
    x = tf.keras.layers.Dense(200, activation="mish")(x)

    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=[img_input, age_input, sex_input], outputs=output)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])




# MCDropout
# https://arxiv.org/abs/1506.02142
class MCDropout(tf.keras.layers.Dropout):
    def call(self, inputs, training=False):
        return super().call(inputs, training=True)

if __name__=="__main__":
    train_ai()