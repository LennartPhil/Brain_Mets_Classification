# This script executes the code that trains the 3D CNN network
# For more information about the code used in this script please see the 3D_CNN.ipynb file

# To-do:
# - Let user choose how many classes are going to get trained
# - let user choose the folder for the training data
# - let user choose the export folder for the trained model
# - let user choose the tensorboard folder
# - insert code from 3D CNN

import tensorflow as tf
from pathlib import Path
from time import strftime

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
sys.path.append(r"/Users/LennartPhilipp/Desktop/Uni/Prowiss/Code/Brain_Mets_Classification")

import brain_mets_classification.ai_funcs as ai_funcs

def train_ai():

    num_classes = 2

    tensorflow_setup()

    path_to_callback = Path("/home/lennart/Documents/brain_mets/3D_CNN_(whole)/callbacks")

    training_set, validation_set, testing_set = load_data(
        path_to_tfr_folder=Path("/home/lennart/Documents/TFRecords/3D-CNN_(whole)"),
        num_classes=num_classes
    )

    train_images, train_sex, train_ages, train_primaries = training_set
    val_images, val_sex, val_ages, val_primaries = validation_set
    test_images, test_sex, test_ages, test_primaries = validation_set

    class_weigths = get_class_weights(
        primaries = [train_primaries, val_primaries, test_primaries],
        num_classes = num_classes
    )

    callbacks = get_callbacks(path_to_callbacks = path_to_callback)

    model = build_ai(
        train_ages = train_ages,
        train_sex = train_sex,
        num_classes = num_classes,
        class_weights = class_weigths
    )

    training_input = [train_images, train_ages, train_sex]

    history = model.fit(training_input, train_primaries,
                        epochs=30, batch_size=5,
                        validation_data=([val_images, val_ages, val_sex], val_primaries),
                        callbacks = callbacks)
    
    print("training successful")

def tensorflow_setup():

    tf.keras.utils.set_random_seed(42)

    # copied directly from: https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
    
    print("tensorflow_setup successful")

def load_data(path_to_tfr_folder, num_classes):
    '''loads the data from the TFRecord file, depending on the number of classes
    Applies data augmentation to the trainig set

    Args:
    - path_to_tfr_folder (String): path to the folder that contains the different TFRecord files
    - num_classes (Int): the number of primary classes that should get clasified (value between 2 and 6)
    
    Returns the training, validation and testing set'''

    match num_classes:
        case 2:
            path_to_tfr = path_to_tfr_folder / "patient_data_2classes.tfrecord"
        case 3:
            path_to_tfr = path_to_tfr_folder / ""
        case 4:
            path_to_tfr = path_to_tfr_folder / ""
        case 5:
            path_to_tfr = path_to_tfr_folder / ""
        case 6:
            path_to_tfr = path_to_tfr_folder / ""
        case _:
            print("there's no dataset matching this amount of classes")

    feature_description = {
        "image": tf.io.FixedLenFeature([155, 240, 240, 4], tf.float32),
        "sex": tf.io.FixedLenFeature([2], tf.int64, default_value=[0,0]),
        "age": tf.io.FixedLenFeature([], tf.int64, default_value=0),
        "primary": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    }

    def parse(serialize_patient):
        example = tf.io.parse_single_example(serialize_patient, feature_description)
        image = example["image"]
        image = tf.reshape(image, [155, 240, 240, 4])
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
    dataset = parsed_dataset.shuffle(buffer_size=200) #formerly: 200
    train_dataset = dataset.take(train_size).prefetch(buffer_size = tf.data.AUTOTUNE)
    remainder_dataset = dataset.skip(train_size).prefetch(buffer_size = tf.data.AUTOTUNE)
    val_dataset = remainder_dataset.take(val_size).prefetch(buffer_size = tf.data.AUTOTUNE)
    test_dataset = remainder_dataset.skip(val_size).prefetch(buffer_size = tf.data.AUTOTUNE)

    # augmentation Sequential (should only be applied to the training set)
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip(mode = "horizontal"),
        tf.keras.layers.RandomBrightness(factor = (-0.2, 0.4), value_range=(0, 1)),
        tf.keras.layers.RandomRotation(factor = (-0.07, 0.07), fill_mode = "nearest"),
        tf.keras.layers.RandomTranslation(
            height_factor = 0.05,
            width_factor = 0.05,
            fill_mode = "nearest"
        )
    ])

    # split the dataset into images, ages, sexes and primaries
    def split_dataset(dataset, augmentation: bool = False):
        images = []
        ages = []
        sexes = []
        primaries = []
        for image, sex, age, primary in dataset:
            if augmentation:
                augmented_image = data_augmentation(image)
                images.append(augmented_image)
            else:
                images.append(image)
            ages.append(age)
            sexes.append(sex)
            primaries.append(primary)
        return tf.stack(images), tf.stack(sexes), tf.stack(ages), tf.stack(primaries)

    # split each dataset into the MRI images, the sex, age and primary for each patient
    training_set = split_dataset(train_dataset, augmentation=True)
    validation_set = split_dataset(val_dataset)
    testing_set = split_dataset(test_dataset)

    print("load_data successful")

    return training_set, validation_set, testing_set

def get_class_weights(primaries, num_classes):
    '''calculates the class weights for the loss function
    
    Args:
    - primaries (Array): array of the primaries of the dataset to use for the calculation of the class_weights
    - num_classes: (Int): the number of primary classes that should get clasified (value between 2 and 6)
    
    Returns the clas_weights to use for the custom loss function'''

    all_primaries = tf.concat(primaries, -1)
    
    match num_classes:
        case 2:
            class_weights = ai_funcs.compute_class_weights(all_primaries, [1,0])
        case 3:
            pass
        case 4:
            pass
        case 5:
            pass
        case 6:
            pass
        case _:
            print("Wrong num classes set in the get_class_weights func, please pick a number between 2 and 6")
    
    print("get_class_weights successful")

    return class_weights

def get_callbacks(path_to_callbacks: Path):
    '''creates three different callbacks to use for the model that get stored at the path_to_callbacks
    - Checkpoint Callback: to save the weights of the best models
    - Early Stopping Callback: to stop when the accuracy doesn't increase anymore
    - Tensorboard Callback: logs the parameters for tensorboard visualisation

    Arg:
        path_to_callbacks: Path
    '''

    def get_run_logdir(root_logdir= path_to_callbacks / "tensorboard"):
        return Path(root_logdir) / strftime("run_%Y_%m_%d_%H_%M_%S")

    run_logdir = get_run_logdir()

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath = path_to_callbacks / "saved_weights.weights.h5",
                                                    monitor = "val_accuracy",
                                                    mode = "max",
                                                    save_best_only = True,
                                                    save_weights_only = True)

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10,
                                                        restore_best_weights = True)

    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir = run_logdir,
                                                    histogram_freq = 1)
    
    print("get_callbacks successful")

    return [checkpoint_cb, early_stopping_cb, tensorboard_cb]

# Custom Weighted Cross Entropy Loss
class WeightedCrossEntropyLoss(tf.keras.losses.Loss):
    def __init__(self, class_weights):
        super().__init__()
        # Convert class weights to a tensor
        self.class_weights = tf.constant(class_weights, dtype=tf.float32)

    def call(self, y_true, y_pred):
        # Compute the weighted cross-entropy loss
        y_true = tf.cast(y_true, tf.int64)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)  # Avoid log(0) error

        # Convert y_true to one-hot encoding
        y_true_one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[1])

        # Compute cross entropy
        cross_entropy = -tf.reduce_sum(y_true_one_hot * tf.math.log(y_pred), axis=-1)

        # Apply the weights
        weights = tf.gather(self.class_weights, y_true)
        weighted_cross_entropy = weights * cross_entropy

        return tf.reduce_mean(weighted_cross_entropy)

# MCDropout
# https://arxiv.org/abs/1506.02142
class MCDropout(tf.keras.layers.Dropout):
    def call(self, inputs, training=False):
        return super().call(inputs, training=True)

def build_ai(train_ages, train_sex, num_classes, class_weights):
    """build the 3D CNN
    
    Args:
        train_ages: the set of ages used for training, needed for the input layer
        train_sex: the set of sexes used for training, needed for the input layer
        num_classes: the amount of classes the classifier gets trained to set the right amount of output layers"""

    input_shape = (155,240,240,4)

    img_input = tf.keras.layers.Input(shape=input_shape)
    age_input = tf.keras.layers.Input(shape=train_ages.shape[1:])
    sex_input = tf.keras.layers.Input(shape=train_sex.shape[1:])

    batchnormalized_images = tf.keras.layers.BatchNormalization()(img_input)
    output_tensor = ai_funcs.create_res_next(
        img_input = batchnormalized_images,
        depth = [3,4,6,3],
        cardinality = 32,
        width = 4,
        weight_decay = 5e-4,
        pooling = "avg"
    )

    flattened_images = tf.keras.layers.Flatten()(output_tensor)
    flattened_sex_input = tf.keras.layers.Flatten()(sex_input)
    age_input_reshaped = tf.keras.layers.Reshape((1,))(age_input)  # Reshape age_input to have 2 dimensions
    concatenated_inputs = tf.keras.layers.Concatenate()([flattened_images, age_input_reshaped, flattened_sex_input])

    x = MCDropout(0.4)(concatenated_inputs)
    x = tf.keras.layers.Dense(200, activation="mish")(x)
    x = MCDropout(0.4)(x)
    x = tf.keras.layers.Dense(200, activation="mish")(x)
    x = MCDropout(0.4)(x)
    x = tf.keras.layers.Dense(200, activation="mish")(x)

    match num_classes:
        case 2:
            output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        case 3:
            output = tf.keras.layers.Dense(3, activation='softmax')(x)
        case 4:
            output = tf.keras.layers.Dense(4, activation='softmax')(x)
        case 5:
            output = tf.keras.layers.Dense(5, activation='softmax')(x)
        case 6:
            output = tf.keras.layers.Dense(6, activation='softmax')(x)
        case _:
            print("Wrong num classes set in the buil_ai func, please pick a number between 2 and 6")

    model = tf.keras.Model(inputs=[img_input, age_input, sex_input], outputs=output)

    loss = WeightedCrossEntropyLoss(class_weights=class_weights)

    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

    return model


if __name__=="__main__":
    train_ai()