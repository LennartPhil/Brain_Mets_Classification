# This skript is based on the data_preparation.ipynb notebook
# For further information about each function please see the notebook

import tensorflow as tf

import nibabel as nib
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy import ndimage
from pathlib import Path

from tensorflow.train import BytesList, FloatList, Int64List
from tensorflow.train import Feature, Features, Example

from sklearn.utils import class_weight

from tqdm import tqdm

num_classes = 2

path_to_output = "/output" # path to the folder where the TFRecords folder will be saved
path_to_preprocessed_directory = "/data" # path to the folder where the preprocessed data is stored
path_to_csv = "/patients.tsv"

def data_preparation():
    training_patients = patient_table_setup()
    unified_primaries = unify_primaries(training_patients)
    compute_class_weights(unified_primaries)
    sex_encoded = one_hot_encode_sex(training_patients)
    write_tfr_record(sex_encoded, training_patients, unified_primaries)

def patient_table_setup():
    training_patients = pd.read_csv(path_to_csv, sep="\t", index_col=False)

    # drop patient sub-01383503
    patient_to_drop_index = training_patients.index[training_patients["participant_id"] == "sub-01383503"]
    training_patients.drop(index=patient_to_drop_index, inplace=True)
    training_patients.reset_index(drop=True, inplace=True)

    print(f"Before removing unfit files: {len(training_patients)}")
    patient_files_list = os.listdir(path_to_preprocessed_directory)
    for index, row in training_patients.iterrows():
        if training_patients["participant_id"][index] not in patient_files_list:
            training_patients.drop(index=index, inplace=True)

    print(f"After removing unfit files: {len(training_patients)}")
    training_patients.reset_index(drop=True, inplace=True)

    # shuffle dataset
    training_patients = training_patients.sample(frac=1).reset_index(drop=True)
    return training_patients

def unify_primaries(patients_table):
    primaries_array = patients_table["primary"]

    compressed_list = pd.Series(compress_primaries(primaries_array))

    two_classes_primaries = pd.Series(return_modified_primaries(compressed_list, num_classes=num_classes))

    print("unify_primaries successful")

    return two_classes_primaries

def compress_primaries(primaries_array):
    '''moves all the primaries from different subclasses into one class, e.g. 1a-f become 1 etc.'''
    
    letters_removed_primaries = []

    # remove any letters from the list
    for primary in primaries_array:
        clean_primary = ''.join(filter(str.isdigit, primary))
        letters_removed_primaries.append(clean_primary)

    compressed_primaries = []

    for primary in letters_removed_primaries:
        primary_num = int(primary)
        renamed_primary = 0
        # compress all the genitourinary cancers togehter (3-10)
        if primary_num >= 3 and primary_num <= 10:
            renamed_primary = 3
        # compress all the gastrointestinal cancers together (19-25)
        elif primary_num >= 19 and primary_num <= 25:
            renamed_primary = 19
        # compress all the head and neck cancers together (13-18)
        elif primary_num >= 13 and primary_num <= 18:
            renamed_primary = 13
        else:
            renamed_primary = primary_num
        
        compressed_primaries.append(renamed_primary)
        
    return compressed_primaries

def return_modified_primaries(primaries_array, num_classes):
    '''returns an array where all the items are grouped into x classes depening on num_classes
    e.g. if num_classes = 2, then only the most frequent category (lung cancer) gets returned while all the other categories are grouped as \'other\''''
    
    # get most frequent classes
    # go through the list and replace each item that is not in the most frequent classes with "other"
    # the following code is probably one the least efficient ways to solve this problem
    # but it works so who am I to change it
    different_primaries = []

    for primary in primaries_array:
        if primary not in different_primaries:
            different_primaries.append(primary)
    
    count_dict = {}

    print(different_primaries)

    for dif_primary in different_primaries:
        count = list(primaries_array).count(dif_primary)
        count_dict[count] = dif_primary
    
    sorted_dict = sorted(count_dict, reverse=True)

    white_list_count = []

    for n in range(num_classes - 1):
        white_list_count.append(sorted_dict[n])

    white_list = []
    for n in white_list_count:
        white_list.append(count_dict[n])

    modified_array = []

    for primary in primaries_array:
        modified_primary = 0

        if primary not in white_list:
            modified_primary = 0
        else:
            modified_primary = primary
        
        modified_array.append(modified_primary)
    
    return modified_array

def compute_class_weights(primaries):
    labels = primaries.to_numpy()
    
    if num_classes == 2:
        classes = np.array([1, 0])
    else:
        raise Exception("Please adjust the num_classes variable in the compute_class_weights function to the number of classes you have in your data set")
    
    weights = class_weight.compute_class_weight(class_weight="balanced",
                                                classes=classes,
                                                y=labels)

    print("Class weights: ")
    print(weights)
    print()

def one_hot_encode_sex(patients_table):
    sex_array = patients_table["sex (m/f)"]

    print(sex_array.value_counts())

    sex_encoded = []
    for sex in sex_array:
        if sex == "m":
            sex_encoded.append([1, 0])
        elif sex == "f":
            sex_encoded.append([0, 1])
        else:
            print(f"unknown sex: {sex}")
            sex_encoded.append([0, 0])

    print("one_hot_encode_sex successful")

    return sex_encoded

def prepare_images(patientID):
    loaded_images = load_patient(patientID)
    rotated_images = rotate_90_deg(loaded_images)
    return merge_and_transpose_images(rotated_images)

def load_patient(patientID):
    """loads the images for a specific patient and returns a tensorflow tensor"""
    images = []
    # get all four sequences
    patientID = str(patientID)
    patient_path = Path(patientID)
    image_names = os.listdir(Path(path_to_preprocessed_directory) / patient_path / "perc_normalized")
    
    # load them
    for name in image_names:
        path_to_image = Path(path_to_preprocessed_directory) / patient_path / "perc_normalized" / Path(name)
        image = nib.load(path_to_image)
        data = image.get_fdata()
        #tensor = tf.convert_to_tensor(data, dtype = float)
        images.append(data)
    
    if len(images) != 4:
        print(f"Warning: either too many or too few images for {patientID} (#{len(images)})")
    
    # return four images as array
    return images

def rotate_90_deg(images):
    """rotates images by 90 degrees"""
    # rotate images
    rotated_images = []
    for image in images:
        rotated_image = ndimage.rotate(np.array(image), angle = 90)
        #rotated_images.append(tf.convert_to_tensor(rotated_image, dtype = float))
        rotated_images.append(rotated_image)


    # return back
    return rotated_images

def merge_and_transpose_images(images):
    """merge images so that the fourth dimension used for the different sequences"""
    # merge image
    stacked = tf.stack(images, axis = -1)

    new_order = [2, 0, 1, 3]
    transposed = np.transpose(stacked, axes=new_order)
    # transposed = tf.transpose(stacked, perm = new_order)
    return transposed

def write_tfr_record(sex_encoded, training_patients, classes_primaries):

    print("Starting TFRecord writing")

    path_to_tfrecords = Path(path_to_output) / "TFRecords"
    os.makedirs(path_to_tfrecords, exist_ok=True)

    path_to_preprocessed_patients = Path(path_to_preprocessed_directory)

    # Write the dataset to TFRecord
    options = tf.io.TFRecordOptions(compression_type="GZIP") # compress the dataset

    patients = [patient for patient in os.listdir(path_to_preprocessed_patients) if os.path.isdir(os.path.join(path_to_preprocessed_patients, patient))]

    directory_name = f"pats_{num_classes}_singles"
    path_to_all_tfr = path_to_tfrecords / Path(directory_name)
    os.makedirs(path_to_all_tfr, exist_ok=True)

    for pat in tqdm(range(len(patients))):

        file_path = str(path_to_all_tfr) + "/" + patients[pat] + ".tfrecord"
        with tf.io.TFRecordWriter(file_path, options) as writer:

            sex = sex_encoded[pat]
            id = training_patients["participant_id"][pat]
            age = training_patients["age"][pat]
            primary = classes_primaries[pat]
            example = serialize_patient(id, sex, age, primary)
            writer.write(example)
    
    print("TFRecord writing successful")

def serialize_patient(patientID, sex, age, primary):

    image_data = prepare_images(patientID)

    patient_example = Example(
        features = Features(
            feature = {
                'image': tf.train.Feature(float_list=tf.train.FloatList(value=image_data.ravel())),
                'sex': tf.train.Feature(int64_list=tf.train.Int64List(value= sex)),
                'age': tf.train.Feature(int64_list=tf.train.Int64List(value=[age])),
                'primary': tf.train.Feature(int64_list=tf.train.Int64List(value=[primary])),
            }   
        )
    )

    return patient_example.SerializeToString()

if __name__ == "__main__":
    data_preparation()