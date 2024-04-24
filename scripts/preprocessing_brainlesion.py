# This file is basically a script version of the
# preprocessing_brain_lesion.ipynb file
# For more information, look there :)

# Import necessary libraries

import matplotlib.pyplot as plt
from auxiliary.normalization.percentile_normalizer import PercentileNormalizer
from auxiliary.turbopath import turbopath
from tqdm import tqdm
import os
from datetime import datetime

from brainles_preprocessing.brain_extraction import HDBetExtractor
from brainles_preprocessing.modality import Modality
from brainles_preprocessing.preprocessor import Preprocessor
from brainles_preprocessing.registration import (ANTsRegistrator, NiftyRegRegistrator)


def run_preprocessing():
    
    patients_directory = "/Users/LennartPhilipp/Desktop/testing_data/raw_data"
    path_to_output = "/Users/LennartPhilipp/Desktop/testing_data/derivatives"

    # create folder at path to output called Rgb_Brain_Mets_preprocessed
    now = datetime.now()
    timeFormatted = now.strftime("%Y%m%d-%H%M%S")
    path_to_preprocessed_files = f"{path_to_output}/preprocessed_brainlesion_{timeFormatted}"

    preprocessed_folder_exists = False

    for file in os.listdir(path_to_output):
        if "brainlesion" in file:
            path_to_preprocessed_files = f"{path_to_output}/{file}"
            preprocessed_folder_exists = True
    
    if not preprocessed_folder_exists:
        os.mkdir(path_to_preprocessed_files)

    data_dir = turbopath(patients_directory)

    patients = data_dir.dirs()

    for patient in tqdm(patients):
        print("processing: ", patient)
        print(patient.name)
        exam = patient.dirs()[0]
        preprocess_exam_in_brats_style(inputDir = exam,
                                       patID = patient.name,
                                       outputDir = path_to_preprocessed_files)


def preprocess_exam_in_brats_style(inputDir: str, patID: str, outputDir: str) -> None:
    """
    Perform BRATS (Brain Tumor Segmentation) style preprocessing on MRI exam data.

    Args:
        inputDir (str): Path to the directory containing raw MRI files for an exam.

    Raises:
        Exception: If any error occurs during the preprocessing.

    Example:
        brat_style_preprocess_exam("/path/to/exam_directory")

    This function preprocesses MRI exam data following the BRATS style, which includes the following steps:
    1. Normalization using a percentile normalizer.
    2. Registration and correction using NiftyReg.
    3. Brain extraction using HDBet.

    The processed data is saved in a structured directory within the input directory.

    Args:
        inputDir (str): Path to the directory containing raw MRI files for an exam.

    Returns:
        None
    """

    # create subfolder for each patient
    # check if patient directory already exists
    pat_directory = f"{outputDir}/{patID}"
    if patID not in os.listdir(outputDir):
        # if not create new directory for patient
        os.mkdir(pat_directory)
    else:
        print("Warning: patient directory already exists")

    inputDir = turbopath(inputDir)
    outputDir = turbopath(outputDir)
    print("*** start ***")
    brainles_dir = pat_directory
    norm_bet_dir = turbopath(pat_directory) / "preprocessed"

    t1_file = inputDir.files("*T1w.nii.gz")
    t1c_file = inputDir.files("*T1c.nii.gz")
    t2_file = inputDir.files("*T2w.nii.gz")
    flair_file = inputDir.files("*FLAIR.nii.gz")

    # we check that we have only one file of each type
    if len(t1_file) == len(t1c_file) == len(t2_file) == len(flair_file) == 1:
        t1File = t1_file[0]
        t1cFile = t1c_file[0]
        t2File = t2_file[0]
        flaFile = flair_file[0]
        
        # normalizer
        percentile_normalizer = PercentileNormalizer(
            lower_percentile=0.1,
            upper_percentile=99.9,
            lower_limit=0,
            upper_limit=1,
        )
        # define modalities
        center = Modality(
            modality_name="t1c",
            input_path=t1cFile,
            normalized_bet_output_path=norm_bet_dir / patID
            + "_t1c_bet_normalized.nii.gz",
            atlas_correction=True,
            normalizer=percentile_normalizer,
        )

        moving_modalities = [
            Modality(
                modality_name="t1",
                input_path=t1File,
                normalized_bet_output_path=norm_bet_dir / patID
                + "_t1_bet_normalized.nii.gz",
                atlas_correction=True,
                normalizer=percentile_normalizer,
            ),
            Modality(
                modality_name="t2",
                input_path=t2File,
                normalized_bet_output_path=norm_bet_dir / patID
                + "_t2_bet_normalized.nii.gz",
                atlas_correction=True,
                normalizer=percentile_normalizer,
            ),
            Modality(
                modality_name="flair",
                input_path=flaFile,
                normalized_bet_output_path=norm_bet_dir / patID
                + "_fla_bet_normalized.nii.gz",
                atlas_correction=True,
                normalizer=percentile_normalizer,
            ),
        ]

        preprocessor = Preprocessor(
            center_modality=center,
            moving_modalities=moving_modalities,
            registrator=ANTsRegistrator(), # previously NiftRegRegistrator()
            brain_extractor=HDBetExtractor(),
            
            # REMOVE IN PRODUCTION - START
            use_gpu=False,
            # REMOVE IN PRODUCTION - END

            #limit_cuda_visible_devices="0",
        )

        preprocessor.run(
            save_dir_coregistration=brainles_dir + "/co-registration",
            save_dir_atlas_registration=brainles_dir + "/atlas-registration",
            save_dir_atlas_correction=brainles_dir + "/atlas-correction",
            save_dir_brain_extraction=brainles_dir + "/brain-extraction",
        )


if __name__ == "__main__":
    run_preprocessing()