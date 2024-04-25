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

    missing_patients = ['sub-02036251', 'sub-01288350', 'sub-01492723', 'sub-01673701', 'sub-01104996', 'sub-01087386', 'sub-01908576', 'sub-01713022', 'sub-01420310', 'sub-01614295', 'sub-01456959', 'sub-02000864', 'sub-01450871', 'sub-01650072', 'sub-02063398', 'sub-01575055', 'sub-01138456', 'sub-01476909', 'sub-01475524', 'sub-01419998', 'sub-01805334', 'sub-01499528', 'sub-01461078', 'sub-01853095', 'sub-01362907', 'sub-01942928', 'sub-01207036', 'sub-01465229', 'sub-01196057', 'sub-01584596', 'sub-01414540', 'sub-01496608', 'sub-01390721', 'sub-90003562', 'sub-01251946', 'sub-01709242', 'sub-01695930', 'sub-01518885', 'sub-90001992', 'sub-01754011', 'sub-01763867', 'sub-01204563', 'sub-01015961', 'sub-01513891', 'sub-01657294', 'sub-01331487', 'sub-01483116', 'sub-01373703', 'sub-01478990', 'sub-01661279', 'sub-01801060', 'sub-01502083', 'sub-01201482', 'sub-90011887', 'sub-01744565', 'sub-01099901', 'sub-01677324', 'sub-01498464', 'sub-01718213', 'sub-80004059', 'sub-01381621', 'sub-01596127', 'sub-01393875', 'sub-01288245', 'sub-01616246', 'sub-01332588', 'sub-01641510', 'sub-01309950', 'sub-01681275', 'sub-01893873', 'sub-01031243', 'sub-02012594', 'sub-01370265', 'sub-01431720', 'sub-01387984', 'sub-02014685', 'sub-01489395', 'sub-01434869', 'sub-93002557', 'sub-01131702', 'sub-01703264', 'sub-90005031', 'sub-01164986', 'sub-01933711', 'sub-01452858', 'sub-01713570', 'sub-01666008', 'sub-01409764', 'sub-01480742', 'sub-01924748', 'sub-01441531', 'sub-01216717', 'sub-01395836', 'sub-01565091', 'sub-01589112', 'sub-01696845', 'sub-01410235', 'sub-01545797', 'sub-01071055', 'sub-01415245', 'sub-80011453', 'sub-01621161', 'sub-01214172', 'sub-01188297', 'sub-93003757', 'sub-01654658', 'sub-01674416', 'sub-01707721', 'sub-01515235', 'sub-01130856', 'sub-01437004', 'sub-01957247', 'sub-01383503', 'sub-01587295', 'sub-01119720', 'sub-01130173', 'sub-01458719', 'sub-01569328', 'sub-01281168', 'sub-01583797', 'sub-01547588', 'sub-01433377', 'sub-01835095', 'sub-01530724', 'sub-01551183', 'sub-01710250', 'sub-01457167', 'sub-02063373', 'sub-95001254', 'sub-01958155', 'sub-01698789', 'sub-01384142', 'sub-01870024', 'sub-01706562', 'sub-01514331', 'sub-01732456', 'sub-01960441', 'sub-01905848', 'sub-01056884', 'sub-01434617', 'sub-01494236', 'sub-01979997', 'sub-01605537', 'sub-01516618', 'sub-01483723', 'sub-01966470', 'sub-01997658', 'sub-01391534', 'sub-01713725', 'sub-01779701', 'sub-01435731', 'sub-01025630', 'sub-01562247', 'sub-01521599', 'sub-88000225', 'sub-01064662', 'sub-01573094', 'sub-01572564', 'sub-01953116', 'sub-01961566', 'sub-01594137', 'sub-01486069', 'sub-01483526', 'sub-01455312', 'sub-01732889', 'sub-01040149', 'sub-01496804', 'sub-01274157', 'sub-01861511', 'sub-01702596', 'sub-02021781', 'sub-01542729', 'sub-01649133', 'sub-01600788', 'sub-01695173', 'sub-01550202', 'sub-01668785', 'sub-02038513', 'sub-01695094', 'sub-01357275', 'sub-01402283', 'sub-01936520', 'sub-01205745', 'sub-01852952', 'sub-01695080']
    
    # on Lennart's Mac Book: patients_directory = "/Users/LennartPhilipp/Desktop/testing_data/raw_data"
    patients_directory = "/home/lennart/Desktop/brain_mets_regensburg/rawdata"
    # on Lennart's Mac Book: path_to_output = "/Users/LennartPhilipp/Desktop/testing_data/derivatives"
    path_to_output = "/home/lennart/Desktop/brain_mets_regensburg/derivatives"

    # create folder at path to output called Rgb_Brain_Mets_preprocessed
    now = datetime.now()
    timeFormatted = now.strftime("%Y%m%d-%H%M%S")
    path_to_preprocessed_files = f"{path_to_output}/preprocessed_brainlesion_missing_patients_{timeFormatted}"

    preprocessed_folder_exists = False

    for file in os.listdir(path_to_output):
        if "brainlesion_missing" in file:
            path_to_preprocessed_files = f"{path_to_output}/{file}"
            preprocessed_folder_exists = True
    
    if not preprocessed_folder_exists:
        os.mkdir(path_to_preprocessed_files)

    data_dir = turbopath(patients_directory)

    patients = data_dir.dirs()

    for patient in tqdm(patients):
        if patient.name in missing_patients:
            print("skipping patient: ")
            continue
        print("processing: ", patient.name)
        exam = patient.dirs()[0]
        if exam.name != "anat":
            print(f"Warning: this isn't the anat directory {exam}")
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
            use_gpu=True,
            # REMOVE IN PRODUCTION - END

            #limit_cuda_visible_devices="0",
        )

        preprocessor.run(
            save_dir_coregistration=brainles_dir + "/co-registration",
            save_dir_atlas_registration=brainles_dir + "/atlas-registration",
            save_dir_atlas_correction=brainles_dir + "/atlas-correction",
            save_dir_brain_extraction=brainles_dir + "/brain-extraction",
        )

        print(f"finished preprocessing for {patID}")


if __name__ == "__main__":
    run_preprocessing()