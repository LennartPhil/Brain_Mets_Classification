# This file is basically a script version of the
# 02_preprocessing_all_at_once copy.ipynb file
# For more information, look there :)

# Import necessary libraries

import sys
sys.path.append(r"/Users/LennartPhilipp/Desktop/Uni/Prowiss/Code/Brain_Mets_Classification")

import brain_mets_classification.config as config
import brain_mets_classification.custom_funcs as funcs
import brain_mets_classification.preprocessing_funcs as preprocessing

from tqdm import tqdm
from datetime import datetime
import subprocess

import shutil

import pandas as pd
import os
import pathlib
import ants
from typing import Union, List, Tuple
import multiprocessing
import SimpleITK as sitk
from nipype.interfaces.dcm2nii import Dcm2niix
import numpy as np
from nipype.interfaces import fsl
from intensity_normalization.normalize.zscore import ZScoreNormalize

def run_preprocessing():

    path_to_patients = "/Volumes/BrainMets/Rgb_Brain_Mets/brain_mets_classification/rawdata"
    path_to_output = "/Volumes/BrainMets/Rgb_Brain_Mets/brain_mets_classification/derivatives"
    
    # create folder at path to output called Rgb_Brain_Mets_preprocessed
    now = datetime.now()
    timeFormatted = now.strftime("%Y%m%d-%H%M%S")
    path_to_preprocessed_files = f"{path_to_output}/preprocessed_mnit_{timeFormatted}"

    preprocessed_folder_exists = False

    for file in os.listdir(path_to_output):
        if "mnit" in file:
            path_to_preprocessed_files = f"{path_to_output}/{file}"
            preprocessed_folder_exists = True
    
    if not preprocessed_folder_exists:
        os.mkdir(path_to_preprocessed_files)

    # gets only the folders at path and puts them in an array 
    patient_folders = [
        folder for folder in os.listdir(path_to_patients) if os.path.isdir(os.path.join(path_to_patients, folder))
    ]

    path_to_txt = f"{path_to_preprocessed_files}/patients.txt"

    text_file_already_exists = False
    
    for file in os.listdir(path_to_preprocessed_files):
        if file.endswith(".txt"):
            text_file_already_exists = True

    if not text_file_already_exists:
        with open(path_to_txt, "w") as f:
            f.write("\n".join(patient_folders))

    filetypes_to_remove = ["reoriented", "brainextracted", "n4biascorrected", "coregistered"]

    error_patients = {}

    try: path_to_txt
    except: path_to_txt = "/Volumes/BrainMets/Rgb_Brain_Mets/brain_mets_classification/derivatives/preprocessed_20240131-135755/patients.txt"

    try: path_to_preprocessed_files
    except: path_to_preprocessed_files = "/Volumes/BrainMets/Rgb_Brain_Mets/brain_mets_classification/derivatives/preprocessed_20240131-135755"

    txt_file = open(path_to_txt, "r+")
    patients = txt_file.read().splitlines()
    print(f"path to txt: {path_to_txt}")
    print(patients)

    for _ in tqdm(patients):

        # get's first patient in the file
        patientID = patients[0]
        # removes first patient from the array
        patients.pop(0)

        # create folder for patient in path_to_preprocessed_files
        funcs.createFolderForPatient(path_to_preprocessed_files, patientID)

        # get the different sequences for each patient and put them in an array
        niftiSequences = [
            sequence for sequence in os.listdir(os.path.join(path_to_patients, patientID, "anat")) if ".nii" in sequence
        ]

        if len(niftiSequences) < 4:
            error_message = f"Warning: too few nifti files found ({len(niftiSequences)})"
            print(error_message)
            error_patients[patientID] = error_message
            continue
        
        time = datetime.now().strftime("%Y%m%d-%H%M%S")
        print(f"Starting preprocessing for {patientID} - {time}")

        # loop through the nifit sequences
        for niftiSequence in niftiSequences:
            
            # ignore . files
            if niftiSequence.startswith("."):
                continue

            # reorient images
            print(f"{niftiSequence}: Starting reorientation")
            sequenceType = (niftiSequence.split("_")[1]).split(".")[0]
            path_to_input_image = os.path.join(path_to_patients, patientID, "anat", niftiSequence)
            path_to_output_reorientedImage = f"{path_to_preprocessed_files}/{patientID}/{patientID}_{sequenceType}_reoriented.nii"
            preprocessing.reorient_brain(
                path_to_input_image = path_to_input_image,
                path_to_output_image = path_to_output_reorientedImage
            )
            print(f"{niftiSequence}: Finished reorientation")

            # use brain extraction
            print(f"{patientID} {sequenceType}: Starting brain extraction")
            path_to_output_extractedImage = f"{path_to_preprocessed_files}/{patientID}/{patientID}_{sequenceType}_brainextracted.nii"
            extract_brain(
                path_to_input_image = path_to_input_image,
                path_to_output_image = path_to_output_extractedImage,
                device = "cpu")
            print(f"{patientID} {sequenceType}: Finished brain extraction")
    
        # get the brain extracted files
        brainExtractedFiles = [
            sequence for sequence in os.listdir(os.path.join(path_to_preprocessed_files, patientID)) if ("brainextracted" in sequence and "mask" not in sequence)
        ]

        if len(brainExtractedFiles) < 4:
            error_message = f"Warning: too few brain extracted files found ({len(brainExtractedFiles)})"
            print(error_message)
            error_patients[patientID] = error_message
            continue


        # loop through the nifit sequences
        for brainExtractedSequence in brainExtractedFiles:
            print(f"{brainExtractedSequence}: Starting bounding box")

            input_image = sitk.ReadImage(os.path.join(path_to_preprocessed_files, patientID, brainExtractedSequence), imageIO="NiftiImageIO")

            # get and apply a bounding box both to the brain as well as the mask
            croppedImage = apply_bounding_box(
                image = input_image,
                bounding_box = get_bounding_box(image = input_image))
            print(f"{brainExtractedSequence}: Finished bounding box")

            # n4 bias correction
            print(f"{brainExtractedSequence}: Starting n4 bias correction")
            n4correctedImage = apply_bias_correction(image = croppedImage)
            print(f"{brainExtractedSequence}: Finished n4 bias correction")

            # save image
            sequenceType = (brainExtractedSequence.split("_")[1]).split(".")[0]
            path_to_output_image = f"{path_to_preprocessed_files}/{patientID}/{patientID}_{sequenceType}_n4biascorrected.nii.gz"
            sitk.WriteImage(n4correctedImage, path_to_output_image, imageIO = "NiftiImageIO")


        n4correctedFiles = [
            sequence for sequence in os.listdir(os.path.join(path_to_preprocessed_files, patientID)) if ("_n4biascorrected" in sequence)
        ]

        if len(n4correctedFiles) < 4:
            error_message = f"Warning: too few brain n4biascorrected files found ({len(n4correctedFiles)})"
            print(error_message)
            error_patients[patientID] = error_message
            continue


        path_to_mnitemplate = "/Users/LennartPhilipp/Desktop/Uni/Prowiss/Code/preprocessing/icbm152_ext55_model_sym_2020/mni_icbm152_t1_tal_nlin_sym_55_ext.nii"

        # loop through n4corrected files
        for n4correctedFile in n4correctedFiles:
        
            sequenceType = (n4correctedFile.split("_")[1]).split(".")[0]
            coregistered_out_path = f"{path_to_preprocessed_files}/{patientID}/{patientID}_{sequenceType}_coregistered.nii.gz"

            print(f"{patientID} {sequenceType}: Starting coregistration of sequence")
            coregister_antspy(
                fixed_path = path_to_mnitemplate,
                moving_path = os.path.join(path_to_preprocessed_files, patientID, n4correctedFile),
                out_path = coregistered_out_path,
                num_threads = N_PROC)
            print(f"{patientID} {sequenceType}: Finished coregistration of sequence")

            coregistered_image = sitk.ReadImage(coregistered_out_path, imageIO="NiftiImageIO")


            # resample images
            print(f"{patientID} {sequenceType}: Stating resampling of images")
            resampled_image = resample(
                itk_image = coregistered_image,
                out_spacing = (1,1,1),
                is_mask = False
            )
            print(f"{patientID} {sequenceType}: Finished resampling of images")

            # z score normalize images
            print(f"{patientID} {sequenceType}: Starting z score normalization of sequence")
            z_normalized_image = zscore_normalize(resampled_image)
            print(f"{patientID} {sequenceType}: Starting z score normalization of sequence")

            # save z_normalized_image in the finished preprocessing folder
            sitk.WriteImage(z_normalized_image, f"{path_to_preprocessed_files}/{patientID}/{patientID}_{sequenceType}_preprocessed.nii.gz", imageIO = "NiftiImageIO")

        time = datetime.now().strftime("%Y%m%d-%H%M%S")
        print(f"Finished patient {patientID} - {time}")
    
        patientFiles = os.listdir(os.path.join(path_to_preprocessed_files, patientID))

        for file in patientFiles:
            if any(fileType in file for fileType in filetypes_to_remove):
               #print(f"removing file: {file}")
               os.remove(os.path.join(path_to_preprocessed_files, patientID, file))

        # clean up created files (nifti files and brain extracted images)
        print("Cleaned up unnecessary files")

        with open(path_to_txt, "w") as f:
            f.write("\n".join(patients))

    print("Finished preprocessing images")

    if error_patients:
        print("The following error messages occured")
        for key, value in error_patients.items():
            print(f"{key:15} ==> {value:40}")




# Helper functions
N_PROC = multiprocessing.cpu_count() - 1


def extract_brain(path_to_input_image: Union[str, pathlib.Path],
                  path_to_output_image: Union[str, pathlib.Path],
                  device: str):
    """
    runs the hd-bet brain extraction on the input image and returns the extracted brain
    
    Keyword Arguments:
    path_to_input_image: Union[str, pathlib.Path] = file path to input image (brain scan)
    path_to_output_image: Union[str, pathlib.Path] = location to store brain extracted image
    device: str = either "cpu" or "gpu", if you're running this on a macbook, choose "cpu"
    """

    if device == "cpu":
        subprocess.call(["hd-bet", "-i", f"{path_to_input_image}", "-o", f"{path_to_output_image}", "-device", "cpu", "-mode", "fast", "-tta", "0"], stdout=open(os.devnull, "w"), stderr=subprocess.STDOUT)
    elif device == "gpu":
        subprocess.call(["hd-bet", "-i", f"{path_to_input_image}", "-o", f"{path_to_output_image}"], stdout=open(os.devnull, "w"), stderr=subprocess.STDOUT)
    else:
        raise Exception("Wrong device input for the extract_brain method, please use either \"cpu\" or \"gpu\"")

def fill_holes(
    binary_image: sitk.Image,
    radius: int = 3,
) -> sitk.Image:
    """
    Fills holes in binary segmentation

    Keyword Arguments:
    - binary_image: sitk.Image = binary brain segmentation
    - radius: int = kernel radius

    Returns:
    - closed_image: sitk.Image = binary brain segmentation with holes filled
    """

    closing_filter = sitk.BinaryMorphologicalClosingImageFilter()
    closing_filter.SetKernelRadius(radius)
    closed_image = closing_filter.Execute(binary_image)

    return closed_image

def binary_segment_brain(
    image: sitk.Image,
) -> sitk.Image:
    """
    Returns binary segmentation of brain from brain-extracted scan via otsu thresholding

    Keyword Arguments:
    - image: sitk.Image = brain-extracted scan

    Returns:
    - sitk.Image = binary segmentation of brain scan with filled holes
    """

    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(0)
    otsu_filter.SetOutsideValue(1)
    binary_mask = otsu_filter.Execute(image)

    return fill_holes(binary_mask)

def get_bounding_box(
    image: sitk.Image,
) -> Tuple[int]:
    """
    Returns bounding box of brain-extracted scan

    Keyword Arguments:
    - image: sitk.Image = brain-extracted scan

    Returns
    - bounding_box: Tuple(int) = bounding box (startX, startY, startZ, sizeX, sizeY, sizeZ)
    """

    mask_image = binary_segment_brain(image)

    lsif = sitk.LabelShapeStatisticsImageFilter()
    lsif.Execute(mask_image)
    bounding_box = np.array(lsif.GetBoundingBox(1))

    return bounding_box

def apply_bounding_box(
    image: sitk.Image,
    bounding_box: Tuple[int],
) -> sitk.Image:
    """
    Returns image, cropped to bounding box

    Keyword Arguments:
    - image: sitk.Image = image
    - bounding_box: Tuple(ing) = bounding box of kind (startX, startY, startZ, sizeX, sizeY, sizeZ)

    Returns
    - cropped_image: sitk.Image = cropped image
    """

    cropped_image = image[
        bounding_box[0] : bounding_box[3] + bounding_box[0],
        bounding_box[1] : bounding_box[4] + bounding_box[1],
        bounding_box[2] : bounding_box[5] + bounding_box[2],
    ]

    return cropped_image

def apply_bias_correction(
    image: sitk.Image,
) -> sitk.Image:
    """applies N4 bias field correction to image but keeps background at zero

    Keyword Arguments:
    image: sitk.Image = image to apply bias correction to

    Returns:
    image_corrected_masked: sitk.Image = N4 bias field corrected image
    """

    mask_image = binary_segment_brain(image)
    float_image = sitk.Cast(image, sitk.sitkFloat32) # apparently n4biasfieldcorrectionimagefilter doesn't take int16, that's why i added this line
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    image_corrected = corrector.Execute(float_image, mask_image)

    mask_filter = sitk.MaskImageFilter()
    mask_filter.SetOutsideValue(0)
    image_corrected_masked = mask_filter.Execute(image_corrected, mask_image)

    return image_corrected_masked

def coregister_antspy(
    fixed_path: Union[str, pathlib.Path],
    moving_path: Union[str, pathlib.Path],
    out_path: Union[str, pathlib.Path],
    num_threads=N_PROC,
) -> ants.core.ants_image.ANTsImage:
    """
    Coregister moving image to fixed image. Return warped image and save to disk.

    Keyword Arguments:
    fixed_path: path to fixed image
    moving_path: path to moving image
    out_path: path to save warped image to
    num_threads: number of threads
    """

    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(num_threads)

    res = ants.registration(
        fixed=ants.image_read(fixed_path),
        moving=ants.image_read(moving_path),
        type_of_transform="antsRegistrationSyNQuick[s]",  # or "SyNRA"
        initial_transform=None,
        outprefix="",
        mask=None,
        moving_mask=None,
        mask_all_stages=False,
        grad_step=0.2,
        flow_sigma=3,
        total_sigma=0,
        aff_metric="mattes",
        aff_sampling=32,
        aff_random_sampling_rate=0.2,
        syn_metric="mattes",
        syn_sampling=32,
        reg_iterations=(40, 20, 0),
        aff_iterations=(2100, 1200, 1200, 10),
        aff_shrink_factors=(6, 4, 2, 1),
        aff_smoothing_sigmas=(3, 2, 1, 0),
        write_composite_transform=False,
        random_seed=None,
        verbose=False,
        multivariate_extras=None,
        restrict_transformation=None,
        smoothing_in_mm=False,
    )

    warped_moving = res["warpedmovout"]

    ants.image_write(warped_moving, out_path)

    return warped_moving

def resample(
    itk_image: sitk.Image,
    out_spacing: Tuple[float, ...],
    is_mask: bool,
) -> sitk.Image:
    """
    Resamples sitk image to expected output spacing

    Keyword Arguments:
    itk_image: sitk.Image
    out_spacing: Tuple
    is_mask: bool = True if input image is label mask -> NN-interpolation

    Returns
    output_image: sitk.Image = image resampled to out_spacing
    """

    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(round(osz * osp / nsp))
        for osz, osp, nsp in zip(original_size, original_spacing, out_spacing)
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(0)

    if is_mask:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)

    else:
        resample.SetInterpolator(
            sitk.sitkBSpline
        )  # sitk.sitkLinear sitk.sitkNearestNeighbor

    output_image = resample.Execute(itk_image)

    return output_image

def zscore_normalize(image: sitk.Image) -> sitk.Image:
    """
    Applies z score normalization to brain scan using a brain mask

    Keyword Arguments:
    image: sitk.Image = input brain scan

    Returns:
    normalized_brain_image: sitk.Image = normalized brain scan
    """

    brain_mask = binary_segment_brain(image)

    normalizer = ZScoreNormalize()
    normalized_brain_array = normalizer(
        sitk.GetArrayFromImage(image),
        sitk.GetArrayFromImage(brain_mask),
    )

    normalized_brain_image = sitk.GetImageFromArray(normalized_brain_array)
    normalized_brain_image.CopyInformation(image)

    return normalized_brain_image

if __name__=="__main__":
    run_preprocessing()