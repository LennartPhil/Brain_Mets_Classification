# skrip to segment all the patient images using AURORA
# https://github.com/BrainLesion/AURORA
# this is basically the segmentation.ipynb turned into a skript
# for more information, look there or read their tutorial
# https://github.com/BrainLesion/tutorials/blob/main/AURORA/tutorial.ipynb

# Import necessary libraries

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from brainles_aurora.inferer import AuroraInferer, AuroraInfererConfig
from brainles_aurora.inferer.constants import Device
import nibabel as nib
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

missing_segmentation_patients = ['sub-01015961', 'sub-01025630', 'sub-01031243', 'sub-01040149', 'sub-01056884', 'sub-01064662', 'sub-01071055', 'sub-01087386', 'sub-01099901', 'sub-01104996', 'sub-01119720', 'sub-01130173', 'sub-01130856', 'sub-01131702', 'sub-01138456', 'sub-01164986', 'sub-01188297', 'sub-01196057', 'sub-01201482', 'sub-01204563', 'sub-01205745', 'sub-01207036', 'sub-01214172', 'sub-01216717', 'sub-01251946', 'sub-01274157', 'sub-01281168', 'sub-01288245', 'sub-01288350', 'sub-01309950', 'sub-01331487', 'sub-01332588', 'sub-01357275', 'sub-01362907', 'sub-01370265', 'sub-01373703', 'sub-01381621', 'sub-01383503', 'sub-01384142', 'sub-01387984', 'sub-01390721', 'sub-01391534', 'sub-01393875', 'sub-01395836', 'sub-01402283', 'sub-01409764', 'sub-01410235', 'sub-01414540', 'sub-01415245', 'sub-01419998', 'sub-01420310', 'sub-01431720', 'sub-01433377', 'sub-01434617', 'sub-01434869', 'sub-01435731', 'sub-01437004', 'sub-01441531', 'sub-01450871', 'sub-01452858', 'sub-01455312', 'sub-01456959', 'sub-01457167', 'sub-01458719', 'sub-01461078', 'sub-01465229', 'sub-01475524', 'sub-01476909', 'sub-01478990', 'sub-01480742', 'sub-01483116', 'sub-01483526', 'sub-01483723', 'sub-01486069', 'sub-01489395', 'sub-01492723', 'sub-01494236', 'sub-01496608', 'sub-01496804', 'sub-01498464', 'sub-01499528', 'sub-01502083', 'sub-01513891', 'sub-01514331', 'sub-01515235', 'sub-01516618', 'sub-01518885', 'sub-01521599', 'sub-01530724', 'sub-01542729', 'sub-01545797', 'sub-01547588', 'sub-01550202', 'sub-01551183', 'sub-01562247', 'sub-01565091', 'sub-01569328', 'sub-01572564', 'sub-01573094', 'sub-01575055', 'sub-01583797', 'sub-01584596', 'sub-01587295', 'sub-01589112', 'sub-01594137', 'sub-01596127', 'sub-01600788', 'sub-01605537', 'sub-01614295', 'sub-01616246', 'sub-01621161', 'sub-01641510', 'sub-01649133', 'sub-01650072', 'sub-01654658', 'sub-01657294', 'sub-01661279', 'sub-01666008', 'sub-01668785', 'sub-01673701', 'sub-01674416', 'sub-01677324', 'sub-01681275', 'sub-01695080', 'sub-01695094', 'sub-01695173', 'sub-01695930', 'sub-01696845', 'sub-01698789', 'sub-01702596', 'sub-01703264', 'sub-01706562', 'sub-01707721', 'sub-01709242', 'sub-01710250', 'sub-01713022', 'sub-01713570', 'sub-01713725', 'sub-01718213', 'sub-01732456', 'sub-01732889', 'sub-01744565', 'sub-01754011', 'sub-01763867', 'sub-01779701', 'sub-01801060', 'sub-01805334', 'sub-01835095', 'sub-01852952', 'sub-01853095', 'sub-01861511', 'sub-01870024', 'sub-01893873', 'sub-01905848', 'sub-01908576', 'sub-01924748', 'sub-01933711', 'sub-01936520', 'sub-01942928', 'sub-01953116', 'sub-01957247', 'sub-01958155', 'sub-01960441', 'sub-01961566', 'sub-01966470', 'sub-01979997', 'sub-01997658', 'sub-02000864', 'sub-02012594', 'sub-02014685', 'sub-02021781', 'sub-02036251', 'sub-02038513', 'sub-02063373', 'sub-02063398', 'sub-80004059', 'sub-80011453', 'sub-88000225', 'sub-90001992', 'sub-90003562', 'sub-90005031', 'sub-90011887', 'sub-93002557', 'sub-93003757', 'sub-95001254']

# on Lennart's Mac Book: path_to_preprocessed = Path("/Users/LennartPhilipp/Desktop/testing_data/derivatives/preprocessed_brainlesion_20240424-110551")
path_to_preprocessed = Path("/home/lennart/Desktop/brain_mets_regensburg/derivatives/preprocessed_brainlesion_allpatients")
# on Lennart's Mac Book: path_to_output = Path("/Users/LennartPhilipp/Desktop/testing_data/derivatives")
path_to_output = Path("/home/lennart/Desktop/brain_mets_regensburg/derivatives")

# go into path_to_preprocess
# go into each patient folder
# go there into the preprocessed folder
# use all the four sequences
# run segmentation
# in the path_to_segmented create a folder for each patient
# save segmentation there

def run_segmentation():

    list_of_error_patients = []

    # We first need to create an instance of the AuroraInfererConfig class,
    # which will hold the configuration for the inferer.
    # We can then create an instance of the AuroraInferer class, which will be used to perform the inference.

    config = AuroraInfererConfig(
        tta=True,
        # we disable test time augmentations for a quick demo
        # should be set to True for better results
        sliding_window_batch_size=4,
        # The batch size used for the sliding window inference
        # decrease if you run out of memory
        # warning: too small batches might lead to unstable results

        cuda_devices = "0",  # optional, if you have multiple GPUs you can specify which one to use
        device = Device.GPU,  # uncomment this line to force-use CPU
    )   
    
    # create folder at path to output called Rgb_Brain_Mets_preprocessed
    now = datetime.now()
    timeFormatted = now.strftime("%Y%m%d-%H%M%S")
    path_to_segmented_files = f"{path_to_output}/segmented_AURORA_missing_patients_00_{timeFormatted}"

    preprocessed_folder_exists = False

    for file in os.listdir(path_to_output):
        if "AURORA_missing" in file:
            path_to_segmented_files = f"{path_to_output}/{file}"
            preprocessed_folder_exists = True
    
    if not preprocessed_folder_exists:
        os.mkdir(path_to_segmented_files)
    
    # get list of all patients
    patients = [f for f in os.listdir(path_to_preprocessed) if os.path.isdir(os.path.join(path_to_preprocessed, f))]

    for patient in tqdm(patients):
        
        # ignore already segmented patients
        if patient not in missing_segmentation_patients:
            continue

        print("segmenting: ", patient)
        print("getting files for ", patient)
        path_to_preprocessed_patient_files = path_to_preprocessed / patient / "preprocessed"
        if not path_to_preprocessed_patient_files.exists():
            print(f"WARNING: {path_to_preprocessed_patient_files} doesn't exist")
            list_of_error_patients.append(patient)
            continue
        path_to_t1 = Path("")
        path_to_t1c = Path("")
        path_to_t2 = Path("")
        path_to_flair = Path("")
        files = os.listdir(path_to_preprocessed_patient_files)
        for f in files:
            if "_t1_" in f:
                path_to_t1 = path_to_preprocessed_patient_files / f
                print("Check t1")
            elif "_t1c_" in f:
                path_to_t1c = path_to_preprocessed_patient_files / f
                print("Check t1c")
            elif "_t2_" in f:
                path_to_t2 = path_to_preprocessed_patient_files / f
                print("Check t2")
            elif "_fla_" in f:
                path_to_flair = path_to_preprocessed_patient_files / f
                print("Check flair")
            else:
                print(f"ignoring the following file: {f}")
        
        # Instantiate the AuroraInferer
        inferer = AuroraInferer()

        inferer = AuroraInferer(config=config)

        _ = inferer.infer(
            t1=str(path_to_t1),
            t1c=str(path_to_t1c),
            t2=str(path_to_t2),
            fla=str(path_to_flair),
            segmentation_file=f"{path_to_segmented_files}/{patient}/{patient}_multi-modal_segmentation.nii.gz",
            # The unbinarized network outputs for the whole tumor channel (edema + enhancing tumor core + necrosis) channel
            whole_tumor_unbinarized_floats_file=f"{path_to_segmented_files}/{patient}/{patient}_whole_tumor_unbinarized_floats.nii.gz",
            # The unbinarized network outputs for the metastasis (tumor core) channel
            metastasis_unbinarized_floats_file=f"{path_to_segmented_files}/{patient}/{patient}_metastasis_unbinarized_floats.nii.gz",
            log_file=f"{path_to_segmented_files}/{patient}/{patient}_custom_logfile.log",
        )

    print("ERROR PATIENTS:")
    for error_patient in list_of_error_patients:
        print(error_patient)



if __name__ == "__main__":
    run_segmentation()



#!!!!! no files found for sub-02036251