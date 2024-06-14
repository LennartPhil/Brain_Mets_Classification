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

#missing_segmentation_patients = []

# on Lennart's Mac Book: path_to_preprocessed = Path("/Users/LennartPhilipp/Desktop/testing_data/derivatives/preprocessed_brainlesion_20240424-110551")
#path_to_preprocessed = Path("/home/lennart/Desktop/brain_mets_regensburg/derivatives/preprocessed_brainlesion_allpatients")
# on Lennart's Mac Book: path_to_output = Path("/Users/LennartPhilipp/Desktop/testing_data/derivatives")
#path_to_output = Path("/home/lennart/Desktop/brain_mets_regensburg/derivatives")
path_to_preprocessed = "/data"
path_to_output = "/output"

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

        #cuda_devices = "0",  # optional, if you have multiple GPUs you can specify which one to use
        device = Device.GPU,  # uncomment this line to force-use CPU
    )   
    
    # create folder at path to output
    now = datetime.now()
    timeFormatted = now.strftime("%Y%m%d-%H%M%S")
    path_to_segmented_files = f"{path_to_output}/segmented_AURORA_n4_{timeFormatted}"

    preprocessed_folder_exists = False

    for file in os.listdir(path_to_output):
        if "AURORA_n4" in file:
            path_to_segmented_files = f"{path_to_output}/{file}"
            preprocessed_folder_exists = True
    
    if not preprocessed_folder_exists:
        os.mkdir(path_to_segmented_files)
    
    # get list of all patients
    patients = [f for f in os.listdir(path_to_preprocessed) if os.path.isdir(os.path.join(path_to_preprocessed, f))]

    for patient in tqdm(patients):
        
        # # ignore already segmented patients
        # if patient not in missing_segmentation_patients:
        #     continue

        print("segmenting: ", patient)
        print("getting files for ", patient)
        path_to_preprocessed_patient_files = Path(path_to_preprocessed) / Path(patient) / Path("perc_normalized")
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
            
            # ignore any . files
            if str(f).startswith("."):
                continue

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

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number CUDA Devices:', torch.cuda.device_count())
        print('__CUDA Device Name:',torch.cuda.get_device_name(0))
        print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)


    print("*** Starting segmentation at " + datetime.now().strftime("%d/%m/%Y %H:%M:%S") + " ***")
    run_segmentation()
    print("*** Finished segmentation at " + datetime.now().strftime("%d/%m/%Y %H:%M:%S") + " ***")



#!!!!! no files found for sub-02036251