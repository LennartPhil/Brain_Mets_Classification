# This script is meant to run the hd-bet tool and extract the brain for all the patient files

import os
from tqdm import tqdm

def main():
    # To-do:
    # go through all the patient
    # extract the brain for each folder of MRI images

    path_to_patients = "/Users/LennartPhilipp/Desktop/n30"
    path_to_store_extracted_images = "/Users/LennartPhilipp/Desktop/extractedImages"

    # get all the folders at path_to_patients and store them in the patient_folders array
    patient_folders = [
        folder for folder in os.listdir(path_to_patients) if os.path.isdir(os.path.join(path_to_patients, folder))
    ]

    try:
        for patient_folder in tqdm(patient_folders):

            print(f"Working on: {patient_folder}")

            # create new patientfolder at path_to_store_extracted_images
            createFolderForPatient(path_to_store_extracted_images, patient_folder)

            # get the nifti fils for each patient and put them in an array
            niftiFiles = [
                niftiFile for niftiFile in os.listdir(os.path.join(path_to_patients, patient_folder)) if (".nii" in niftiFile)
            ]

            for niftiFile in niftiFiles:

                sequenceType = niftiFile.split("_")[1]

                # hd-bet -i INPUT_FOLDER -o OUTPUT_FOLDER -device cpu -mode fast -tta 0
                os.system(f"hd-bet -i {path_to_patients}/{patient_folder}/{niftiFile} -o {path_to_store_extracted_images}/{patient_folder}/{patient_folder}_{sequenceType}_brainextracted -device cpu -mode fast -tta 0")
            

            print(f"Finished: {patient_folder}!")
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected, stoped program")

def createFolderForPatient(path, patientID: str):
    '''a function that creates a folder with the patientID as the name if it doesn't exist yet'''

    pathFiles = os.listdir(path)
    if not patientID in pathFiles:
        os.mkdir(f"{path}/{patientID}")

if __name__ == "__main__":
    main()