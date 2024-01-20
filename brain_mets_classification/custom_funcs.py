import os
from datetime import datetime
import shutil

import sys
sys.path.append(r"/Users/LennartPhilipp/Desktop/Uni/Prowiss/Code/Brain_Mets_Classification")

import brain_mets_classification.config as config


def createFolderForPatient(path, patientID: str):
    '''a function that creates a folder with the patientID as the name if it doesn't exist yet'''

    pathFiles = os.listdir(path)
    if not patientID in pathFiles:
        os.mkdir(f"{path}/{patientID}")

def createSequenceFolder(path, patientID, sequence, sequence_list, original_sequence_name):
    '''a helper function that creates a folder for a MRI sequence if it doesn't exist yet in the given path

    keyword arguments:
    - path: Union[str, pathlib.Path] = path. where the new folder should be created
    - patientID: str = the individual patientID that will be part of the name of the new folder
    - sequence: str = type of sequence that will be used (should be one of the following: T1, T1CE, T2, FLAIR)
    - sequence_list: [str] = needed to ensure correct numbering of files
    - orginal_sequence_name: str = original sequence name to append to the file name

    Returns:
    - path_to_new_folder: str = path as a string to the newly created folder
    '''

    sequence_number = len(sequence_list)
    folderName = f"{patientID}_{sequence}_{sequence_number}_{original_sequence_name}"
    pathFiles = os.listdir(path)

    if not folderName in pathFiles:
        path_to_new_folder = f"{path}/{folderName}"
        os.mkdir(path_to_new_folder)
        return path_to_new_folder
    else:
        print("WARNING: Couldn't create sequence folder as folder with same name already exists!")

def copyFilesFromDirectoryToNewDirectory(path_to_original_directory, path_to_new_directory):

    # get list of all the dicom files for the T1 sequence
    filesInDirectory = os.listdir(path_to_original_directory)

    # loops through the list of dicom files
    for file in filesInDirectory:
        # ignores the ds_folders
        if config.dsStore in file:
            continue

        # copy each file individually into the path_to_sequence folder
        shutil.copyfile(os.path.join(path_to_original_directory, file), os.path.join(path_to_new_directory, file))

def getUnrenamedFile(path):
    '''a function that returns the path to the file that hasn't been renamed yet'''
    files = os.listdir(path)

    for file in files:

        try:
            patientID = str(file.split("_")[0])
        except RuntimeError as e:
            print("Couldn't split filename: ", e)

        if not len(patientID) == 8: # all patient IDs are 8 numbers long
            return f"{path}/{file}"
        else:
            print("patientID not 8 numbers long")


def createNewPreprocessingStepFolder(step):
    '''a function that creates a folder for the individual preprocessing step
    arguments:
    step: str = description of the current preprocessing step

    outputs:
    pathToPreprocessingFolder: String = the path to the newly created folder

    the folder is named such as the following Rgb_Brain_Mets_Preprocessing#X_202X-XX-XX_XX_XX_XX
    '''
    now = datetime.now()
    timeFormatted = now.strftime("%Y%m%d-%H%M%S")
    pathToPreprocessingFolder = f"{config.path_to_ssd}/Rgb_Brain_Mets_Preprocessing_{step}_{timeFormatted}"
    os.mkdir(pathToPreprocessingFolder)

    return pathToPreprocessingFolder