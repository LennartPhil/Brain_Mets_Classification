import os
from datetime import datetime

import sys
sys.path.append(r"/Users/LennartPhilipp/Desktop/Uni/Prowiss/Code/Brain_Mets_Classification")

import brain_mets_classification.config as config


def createFolderForPatient(path, patientID):
    '''a function that creates a folder with the patientID as the name if it doesn't exist yet'''

    pathFiles = os.listdir(path)
    if not patientID in pathFiles:
        os.mkdir(f"{path}/{patientID}")


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
    step: Int = the number of the current preprocessing step

    outputs:
    pathToPreprocessingFolder: String = the path to the newly created folder

    the folder is named such as the following Rgb_Brain_Mets_Preprocessing#X_202X-XX-XX_XX_XX_XX
    '''
    now = datetime.now()
    timeFormatted = now.strftime("%Y-%m-%d_%H-%M-%S")
    pathToPreprocessingFolder = f"{config.path}/Rgb_Brain_Mets_Preprocessing#{step}_{timeFormatted}"
    os.mkdir(pathToPreprocessingFolder)

    return pathToPreprocessingFolder