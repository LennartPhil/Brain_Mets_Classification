import os

def createFolderForPatient(path, patientID):
    '''a function that creates a folder with the patientID as the name if it doesn't exist yet'''

    pathFiles = os.listdir(path)
    if not patientID in pathFiles:
        os.mkdir(f"{path}/{patientID}")



def getUnrenamedFile(path):
    '''a function that returns the path to the file that hasn't been renamed yet'''
    files = os.listdir(path)

    for file in files:

        patientID = str(file.split("_")[0])

        if not len(patientID) == 8: # all patient IDs are 8 numbers long
            return f"{path}/{file}"

    