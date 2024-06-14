import os
from pathlib import Path
from datetime import datetime

path_to_derivative = Path("/raid/lennart/derivatives/preprocessed_n4_brainlesion_percentile_20240612-083743")

def run_check():
    # To-do:
    # check n4 correction status
    # check preproc status
    # check percentile normalization status

    n4_failures = []
    preproc_failures = []
    perc_norm_failures = []

    patients = [pat for pat in os.listdir(path_to_derivative) if os.path.isdir(os.path.join(path_to_derivative, pat))]

    for patient in patients:

        print(f"Checking patient {patient}")
        print()

        path_to_patient = Path(path_to_derivative) / Path(patient)

        n4_status = check_n4(path_to_patient)
        preproc_status = check_preproc(path_to_patient)
        perc_norm_status = check_perc_norm(path_to_patient)

        if n4_status == False:
            n4_failures.append(patient)
            print(f"{patient} n4 correction failed")

        if preproc_status == False:
            preproc_failures.append(patient)
            print(f"{patient} preprocessing failed")

        if perc_norm_status == False:
            perc_norm_failures.append(patient)
            print(f"{patient} percentile normalization failed")
        
        if n4_status == True and preproc_status == True and perc_norm_status == True:
            print(f"{patient} all preprocessing steps succeeded")
        
        print()
    
    print(f"n4 failures: {n4_failures}")
    print()
    print(f"preprocessing failures: {preproc_failures}")
    print()
    print(f"percentile normalization failures: {perc_norm_failures}")


def check_n4(path_to_patient):
    path_to_n4 = path_to_patient / Path("n4_normalized")
    if not os.path.exists(path_to_n4):
        return False
    else:
        if len(os.listdir(path_to_n4)) < 4:
            return False
        else:
            return True
        
def check_preproc(path_to_patient):
    path_to_n4 = path_to_patient / Path("raw_bet")
    if not os.path.exists(path_to_n4):
        return False
    else:
        if len(os.listdir(path_to_n4)) < 4:
            return False
        else:
            return True
        
def check_perc_norm(path_to_patient):
    path_to_n4 = path_to_patient / Path("perc_normalized")
    if not os.path.exists(path_to_n4):
        return False
    else:
        if len(os.listdir(path_to_n4)) < 4:
            return False
        else:
            return True

if __name__ == "__main__":
    print("*** Starting preprocessing check at " + datetime.now().strftime("%d/%m/%Y %H:%M:%S") + " ***")
    run_check()