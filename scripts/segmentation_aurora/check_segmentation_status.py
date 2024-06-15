import os
from pathlib import Path

path_to_segmentation = Path("/raid/lennart/derivatives/segmented_AURORA_n4_20240614-170748")

def check_segmentation_status():

    missing_patients = []

    segmentation_patients = [f for f in os.listdir(path_to_segmentation) if os.path.isdir(os.path.join(path_to_segmentation, f))]
    print(len(segmentation_patients), "segmentation patients found")

    for patient in segmentation_patients:
        path_to_patient = Path(path_to_segmentation) / Path(patient)

        if len(os.listdir(path_to_patient)) < 4:
            print(f"{patient} is not complete")
            missing_patients.append(patient)
    
    print("\n", len(missing_patients), "segmentation patients missing")
    print(missing_patients)


if __name__ == "__main__":
    check_segmentation_status()