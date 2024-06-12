# This file is basically a script version of the
# preprocessing_brain_lesion.ipynb file
# For more information, look there :)

# Import necessary libraries

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from auxiliary.normalization.percentile_normalizer import PercentileNormalizer
from auxiliary.turbopath import turbopath
from tqdm import tqdm
from datetime import datetime

from brainles_preprocessing.brain_extraction import HDBetExtractor
from brainles_preprocessing.modality import Modality
from brainles_preprocessing.preprocessor import Preprocessor
from brainles_preprocessing.registration import (ANTsRegistrator)

from intensity_normalization.normalize.zscore import ZScoreNormalize

import SimpleITK as sitk
from pathlib import Path

import numpy as np

class MyException(Exception):
    def __init__(self, message, *args):
        self.message = message # without this you may get DeprecationWarning
        # Special attribute you desire with your Error, 
        # perhaps the value that caused the error?:
        # allow users initialize misc. arguments as any other builtin Error
        super(MyException, self).__init__(message, *args) 


path_to_output = "/output"
patients_directory = "/data"
#patients_directory = "/Users/LennartPhilipp/Desktop/testing_data/raw_data"
#patients_directory = "/Users/LennartPhilipp/Desktop/testing_data/one_pat"
#path_to_output = "/Users/LennartPhilipp/Desktop/testing_data/derivatives"
output_folder_keywords = "preprocessed_n4_brainlesion_percentile"
n4_norm_folder = "n4_normalized"

def run_preprocessing():
    patients_to_preprocesse = ['sub-01005097', 'sub-01005630', 'sub-01006290', 'sub-01009590', 'sub-01015961', 'sub-01018613', 'sub-01021714', 'sub-01021993', 'sub-01022787', 'sub-01025630', 'sub-01031243', 'sub-01038520', 'sub-01040149', 'sub-01041137', 'sub-01055292', 'sub-01056598', 'sub-01056884', 'sub-01064662', 'sub-01071055', 'sub-01072344', 'sub-01083248', 'sub-01087386', 'sub-01098043', 'sub-01099901', 'sub-01104996', 'sub-01106844', 'sub-01108350', 'sub-01109318', 'sub-01111974', 'sub-01117914', 'sub-01117958', 'sub-01119720', 'sub-01122863', 'sub-01125016', 'sub-01130173', 'sub-01130856', 'sub-01131702', 'sub-01134825', 'sub-01138456', 'sub-01147272', 'sub-01150136', 'sub-01152379', 'sub-01164049', 'sub-01164986', 'sub-01169240', 'sub-01188297', 'sub-01189050', 'sub-01190670', 'sub-01190738', 'sub-01196057', 'sub-01199093', 'sub-01201117', 'sub-01201482', 'sub-01204563', 'sub-01205171', 'sub-01205745', 'sub-01207036', 'sub-01213140', 'sub-01214172', 'sub-01214417', 'sub-01216717', 'sub-01220269', 'sub-01227960', 'sub-01231700', 'sub-01241505', 'sub-01243841', 'sub-01251946', 'sub-01261127', 'sub-01262362', 'sub-01269967', 'sub-01274157', 'sub-01281168', 'sub-01288036', 'sub-01288245', 'sub-01288350', 'sub-01288896', 'sub-01307298', 'sub-01309950', 'sub-01311383', 'sub-01314225', 'sub-01321873', 'sub-01331487', 'sub-01332588', 'sub-01335279', 'sub-01340749', 'sub-01349100', 'sub-01357275', 'sub-01358619', 'sub-01362907', 'sub-01370265', 'sub-01373703', 'sub-01373833', 'sub-01377175', 'sub-01377261', 'sub-01381621', 'sub-01384142', 'sub-01387984', 'sub-01390721', 'sub-01391534', 'sub-01391984', 'sub-01393875', 'sub-01395836', 'sub-01398968', 'sub-01400779', 'sub-01402283', 'sub-01405609', 'sub-01409764', 'sub-01410235', 'sub-01410317', 'sub-01414189', 'sub-01414540', 'sub-01415245', 'sub-01417614', 'sub-01419998', 'sub-01420310', 'sub-01421533', 'sub-01423083', 'sub-01425882', 'sub-01431720', 'sub-01432274', 'sub-01433377', 'sub-01434869', 'sub-01435731', 'sub-01437004', 'sub-01441531', 'sub-01443624', 'sub-01450871', 'sub-01452858', 'sub-01455312', 'sub-01456719', 'sub-01456959', 'sub-01457167', 'sub-01458719', 'sub-01461078', 'sub-01465229', 'sub-01476909', 'sub-01478990', 'sub-01479403', 'sub-01480742', 'sub-01483116', 'sub-01483526', 'sub-01483723', 'sub-01484016', 'sub-01486069', 'sub-01489395', 'sub-01492604', 'sub-01492723', 'sub-01494236', 'sub-01494733', 'sub-01496608', 'sub-01496804', 'sub-01497790', 'sub-01498464', 'sub-01499528', 'sub-01501379', 'sub-01502083', 'sub-01505384', 'sub-01513891', 'sub-01514331', 'sub-01515235', 'sub-01516618', 'sub-01516702', 'sub-01518885', 'sub-01521599', 'sub-01521835', 'sub-01529963', 'sub-01530724', 'sub-01542729', 'sub-01543367', 'sub-01545797', 'sub-01547588', 'sub-01548397', 'sub-01549022', 'sub-01550202', 'sub-01551183', 'sub-01562247', 'sub-01565091', 'sub-01569328', 'sub-01572564', 'sub-01573094', 'sub-01578955', 'sub-01583797', 'sub-01584596', 'sub-01587295', 'sub-01589112', 'sub-01592849', 'sub-01594137', 'sub-01596127', 'sub-01600788', 'sub-01605537', 'sub-01607473', 'sub-01609293', 'sub-01616246', 'sub-01616767', 'sub-01619086', 'sub-01621161', 'sub-01641960', 'sub-01649133', 'sub-01650072', 'sub-01652130', 'sub-01654658', 'sub-01657294', 'sub-01659187', 'sub-01660211', 'sub-01661279', 'sub-01666008', 'sub-01668785', 'sub-01670714', 'sub-01673666', 'sub-01673701', 'sub-01674416', 'sub-01677324', 'sub-01681275', 'sub-01686968', 'sub-01691369', 'sub-01695080', 'sub-01695094', 'sub-01695173', 'sub-01695930', 'sub-01696845', 'sub-01696904', 'sub-01698789', 'sub-01699419', 'sub-01699532', 'sub-01702596', 'sub-01703264', 'sub-01705952', 'sub-01706146', 'sub-01706562', 'sub-01707275', 'sub-01707721', 'sub-01709242', 'sub-01710250', 'sub-01710310', 'sub-01713022', 'sub-01713570', 'sub-01713725', 'sub-01717958', 'sub-01718213', 'sub-01729917', 'sub-01732456', 'sub-01732889', 'sub-01744565', 'sub-01754011', 'sub-01755816', 'sub-01760947', 'sub-01762556', 'sub-01763867', 'sub-01764802', 'sub-01769144', 'sub-01771120', 'sub-01773716', 'sub-01779701', 'sub-01781732', 'sub-01789555', 'sub-01792136', 'sub-01792771', 'sub-01795656', 'sub-01796937', 'sub-01797781', 'sub-01798755', 'sub-01801060', 'sub-01804484', 'sub-01805334', 'sub-01812518', 'sub-01812578', 'sub-01819252', 'sub-01830168', 'sub-01835095', 'sub-01840035', 'sub-01852952', 'sub-01854308', 'sub-01861511', 'sub-01864584', 'sub-01870024', 'sub-01871625', 'sub-01874079', 'sub-01876862', 'sub-01878754', 'sub-01879950', 'sub-01881784', 'sub-01882333', 'sub-01882989', 'sub-01883957', 'sub-01885520', 'sub-01890298', 'sub-01892684', 'sub-01893873', 'sub-01895825', 'sub-01905692', 'sub-01905848', 'sub-01906150', 'sub-01908576', 'sub-01908872', 'sub-01914558', 'sub-01921604', 'sub-01924748', 'sub-01924997', 'sub-01933711', 'sub-01935938', 'sub-01936520', 'sub-01936903', 'sub-01942928', 'sub-01943022', 'sub-01946271', 'sub-01946372', 'sub-01947074', 'sub-01952689', 'sub-01953116', 'sub-01957247', 'sub-01957841', 'sub-01958155', 'sub-01960441', 'sub-01961554', 'sub-01961566', 'sub-01966470', 'sub-01969755', 'sub-01978091', 'sub-01979317', 'sub-01979997', 'sub-01981993', 'sub-01982853', 'sub-01983233', 'sub-01983705', 'sub-01985065', 'sub-01990699', 'sub-01991292', 'sub-01997658', 'sub-02000864', 'sub-02009523', 'sub-02010452', 'sub-02011152', 'sub-02012594', 'sub-02014068', 'sub-02014467', 'sub-02014685', 'sub-02015217', 'sub-02015335', 'sub-02015730', 'sub-02018743', 'sub-02019520', 'sub-02019676', 'sub-02020169', 'sub-02020631', 'sub-02021781', 'sub-02026140', 'sub-02026964', 'sub-02031256', 'sub-02031868', 'sub-02034046', 'sub-02035864', 'sub-02036053', 'sub-02036130', 'sub-02036251', 'sub-02036854', 'sub-02038513', 'sub-02046093', 'sub-02047436', 'sub-02051037', 'sub-02055312', 'sub-02059459', 'sub-02063373', 'sub-02063398', 'sub-02063986', 'sub-02064363', 'sub-02066389', 'sub-02066445', 'sub-02070580', 'sub-02070606', 'sub-02070903', 'sub-02071087', 'sub-02071305', 'sub-02071509', 'sub-02073121', 'sub-02073198', 'sub-02074050', 'sub-02075769', 'sub-02080563', 'sub-02082001', 'sub-02082498', 'sub-02084961', 'sub-02085355', 'sub-02086122', 'sub-02087023', 'sub-02087621', 'sub-02088404', 'sub-02088565', 'sub-02089657', 'sub-02090169', 'sub-02090355', 'sub-02090905', 'sub-02092748', 'sub-02094018', 'sub-02094355', 'sub-02094837', 'sub-02095181', 'sub-02095303', 'sub-02095961', 'sub-02096374', 'sub-02097980', 'sub-02100576', 'sub-02104370', 'sub-02105939', 'sub-02106388', 'sub-02110064', 'sub-02113470', 'sub-02113718', 'sub-02115365', 'sub-02115377', 'sub-02116290', 'sub-02119444', 'sub-02119712', 'sub-02120805', 'sub-02120806', 'sub-02122538', 'sub-02124336', 'sub-02126982', 'sub-02127770', 'sub-02127867', 'sub-02128777', 'sub-02131818', 'sub-02132336', 'sub-02134230', 'sub-02134991', 'sub-02135803', 'sub-02136965', 'sub-02137062', 'sub-02137073', 'sub-02138280', 'sub-02139997', 'sub-02140009', 'sub-02140670', 'sub-02142561', 'sub-02145605', 'sub-02145622', 'sub-02145747', 'sub-02145870', 'sub-02146286', 'sub-02148184', 'sub-02148372', 'sub-02152734', 'sub-02153522', 'sub-02154718', 'sub-02155605', 'sub-02163033', 'sub-02164825', 'sub-02165732', 'sub-02167792', 'sub-02167890', 'sub-02168956', 'sub-02172003', 'sub-02172137', 'sub-02173158', 'sub-02174928', 'sub-02177752', 'sub-02178883', 'sub-02179132', 'sub-02181503', 'sub-02183443', 'sub-02184584', 'sub-02185066', 'sub-02188930', 'sub-02190005', 'sub-02190209', 'sub-02196769', 'sub-02197114', 'sub-02197683', 'sub-02199356', 'sub-02209727', 'sub-80004059', 'sub-80011453', 'sub-88000225', 'sub-90001992', 'sub-90003334', 'sub-90003562', 'sub-90005031', 'sub-90005297', 'sub-90011887', 'sub-93002557', 'sub-93003757', 'sub-95001254', 'sub-96003928', 'sub-96004436', 'sub-99000041']
    
    print(f"amount of patients to preprocess: {len(patients_to_preprocesse)}")

    # on Lennart's Mac Book:
    #patients_directory = "/Users/LennartPhilipp/Desktop/testing_data/raw_data"

    # with docker
    #patients_directory = "/data"

    # on Lennart's Mac Book:
    #path_to_output = "/Users/LennartPhilipp/Desktop/testing_data/derivatives"

    # with docker
    #path_to_output = "/output"


    path_to_preprocessed_files = ""
    preprocessed_folder_exists = False

    for file in os.listdir(path_to_output):
        if output_folder_keywords in file:
            path_to_preprocessed_files = f"{path_to_output}/{file}"
            preprocessed_folder_exists = True
    
    if preprocessed_folder_exists == False:
        raise MyException("preprocessed folder doesn't exist")
    
    data_dir = turbopath(path_to_preprocessed_files)

    patients = data_dir.dirs()

    for patient in tqdm(patients):

        if patient.name not in patients_to_preprocesse:
            print(f"skipping patient: {patient.name}")
            continue
        print("processing: ", patient.name)
        preprocessed_folder = [folder for folder in patient.dirs() if n4_norm_folder in folder.name][0]
        preprocess_exam_in_brats_style(inputDir = preprocessed_folder,
                                       patID = patient.name,
                                       outputDir = path_to_preprocessed_files)

    print("*** finished preprocessing ***")

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
    raw_bet_dir = turbopath(pat_directory) / "raw_bet"

    t1_file = inputDir / patID + "_n4_normalized_" + "T1w.nii.gz"
    t1c_file = inputDir / patID + "_n4_normalized_" + "T1c.nii.gz"
    t2_file = inputDir / patID + "_n4_normalized_" + "T2w.nii.gz"
    flair_file = inputDir / patID + "_n4_normalized_" + "FLAIR.nii.gz"
    
    
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
        input_path=t1c_file,
        raw_bet_output_path=raw_bet_dir / patID
        + "_t1c_bet_normalized.nii.gz",
        atlas_correction=True,
        #normalizer=percentile_normalizer,
    )

    moving_modalities = [
        Modality(
            modality_name="t1",
            input_path=t1_file,
            raw_bet_output_path=raw_bet_dir / patID
            + "_t1_bet_raw.nii.gz",
            atlas_correction=True,
            #normalizer=percentile_normalizer,
        ),
        Modality(
            modality_name="t2",
            input_path=t2_file,
            raw_bet_output_path=raw_bet_dir / patID
            + "_t2_bet_raw.nii.gz",
            atlas_correction=True,
            #normalizer=percentile_normalizer,
        ),
        Modality(
            modality_name="flair",
            input_path=flair_file,
            raw_bet_output_path=raw_bet_dir / patID
            + "_fla_bet_raw.nii.gz",
            atlas_correction=True,
            #normalizer=percentile_normalizer,
        ),
    ]

    print("preparing preprocessor")

    preprocessor = Preprocessor(
        center_modality=center,
        moving_modalities=moving_modalities,
        registrator=ANTsRegistrator(), # previously NiftRegRegistrator()
        brain_extractor=HDBetExtractor(),
        use_gpu=True,
    )

    preprocessor.run(
        save_dir_coregistration=brainles_dir + "/co-registration",
        save_dir_atlas_registration=brainles_dir + "/atlas-registration",
        save_dir_atlas_correction=brainles_dir + "/atlas-correction",
        save_dir_brain_extraction=brainles_dir + "/brain-extraction",
    )

    print(f"finished preprocessing for {patID}")

def normalize():

    now = datetime.now()
    timeFormatted = now.strftime("%Y%m%d-%H%M%S")

    path_to_preprocessed_files = f"{path_to_output}/preprocessed_n4_brainlesion_percentile_{timeFormatted}"

    preprocessed_folder_exists = False

    for file in os.listdir(path_to_output):
        if "n4_brainlesion_percentile" in file:
            path_to_preprocessed_files = f"{path_to_output}/{file}"
            preprocessed_folder_exists = True
    
    if preprocessed_folder_exists == False:
        os.mkdir(path_to_preprocessed_files)

    #os.makedirs(path_to_preprocessed_files, exist_ok=True)

    patients = [patient for patient in os.listdir(patients_directory) if os.path.isdir(os.path.join(patients_directory, patient))]
    print(f"found {len(patients)} patients")

    for patient in tqdm(patients):
        print(f"currently normalizing patient {patient}")

        path_to_patient_files = Path(patients_directory) / Path(patient) / Path("anat")

        raw_files = [file for file in os.listdir(path_to_patient_files) if ".nii.gz" in file]

        # create subfolder for each patient
        path_to_normalized_folder = Path(path_to_preprocessed_files) / Path(patient) / "n4_normalized"
        os.makedirs(path_to_normalized_folder, exist_ok=True)

        for raw_file in raw_files:

            if raw_file.startswith("."):
                continue

            sequence_type = raw_file.split("_")[1].split(".nii")[0]
            file_name = patient + "_n4_normalized_" + sequence_type + ".nii.gz"
            path_to_normalized_file = path_to_normalized_folder / Path(file_name)

            # print(f"starting z_score bias correction for {raw_file}")

            path_to_file = Path(path_to_patient_files) / raw_file
            sitk_image = sitk.ReadImage(path_to_file)
            sitk_image = sitk.Cast(sitk_image, sitk.sitkFloat32)
            # z_score_normalized_image = z_score_normalize(path_to_file)

            # print(f"finished z_score bias correction for {raw_file}")

            print(f"starting n4 bias correction for {raw_file}")

            n4_bias_correction(sitk_image, path_to_normalized_file)
            
            print(f"finished n4 bias correction for {raw_file}")

def perc_normalize_and_save():
    """
    Performs percentile normalization and saving of preprocessed files.

    This function iterates through the files in the specified output directory. If a file contains the string
    "z_score_n4_brainlesion_percentile", the path to that file is stored in the `path_to_preprocessed` variable.
    If no such file is found, a `MyException` is raised.

    Next, the function finds all patient directories in the `path_to_preprocessed` directory and stores their names
    in the `patients` list.

    Then, for each patient, the function prints the current patient being normalized and retrieves the paths to the
    `raw_bet` directory and all files in that directory with the extension ".nii.gz". For each file, the function
    prints the start of the n4 bias correction process and retrieves the sequence type from the file name.

    The function then reads the mask image from the `brain-extraction` directory and casts it to `sitk.sitkUInt8`.

    Next, the function reads the image from the current file and performs percentile normalization on the image array.
    """

    for file in os.listdir(path_to_output):
        if "n4_brainlesion_percentile" in file:
            path_to_preprocessed = f"{path_to_output}/{file}"
    
    if path_to_preprocessed == "":
        raise MyException("Could not find preprocessed data in output folder")
    
    # find patient files to normalize
    patients = [patient for patient in os.listdir(path_to_preprocessed) if os.path.isdir(os.path.join(path_to_preprocessed, patient))]


    # loop through all preprocessed patients
    for patient in tqdm(patients):
        print(f"currently percentile normalizing patient {patient}")

        path_to_patient = Path(path_to_preprocessed) / Path(patient)
        path_to_raw_bet = path_to_patient / "raw_bet"

        raw_bet_files = [file for file in os.listdir(path_to_raw_bet) if ".nii.gz" in file]
        # loop through all raw_bet files
        for file in raw_bet_files:

            if file.startswith("."):
                continue

            path_to_file = Path(path_to_raw_bet) / file
            print(f"starting n4 bias correction for  {file}")

            sequence_type = file.split("_")[1]

            path_to_mask_image = path_to_patient / "brain-extraction" / "atlas_bet_t1c_mask.nii.gz"
            mask_image = sitk.ReadImage(str(path_to_mask_image), imageIO="NiftiImageIO")
            mask_image = sitk.Cast(mask_image, sitk.sitkUInt8)

            # n4 bias correct files
            image = sitk.ReadImage(str(path_to_file), imageIO="NiftiImageIO")

            # percentile normalize
            print(f"starting percentile normalization {file}")
            percentile_normalized_image = percentile_normalize(
                sitk.GetArrayFromImage(image),
                lower_percentile=0.1,
                upper_percentile=99.9,
                lower_limit=0,
                upper_limit=1,
            )

            print(f"successfully percentile normalized {file}")
            print()

            # create new n4 corrected directory
            path_to_normalized = path_to_patient / "perc_normalized"
            Path(path_to_normalized).mkdir(exist_ok=True)

            # save in new directory
            normalized_file_name = f"{patient}_{sequence_type}_perc_normalized.nii.gz"
            path_to_percentile_corrected_file = path_to_normalized / normalized_file_name
            sitk.WriteImage(sitk.GetImageFromArray(percentile_normalized_image), str(path_to_percentile_corrected_file))
        
        print(f"finished normalizing patient {patient}")

def z_score_normalize(path_to_image_file):
    """
    Applies z score normalization to a brain scan using the ZScoreNormalize class from the SimpleITK library.

    Args:
        path_to_image_file (str): The path to the input image file.

    Returns:
        sitk.Image: The normalized brain scan image.
    """

    image = sitk.ReadImage(path_to_image_file)

    normalizer = ZScoreNormalize()
    normalized_brain_array = normalizer(
        sitk.GetArrayFromImage(image),
    )

    normalized_brain_image = sitk.GetImageFromArray(normalized_brain_array)
    normalized_brain_image.CopyInformation(image)

    return normalized_brain_image

def normalize_old():

    path_to_preprocessed = ""

    for file in os.listdir(path_to_output):
        if "z_score_n4_brainlesion_percentile" in file:
            path_to_preprocessed = f"{path_to_output}/{file}"
    
    if path_to_preprocessed == "":
        raise Exception("Could not find preprocessed data in output folder")

    # find patient files to normalize
    patients = [patient for patient in os.listdir(path_to_preprocessed) if os.path.isdir(os.path.join(path_to_preprocessed, patient))]

    # loop through all preprocessed patients
    for patient in tqdm(patients):
        print(f"currently normalizing patient {patient}")

        path_to_patient = Path(path_to_preprocessed) / Path(patient)
        path_to_raw_bet = path_to_patient / "raw_bet"

        raw_bet_files = [file for file in os.listdir(path_to_raw_bet) if ".nii.gz" in file]
        # loop through all raw_bet files
        for file in raw_bet_files:
            path_to_file = Path(path_to_raw_bet) / file
            print(f"starting n4 bias correction for  {file}")

            sequence_type = file.split("_")[1]

            path_to_mask_image = path_to_patient / "brain-extraction" / "atlas_bet_t1c_mask.nii.gz"
            mask_image = sitk.ReadImage(str(path_to_mask_image), imageIO="NiftiImageIO")
            mask_image = sitk.Cast(mask_image, sitk.sitkUInt8)

            # n4 bias correct files
            image = sitk.ReadImage(str(path_to_file), imageIO="NiftiImageIO")
            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            image_corrected = corrector.Execute(image, mask_image)

            mask_filter = sitk.MaskImageFilter()
            mask_filter.SetOutsideValue(0)
            image_corrected_masked = mask_filter.Execute(image_corrected, mask_image)

            print(f"successfully n4 bias corrected {file}")

            # create new n4 corrected directory
            path_to_n4_corrected = path_to_patient / "n4_corrected"
            Path(path_to_n4_corrected).mkdir(exist_ok=True)

            # save in new directory
            n4_file_name = f"{patient}_{sequence_type}_n4_corrected.nii.gz"
            path_to_n4_corrected_file = path_to_n4_corrected / n4_file_name
            sitk.WriteImage(image_corrected_masked, str(path_to_n4_corrected_file))

            print(f"successfully saved n4 corrected {file}")

            # percentile normalize
            print(f"starting percentile normalization {file}")
            percentile_normalized_image = percentile_normalize(sitk.GetArrayFromImage(image_corrected_masked))

            print(f"successfully percentile normalized {file}")

            # create new n4 corrected directory
            path_to_normalized = path_to_patient / "normalized"
            Path(path_to_normalized).mkdir(exist_ok=True)

            # save in new directory
            normalized_file_name = f"{patient}_{sequence_type}_normalized.nii.gz"
            path_to_percentile_corrected_file = path_to_normalized / normalized_file_name
            sitk.WriteImage(sitk.GetImageFromArray(percentile_normalized_image), str(path_to_percentile_corrected_file))
        
        print(f"finished normalizing patient {patient}")
            
def n4_bias_correction(image, path_to_output):
    """
    Applies N4 bias field correction to the given image and saves the corrected image to the specified output path.

    Parameters:
        image (sitk.Image): The input image to be corrected.
        path_to_output (str or pathlib.Path): The path to save the corrected image.

    Returns:
        None
    """

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    image_corrected = corrector.Execute(image)

    sitk.WriteImage(image_corrected, str(path_to_output))

def percentile_normalize(image: np.ndarray, lower_percentile: float = 0.0, upper_percentile: float = 100.0, lower_limit: float = 0, upper_limit: float = 1):
    """
    Normalizes an image using percentile-based mapping.

    Args:
        image (numpy.ndarray): The input image.
        lower_percentile (float, optional): The lower percentile for mapping. Defaults to 0.0.
        upper_percentile (float, optional): The upper percentile for mapping. Defaults to 100.0.
        lower_limit (float, optional): The lower limit for normalized values. Defaults to 0.
        upper_limit (float, optional): The upper limit for normalized values. Defaults to 1.

    Returns:
        numpy.ndarray: The percentile-normalized image.

    Description:
        This function takes an image as input and normalizes it using percentile-based mapping. It calculates the lower
        and upper values of the image based on the provided percentiles. The normalized image is then calculated by
        subtracting the lower value from each pixel value, dividing the result by the difference between the upper and
        lower values, and clipping the values between 0 and 1. Finally, the normalized image is scaled to the specified
        lower limit and upper limit.

    Example:
        >>> image = np.array([10, 20, 30, 40, 50])
        >>> percentile_normalize(image, lower_percentile=10, upper_percentile=90)
        array([0.        , 0.11111111, 0.22222222, 0.33333333, 0.44444444])
    """


    lower_value = np.percentile(image, lower_percentile)
    upper_value = np.percentile(image, upper_percentile)

    normalized_image = np.clip(
        (image - lower_value) / (upper_value - lower_value), 0, 1
    )
    normalized_image = (
        normalized_image * (upper_limit - lower_limit) + lower_limit
    )

    normalized_upper_value = np.percentile(normalized_image, upper_percentile)
    normalized_lower_value = np.percentile(normalized_image, lower_percentile)

    if normalized_upper_value != 1 or normalized_lower_value != 0:
        print(f"Percentile normalization failed. Upper value: {normalized_upper_value}, lower value: {normalized_lower_value}")

    return normalized_image


if __name__ == "__main__":

    print("*** Preprocessing started at " + datetime.now().strftime("%d/%m/%Y %H:%M:%S") + " ***")

    normalize()
    run_preprocessing()
    perc_normalize_and_save()

    print()
    print("*** Preprocessing finished at " + datetime.now().strftime("%d/%m/%Y %H:%M:%S") + " ***")