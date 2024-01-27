from nipype.interfaces.dcm2nii import Dcm2niix
import pathlib
from typing import Union, List, Tuple
from nipype.interfaces import fsl


def convert_dicom_to_nifti (path_to_sequence_folder: str, path_to_output_directory: str, new_filename: str):
    '''Converts a Dicom folder into a nifti file
    
    key arguments:
    - path_to_sequence_folder: str = path that leads to the dicom sequence that needs to be converted to nifti
    - path_to_output_directory: str = path to output directory
    - new_filename: str = name of the newly created nifti file'''
    
    converter = Dcm2niix()
    converter.inputs.source_dir = path_to_sequence_folder
    converter.inputs.compress = "y" # uses compression, "y" = yes
    converter.inputs.merge_imgs = True
    converter.inputs.bids_format = True
    converter.inputs.out_filename = new_filename
    converter.inputs.output_dir = path_to_output_directory
    converter.run()

def reorient_brain(
    path_to_input_image: Union[str, pathlib.Path],
    path_to_output_image: Union[str, pathlib.Path],
):
    """
    applies FSL.Reorient2Std() to input brain scan
    and returns reoriented image

    Keyword Arguments:
    path_to_input_image: Union[str, pathlib.Path] = file path to input image (brain scan)
    path_to_output_image: Union[str, pathlib.Path] = location to store reoriented image
    """

    reorient = fsl.Reorient2Std()
    reorient.inputs.in_file = path_to_input_image
    reorient.inputs.out_file = path_to_output_image
    reorient.run()