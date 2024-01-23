from nipype.interfaces.dcm2nii import Dcm2niix

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