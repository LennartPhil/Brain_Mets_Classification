# Brain_Mets_Classification

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li><a href="#road-map">Roadmap</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

## About The Project
The aim of this project is to write an ai that classifies brain metastases based on their primary cancers.

### Built with
The following dependencies, libraries and ressources were used:
HD-BET
Tensorflow
...

<!-- ROADMAP -->
## Road Map

### Work in Progress
- [X] segmentation
    - [X] redo segmentation on n4 bias corrected files
    - [ ] manually adjust segmentation
- [ ] Build AIs
    - [ ] 2D CNN (only metastasis cutout)
    - [ ] 2D CNN* (transfer learning)
    - [ ] Vision Transformer (maybe)

### To-do
- [ ] Train ai
- [ ] Explore results

### Done
- [X] Acquire patient data
- [X] Find correct sequences for each patient
- [X] Preprocessing
    - [X] Extract dicom metadata
    - [X] Convert dicom to nifti
    - [X] Extract brain
        - [X] extract patients brain using HD-BET
        - [X] compare HD-BET images with synthstrip images (chose HD-BET)
    - [X] Fill holes
    - [X] Binary Segment
    - [X] Cropy images
    - [X] Bias correction
    - [X] Coregister images
    - [X] Resample images
    - [X] Z-score normalization
    - [X] Merge images
- [X] redo preprocessing using the brats-toolkit
- [X] redo preprocessing with n4 bias correction (last time, I swear)
- [ ] Build AIs (currently working on this)
    - [X] 3D CNN (entire brain) -> unfortunately unsuccessfull :/
        - [X] Transfer ResNeXt architecture to 3D
        - [X] custom scheduler
        - [X] custom ai architecture (input: images, age, sex)