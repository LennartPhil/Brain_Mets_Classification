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
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

## About The Project
The aim of this project is to write an ai that classifies brain metastases based on their primary cancers.

<!-- ROADMAP -->
## Road Map
- [X] Acquire patient data
- [ ] Find correct sequences for each patient (currently working on this)
- [ ] Preprocessing (currently working on this)
    - [X] Extract dicom metadata
    - [X] Convert dicom to nifti
    - [ ] Extract brain (currently working on this)
    - [X] Fill holes
    - [X] Binary Segment
    - [X] Cropy images
    - [X] Bias correction
    - [X] Coregister images
    - [X] Resample images
    - [X] Z-score normalization
- [ ] Write ai using tensorflow
- [ ] Train ai
- [ ] Explore results