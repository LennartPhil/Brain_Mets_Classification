'''
The primaries_shortcuts_brats was originally used in this project, but has now been replaced by the cancer_primaries_dict.
The former version was a modified dict, which had been used by the BRATS dataset.
'''

primaries_shortcuts_brats = { '1a': 'small cell lung cancer',
                        '1b': 'squamous cell lung cancer',
                        '1c': 'adenocarcinoma of the lung',
                        '1d': 'large cell lung cancer',
                        '1e': 'lung cancer, other',
                        '1f': 'lung cancer, exact histology unknown',
                        '1g': 'NSCLC',
                        '2': 'breast cancer',
                        '3': 'melanoma',
                        '4': 'prostate cancer',
                        '5': 'renal cell carcinoma',
                        '6': 'transitional cell carcinoma', # also called urothelial carcinoma
                        '7': 'colorectal cancer',
                        '8': 'plasmacytoma',
                        '9': 'testicular cancer',
                        '10': 'thyroid cancer',
                        '11': 'laryngeal cancer',
                        '12': 'cervical cancer',
                        '13': 'cholangiocellular carcinoma',
                        '14': 'ovarian cell carcinoma',
                        '15': 'rhabdomyosarcoma',
                        '16': 'gastric cancer',
                        '17': 'endometrial cancer',
                        '18': 'osteosarcoma',
                        '20': 'esophageal cancer',
                        '21': 'oral squamous cell carcinoma',
                        '22': 'synovial cell carcinoma',
                        '23': 'leiomyosarcoma',
                        '24': 'vulvar carcinoma',
                        '25': 'salivary gland cancer',
                        '26': 'cancer of unknown primary',
}


'''
The cancer_primaries_dict is sorted by the most common sources of brain metastases, i.e.
lung cancer (1a-1g),
breast cancer (2),
genitourinary tract cancers (3-10),
sarcoma (11a-h),
melanoma (12),
head and neck cancer (13-18),
gastrointestinal cancers (19-25)

at the end are other cancer primaries like
- cancers originating from mesothelium (26a-b),
- thymoma (27)
- plasmocytoma (28)

and lastly CUP (29)
'''
cancer_primaries_dict = {   '1a': 'small cell lung cancer',
                            '1b': 'squamous cell lung cancer',
                            '1c': 'adenocarcinoma of the lung',
                            '1d': 'large cell lung cancer',
                            '1e': 'NSCLC',
                            '1e': 'lung cancer, other',
                            '1g': 'lung cancer, exact histology unknown',
                            '2': 'breast cancer',
                            '3': 'prostate cancer',
                            '4': 'testicular cancer',
                            '5': 'renal cell carcinoma',
                            '6': 'transitional cell carcinoma', # also called urothelial carcinoma
                            '7': 'vulvar carcinoma',
                            '8': 'cervical cancer',
                            '9': 'endometrial cancer',
                            '10': 'ovarian cell carcinoma',
                            '11a': 'osteosarcoma',
                            '11b': 'liposarcoma',
                            '11c': 'leiomyosarcoma',
                            '11d': 'rhabdomyosarcoma',
                            '11e': 'synovial sarcoma',
                            '11f': 'solitary fibrous tumor',
                            '11g': 'sarcoma, other',
                            '11h': 'sarcoma, exact histology unknown',
                            '12': 'melanoma',
                            '13': 'lacrimal gland carcinoma',
                            '14': 'salivary gland cancer',
                            '15': 'oral squamous cell carcinoma',
                            '16': 'adenoid cystic carcinoma',
                            '17': 'laryngeal cancer',
                            '18': 'thyroid cancer',
                            '19': 'esophageal cancer',
                            '20': 'gastric cancer',
                            '21': 'pancreatic cancer',
                            '22': 'cholangiocellular carcinoma',
                            '23': 'colorectal cancer',
                            '24': 'NET', # = neuroendocrine tumors
                            '25': 'MiNEN', # = mixed neuroendocrine non-neuroendocine neurplasms
                            '26a': 'DSRCT', # = Desmoplastic small-round-cell Tumor
                            '26b': 'mesothelioma, other',
                            '27': 'thymoma',
                            '28': 'plasmacytoma',
                            '29': 'cancer of unknown primary'
}