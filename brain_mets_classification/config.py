# modified path
path_to_n30 = "/Users/LennartPhilipp/Desktop/Uni/Prowiss/Code/Brain_Mets_Classification/Rgb_Brain_Mets_Dataset/N30"

# path to SSD
path_to_ssd = "/Volumes/BrainMets/Rgb_Brain_Mets/allPatients"

# path to revision
path_to_revision = "/Volumes/BrainMets/Rgb_Brain_Mets/brain_mets_revision"

path_to_dataset = "/Volumes/BrainMets/Rgb_Brain_Mets/Rgb_Brain_Mets_Dataset"

dsStore = ".DS_Store"

#desiredSequences = ["FLAIR", "T1", "T1CE", "T2"]
from enum import Enum
class desiredSequences(Enum):
    T1 = "T1"
    T1CE = "T1CE"
    T2 = "T2"
    FLAIR = "FLAIR"
    STERN = "STERN"
    DWI = "DWI"
    ADC = "ADC"
    MPR = "MPRAGE"

class patient:

    def __init__(self, id):
        self.patientID = id
        self.T1_sequences: list[str] = []
        self.T1_amount: int = 0
        self.T1CE_sequences: list[str] = []
        self.T1CE_amount: int = 0
        self.T2_sequences: list[str] = []
        self.T2_amount: int = 0
        self.FLAIR_sequences: list[str] = []
        self.FLAIR_amount: int = 0
        self.STERN_sequences: list[str] = []
        self.STERN_amount: int = 0
        self.DWI_sequences: list[str] = []
        self.DWI_amount: int = 0
        self.ADC_sequences: list[str] = []
        self.ADC_amount: int = 0
        self.MPR_sequences: list[str] = []
        self.MPR_amount: int = 0

        self.other_sequences: list[str] = []
        self.other_amount: int = 0

        self.birthdate: str = ""
        self.mri_date: str = ""
        self.age: int = 0
        self.sex: str = ""
        self.primary: str = ""

    def __str__(self):
        return f"""
id: {self.patientID}
T1 sequences: {self.T1_sequences} ({self.T1_amount})
T1CE sequences: {self.T1CE_sequences} ({self.T1CE_amount})
T2 sequences: {self.T2_sequences} ({self.T2_amount})
FLAIR sequences: {self.FLAIR_sequences} ({self.FLAIR_amount})
STERN sequences: {self.STERN_sequences} ({self.STERN_amount})
DWI sequences: {self.DWI_sequences} ({self.DWI_amount})
ADC sequences: {self.ADC_sequences} ({self.ADC_amount})
MRP sequences: {self.MPR_sequences} ({self.MPR_amount})
birthdate: {self.birthdate}, mri date: {self.mri_date}
age: {self.age}, sex: {self.sex}, primary: {self.primary}
"""


seq_whitelist = ["ax", "axial", "tra", "transversal", "diff", "adc", "dti"]
seq_blacklist = ["lws", "hand", "roi", "scout", "hws", "bws", "sag", "posdisp", "cor"]

FLAIRlist = ['flair', 'tirm', '110b19']
T1list = ['t1']
KMlist = ['km', 'contrast', 'post', 'ce', 'enhaced', '+', '17rb130', 'rephasiert', 'dotarem']
T2list = ['t2']
T2STERNlist = ['t2stern', 'ciss', 'blutung', 'stern']
DWIlist = ['diffusion', 'diff', 'dwi']
ADClist = ['adc']
sub = ['sub']


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
                            '1f': 'lung cancer, other',
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
                            '25': 'MiNEN', # = mixed neuroendocrine non-neuroendocine neuroplasms
                            '26a': 'DSRCT', # = Desmoplastic small-round-cell Tumor
                            '26b': 'mesothelioma, other',
                            '27': 'thymoma',
                            '28': 'plasmacytoma',
                            '29': 'cancer of unknown primary'
}