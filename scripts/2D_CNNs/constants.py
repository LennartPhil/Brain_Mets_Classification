import os
from enum import Enum, auto
from pathlib import Path

class Environment(Enum):
    LOCAL = auto()
    DOCKER = auto()
    UKR_AI_SERVER = auto()

ENVIRONMENT: Environment = Environment.DOCKER

if ENVIRONMENT == Environment.DOCKER:
    # --- Docker Paths ---
    path_to_logs = Path("/logs")
    path_to_tfr_dirs = Path("/tfrs")
    print(f"INFO: Using Docker paths for logs and tfrecords: {path_to_logs.resolve()}, {path_to_tfr_dirs.resolve()}")

elif ENVIRONMENT == Environment.UKR_AI_SERVER:
    # --- UKR AI Server Paths ---
    # non-docker setup

    # Read the base path from the environment variable "PROJECT_BASE_DIR"
    # If not set, use the old relative path
    base_dir_str = os.getenv("PROJECT_BASE_DIR", "/home/lennart")
    path_to_base = Path(base_dir_str)

    path_to_logs = path_to_base / "logs"
    path_to_tfr_dirs = path_to_base / "tfrs"

    print(f"INFO: Using project base directory: {path_to_base.resolve()}")

elif ENVIRONMENT == Environment.LOCAL:
    # --- Local Paths ---
    path_to_logs = Path("/Users/LennartPhilipp/Desktop/Uni/Prowiss/Training/training_evaluation_09_25")
    path_to_tfr_dirs = Path("/Users/LennartPhilipp/Desktop/Uni/Prowiss/Datensatz_RGB/regensburg_slices_tfrecords")
    print(f"INFO: Using local paths for logs and tfrecords: {path_to_logs.resolve()}, {path_to_tfr_dirs.resolve()}")

else:
    raise ValueError(f"Unknown environment: {ENVIRONMENT}, Please set ENVIRONMENT to one of the defined Environment enum values (LOCAL, DOCKER, UKR_AI_SERVER)")


# --- Path constants ---
# to use for AI server

path_to_splits = path_to_tfr_dirs / "split_text_files"
paths_to_rough_pretraining = path_to_tfr_dirs / "rough_pretraining" / "rough_train.tfrecord", path_to_tfr_dirs / "rough_pretraining" / "rough_val.tfrecord"
path_to_fine_pretraining = path_to_tfr_dirs / "fine_pretraining"

# to use for local / docker development
#path_to_logs = Path("/logs")
#path_to_tfr_dirs = Path("/tfrs")
#path_to_splits = path_to_tfr_dirs / Path("split_text_files") # Directory containing train/val/test split files
#paths_to_rough_pretraining = "/rough_pretraining/rough_train.tfrecord", "/rough_pretraining/rough_val.tfrecord"
#path_to_fine_pretraining = Path("/fine_pretraining")

# --- Data setup constants ---
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

shuffle_buffer_size = 200
repeat_count = 1

AGE_MIN = 0
AGE_MAX = 110

LAYER_MIN = 0
LAYER_MAX = 154

IMG_SIZE = 240

ROUGH_NUM_IMAGES = 3 # 3 rgb images

# --- Sequence / Channel Mapping ---
SEQUENCE_TO_INDEX = {
    "t1": 0,
    "t1c": 1,
    "t2": 2,
    "flair": 3,
    "mask": 4,
}
# Example: selected_sequences = ["t1c", "flair", "mask"] -> indices = [1, 3, 4]

# --- Training constants ---
kernel_initializer = "he_normal"
activation_func = "mish"

early_stopping_patience = 300 #200
early_stopping_patience_upper_layer = 20

LEARNING_RATE_EPOCHS = 400
MAX_TRAINING_EPOCHS = 2500

REGULAR_DROPOUT_RATE = 0.4
REGULAR_L2_REGULARIZATION = 0.0001

# --- Model constants ---
# INPUT_SHAPE_RGB = (IMG_SIZE, IMG_SIZE, 3)
# INPUT_SHAPE_4_SEQ = (IMG_SIZE, IMG_SIZE, 4)
# INPUT_SHAPE_4_SEQ_MASK = (IMG_SIZE, IMG_SIZE, 5) 

# --- Weights constants ---
#two_class_weights = {1: 0.92156863, 0 :1.09302326}
class_weights_dict = {
    2: {0: 1.0710093896713615, 1: 0.9378211716341213},
    3: {0: 1.1223862238622386, 1: 0.6252141144227475, 2: 1.9623655913978495},
    4: {0: 1.683579335793358, 1: 0.46891058581706063, 2: 1.471774193548387, 3: 1.683579335793358},
    5: {0: 2.122093023255814, 1: 0.3751284686536485, 2: 1.1774193548387097, 3: 1.3468634686346863, 4: 3.686868686868687},
    6: {0: 3.306159420289855, 1: 0.3126070572113738, 2: 0.9811827956989247, 3: 1.1223862238622386, 4: 3.0723905723905722, 5: 3.8020833333333335}
}
normal_two_class_weights = {0: 1.09302326, 1: 0.92156863}
rough_class_weights = {0: 0.7078619089062953, 1: 1.4294083186877562, 2: 1.126500461680517} # for the exact calculation see the preprocessing_brain_tumor_dataset.ipynb file
fine_two_class_weights = {0: 2.588372093023256, 1: 0.6197104677060133} # for the calculation see the other_dataset.ipynb file

# --- Type of dataset to use ---
class Dataset(Enum):
    NORMAL = auto()
    PRETRAIN_ROUGH = auto() # external brain_tumor_dataset
    PRETRAIN_FINE = auto() # external high quality brain leasion dataset

# --- Type of training to run ---
class Training(Enum):
    LEARNING_RATE_TUNING = auto()
    NORMAL = auto() # Standard training on one split
    K_FOLD = auto() # K-Fold cross-validation
    UPPER_LAYER = auto() # Transfer Learning