from enum import Enum, auto
from pathlib import Path

# --- Path constants ---
path_to_logs = Path("/logs")
path_to_tfr_dirs = Path("/tfrs")
path_to_splits = path_to_tfr_dirs / Path("split_text_files") # Directory containing train/val/test split files
paths_to_rough_pretraining = path_to_tfr_dirs / "rough_train.tfrecord", path_to_tfr_dirs / "rough_val.tfrecord" #"/tfrs/rough_pretraining/train.tfrecord", "tfrs/rough_pretraining/val.tfrecord"

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

# --- Model constants ---
# INPUT_SHAPE_RGB = (IMG_SIZE, IMG_SIZE, 3)
# INPUT_SHAPE_4_SEQ = (IMG_SIZE, IMG_SIZE, 4)
# INPUT_SHAPE_4_SEQ_MASK = (IMG_SIZE, IMG_SIZE, 5) 

# --- Weights constants ---
#two_class_weights = {1: 0.92156863, 0 :1.09302326}
normal_two_class_weights = {0: 1.09302326, 1: 0.92156863}

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