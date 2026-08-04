"""
Central configuration. Edit values HERE only; every other module imports from this.
Keeping one config is what makes the ResNet vs EfficientNet comparison fair:
both models read the exact same split, image size, epochs, and augmentation.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Data location
# --------------------------------------------------------------------------
# On KAGGLE Notebooks the competition data is already mounted here (no download):
#   /kaggle/input/rsna-intracranial-hemorrhage-detection
# On COLAB you download it with the Kaggle API (see README) into this path.
DATA_ROOT = Path("/kaggle/input/rsna-intracranial-hemorrhage-detection")
TRAIN_DIR = DATA_ROOT / "stage_2_train"      # folder of .dcm files
TRAIN_CSV = DATA_ROOT / "stage_2_train.csv"  # long-format labels

WORK_DIR = Path("./work")                    # caches / splits / checkpoints
WORK_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Labels  (order is fixed and used everywhere -> do not reorder)
# --------------------------------------------------------------------------
SUBTYPES = ["epidural", "intraparenchymal", "intraventricular",
            "subarachnoid", "subdural", "any"]
N_CLASSES = len(SUBTYPES)

# --------------------------------------------------------------------------
# Subsetting  (CRUCIAL on Colab)
# --------------------------------------------------------------------------
# Full set is ~19k patients / ~750k slices (~180 GB of DICOM) and will NOT fit
# on free Colab disk. Sample this many PATIENTS (patient-level, no leakage).
# Set to None to use everything (only on Kaggle Notebooks or a big-disk machine).
SUBSET_N_PATIENTS = 3000
SEED = 42

# --------------------------------------------------------------------------
# Patient-level split fractions
# --------------------------------------------------------------------------
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# --------------------------------------------------------------------------
# Image / windowing  (3 CT windows -> 3 channels, the standard ICH recipe)
# --------------------------------------------------------------------------
IMG_SIZE = 256
WINDOWS = [(40, 80),    # brain
           (80, 200),   # subdural / blood
           (40, 380)]   # soft tissue

# --------------------------------------------------------------------------
# Training  (IDENTICAL for every model -> that is the whole point)
# --------------------------------------------------------------------------
BATCH_SIZE = 32
EPOCHS = 8
LR = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2
USE_AMP = True          # mixed precision -> ~2x faster on Colab T4
