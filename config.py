"""
Central configuration. Edit values HERE only; every other module imports from this.
Keeping one config is what makes the ResNet vs EfficientNet comparison fair:
both models read the exact same split, image size, epochs, and augmentation.
"""
import glob
from pathlib import Path

# --------------------------------------------------------------------------
# Data location  (auto-detected)
# --------------------------------------------------------------------------
# Kaggle mounts competitions in different places depending on how they're added
# (e.g. /kaggle/input/<slug> OR /kaggle/input/competitions/<slug>), so instead of
# hard-coding a path we search a few shallow locations for stage_2_train.csv.
def _find_data_root() -> Path:
    preferred = Path("/kaggle/input/rsna-intracranial-hemorrhage-detection")
    if (preferred / "stage_2_train.csv").exists():
        return preferred
    # bounded-depth globs -> fast, never descends into the 750k-image folder
    for pattern in ("/kaggle/input/*/stage_2_train.csv",
                    "/kaggle/input/*/*/stage_2_train.csv",
                    "/kaggle/input/*/*/*/stage_2_train.csv"):
        hits = glob.glob(pattern)
        if hits:
            return Path(hits[0]).parent
    return preferred        # nothing found; scripts will raise a clear error

DATA_ROOT = _find_data_root()


def _find_train_dir(root: Path) -> Path:
    """Locate the folder of training .dcm files. Names vary across mounts
    (stage_2_train_images / stage_1_train_images / stage_2_train / ...), so try
    the common names, then fall back to any subfolder that actually holds .dcm."""
    for name in ("stage_2_train_images", "stage_1_train_images",
                 "stage_2_train", "stage_1_train", "train_images", "train"):
        p = root / name
        if p.is_dir():
            return p
    if root.is_dir():
        subdirs = sorted((d for d in root.iterdir() if d.is_dir()),
                         key=lambda d: (0 if "train" in d.name.lower() else 1, d.name))
        for d in subdirs:
            if any(True for _ in d.glob("*.dcm")):   # lazy: stops at first hit
                return d
    return root / "stage_2_train_images"             # fallback


TRAIN_DIR = _find_train_dir(DATA_ROOT)               # folder of .dcm files
TRAIN_CSV = DATA_ROOT / "stage_2_train.csv"          # long-format labels

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
SUBSET_N_PATIENTS = None
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
EPOCHS = 2
LR = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2
USE_AMP = True          # mixed precision -> ~2x faster on Colab T4
