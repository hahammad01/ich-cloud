"""
DICOM -> 3-window image -> tensor, plus the PyTorch Dataset.

The three-window trick (brain / blood / soft tissue on the R,G,B channels) is
where most of the accuracy in ICH classification actually comes from -- more than
the backbone choice. Get this right before worrying about models.
"""
import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import cv2

import config as C


def _get_hu(dcm) -> np.ndarray:
    """Convert stored pixels to Hounsfield Units, correcting the known RSNA
    intercept bug (some slices ship with a wrong RescaleIntercept)."""
    img = dcm.pixel_array.astype(np.float32)
    slope = float(getattr(dcm, "RescaleSlope", 1))
    intercept = float(getattr(dcm, "RescaleIntercept", 0))
    # RSNA fix: a subset of images store 12-bit pixels with intercept ~ -1000.
    if intercept > -100:
        return img * slope + intercept
    corrected = img + 1000
    corrected[corrected >= 4096] -= 4096
    return corrected * slope - 1000


def _window(hu: np.ndarray, wl: float, ww: float) -> np.ndarray:
    lo, hi = wl - ww / 2.0, wl + ww / 2.0
    x = np.clip(hu, lo, hi)
    return (x - lo) / (hi - lo + 1e-6)          # -> 0..1


def dicom_to_image(path: str) -> np.ndarray:
    """Return an (IMG_SIZE, IMG_SIZE, 3) float32 image in [0,1]."""
    dcm = pydicom.dcmread(path)
    hu = _get_hu(dcm)
    chans = [_window(hu, wl, ww) for wl, ww in C.WINDOWS]
    img = np.stack(chans, axis=-1).astype(np.float32)
    img = cv2.resize(img, (C.IMG_SIZE, C.IMG_SIZE), interpolation=cv2.INTER_AREA)
    return img


class ICHDataset(Dataset):
    def __init__(self, labels_df: pd.DataFrame, split: str, train: bool):
        self.df = labels_df[labels_df["split"] == split].reset_index(drop=True)
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        path = str(C.TRAIN_DIR / f"{row['image_id']}.dcm")
        try:
            img = dicom_to_image(path)
        except Exception:
            img = np.zeros((C.IMG_SIZE, C.IMG_SIZE, 3), dtype=np.float32)
        x = torch.from_numpy(img).permute(2, 0, 1)             # C,H,W

        if self.train:                                          # light augmentation
            if torch.rand(1).item() < 0.5:
                x = TF.hflip(x)
            angle = float(torch.empty(1).uniform_(-10, 10).item())
            x = TF.rotate(x, angle)

        y = torch.tensor(row[C.SUBTYPES].values.astype(np.float32))
        return x, y
