"""
Build the label table and a PATIENT-LEVEL train/val/test split.

Why patient-level: slices from one patient are highly correlated. If some of a
patient's slices land in train and others in test, the model "cheats" and your
metrics are inflated. Splitting by PatientID is the single most important
validity control in this whole project. Do not change it to a random slice split.

Run once:  python prepare_data.py
Outputs:   work/labels.csv   (image_id + 6 binary columns + patient_id)
           work/splits.csv   (image_id -> split in {train,val,test})
"""
import multiprocessing as mp
from functools import partial

import numpy as np
import pandas as pd
import pydicom
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

import config as C


def load_wide_labels() -> pd.DataFrame:
    """stage_2_train.csv is long-format: 'ID_<img>_<subtype>, Label'. Pivot to wide."""
    df = pd.read_csv(C.TRAIN_CSV)
    df = df.drop_duplicates(subset="ID")                       # known dup rows
    df["subtype"] = df["ID"].str.rsplit("_", n=1).str[1]
    df["image_id"] = df["ID"].str.rsplit("_", n=1).str[0]
    wide = df.pivot(index="image_id", columns="subtype", values="Label")
    wide = wide[C.SUBTYPES].reset_index()                      # fixed column order
    return wide


def _read_patient_id(image_id: str):
    path = C.TRAIN_DIR / f"{image_id}.dcm"
    try:
        d = pydicom.dcmread(str(path), stop_before_pixels=True)  # header only = fast
        return image_id, str(d.PatientID)
    except Exception:
        return image_id, None                                   # corrupt/missing


def attach_patient_ids(wide: pd.DataFrame) -> pd.DataFrame:
    """Read DICOM headers to get PatientID for every slice. Cached to disk."""
    cache = C.WORK_DIR / "image_to_patient.csv"
    if cache.exists():
        mapping = pd.read_csv(cache)
    else:
        ids = wide["image_id"].tolist()
        with mp.Pool(processes=max(1, mp.cpu_count())) as pool:
            rows = list(tqdm(pool.imap(_read_patient_id, ids, chunksize=256),
                             total=len(ids), desc="Reading DICOM headers"))
        mapping = pd.DataFrame(rows, columns=["image_id", "patient_id"])
        mapping.to_csv(cache, index=False)
    out = wide.merge(mapping, on="image_id", how="inner")
    out = out.dropna(subset=["patient_id"]).reset_index(drop=True)
    return out


def subset_patients(df: pd.DataFrame) -> pd.DataFrame:
    if C.SUBSET_N_PATIENTS is None:
        return df
    rng = np.random.RandomState(C.SEED)
    patients = df["patient_id"].unique()
    keep = rng.choice(patients, size=min(C.SUBSET_N_PATIENTS, len(patients)),
                      replace=False)
    return df[df["patient_id"].isin(keep)].reset_index(drop=True)


def make_split(df: pd.DataFrame) -> pd.DataFrame:
    """Two GroupShuffleSplits by patient: first carve test, then val from remainder."""
    groups = df["patient_id"].values
    gss1 = GroupShuffleSplit(n_splits=1, test_size=C.TEST_FRAC, random_state=C.SEED)
    trainval_idx, test_idx = next(gss1.split(df, groups=groups))
    df.loc[df.index[test_idx], "split"] = "test"

    tv = df.iloc[trainval_idx]
    val_rel = C.VAL_FRAC / (1 - C.TEST_FRAC)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_rel, random_state=C.SEED)
    tr_idx, va_idx = next(gss2.split(tv, groups=tv["patient_id"].values))
    df.loc[tv.index[tr_idx], "split"] = "train"
    df.loc[tv.index[va_idx], "split"] = "val"

    # sanity: no patient appears in more than one split
    overlap = df.groupby("patient_id")["split"].nunique().max()
    assert overlap == 1, "Patient leakage across splits!"
    return df


def main():
    wide = load_wide_labels()
    wide = attach_patient_ids(wide)
    wide = subset_patients(wide)
    wide = make_split(wide)

    wide.to_csv(C.WORK_DIR / "labels.csv", index=False)
    wide[["image_id", "split"]].to_csv(C.WORK_DIR / "splits.csv", index=False)

    print(f"Patients: {wide['patient_id'].nunique()}  |  Slices: {len(wide)}")
    print(wide["split"].value_counts())
    print("\nPositive slices per subtype (train split):")
    print(wide[wide.split == "train"][C.SUBTYPES].sum().astype(int))


if __name__ == "__main__":
    main()
