# Cloud ICH subtype classifier (RSNA 2019)

Multi-label classification of intracranial hemorrhage subtypes
(`epidural, intraparenchymal, intraventricular, subarachnoid, subdural, any`)
from non-contrast head CT, built to run on **Google Colab** or **Kaggle Notebooks**.

## 1. Get the dataset
Official source: Kaggle competition **`rsna-intracranial-hemorrhage-detection`**.
You must sign in to Kaggle and accept the competition rules once, then either:

- **On Kaggle Notebooks (easiest):** click *Add Data* → the competition. The files
  mount at `/kaggle/input/rsna-intracranial-hemorrhage-detection` with **no
  download** — `config.py` already points there.
- **On Colab:** use the Kaggle API.
  ```bash
  pip install kaggle
  mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
  kaggle competitions download -c rsna-intracranial-hemorrhage-detection
  ```
  The full DICOM set is ~180 GB, so on Colab keep `SUBSET_N_PATIENTS` set (see below).

## 2. Where to run — honest guidance
- **Full run → Kaggle Notebooks.** Data is pre-mounted, ~30 GPU-hrs/week free.
  Set `SUBSET_N_PATIENTS = None` and `DATA_ROOT` to the mounted path.
- **Prototyping → Colab (free or Pro).** Free Colab can't hold 180 GB, so keep
  `SUBSET_N_PATIENTS` at a few thousand to get a working end-to-end pipeline, then
  scale up on Kaggle. Colab Pro adds longer runtimes and better GPUs (L4/A100).

## 3. Install & run
```bash
pip install -r requirements.txt

python prepare_data.py                       # labels + patient-level split (run once)
python train.py --model resnet50             # baseline
python train.py --model efficientnet_b0      # efficient / deployable model
python evaluate.py --model efficientnet_b0 --threshold 0.5
```

## 4. Which model?
- **`efficientnet_b0`** — recommended primary. Best accuracy-per-FLOP, so it fits the
  low-resource-deployment goal. Use `efficientnet_b3` if you have compute to spare.
- **`resnet50`** — the standard strong baseline everyone compares against.

The backbone matters less than the **3-window preprocessing** (`dataset.py`) and
per-patient splitting. If you later want a real accuracy jump, add **2.5D input**
(neighbouring slices) — that helps the rare extra-axial subtypes (EDH/SDH) most.

## 5. Using Claude Code on this repo
Open the folder in Claude Code and start with:
> "Read CLAUDE.md, then review dataset.py and prepare_data.py for correctness."

`CLAUDE.md` lists the invariants Claude Code must not break (patient-level split,
multi-label, matched configs) and a list of good next tasks.
